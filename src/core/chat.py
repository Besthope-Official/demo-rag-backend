"""会话管理"""

import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator

from core.profile import ProfileService
from loguru import logger
from pymongo.database import Database as MongoDatabase

from src.cache import Cache
from src.llm import LLMClient
from src.predoc import PredocClient
from src.schema import (
    AnswerResponse,
    Attachment,
    ChatStatus,
    Chunk,
    EndResponse,
    InitResponse,
    Message,
    QueryRewriteResponse,
    SearchResponse,
)
from src.utils import cosine_similarity

from .history import HistoryService
from .prompt import (
    PROFILE_ANSWER_PROMPT,
    QUERY_REWRITE_PROMPT,
    RAG_ANSWER_PROMPT,
    SYSTEM_PROMPT,
    TITLE_GENERATION_PROMPT,
)
from .user import UserDAO


class ChatInterface(ABC):
    """抽象的聊天接口层。"""

    def __init__(
        self,
        cache: Cache,
        llm_client: LLMClient | None = None,
        mongo_db: MongoDatabase | None = None,
        **_,
    ):
        self.cache: Cache = cache
        self.llm_client: LLMClient | None = llm_client
        self.storage = UserDAO(database=mongo_db)
        self.history_service = HistoryService(self.storage)

    async def _set_chat_status(self, chat_id: str, status: ChatStatus):
        await self.cache.set(chat_id, status.value)

    async def generate(
        self,
        chat_id: str,
        message_id: str,
        messages: list[Message] | None = None,
        username: str | None = "anonymous",
    ) -> AsyncGenerator[str, None]:
        """
        生成流式会话，标记会话初始化与结束阶段，内部业务 `chat_workflow` 在子类实现
        """
        init_resp = InitResponse.create(chat_id=chat_id, message_id=message_id)
        if chat_id and self.cache:
            await self._set_chat_status(chat_id, ChatStatus.ACTIVE)
        yield init_resp.to_jsonl()

        try:
            async for res in self.chat_workflow(
                chat_id=chat_id, messages=messages, username=username
            ):
                if res is not None:
                    yield res
        except Exception as e:
            logger.error(f"error: {e}")
            await self._set_chat_status(chat_id, ChatStatus.TERMINATED)
        finally:
            chat_status = await self.cache.get(chat_id)
            if str(chat_status) != str(ChatStatus.TERMINATED.value):
                await self._set_chat_status(chat_id, ChatStatus.COMPLETED)
            if messages:
                title = await self.generate_chat_title(messages[-1].content)
                await self.history_service.save_chat_history(username, chat_id, messages, title)
            yield EndResponse.from_status(chat_status).to_jsonl()

    async def halt_chat(self, chat_id: str) -> None:
        if not chat_id:
            raise ValueError("Missing chat_id for halt operation")
        try:
            await self._set_chat_status(chat_id, ChatStatus.TERMINATED)
            logger.info(f"Chat {chat_id} has been terminated")
        except Exception as e:
            logger.error(f"Failed to halt chat {chat_id}: {str(e)}")
            raise

    async def generate_chat_title(self, query: str) -> str:
        """为一段会话总结标题"""
        if not query:
            logger.warning("generate_chat_title 收到空 query，返回默认标题")
            return "空查询"
        if not self.llm_client:
            logger.error("llm_client 未初始化")
            raise ValueError("llm_client is not configured")

        try:
            prompt = TITLE_GENERATION_PROMPT.format(compact=query)
            messages = [Message(role="user", content=prompt)]
            logger.debug(f"生成标题 prompt: {prompt[:100]}...")
            title = await self.llm_client.generate(messages=messages)
            idx = title.lower().find("</think>")
            if idx != -1:
                title = title[idx + len("</think>") :].strip()
                logger.debug(f"去除 </think> 后标题: {title}")
            else:
                logger.debug("标题中无 </think> 标签，使用原始标题")
            title = (
                title.strip()
                .replace("。", "")
                .replace("，", "")
                .replace("！", "")
                .replace("？", "")[:15]
            )
            logger.info(f"生成标题: {title}")
            return title if title else "无标题"
        except Exception as e:
            logger.error(f"生成标题失败: {e}")
            return "生成失败"

    @abstractmethod
    async def chat_workflow(
        self, chat_id: str, messages: list[Message] | None = None, username: str | None = None
    ) -> AsyncGenerator[str, None]:
        """聊天工作流程，返回流式响应"""
        raise NotImplementedError()


class ChatService(ChatInterface):
    """无 RAG 模式的聊天业务"""

    async def chat_workflow(
        self, chat_id: str, messages: list[Message] | None = None, username: str | None = None
    ) -> AsyncGenerator[str, None]:
        if not chat_id:
            raise ValueError("Missing chat_id for chat workflow")
        try:
            if self.llm_client is None:
                raise ValueError("llm_client is not configured")
            assistant_content = ""
            async for chunk in self.llm_client.generate_stream(messages=messages):
                chat_status = await self.cache.get(chat_id)
                if str(chat_status) == str(ChatStatus.ACTIVE.value):
                    text = chunk.encode("utf-8").decode("utf-8", errors="replace")
                    yield AnswerResponse.from_text(text).to_jsonl()
                    assistant_content += text
                else:
                    yield None
                    break
            if assistant_content:
                messages.append(Message(role="assistant", content=assistant_content))
        except Exception as e:
            logger.error(f"生成流式答案时出错: {str(e)}")
            if self.cache:
                await self.cache.set(chat_id, ChatStatus.TERMINATED.value)
            yield None


class RAGService(ChatInterface):
    """RAG 模式的聊天业务"""

    def __init__(self, cache, llm_client=None, predoc_client=None, mongo_db=None, **_):
        self.predoc_client = predoc_client
        super().__init__(cache, llm_client, mongo_db, **_)

    def context_inject(self, origin_query: str, rewritten_query: str, context: list[Chunk]) -> str:
        template = f"""用户原始问题：{origin_query}\n重写问题：{rewritten_query}\n上下文信息：\n"""
        for chunk in context:
            template += f"- {chunk.text}\n"
        return template

    async def remove_think(self, text: str) -> str:
        idx = text.lower().find("</think>")
        return text[idx + len("</think>") :] if idx != -1 else text

    async def chat_workflow(
        self, chat_id: str, messages: list[Message] | None = None, username: str | None = None
    ) -> AsyncGenerator[str, None]:
        if not chat_id:
            raise ValueError("Missing chat_id for chat workflow")
        try:
            if self.llm_client is None:
                raise ValueError("llm_client is not configured")
            origin_user_query = messages[0].content
            rewritten_query = ""
            async for chunk in self.query_rewrite(origin_user_query):
                chat_status = await self.cache.get(chat_id)
                if str(chat_status) == str(ChatStatus.ACTIVE.value):
                    text = chunk.encode("utf-8").decode("utf-8", errors="replace")
                    yield QueryRewriteResponse.from_text(text).to_jsonl()
                    rewritten_query += text
                else:
                    yield None
                    break
            rewritten_query = await self.remove_think(rewritten_query)
            messages.append(Message(role="system", content=rewritten_query))
            attachment = await self.search_relevant_docs(rewritten_query)
            yield SearchResponse.from_attachment(attachment).to_jsonl()
            messages.append(Message(role="system", content="", attachment=attachment))
            context = attachment.chunks
            prompt = self.context_inject(origin_user_query, rewritten_query, context)
            final_query = RAG_ANSWER_PROMPT.format(context=prompt)
            assistant_content = ""
            async for chunk in self.llm_client.generate_stream(
                messages=[Message(role="user", content=final_query)]
            ):
                chat_status = await self.cache.get(chat_id)
                if str(chat_status) == str(ChatStatus.ACTIVE.value):
                    text = chunk.encode("utf-8").decode("utf-8", errors="replace")
                    yield AnswerResponse.from_text(text).to_jsonl()
                    assistant_content += text
                else:
                    yield None
                    break
            if assistant_content:
                messages.append(Message(role="assistant", content=assistant_content))
        except Exception as e:
            logger.error(f"生成流式答案时出错: {str(e)}")
            if self.cache:
                await self.cache.set(chat_id, ChatStatus.TERMINATED.value)
            yield None

    async def search_relevant_docs(self, query: str) -> Attachment:
        try:
            client = self.predoc_client
            return client.search(query, topK=10)
        except Exception as e:
            logger.warning(f"Predoc 搜索失败: {e}")
            return Attachment(doc=[], chunks=[])

    async def query_rewrite(self, query: str) -> AsyncGenerator[str, None]:
        prompt = QUERY_REWRITE_PROMPT.format(query=query)
        messages = [Message(role="user", content=prompt)]
        async for response in self.llm_client.generate_stream(messages=messages):
            yield response


class PersonaMixin:
    def __init__(self, cache, predoc_client, mongo_db=None):
        self.cache = cache
        self.predoc_client = predoc_client
        self.profile_service = ProfileService(self.cache, self.predoc_client, mongo_db)

    async def profile_inject(self, username: str, query: str) -> str:
        if not username:
            logger.warning(f"未提供 username: {username}")
            return ""
        if not self.profile_service:
            logger.warning("profile_service 未初始化，跳过画像注入")
            return ""
        try:
            logger.info(f"开始为用户 {username} 生成画像和补充信息，query={query}")
            label_processor = self.profile_service.label_processor
            user = self.profile_service.storage.get_user(username)
            if not user:
                logger.error(f"用户 {username} 不存在")
                return ""
            profile_key = f"profile:{username}"
            label_key = f"label_results:{username}"
            supp_key = f"supplement:{username}"
            profile_cached = await self.cache.get(profile_key)
            label_cached = await self.cache.get(label_key)
            supp_cached = await self.cache.get(supp_key)

            def _to_attachment(val):
                if val is None:
                    return None
                if isinstance(val, Attachment):
                    return val
                if isinstance(val, dict):
                    try:
                        return Attachment.model_validate(val)
                    except Exception:
                        chunks = []
                        for c in (
                            val.get("chunks", []) if isinstance(val.get("chunks", []), list) else []
                        ):
                            if isinstance(c, str):
                                chunks.append(Chunk(text=c))
                            elif isinstance(c, dict):
                                try:
                                    chunks.append(Chunk.model_validate(c))
                                except Exception:
                                    try:
                                        chunks.append(Chunk(**c))
                                    except Exception:
                                        pass
                        return Attachment(doc=val.get("doc", []), chunks=chunks)
                if isinstance(val, list):
                    if all(isinstance(i, dict) for i in val):
                        try:
                            return Attachment.model_validate({"doc": [], "chunks": val})
                        except Exception:
                            chunks = []
                            for c in val:
                                try:
                                    chunks.append(Chunk.model_validate(c))
                                except Exception:
                                    try:
                                        chunks.append(Chunk(**c))
                                    except Exception:
                                        pass
                            return Attachment(doc=[], chunks=chunks)
                    if all(isinstance(i, str) for i in val):
                        return Attachment(doc=[], chunks=[Chunk(text=s) for s in val])
                    chunks = []
                    for item in val:
                        if isinstance(item, str):
                            chunks.append(Chunk(text=item))
                        elif isinstance(item, dict):
                            try:
                                chunks.append(Chunk.model_validate(item))
                            except Exception:
                                try:
                                    chunks.append(Chunk(**item))
                                except Exception:
                                    pass
                    return Attachment(doc=[], chunks=chunks)
                if isinstance(val, str):
                    try:
                        import json

                        parsed = json.loads(val)
                        return _to_attachment(parsed)
                    except Exception:
                        return Attachment(doc=[], chunks=[Chunk(text=val)])
                return None

            label_cached = _to_attachment(label_cached) if label_cached is not None else None
            supp_cached = _to_attachment(supp_cached) if supp_cached is not None else None
            profile = profile_cached if profile_cached is not None else None
            label_supplement = label_cached if label_cached is not None else None
            desc_supplement = supp_cached if supp_cached is not None else None

            tasks = []
            task_keys = []
            if profile is None:
                tasks.append(self.profile_service.generate_profile(user, label_processor))
                task_keys.append("profile")
            if label_supplement is None:
                tasks.append(self.profile_service.search_by_labels(username))
                task_keys.append("label")
            if desc_supplement is None:
                tasks.append(self.profile_service.get_supplement(username, label_processor))
                task_keys.append("supp")

            if tasks:
                try:
                    results = await asyncio.gather(*tasks)
                except Exception as e:
                    logger.error(f"生成画像/补充信息时出错: {e}")
                    results = [None] * len(tasks)

                for key, res in zip(task_keys, results, strict=False):
                    try:
                        if key == "profile":
                            profile = res or ""
                            await self.cache.set(profile_key, profile)
                        elif key == "label":
                            label_supplement = res or Attachment(doc=[], chunks=[])
                            await self.cache.set(label_key, label_supplement)
                        elif key == "supp":
                            desc_supplement = res or Attachment(doc=[], chunks=[])
                            await self.cache.set(supp_key, desc_supplement)
                    except Exception:
                        logger.warning(f"无法将 {key} 写入缓存: {username}")

            profile = profile or ""
            label_supplement = label_supplement or Attachment(doc=[], chunks=[])
            desc_supplement = desc_supplement or Attachment(doc=[], chunks=[])
            combined_chunks: list[Chunk] = label_supplement.chunks + desc_supplement.chunks
            logger.info(f"组合 chunks 总数: {len(combined_chunks)}")

            seen = set()
            unique_chunks: list[Chunk] = []
            for chunk in combined_chunks:
                text = chunk.text
                if text and text not in seen:
                    seen.add(text)
                    unique_chunks.append(chunk)
            logger.info(f"去重后 chunks 数量: {len(unique_chunks)}")

            chunk_scores: list[tuple[Chunk, float]] = []
            if unique_chunks and query:
                try:
                    query_embedding = await asyncio.to_thread(self.predoc_client.embedding, query)
                    chunk_embeddings_tasks = [
                        asyncio.to_thread(self.predoc_client.embedding, chunk.text)
                        for chunk in unique_chunks
                    ]
                    chunk_embeddings = await asyncio.gather(*chunk_embeddings_tasks)
                    for i, chunk in enumerate(unique_chunks):
                        score = cosine_similarity(query_embedding, chunk_embeddings[i])
                        chunk_scores.append((chunk, score))
                    logger.info(
                        f"计算 {len(unique_chunks)} 个 chunks 的相关性分数 (query: {query})"
                    )
                    for chunk, score in chunk_scores:
                        logger.debug(f"Chunk: {chunk.text[:50]}... Score: {score:.4f}")
                except Exception as e:
                    logger.error(f"嵌入计算失败: {e}，使用默认 score=0.0")
                    chunk_scores = [(chunk, 0.0) for chunk in unique_chunks]
            else:
                chunk_scores = [(chunk, 0.0) for chunk in unique_chunks]

            sorted_chunk_scores = sorted(chunk_scores, key=lambda x: x[1], reverse=True)
            THRESHOLD = 0.4
            filtered_chunk_scores = [
                (chunk, score) for chunk, score in sorted_chunk_scores if score >= THRESHOLD
            ]
            logger.info(
                f"阈值过滤 (score >= {THRESHOLD}): 保留 {len(filtered_chunk_scores)} 个 chunks"
            )
            if filtered_chunk_scores:
                logger.debug("过滤后 chunks:")
                for chunk, score in filtered_chunk_scores:
                    logger.debug(f"Chunk: {chunk.text[:50]}... Score: {score:.4f}")
            else:
                logger.warning(f"阈值过滤后无 chunks，阈值 {THRESHOLD} 可能过高")

            k = min(len(filtered_chunk_scores), 5)
            top_chunks = filtered_chunk_scores[:k]
            top_texts = [chunk.text for chunk, _ in top_chunks if chunk.text]
            sup_str = "\n- " + "\n- ".join(top_texts) if top_texts else "无补充信息"
            profile_prompt = f"用户画像: {profile}\n补充信息:{sup_str}\n"
            logger.info(f"画像注入成功: {profile_prompt[:100]}... \n(总补充条数: {len(top_texts)})")
            logger.info(f"Top {k} chunks 选择:")
            for i, (chunk, score) in enumerate(top_chunks, 1):
                logger.info(f"{i}. {chunk.text[:50]}... (Score: {score:.4f})")
            return profile_prompt
        except Exception as e:
            logger.error(f"画像注入失败: {e}")
            return ""


class PersonaChat(ChatService, PersonaMixin):
    """注入用户画像的 ChatService"""

    def __init__(self, cache, llm_client=None, predoc_client=None, mongo_db=None, **_):
        self.cache: Cache = cache
        self.llm_client: LLMClient = llm_client
        self.predoc_client = predoc_client or PredocClient()
        self.profile_service = ProfileService(self.cache, self.predoc_client, mongo_db)
        PersonaMixin.__init__(self, cache, self.predoc_client, mongo_db)
        super().__init__(cache, llm_client, mongo_db, **_)

    async def chat_workflow(
        self, chat_id: str, messages: list[Message] | None = None, username: str | None = None
    ) -> AsyncGenerator[str, None]:
        if not chat_id:
            raise ValueError("Missing chat_id for chat workflow")
        try:
            if self.llm_client is None:
                raise ValueError("llm_client is not configured")
            profile_prompt = (
                await self.profile_inject(username, messages[0].content) if username else ""
            )
            prompt = PROFILE_ANSWER_PROMPT.format(
                profile_prompt=profile_prompt, query=messages[0].content
            )
            final_prompt = f"{SYSTEM_PROMPT}\n{prompt}"
            logger.info(f"ChatService 最终 prompt: {final_prompt[:100]}...")
            assistant_content = ""
            async for chunk in self.llm_client.generate_stream(
                [Message(role="user", content=final_prompt)]
            ):
                chat_status = await self.cache.get(chat_id)
                if str(chat_status) == str(ChatStatus.ACTIVE.value):
                    text = chunk.encode("utf-8").decode("utf-8", errors="replace")
                    yield AnswerResponse.from_text(text).to_jsonl()
                    assistant_content += text
                else:
                    yield None
                    break
            if assistant_content:
                messages.append(Message(role="assistant", content=assistant_content))
        except Exception as e:
            logger.error(f"生成流式答案时出错: {str(e)}")
            if self.cache:
                await self.cache.set(chat_id, ChatStatus.TERMINATED.value)
            yield None


class PersonaRAG(RAGService, PersonaMixin):
    """注入用户画像的 RAGService"""

    def __init__(self, cache, llm_client=None, predoc_client=None, mongo_db=None, **_):
        PersonaMixin.__init__(self, cache, predoc_client, mongo_db)
        super().__init__(cache, llm_client, predoc_client, mongo_db, **_)

    async def chat_workflow(
        self, chat_id: str, messages: list[Message] | None = None, username: str | None = None
    ) -> AsyncGenerator[str, None]:
        if not chat_id:
            raise ValueError("Missing chat_id for chat workflow")
        try:
            if self.llm_client is None:
                raise ValueError("llm_client is not configured")
            origin_user_query = messages[0].content
            rewritten_query = ""
            async for chunk in self.query_rewrite(origin_user_query):
                chat_status = await self.cache.get(chat_id)
                if str(chat_status) == str(ChatStatus.ACTIVE.value):
                    text = chunk.encode("utf-8").decode("utf-8", errors="replace")
                    yield QueryRewriteResponse.from_text(text).to_jsonl()
                    rewritten_query += text
                else:
                    yield None
                    break
            rewritten_query = await self.remove_think(rewritten_query)
            messages.append(Message(role="system", content=rewritten_query))
            profile_prompt = (
                await self.profile_inject(username, rewritten_query) if username else ""
            )
            attachment = await self.search_relevant_docs(rewritten_query)
            yield SearchResponse.from_attachment(attachment).to_jsonl()
            messages.append(Message(role="system", content="", attachment=attachment))
            context = attachment.chunks
            context_prompt = self.context_inject(origin_user_query, rewritten_query, context)
            prompt = PROFILE_ANSWER_PROMPT.format(
                profile_prompt=profile_prompt, query=origin_user_query
            )
            final_prompt = (
                f"{SYSTEM_PROMPT}\n{prompt}\n{RAG_ANSWER_PROMPT.format(context=context_prompt)}"
            )
            logger.info(f"RAGService 最终 prompt: {final_prompt[:100]}...")
            assistant_content = ""
            async for chunk in self.llm_client.generate_stream(
                messages=[Message(role="user", content=final_prompt)]
            ):
                chat_status = await self.cache.get(chat_id)
                if str(chat_status) == str(ChatStatus.ACTIVE.value):
                    text = chunk.encode("utf-8").decode("utf-8", errors="replace")
                    yield AnswerResponse.from_text(text).to_jsonl()
                    assistant_content += text
                else:
                    yield None
                    break
            if assistant_content:
                messages.append(Message(role="assistant", content=assistant_content))
        except Exception as e:
            logger.error(f"生成流式答案时出错: {str(e)}")
            if self.cache:
                await self.cache.set(chat_id, ChatStatus.TERMINATED.value)
            yield None
