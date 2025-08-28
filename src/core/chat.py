"""会话管理"""

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator

from loguru import logger

from src.cache import Cache
from src.llm import LLMClient
from src.schema import (
    AnswerResponse,
    Attachment,
    Author,
    ChatStatus,
    Chunk,
    Document,
    EndResponse,
    InitResponse,
    Message,
    QueryRewriteResponse,
    SearchResponse,
    Source,
)

from .prompt import QUERY_REWRITE_PROMPT, RAG_ANSWER_PROMPT


class ChatInterface(ABC):
    """抽象的聊天接口层。"""

    def __init__(self, cache: Cache, llm_client: LLMClient | None = None, **_):
        self.cache: Cache = cache
        self.llm_client: LLMClient | None = llm_client

    async def _set_chat_status(self, chat_id: str, status: ChatStatus):
        await self.cache.set(chat_id, status.value)

    async def generate(
        self, chat_id: str, message_id: str, messages: list[Message] | None = None
    ) -> AsyncGenerator[str, None]:
        """
        生成流式会话, 标记会话初始化与结束阶段，
        内部业务 `chat_workflow` 在子类实现
        """
        init_resp = InitResponse.create(chat_id=chat_id, message_id=message_id)

        if chat_id and self.cache:
            await self._set_chat_status(chat_id, ChatStatus.ACTIVE)

        yield init_resp.to_jsonl()

        try:
            async for res in self.chat_workflow(chat_id=chat_id, messages=messages):
                if res is not None:
                    yield res
        except Exception as e:
            logger.error(f"error: {e}")
            await self._set_chat_status(chat_id, ChatStatus.TERMINATED)
        finally:
            chat_status = await self.cache.get(chat_id)
            if str(chat_status) != str(ChatStatus.TERMINATED.value):
                await self._set_chat_status(chat_id, ChatStatus.COMPLETED)

        chat_status = await self.cache.get(chat_id)
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

    async def generate_chat_title(self, query: str) -> str | None:
        """根据用户提问，生成聊天标题"""
        # TODO: use qwen3 to generate title
        return query[:10]

    async def generate_new(
        self,chat_id:str,message_id:str,messages:list[Message]|None=None
    )->AsyncGenerator[str,None]:
        """
        模拟生成过程
        """
        return "成功"

    # async def get_summary(self, chat_id: str, user_id: str) -> str | None:
    #     """获取会话摘要"""
    #     # TODO: use qwen3 to generate summary
    #     return "这是一个摘要"
    
    # async def get_chat_list(self, user_id: str) -> list[dict]:
    #     return []

    # async def get_chat_history(self, chat_id: str, user_id: str) -> list[dict]:
    #     return []

    # async def create_chat(self, user_id: str) -> str | None:
    #     """创建新会话"""
    #     # TODO: use qwen3 to create chat
    #     return "new_chat_id"

    # async def send_text(self, user_id: str, text: str) -> None:
    #     """发送文本消息"""
    #     # TODO: use qwen3 to send text
    #     pass

    @abstractmethod
    async def chat_workflow(
        self, chat_id: str, messages: list[Message] | None = None
    ) -> AsyncGenerator[str, None]:
        """聊天工作流程，返回流式响应"""
        raise NotImplementedError()


class ChatService(ChatInterface):
    """
    无 RAG 模式的聊天业务
    """

    async def chat_workflow(
        self, chat_id: str, messages: list[Message] | None = None
    ) -> AsyncGenerator[str, None]:
        if not chat_id:
            raise ValueError("Missing chat_id for chat workflow")

        try:
            if self.llm_client is None:
                raise ValueError("llm_client is not configured")

            async for chunk in self.llm_client.generate_stream(messages=messages):
                chat_status = await self.cache.get(chat_id)
                if str(chat_status) == str(ChatStatus.ACTIVE.value):
                    yield AnswerResponse.from_text(chunk).to_jsonl()
                else:
                    yield None
                    break
        except Exception as e:
            logger.error(f"生成流式答案时出错: {str(e)}")
            if self.cache:
                await self.cache.set(chat_id, ChatStatus.TERMINATED.value)
            yield None


class RAGService(ChatInterface):
    """
    RAG 模式的聊天业务
    """

    def context_inject(self, origin_query: str, rewritten_query: str, context: list[Chunk]) -> str:
        template = f"""用户原始问题：{origin_query}\n重写问题：{rewritten_query}\n上下文信息：\n"""
        for chunk in context:
            template += f"- {chunk.text}\n"
        return template

    async def remove_think(self, text: str) -> str:
        """去除</think>标签前的全部思考部分"""
        idx = text.lower().find("</think>")
        return text[idx + len("</think>") :] if idx != -1 else text

    async def chat_workflow(
        self, chat_id: str, messages: list[Message] | None = None
    ) -> AsyncGenerator[str, None]:
        if not chat_id:
            raise ValueError("Missing chat_id for chat workflow")

        try:
            if self.llm_client is None:
                raise ValueError("llm_client is not configured")
            # STEP 1: Query rewrite
            origin_user_query = messages[0].content
            rewritten_query = ""

            async for chunk in self.query_rewrite(origin_user_query):
                chat_status = await self.cache.get(chat_id)
                if str(chat_status) == str(ChatStatus.ACTIVE.value):
                    yield QueryRewriteResponse.from_text(chunk).to_jsonl()
                    rewritten_query += chunk
                else:
                    yield None
                    break

            rewritten_query = await self.remove_think(rewritten_query)
            # STEP 2: Search
            # use rewritten query to search relevant chunks
            attachment = await self.search_relevant_docs(rewritten_query)
            yield SearchResponse.from_attachment(attachment).to_jsonl()

            context = attachment.chunks

            prompt = self.context_inject(origin_user_query, rewritten_query, context)
            final_query = RAG_ANSWER_PROMPT.format(context=prompt)
            messages = [Message(role="user", content=final_query)]

            # logger.debug(f"Final RAG query: {final_query}")

            # STEP 3: Answer
            async for chunk in self.llm_client.generate_stream(messages=messages):
                chat_status = await self.cache.get(chat_id)
                if str(chat_status) == str(ChatStatus.ACTIVE.value):
                    yield AnswerResponse.from_text(chunk).to_jsonl()
                else:
                    yield None
                    break
        except Exception as e:
            logger.error(f"生成流式答案时出错: {str(e)}")
            if self.cache:
                await self.cache.set(chat_id, ChatStatus.TERMINATED.value)
            yield None

    async def search_relevant_docs(self, query: str) -> Attachment:
        """搜索相关文档"""
        # mock request to predoc server
        # TODO: config predoc project to make real API request
        import json
        import pathlib

        with open(pathlib.Path(__file__).with_name("example_search_result.json")) as f:
            example_json = json.load(f)

            attachment_data = example_json.get("data", {})

            docs = []
            for doc_data in attachment_data.get("doc", []):
                doc = Document(
                    idx=doc_data.get("idx", 0),
                    title=doc_data.get("title", ""),
                    authors=[
                        Author(
                            name=author.get("name", ""),
                            institution=author.get("institution", ""),
                        )
                        for author in doc_data.get("authors", [])
                    ],
                    publicationDate=doc_data.get("publicationDate", ""),
                    language=doc_data.get("language", ""),
                    keywords=doc_data.get("keywords", []),
                    publisher=doc_data.get("publisher", ""),
                    journal=doc_data.get("journal", ""),
                )
                docs.append(doc)

            chunks = []
            for chunk_data in attachment_data.get("chunks", []):
                chunk = Chunk(
                    id=chunk_data.get("id", 0),
                    doc_id=chunk_data.get("doc_id", 0),
                    text=chunk_data.get("text", ""),
                    source=[
                        Source(
                            type="document",
                            id=chunk_data.get("id", 0),
                            url=chunk_data.get("url", ""),
                        )
                    ],
                )
                chunks.append(chunk)

            attachment = Attachment(doc=docs, chunks=chunks)
            return attachment

    async def query_rewrite(self, query: str) -> AsyncGenerator[str, None]:
        """对用户查询进行重写"""
        prompt = QUERY_REWRITE_PROMPT.format(query=query)
        messages = [Message(role="user", content=prompt)]
        async for response in self.llm_client.generate_stream(messages=messages):
            yield response
