"""会话管理"""

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator

from loguru import logger

from src.cache import Cache
from src.llm import LLMClient
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

    def __init__(self, cache, llm_client=None, predoc_client=None, **_):
        self.predoc_client = predoc_client
        super().__init__(cache, llm_client, **_)

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
        try:
            client = self.predoc_client
            return client.search(query, topK=10)
        except Exception as e:
            logger.warning(f"Predoc 搜索失败: {e}")

    async def query_rewrite(self, query: str) -> AsyncGenerator[str, None]:
        """对用户查询进行重写"""
        prompt = QUERY_REWRITE_PROMPT.format(query=query)
        messages = [Message(role="user", content=prompt)]
        async for response in self.llm_client.generate_stream(messages=messages):
            yield response
