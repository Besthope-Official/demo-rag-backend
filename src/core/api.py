"""API 路由"""

from fastapi import APIRouter, Depends, Request
from fastapi.responses import RedirectResponse, StreamingResponse

from src.cache import Cache
from src.llm import LLMClient
from src.predoc import PredocClient
from src.response import ApiResponse
from src.schema import Message

from .chat import ChatService, RAGService
from .dto import ChatRequest

router = APIRouter()


def get_qwq_client() -> LLMClient:
    return LLMClient()


def get_predoc_client() -> PredocClient:
    return PredocClient()


def get_cache(request: Request) -> Cache:
    cache = getattr(request.app.state, "cache", None)
    if cache is None:
        raise RuntimeError("Cache is not configured on app.state.cache")
    return cache


# Avoid calling Depends(...) inside default args (ruff B008)
CACHE_DEP = Depends(get_cache)
QWQ_LLM_DEP = Depends(get_qwq_client)
PREDOC_DEP = Depends(get_predoc_client)


@router.get("/")
async def index():
    return RedirectResponse(url="/docs")


@router.post("/v1/chat/completions")
async def chat_completions(
    request: ChatRequest,
    cache: Cache = CACHE_DEP,
    llm_client: LLMClient = QWQ_LLM_DEP,
    predoc_client: PredocClient | None = PREDOC_DEP,
):
    """
    对话补全接口，分为 RAG 启用/关闭两类

    流式响应的内容格式是 `text/event-stream`;
    响应保证：单次请求中每条 JSON 结构是完整的.
    """

    if request.rag_enable:
        if predoc_client is None:
            raise Exception("Knowledge Base cannot be accessed")
        chat_service = RAGService(cache=cache, llm_client=llm_client, predoc_client=predoc_client)
    else:
        chat_service = ChatService(cache=cache, llm_client=llm_client)

    async def stream_generator():
        async for chunk in chat_service.generate(
            chat_id=request.chat_id,
            message_id=request.message_id,
            messages=request.query,
        ):
            yield chunk

    try:
        return StreamingResponse(stream_generator(), media_type="text/event-stream")
    except Exception as e:
        return ApiResponse.fail(msg=str(e))


@router.post("/v1/chat/halt")
async def chat_halt(chat_id: str, cache: Cache = CACHE_DEP):
    """
    会话终止接口，截断RAG的回答链路，以减小服务器端压力

    客户端接收到 /completions 里的 end 事件后，终止当前会话请求.
    """
    service = ChatService(cache=cache)
    await service.halt_chat(chat_id)
    return ApiResponse.success(data={"chat_id": chat_id, "status": "halted"})


@router.post("/v1/chat/summarize")
async def chat_summarize(messages: list[Message], cache: Cache = CACHE_DEP):
    """会话标题生成接口，为一段会话总结标题"""
    service = ChatService(cache=cache)

    query = messages[-1].content
    title = await service.generate_chat_title(query)

    return ApiResponse.success(data={"title": title})
