"""API 路由"""

from core.assessment import AssessmentResponse
from core.label_processor import LabelProcessor
from core.profile import ProfileService
from exceptions import ConfigException
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import RedirectResponse, StreamingResponse
from loguru import logger
from pymongo.database import Database as MongoDatabase

from src.cache import Cache
from src.llm import LLMClient
from src.predoc import PredocClient
from src.response import ApiResponse
from src.schema import Message

from .chat import ChatService, PersonaChat, PersonaRAG, RAGService
from .dto import ChatRequestWithUser
from .history import HistoryService
from .user import UserBasicInfo, UserDAO

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


def get_mongo(request: Request) -> MongoDatabase:
    db = getattr(request.app.state, "mongo_db", None)
    if db is None:
        raise RuntimeError("Mongo is not configured on app.state.mongo_db")
    return db


CACHE_DEP = Depends(get_cache)
QWQ_LLM_DEP = Depends(get_qwq_client)
PREDOC_DEP = Depends(get_predoc_client)
MONGO_DEP = Depends(get_mongo)


async def stream_chat(
    chat_service,
    chat_id: str,
    message_id: str,
    messages: list[Message],
    username: str | None = None,
):
    async for chunk in chat_service.generate(
        chat_id=chat_id, message_id=message_id, messages=messages, username=username
    ):
        yield chunk


@router.get("/")
async def index():
    return RedirectResponse(url="/docs")


@router.post("/v1/chat/completions")
async def chat_completions(
    request: ChatRequestWithUser,
    cache: Cache = CACHE_DEP,
    llm_client: LLMClient = QWQ_LLM_DEP,
    predoc_client: PredocClient | None = PREDOC_DEP,
    mongo_db: MongoDatabase = MONGO_DEP,
):
    if request.rag_enable:
        if predoc_client is None:
            raise Exception("Knowledge Base cannot be accessed")
        chat_service = RAGService(
            cache=cache, llm_client=llm_client, predoc_client=predoc_client, mongo_db=mongo_db
        )
    else:
        chat_service = ChatService(cache=cache, llm_client=llm_client, mongo_db=mongo_db)
    try:
        return StreamingResponse(
            stream_chat(
                chat_service, request.chat_id, request.message_id, request.query, request.username
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache, no-transform"},
        )
    except Exception as e:
        return ApiResponse.fail(msg=str(e))


@router.post("/v2/chat/completions")
async def chat_completions_with_user(
    request: ChatRequestWithUser,
    cache: Cache = CACHE_DEP,
    llm_client: LLMClient = QWQ_LLM_DEP,
    predoc_client: PredocClient | None = PREDOC_DEP,
    mongo_db: MongoDatabase = MONGO_DEP,
):
    if request.rag_enable:
        if predoc_client is None:
            raise Exception("Knowledge Base cannot be accessed")
        chat_service = PersonaRAG(
            cache=cache, llm_client=llm_client, predoc_client=predoc_client, mongo_db=mongo_db
        )
    else:
        chat_service = PersonaChat(
            cache=cache, llm_client=llm_client, predoc_client=predoc_client, mongo_db=mongo_db
        )
    try:
        return StreamingResponse(
            stream_chat(
                chat_service, request.chat_id, request.message_id, request.query, request.username
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache, no-transform"},
        )
    except Exception as e:
        return ApiResponse.fail(msg=str(e))


@router.post("/v1/chat/halt")
async def chat_halt(chat_id: str, cache: Cache = CACHE_DEP):
    service = ChatService(cache=cache)
    await service.halt_chat(chat_id)
    return ApiResponse.success(data={"chat_id": chat_id, "status": "halted"})


@router.post("/v1/chat/summarize")
async def chat_summarize(
    messages: list[Message],
    cache: Cache = CACHE_DEP,
    llm_client: LLMClient = QWQ_LLM_DEP,
):
    service = ChatService(cache=cache, llm_client=llm_client)
    query = messages[-1].content if messages else ""
    title = await service.generate_chat_title(query)
    return ApiResponse.success(data={"title": title})


@router.post("/v1/user/basic")
async def create_user(
    basic_info: UserBasicInfo,
    db: MongoDatabase = MONGO_DEP,
    cache: Cache = CACHE_DEP,
    predoc_client: PredocClient = PREDOC_DEP,
):
    try:
        userDAO = UserDAO(database=db)
        userDAO.save_user(basic_info)
        label_processor = LabelProcessor(userDAO)
        service = ProfileService(cache, predoc_client, db)
        assessments = userDAO.get_assessments(basic_info.username)
        if assessments:
            profile = await service.generate_profile(basic_info, label_processor)
            label_results = await service.search_by_labels(basic_info.username)
            supplement = await service.get_supplement(basic_info.username, label_processor)
            labels = label_results.chunks
            await cache.set(
                f"label_results:{basic_info.username}", [label.text for label in labels]
            )
            await cache.set(f"supplement:{basic_info.username}", supplement.model_dump())
        else:
            profile = await service.generate_profile(basic_info, label_processor)
            await cache.set(f"profile:{basic_info.username}", profile)
        return ApiResponse.success(data={"profile": profile}, msg="用户信息保存成功")
    except ConfigException as e:
        logger.error(f"处理用户基本信息失败: {e}")
        return ApiResponse.fail(msg=str(e), code="400")
    except Exception as e:
        logger.error(f"处理用户基本信息失败: {e}")
        return ApiResponse.fail(msg=str(e), code="500")


@router.post("/v1/user/assessment/{username}")
async def save_user_assessment(
    username: str,
    assessments: list[AssessmentResponse],
    db: MongoDatabase = MONGO_DEP,
    cache: Cache = CACHE_DEP,
    predoc_client: PredocClient | None = PREDOC_DEP,
):
    try:
        storage = UserDAO(database=db)
        basic_info = storage.get_user(username)
        if not basic_info:
            raise ConfigException(f"用户 {username} 不存在")
        for assessment in assessments:
            storage.save_assessment(username, assessment)
        label_processor = LabelProcessor(storage)
        service = ProfileService(cache, predoc_client or PredocClient(), db)
        profile = await service.generate_profile(basic_info, label_processor)
        label_results = await service.search_by_labels(username)
        supplement = await service.get_supplement(username, label_processor)
        label_texts = service.deduplicate_chunks(label_results, k=5)
        supplement_texts = service.deduplicate_chunks(supplement, k=5)
        await cache.set(f"label_results:{username}", label_texts)
        await cache.set(f"supplement:{username}", supplement_texts)
        return ApiResponse.success(
            data={
                "profile": profile,
                "label_results": label_texts,
                "supplement": supplement_texts,
                "message": "用户测评信息处理完成",
            }
        )
    except ConfigException as e:
        logger.error(f"处理用户 {username} 测评信息失败: {e}")
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error(f"处理用户 {username} 测评信息失败: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/v1/debug/cache")
async def debug_cache(username: str, cache: Cache = CACHE_DEP):
    try:
        profile = await cache.get(f"profile:{username}")
        label_results = await cache.get(f"label_results:{username}")
        supplement = await cache.get(f"supplement:{username}")
        return {"profile": profile, "label_results": label_results, "supplement": supplement}
    except Exception as e:
        logger.error(f"获取 Cache 数据失败: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/v1/chat/list/{username}")
async def get_chat_list(username: str, db: MongoDatabase = MONGO_DEP):
    """获取用户所有聊天列表。"""
    try:
        dao = UserDAO(database=db)
        history_service = HistoryService(dao)
        chats = await history_service.get_chat_list(username)
        return ApiResponse.success(data=chats)
    except Exception as e:
        logger.error(f"获取聊天列表失败: {e}")
        return ApiResponse.fail(msg=str(e))


@router.get("/v1/chat/history/{chat_id}")
async def get_chat_history(chat_id: str, db: MongoDatabase = MONGO_DEP):
    """获取指定 chat_id 的完整历史。"""
    try:
        dao = UserDAO(database=db)
        history_service = HistoryService(dao)
        history = await history_service.get_chat_history(chat_id)
        if history:
            return ApiResponse.success(data=history)
        return ApiResponse.fail(msg=f"Chat {chat_id} not found", code="404")
    except Exception as e:
        logger.error(f"获取聊天 {chat_id} 历史失败: {e}")
        return ApiResponse.fail(msg=str(e))
