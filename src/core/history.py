"""管理会话历史"""

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
    UserWindow,
    WindowInform,
)

from .prompt import QUERY_REWRITE_PROMPT, RAG_ANSWER_PROMPT
from .chat import ChatService

# class ChatHistoryInterface(ABC):
#     """聊天历史接口"""
#     def __init__(self, cache: Cache,**_):
#         self.cache:Cache=cache

#     @abstractmethod
#     async def get_chat_history(self, window_id: str, user_id: str) -> list[dict]:
#         """获取聊天历史"""
#         raise NotImplementedError()

#     @abstractmethod
#     async def get_chat_list(self, user_id: str) -> list[dict]:
#         """获取用户的聊天列表"""
#         raise NotImplementedError() 
    
#     @abstractmethod
#     async def create_chat(self, user_id: str) -> str | None:
#         """创建新会话"""
#         raise NotImplementedError()

#     @abstractmethod
#     async def send_text(self, user_id: str,window_id:str, text: str) -> None:
#         raise NotImplementedError()

class ChatHistoryService:
    """聊天历史服务实现"""
    def __init__(self, cache: Cache):
        self.cache:Cache=cache

    async def get_chat_history(self, window_id: str, user_id: str) -> (list,list):
        """获取聊天历史"""
        if self.cache.exists((user_id,window_id)):
            history = await self.cache.get_new((user_id,window_id))
            if history is not None:
                user_res=[]
                ai_res=[]
                user_text_list=history.get("user_text")
                ai_text_list=history.get("ai_text")
                async for user_text in user_text_list:
                    user_res.append(user_text)
                async for ai_text in ai_text_list:
                    ai_res.append(ai_text)
                yield (user_res,ai_res)
            else:
                yield
        else:
            yield

    async def get_chat_list(self, user_id: str) -> str |None:
        """获取用户的聊天列表"""
        
        if self.cache.exists(user_id):
            window_list = await self.cache.get(user_id)
            if window_list is not None:
                window=window_list.get("window_id")
                async for window_id in window:
                    yield window_id
            else:
                yield
        else:
            yield

    async def create_chat(self, user_id: str) -> str | None:
        """创建新会话"""
        import uuid
        new_window_id = str(uuid.uuid4())
        if self.cache.exists(user_id) :
            user_inform=await self.cache.get(user_id)
            if user_inform is not None:
                print(user_inform)
                window_list=user_inform.get("window_id")
                print(window_list)
                if len(window_list)==0:
                    window_list=[new_window_id]
                else:
                    window_list.append(new_window_id)
                user_inform["window_id"]=window_list
                await self.cache.set(user_id,user_inform)
            else:
                window_ids=[new_window_id]
                user_inform=UserWindow(window_id=window_ids)
                await self.cache.set(user_id,user_inform)
        else:
            window_ids=[new_window_id]
            user_inform=UserWindow(window_id=window_ids)
            await self.cache.set(user_id,user_inform)
        yield new_window_id

    # 这个未测试
    async def send_text(self, user_id: str,window_id:str, text: str) -> None:
        service = ChatService(cache=cache)
        message=Messgae(role="user",content=text)
        messgaes=List[messgae]
        import uuid
        message_id = str(uuid.uuid4())
        response=service.generate_new(window_id,message_id,messages)
        if self.cache.exists_new((user_id,window_id)):
            history=self.cache.get_new((user_id,window_id))
            user_list=history.user_chat
            ai_list=history.ai_chat
            user_list.append(text)
            ai_list.append(response)
            history.user_chat=user_list
            history.ai_chat=ai_list
            self.cache.set_new((user_id,window_id),history)
        else:
            user_chat=[text]
            ai_chat=[response]
            summary=service.generate_chat_title(text)
            history=user_inform(user_chat=user_chat,ai_chat=ai_chat,summary=summary)
            self.cache.set_new((user_id,window_id),history)
        return 200