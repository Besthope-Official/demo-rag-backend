"""数据模型定义"""

from datetime import datetime
from enum import Enum
from typing import Any, Literal, Union,Optional

from pydantic import BaseModel, Field


class Author(BaseModel):
    """作者信息"""

    name: str
    institution: str


class Document(BaseModel):
    """文档信息"""

    idx: int
    title: str
    authors: list[Author]
    publicationDate: str
    language: str
    keywords: list[str]
    publisher: str
    journal: str


class Source(BaseModel):
    """文档来源信息"""

    type: str
    id: int
    url: str


class Chunk(BaseModel):
    """文档分块信息"""

    id: int
    doc_id: int
    text: str
    source: list[Source]


class Attachment(BaseModel):
    """附件信息"""

    doc: list[Document]
    chunks: list[Chunk]


class Message(BaseModel):
    """对话消息"""

    role: Literal["user", "assistant"]
    content: str
    attachment: Attachment | None = None


class ResponseEventType(str, Enum):
    """会话响应事件类型"""

    INIT = "Init"
    QUERY_REWRITE = "Query Rewrite"
    SEARCH = "Search"
    ANSWER = "Answer"
    END = "End"


class ChatStatus(Enum):
    """会话状态"""

    ACTIVE = 1
    COMPLETED = 0
    TERMINATED = -1


class ChatResponse(BaseModel):
    """会话响应基类，提供如下的基本结构，data 由继承的事件响应注入

    ```json
    {"event": "...", "data": {} }
    ```

    使用 Pydantic 的 model_dump() 和 model_dump_json() 来序列化和反序列化
    方便起见，流式输出推荐用 to_jsonl() 来输出单行 JSON 格式
    """

    event: ResponseEventType
    data: Any

    def to_jsonl(self):
        return self.model_dump_json() + "\n"


class LLMResponse(BaseModel):
    """模型响应"""

    content: str


class InitData(BaseModel):
    """会话初始化数据"""

    chat_id: str
    message_id: str
    created_at: float = Field(default_factory=lambda: datetime.now().timestamp())


class EndData(BaseModel):
    """会话终止数据"""

    completion_time: float = Field(default_factory=lambda: datetime.now().timestamp())
    end_reason: int = Field(default=0, description="0: 正常结束, -1: 提前退出")


class InitResponse(ChatResponse):
    event: ResponseEventType = ResponseEventType.INIT
    data: InitData

    @classmethod
    def create(cls, chat_id: str, message_id: str) -> "InitResponse":
        return cls(data=InitData(chat_id=chat_id, message_id=message_id))


class QueryRewriteResponse(ChatResponse):
    event: ResponseEventType = ResponseEventType.QUERY_REWRITE
    data: LLMResponse

    @classmethod
    def from_text(cls, content: str) -> "QueryRewriteResponse":
        return cls(data=LLMResponse(content=content))


class SearchResponse(ChatResponse):
    event: ResponseEventType = ResponseEventType.SEARCH
    data: Attachment

    @classmethod
    def from_attachment(cls, attachment: Attachment) -> "SearchResponse":
        return cls(data=attachment)


class AnswerResponse(ChatResponse):
    event: ResponseEventType = ResponseEventType.ANSWER
    data: LLMResponse

    @classmethod
    def from_text(cls, content: str) -> "AnswerResponse":
        """根据文本内容构建回答响应"""
        return cls(data=LLMResponse(content=content))


class EndResponse(ChatResponse):
    event: ResponseEventType = ResponseEventType.END
    data: EndData

    @classmethod
    def from_status(cls, status: Union[int, str, "ChatStatus"]) -> "EndResponse":
        """根据会话状态生成结束响应"""
        if isinstance(status, str):
            end_reason = int(status)
        elif isinstance(status, int):
            end_reason = status
        else:  # ChatStatus
            end_reason = status.value
        return cls(data=EndData(end_reason=end_reason))

class UserWindow(BaseModel):
    """用户的会话历史,之后的用户画像也可以加载这里面"""
    window_id:list[str]

class WindowInform(BaseModel):
    """用户的会话记录"""
    user_chat:Optional[list[str]]=[]
    ai_chat:Optional[list[str]]=[]
    summary:Optional[str]=""
