"""接口请求的数据模型"""

from core.assessment import AssessmentResponse
from core.user import UserBasicInfo
from pydantic import BaseModel, Field, field_validator

from src.schema import Message


class ChatRequest(BaseModel):
    """会话请求"""

    rag_enable: bool
    chat_id: str
    message_id: str
    query: list[Message] = Field(..., min_length=1)

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: list[Message]) -> list[Message]:
        if not v:
            raise ValueError("query must not be empty")
        return v


class ChatRequestWithUser(ChatRequest):
    """带用户信息的会话请求"""

    username: str = Field(default="test")


class UserProfileRequest(BaseModel):
    """用户画像请求模型。

    前端提交的基本信息和测评结果。
    """

    basic_info: UserBasicInfo = Field(..., description="用户基本信息")
    assessments: list[AssessmentResponse] = Field(
        default_factory=list, description="测评结果列表，允许部分未完成"
    )
