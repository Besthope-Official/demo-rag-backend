"""用户数据持久化"""

from datetime import date

from pydantic import BaseModel, Field
from pymongo import ReturnDocument
from pymongo.database import Database as MongoDatabase

from src.database import PyObjectId


class UserBasicInfo(BaseModel):
    """用户个人资料数据模型
    既是前端传输的 DTO, 也是实际的 Data Model
    """

    id: PyObjectId | None = Field(alias="_id", default=None)
    # can be null if passed in DTO
    name: str | None = "test"

    gender: str | None = None
    birthday: date | None = None
    city: str | None = None
    identity: str | None = None

    careTopics: list[str] = Field(default_factory=list)
    interestTopics: list[str] = Field(default_factory=list)

    class Config:
        from_attributes = True


def save_user_profile(db: MongoDatabase, user_profile: UserBasicInfo) -> str:
    """创建或更新用户资料, 更新成功返回文档 id"""
    collection = db["users"]

    user_data = user_profile.model_dump(
        exclude_unset=True,
        exclude_none=True,
        exclude={"id"},
        mode="json",
    )

    name = user_profile.name or "test"

    doc = collection.find_one_and_update(
        {"name": name},
        {"$set": user_data},
        upsert=True,
        projection={"_id": 1},
        return_document=ReturnDocument.AFTER,
    )

    if not doc or "_id" not in doc:
        raise RuntimeError(f"Failed to upsert or fetch user '{name}'.")

    return str(doc["_id"])
