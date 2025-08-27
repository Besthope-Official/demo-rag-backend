"""数据库连接相关配置"""

from __future__ import annotations

from typing import Annotated

from pydantic import BeforeValidator

from src.config import BaseConfig

# Represents an ObjectId field in the database.
# It will be represented as a `str` on the model so that it can be serialized to JSON.
PyObjectId = Annotated[str, BeforeValidator(str)]


class MongoConfig(BaseConfig):
    """Mongo 配置模型。支持从 config.yaml -> mongo 节读取。"""

    url: str = "mongodb://localhost:27017"
    database: str = "default"

    @classmethod
    def from_yaml(cls, path: str | None = None) -> MongoConfig:
        return super().from_yaml(path)  # type: ignore[return-value]
