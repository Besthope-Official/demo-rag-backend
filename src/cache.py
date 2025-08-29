"""缓存相关接口"""

from abc import ABC, abstractmethod
from typing import Any, Optional

from .config import BaseConfig


class Cache(ABC):
    @abstractmethod
    async def set(self, key: str, value: Any, ex: int | None = None) -> None:
        """
        设置缓存键值。
        """
        pass

    @abstractmethod
    async def get(self, key: str) -> Any:
        """
        获取指定 key 的缓存值。
        """
        pass

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """
        判断指定 key 是否存在于缓存中。
        """
        pass

    @abstractmethod
    async def delete(self, key: str) -> None:
        """
        删除指定 key 的缓存。
        """
        pass

    @abstractmethod
    async def keys(self, pattern: str = "*") -> list[str]:
        """
        获取所有匹配 pattern 的缓存 key 列表。
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """
        关闭缓存连接或清理资源。
        """
        pass


class LocalCache(Cache):
    """Dict 模拟缓存操作"""

    def __init__(self):
        self._store: dict[str, Any] = {}

    async def set(self, key: str, value: Any, ex: int | None = None) -> None:
        self._store[key] = value

    async def get(self, key: str) -> Any:
        return self._store.get(key)

    async def exists(self, key: str) -> bool:
        return key in self._store

    async def delete(self, key: str) -> None:
        if key in self._store:
            del self._store[key]

    async def keys(self, pattern: str = "*") -> list[str]:
        if pattern == "*":
            return list(self._store.keys())
        else:
            if pattern.endswith("*"):
                prefix = pattern[:-1]
                return [k for k in self._store if k.startswith(prefix)]
            return [k for k in self._store if k == pattern]

    async def close(self) -> None:
        self._store.clear()


class RedisConfig(BaseConfig):
    """
    Redis 配置模型，包含连接所需的基本参数。
    """

    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: str | None = None
    encoding: str = "utf-8"

    @classmethod
    def from_yaml(cls, path: str | None = None) -> "RedisConfig":
        return super().from_yaml(path)  # type: ignore[return-value]

    @classmethod
    def from_yaml_dict(cls, path: str | None = None) -> dict:
        return super().from_yaml_dict(path)


class RedisCache(Cache):
    """Redis 缓存实现（基于 aioredis）"""

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config: Optional["RedisConfig"] = None):
        if hasattr(self, "_initialized") and self._initialized:
            return
        self._redis = None
        self._initialized = False
        self._config = config

        self.init()

    def init(self):
        if not self._config:
            raise ValueError("RedisConfig not found.")

        try:
            import aioredis  # pylint: disable=import-outside-toplevel
        except Exception as e:
            raise RuntimeError("aioredis not installed. Please install 'aioredis'.") from e

        url = f"redis://{self._config.host}:{self._config.port}"
        # we will store base64(pickle(obj)) as text, so keep decode_responses=True
        self._redis = aioredis.from_url(  # type: ignore[attr-defined]
            url,
            password=self._config.password,
            db=self._config.db,
            encoding=self._config.encoding,
            decode_responses=True,
        )
        self._initialized = True

    def _get_redis(self):
        if not self._redis:
            raise RuntimeError("RedisCache not init. Please await cache.init() first")
        return self._redis

    async def set(self, key: str, value: Any, ex: int | None = None) -> None:
        redis = self._get_redis()
        # Try pickle+base64 and store with a prefix so we can reliably detect
        # the serialization format when reading back.
        try:
            import base64
            import json  # local import
            import pickle

            payload_b64 = base64.b64encode(pickle.dumps(value)).decode("ascii")
            payload = f"PICKLE:{payload_b64}"
        except Exception:
            try:
                import json

                payload_json = json.dumps(value)
                payload = f"JSON:{payload_json}"
            except Exception:
                payload = f"STR:{str(value)}"

        if ex is not None:
            await redis.set(key, payload, ex=ex)
        else:
            await redis.set(key, payload)

    async def get(self, key: str) -> Any:
        redis = self._get_redis()
        value = await redis.get(key)
        # value is expected to be a str when decode_responses=True
        if value is None:
            return None

        # New-format: check prefix
        try:
            import base64
            import json  # local import
            import pickle

            if isinstance(value, str):
                if value.startswith("PICKLE:"):
                    b = base64.b64decode(value[len("PICKLE:") :])
                    return pickle.loads(b)
                if value.startswith("JSON:"):
                    return json.loads(value[len("JSON:") :])
                if value.startswith("STR:"):
                    return value[len("STR:") :]

            # Legacy/unknown format: best-effort
            try:
                raw = base64.b64decode(value)
                return pickle.loads(raw)
            except Exception:
                pass

            try:
                return json.loads(value)
            except Exception:
                return value
        except Exception:
            return value

    async def exists(self, key: str) -> bool:
        redis = self._get_redis()
        value = await redis.exists(key)
        return bool(value)

    async def delete(self, key: str) -> None:
        redis = self._get_redis()
        await redis.delete(key)

    async def keys(self, pattern: str = "*") -> list[str]:
        redis = self._get_redis()
        return await redis.keys(pattern)

    async def close(self) -> None:
        if self._redis:
            await self._redis.close()
            self._redis = None
            self._initialized = False
