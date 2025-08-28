"""知识库搜索客户端
参考 https://github.com/Besthope-Official/predoc
快速部署镜像: https://hub.docker.com/repository/docker/besthope/predoc/
"""

from __future__ import annotations

import os
from typing import Any

import requests
from loguru import logger
from pydantic import ValidationError

from src.config import BaseConfig
from src.schema import Attachment


class PredocConfig(BaseConfig):
    """Predoc 服务配置。

    从 config.yaml 的 `predoc:` 节读取，或由环境变量覆盖：
    - PREDOC_URL: 基础地址，如 http://localhost:8080
    - PREDOC_TIMEOUT: 请求超时（秒）
    """

    url: str
    timeout: int = 30

    def __init__(self, **data):
        if not data:
            data = self.from_yaml_dict()

        url = os.getenv("PREDOC_URL", data.get("url"))
        timeout_env = os.getenv("PREDOC_TIMEOUT")
        try:
            timeout = int(timeout_env) if timeout_env is not None else data.get("timeout", 30)
        except Exception:
            timeout = data.get("timeout", 30)

        super().__init__(url=url, timeout=timeout)

    @classmethod
    def from_yaml(cls, path: str | None = None) -> PredocConfig:
        return super().from_yaml(path)  # type: ignore[return-value]

    @classmethod
    def from_yaml_dict(cls, path: str | None = None) -> dict:
        return super().from_yaml_dict(path)


class PredocClient:
    """文档预处理客户端"""

    config: PredocConfig

    def __init__(self, config: PredocConfig | None = None):
        self.config = config or PredocConfig()
        self._session = requests.Session()

    def search(
        self, query: str, topK: int = 10, collection: str = "default_collection"
    ) -> Attachment:
        """检索知识库，得到相关文档并返回附件"""
        if not query:
            raise ValueError("query 不能为空")

        url = f"{self.config.url.rstrip('/')}/retrieval"
        payload = {"query": query, "topK": topK, "collection": collection}

        try:
            resp = self._session.post(url, json=payload, timeout=self.config.timeout)
            resp.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"Predoc 请求失败: {e}")
            raise

        try:
            body = resp.json()
        except ValueError as e:
            logger.error(f"Predoc 响应非 JSON: {e}; text={resp.text[:200]}")
            raise

        data = body.get("data", body) if isinstance(body, dict) else {}
        return self._to_attachment(data)

    def embedding(self, text: str) -> list[float]:
        url = f"{self.config.url.rstrip('/')}/embedding"
        payload = {"text": text}

        try:
            resp = self._session.post(url, json=payload, timeout=self.config.timeout)
            resp.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"Predoc 请求失败: {e}")
            raise

        try:
            body = resp.json()
        except ValueError as e:
            logger.error(f"Predoc 响应非 JSON: {e}; text={resp.text[:200]}")
            raise

        data = body.get("data", body) if isinstance(body, dict) else {}
        return data.get("embedding", [])

    def _to_attachment(self, data: dict[str, Any]) -> Attachment:
        """反序列化"""
        try:
            return Attachment.model_validate(data)
        except ValidationError as e:
            logger.error(f"Predoc 数据反序列化失败: {e}")
            raise


if __name__ == "__main__":
    cfg = PredocConfig()
    logger.info(f"Predoc Config: url={cfg.url}, timeout={cfg.timeout}s")

    client = PredocClient(cfg)
    att = client.search(query="合作", topK=10, collection="test")

    logger.info(f"retrieved {len(att.doc)} docs, {len(att.chunks)} chunks")

    preview_n = min(3, len(att.chunks))
    for i in range(preview_n):
        ch = att.chunks[i]
        snippet = (ch.text or "").replace("\n", " ")[:120]
        print(f"{i+1}. [doc_id={ch.doc_id}, chunk_id={ch.id}] {snippet}")

    embedding = client.embedding(text="合作")
    logger.info(f"Embedding: {embedding}")
