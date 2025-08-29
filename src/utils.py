"""新增：嵌入生成和重排序工具模块。

提供文本嵌入生成和重排序功能，用于用户画像模块的三层检索。
兼容 chat.py 和 predoc.py 的风格，使用 typing 和 Loguru。
"""

from math import sqrt

from loguru import logger

from src.exceptions import ConfigException
from src.predoc import PredocClient, PredocConfig


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """计算两个向量的余弦相似度"""
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0
    dot_product = sum(x * y for x, y in zip(vec1, vec2, strict=False))
    norm1 = sqrt(sum(x**2 for x in vec1))
    norm2 = sqrt(sum(y**2 for y in vec2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)


class EmbeddingUtils:
    """嵌入生成和重排序工具类。

    初始化 Predoc 客户端，提供嵌入生成和重排序方法。
    """

    def __init__(self):
        """初始化 Predoc 客户端用于嵌入生成。

        使用 config.yaml 中的 Predoc 服务。
        """
        try:
            self.client = PredocClient(PredocConfig())
            logger.info(f"Predoc 客户端初始化成功, URL: {self.client.config.url}")
        except Exception as e:
            logger.error(f"初始化 Predoc 客户端失败: {e}")
            raise ConfigException(f"初始化 Predoc 客户端失败: {e}") from e

    def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        """生成文本嵌入。

        Args:
            texts: 文本列表。
        Returns:
            嵌入向量列表，每个元素为 list[float]。
        """
        try:
            if not texts:
                logger.warning("嵌入生成输入文本列表为空")
                return []
            embeddings = [self.client.embedding(text) for text in texts]
            if not all(embeddings) and any(embeddings):
                failed_indices = [i for i, emb in enumerate(embeddings) if not emb]
                logger.warning(f"部分嵌入生成失败，索引: {failed_indices}")
            return embeddings
        except Exception as e:
            logger.error(f"生成嵌入失败: {e}")
            raise ConfigException(f"生成嵌入失败: {e}") from e

    def re_rank(
        self, query: str, chunks: list[tuple[str, list[str]]], k: int = 5
    ) -> list[tuple[str, list[str]]]:
        """基于查询嵌入对文本块和元数据对重排序并取 top-k。

        使用余弦相似度排序，去除重复文本。
        Args:
            query: 查询文本。
            chunks: (文本, 元数据列表) 元组列表。
            k: 返回 top-k 结果。
        Returns:
            top-k (文本, 元数据列表) 元组列表。
        """
        try:
            if not chunks:
                return []
            texts = [chunk[0] for chunk in chunks]
            query_emb = self.client.embedding(query)
            chunk_embs = self.generate_embeddings(texts)
            if not query_emb or not all(chunk_embs):
                logger.warning("嵌入数据为空，降级返回前 k 个结果")
                return [(text, meta) for text, meta in chunks[:k]]  # 降级逻辑
            # 预计算所有得分
            score_indices = [
                (self._cos_sim(query_emb, emb), i) for i, emb in enumerate(chunk_embs) if emb
            ]
            score_indices.sort(reverse=True)  # 降序排序
            seen = set()
            unique_indices = []
            for _score, idx in score_indices:
                text = texts[idx]
                if text not in seen:
                    seen.add(text)
                    unique_indices.append(idx)
                if len(unique_indices) >= k:
                    break
            return [(texts[i], chunks[i][1]) for i in unique_indices[:k]]
        except Exception as e:
            logger.error(f"重排序失败: {e}")
            raise ConfigException(f"重排序失败: {e}") from e

    def _cos_sim(self, v1: list[float], v2: list[float]) -> float:
        """计算两个向量的余弦相似度。

        Args:
            v1: 第一个向量。
            v2: 第二个向量。
        Returns:
            余弦相似度值 [0, 1]。
        """
        try:
            if not v1 or not v2:
                return 0.0
            dot_product = sum(a * b for a, b in zip(v1, v2, strict=False))
            norm_v1 = sum(a * a for a in v1) ** 0.5
            norm_v2 = sum(a * a for a in v2) ** 0.5
            return dot_product / (norm_v1 * norm_v2) if norm_v1 * norm_v2 != 0 else 0.0
        except Exception as e:
            logger.error(f"余弦相似度计算失败: {e}")
            return 0.0
