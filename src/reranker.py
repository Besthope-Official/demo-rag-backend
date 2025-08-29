"""第三层检索模块。

独立处理聚合、重排序和去重，用于问答助手模块。
兼容 chat.py 和 embedding_utils.py 的风格，使用 async 方法、typing 和 Loguru。
"""

from loguru import logger

from .exceptions import ConfigException
from .schema import Attachment
from .utils import EmbeddingUtils


class ReRanker:
    """第三层检索类。

    提供聚合和重排序功能，支持测评文献知识库（collection="test"）的去重和基于查询的重排序。
    """

    def __init__(self):
        """初始化重排序工具。

        使用 EmbeddingUtils 生成嵌入。
        """
        try:
            self.embedding_utils = EmbeddingUtils()
            logger.info("重排序工具初始化成功")
        except Exception as e:
            logger.error(f"初始化重排序工具失败: {e}")
            raise ConfigException(f"初始化重排序工具失败: {e}") from e

    async def aggregate_and_re_rank(
        self,
        query: str,
        label_results: list[tuple[Attachment, list[str]]],
        profile_result: Attachment,
        k: int = 5,
        collection: str = "test",
    ) -> list[tuple[str, list[str]]]:
        """聚合第一/第二层结果，去重并重排序。

        Args:
            query: 用户查询（原始或重构）。
            label_results: 第一层检索结果（标签+解释的 Attachment 和标签列表）。
            profile_result: 第二层检索结果（画像的 Attachment）。
            k: top-k 结果。
            collection: 知识库集合（默认 "test"，测评文献知识库）。
        Returns:
            top-k (文本, 元数据列表) 元组列表。
        """
        try:
            if not label_results and not profile_result.chunks:
                logger.warning(f"输入结果为空，collection: {collection}")
                return []

            # 聚合第一层（标签+解释）和第二层（画像）结果
            all_chunks = []
            for result, labels in label_results:
                for chunk in result.chunks[:100]:  # 限制最多 100 个 chunk
                    all_chunks.append((chunk.text, labels + [collection]))
            for chunk in profile_result.chunks[:100]:
                all_chunks.append((chunk.text, ["profile_supplement", collection]))

            # 去重：使用 set 基于 chunk.text
            seen = set()
            unique_chunks = [
                (chunk, meta)
                for chunk, meta in all_chunks
                if not (chunk in seen or seen.add(chunk))
            ]
            if not unique_chunks:
                logger.warning(f"去重后结果为空，collection: {collection}")
                return []

            # 重排序：基于 query 嵌入
            ranked_chunks = self.embedding_utils.re_rank(query, unique_chunks, k=k)
            if not ranked_chunks and unique_chunks:
                logger.warning(f"重排序失败，降级返回前 k 个结果，collection: {collection}")
                return unique_chunks[:k]
            return ranked_chunks
        except Exception as e:
            logger.error(f"聚合和重排序失败: {e}, collection: {collection}")
            raise ConfigException(f"聚合和重排序失败: {e}") from e
