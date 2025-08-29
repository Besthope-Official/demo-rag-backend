"""用户画像服务模块。
实现用户画像生成（模板驱动）、测评文献知识库检索（两层：标签+描述），提供去重功能。
兼容 chat.py 的风格，使用 async 方法、PredocClient 和类型注解。
"""

import asyncio
from datetime import datetime

from loguru import logger
from pymongo import MongoClient
from pymongo.database import Database as MongoDatabase

from src.cache import Cache
from src.database import MongoConfig
from src.predoc import PredocClient
from src.schema import Attachment

from .label_processor import LabelProcessor
from .user import UserBasicInfo, UserDAO

PROFILE_TEMPLATE = """
用户画像（已授权可用）：
基本信息：
用户名：{username}，性别：{gender}，出生日期：{birth_date}，
年龄：{age} 岁，城市：{city}
{role_str}
{concerns_str}
{interests_str}
测评标签（基于行为经济学、实验经济学、行为心理学、决策理论和
人格心理学等领域的理论）：{labels_str}
"""


class ProfileService:
    """用户画像服务类。
    初始化测评文献知识库客户端，提供画像生成、两层检索（标签+描述）和去重功能。
    """

    def __init__(self, cache: Cache, predoc_client: PredocClient, database: MongoDatabase = None):
        """初始化服务。
        Args:
            cache: Cache 实例，用于存储画像。
            predoc_client: PredocClient 实例，用于检索。
        """
        self.cache = cache
        if database is None:
            config = MongoConfig().from_yaml()
            client = MongoClient(config.url)
            db = client[config.database]
            database = db
        self.storage = UserDAO(database)
        self.predoc_client = predoc_client
        self.label_processor = LabelProcessor(self.storage)
        self.profile_template = PROFILE_TEMPLATE

    async def generate_profile(
        self, basic_info: UserBasicInfo, label_processor: LabelProcessor
    ) -> str:
        """异步生成用户画像。
        使用模板填充基本信息和测评标签，动态计算年龄。
        Args:
            basic_info: 用户基本信息。
            label_processor: LabelProcessor 实例。
        Returns:
            用户画像文本描述。
        """
        try:
            labels = await asyncio.to_thread(
                label_processor.process_assessments, basic_info.username
            )
            birth_date = basic_info.birth_date
            # birth_date = datetime.strptime(basic_info.birth_date, "%Y-%m-%d")
            current_date = datetime.now()
            age = (
                current_date.year
                - birth_date.year
                - ((current_date.month, current_date.day) < (birth_date.month, birth_date.day))
            )
            if (current_date.month, current_date.day) == (birth_date.month, birth_date.day):
                age += 1
            role_str = f"，身份：{basic_info.role}" if basic_info.role else ""
            concerns_str = (
                f"，关心：{'、'.join(basic_info.concerns)}" if basic_info.concerns else ""
            )
            interests_str = (
                f"，兴趣：{'、'.join(basic_info.interests)}" if basic_info.interests else ""
            )
            labels_str = "\n- " + "\n- ".join(labels) if labels else "无测评标签"
            profile = self.profile_template.format(
                username=basic_info.username,
                gender=basic_info.gender,
                birth_date=basic_info.birth_date,
                city=basic_info.city,
                age=age,
                role_str=role_str,
                concerns_str=concerns_str,
                interests_str=interests_str,
                labels_str=labels_str,
            )
            await self.cache.set(f"profile:{basic_info.username}", profile)
            return profile
        except Exception as e:
            logger.error(f"生成用户 {basic_info.username} 的画像失败: {e}")
            return ""

    def deduplicate_chunks(self, attachment: Attachment, k: int = 3) -> list[str]:
        """去重 Attachment 中的文本块，返回前 k 个唯一文本。
        Args:
            attachment: Attachment 实例，包含检索结果的 chunks。
            k: 返回 top-k 结果。
        Returns:
            去重后的文本列表。
        """
        try:
            if not attachment.chunks:
                return []
            seen = set()
            unique_texts = []
            for chunk in attachment.chunks:
                text = chunk.text
                if text and text not in seen:
                    seen.add(text)
                    unique_texts.append(text)
                    if len(unique_texts) >= k:
                        break
            return unique_texts
        except Exception as e:
            logger.error(f"去重失败: {e}")
            return [chunk.text for chunk in attachment.chunks[:k] if chunk.text]

    async def search_by_labels(self, username: str, k: int = 3) -> Attachment:
        """第一层检索：按标签和解释检索测评文献知识库。
        Args:
            username: 用户名。
            k: top-k 结果。
        Returns:
            Attachment 实例，包含检索结果。
        """
        try:
            assessments = self.storage.get_assessments(username)
            if not assessments:
                logger.info(f"用户 {username} 无测评数据，返回空 Attachment")
                return Attachment(doc=[], chunks=[])
            all_chunks = []
            for assess in assessments:
                labels = self.storage.get_user_labels(username, assess.assessment_id)
                for label in labels:
                    explanation = (
                        self.storage.get_label_explanation(assess.assessment_id, label) or ""
                    )
                    query = f"{label}: {explanation}".strip()
                    if not query:
                        logger.warning(f"标签 {label} 的查询为空，跳过")
                        continue
                    result = await asyncio.to_thread(
                        self.predoc_client.search, query, topK=k, collection="test"
                    )
                    all_chunks.extend(result.chunks)
            return Attachment(doc=[], chunks=all_chunks)
        except Exception as e:
            logger.error(f"基于标签的检索失败: {e}")
            return Attachment(doc=[], chunks=[])

    async def get_supplement(
        self, username: str, label_processor: LabelProcessor, k: int = 3
    ) -> Attachment:
        """第二层检索：基于用户画像检索测评文献知识库。
        Args:
            username: 用户名。
            label_processor: LabelProcessor 实例。
            k: top-k 结果。
        Returns:
            Attachment 实例。
        """
        try:
            basic_info = self.storage.get_user(username)
            if not basic_info:
                logger.error(f"用户 {username} 不存在")
                return Attachment(doc=[], chunks=[])
            profile = await self.generate_profile(basic_info, label_processor)
            logger.debug(f"Supplement query: {profile[:100]}...")
            return await asyncio.to_thread(
                self.predoc_client.search, profile, topK=k, collection="test"
            )
        except Exception as e:
            logger.error(f"基于画像的检索失败: {e}")
            return Attachment(doc=[], chunks=[])
