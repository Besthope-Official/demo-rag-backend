"""用户数据持久化"""

import json
from datetime import date, datetime

from loguru import logger
from pydantic import BaseModel, Field
from pymongo import MongoClient
from pymongo.database import Database as MongoDatabase
from pymongo.errors import PyMongoError

from src.database import MongoConfig, PyObjectId
from src.exceptions import ConfigException

from .assessment import (
    AssessmentLabelMapping,
    AssessmentResponse,
    ChoiceAnswer,
    LabelExplanation,
    QuestionResponse,
    TextAnswer,
)


class UserBasicInfo(BaseModel):
    """用户个人资料数据模型
    既是前端传输的 DTO, 也是实际的 Data Model
    """

    id: PyObjectId | None = Field(alias="_id", default=None)
    # can be null if passed in DTO
    username: str | None = Field(default="test")

    gender: str | None = None
    birth_date: date | None = Field(default=None)
    city: str | None = None
    role: str | None = None

    concerns: list[str] = Field(default_factory=list, alias="careTopics")
    interests: list[str] = Field(default_factory=list, alias="interestTopics")

    model_config = {"from_attributes": True}


class UserDAO:
    """用户画像的数据访问层。

    使用 MongoDB 连接，提供保存和获取数据的方法。
    """

    def __init__(self, database: MongoDatabase = None, config: MongoConfig = None):
        """初始化数据库连接。

        Args:
            config: MongoConfig 实例，默认为从 config.yaml 加载。
        """
        try:
            if database is not None:
                logger.info("Using provided MongoDatabase instance")
                self.db = database
            else:
                self.config = config or MongoConfig()
                self.client = MongoClient(self.config.url)
                self.db = self.client[self.config.database]
                logger.info(f"MongoDB 初始化成功: {self.config.url}, {self.db.name}")
        except PyMongoError as e:
            logger.error(f"MongoDB 初始化失败: {e}")
            raise ConfigException("MongoDB 初始化失败") from e

    def save_user(self, info: UserBasicInfo):
        """保存用户基本信息。

        Args:
            info: UserBasicInfo 实例。
        """
        try:
            collection = self.db["users"]
            # prepare data and filter before update
            data = info.model_dump(exclude_none=True)
            data["concerns"] = ",".join(info.concerns) if info.concerns else ""
            data["interests"] = ",".join(info.interests) if info.interests else ""
            # sanitize date/datetime values to strings (MongoDB bson cannot encode datetime.date)
            for k, v in list(data.items()):
                if isinstance(v, datetime):
                    # store as YYYY-MM-DD string for consistency with other code
                    data[k] = v.date().isoformat()
                elif isinstance(v, date):
                    data[k] = v.isoformat()

            filter_q = {"username": info.username}

            update_result = collection.update_one(filter_q, {"$set": data}, upsert=True)
            logger.debug(
                f"保存用户 {info.username}, \
                  结果: upserted_id={update_result.upserted_id} \
                        modified={update_result.modified_count}"
            )
        except PyMongoError as e:
            logger.error(f"保存用户 {info.username} 失败: {e}")
            raise ConfigException("保存用户失败") from e

    def save_assessment(self, username: str, response: AssessmentResponse):
        """保存测评结果。

        Args:
            username: 用户名。
            response: AssessmentResponse 实例。
        """
        try:
            assessments_col = self.db["assessments"]
            questions_col = self.db["questions"]
            assessments_col.replace_one(
                {"username": username, "assessment_id": response.assessment_id},
                {"username": username, "assessment_id": response.assessment_id},
                upsert=True,
            )
            for question in response.questions:
                if question.answer:
                    answer_type = "choice" if isinstance(question.answer, ChoiceAnswer) else "text"
                    try:
                        answer_value = json.dumps(question.answer.model_dump(exclude_none=True))
                    except Exception as e:
                        logger.error(
                            f"序列化答案失败 {username} \
                              {response.assessment_id} {question.question_id}: {e}"
                        )
                        continue
                    result = questions_col.replace_one(
                        {
                            "username": username,
                            "assessment_id": response.assessment_id,
                            "question_id": question.question_id,
                        },
                        {
                            "username": username,
                            "assessment_id": response.assessment_id,
                            "question_id": question.question_id,
                            "answer_type": answer_type,
                            "answer_value": answer_value,
                        },
                        upsert=True,
                    )
                    logger.debug(
                        f"保存问题 {question.question_id}, \
                          结果: {result.upserted_id or result.modified_count}"
                    )
        except PyMongoError as e:
            logger.error(f"保存用户 {username} 的测评 {response.assessment_id} 失败: {e}")
            raise ConfigException("保存测评失败") from e

    def save_label_mapping(self, mapping: AssessmentLabelMapping):
        """保存测评标签映射规则。

        Args:
            mapping: AssessmentLabelMapping 实例。
        """
        try:
            collection = self.db["label_mappings"]
            for score_range, labels in mapping.label_mapping.items():
                result = collection.replace_one(
                    {"assessment_id": mapping.assessment_id, "score_range": score_range},
                    {
                        "assessment_id": mapping.assessment_id,
                        "score_range": score_range,
                        "labels": ",".join(labels),
                    },
                    upsert=True,
                )
                logger.debug(
                    f"保存标签映射 {score_range}, \
                      结果: {result.upserted_id or result.modified_count}"
                )
        except PyMongoError as e:
            logger.error(f"保存测评 {mapping.assessment_id} 的标签映射失败: {e}")
            raise ConfigException("保存标签映射失败") from e

    def save_label_explanation(self, explanation: LabelExplanation):
        """保存测评标签解释。

        Args:
            explanation: LabelExplanation 实例。
        """
        try:
            collection = self.db["labels"]
            result = collection.replace_one(
                {"assessment_id": explanation.assessment_id, "label": explanation.label},
                {
                    "assessment_id": explanation.assessment_id,
                    "label": explanation.label,
                    "explanation": explanation.explanation,
                },
                upsert=True,
            )
            logger.debug(
                f"保存标签解释 {explanation.label}, \
                  结果: {result.upserted_id or result.modified_count}"
            )
        except PyMongoError as e:
            logger.error(f"保存测评 {explanation.assessment_id} 的标签解释失败: {e}")
            raise ConfigException("保存标签解释失败") from e

    def save_user_labels(self, username: str, assessment_id: str, labels: list[str]):
        """保存用户测评标签。

        Args:
            username: 用户名。
            assessment_id: 测评 ID。
            labels: 标签列表。
        """
        try:
            collection = self.db["user_labels"]
            result = collection.replace_one(
                {"username": username, "assessment_id": assessment_id},
                {"username": username, "assessment_id": assessment_id, "labels": ",".join(labels)},
                upsert=True,
            )
            logger.debug(
                f"保存用户标签 {username}, 结果: {result.upserted_id or result.modified_count}"
            )
        except PyMongoError as e:
            logger.error(f"保存用户 {username} 的测评 {assessment_id} 标签失败: {e}")
            raise ConfigException("保存用户标签失败") from e

    def get_user(self, username: str) -> UserBasicInfo | None:
        """获取用户基本信息。

        Args:
            username: 用户名。
        Returns:
            UserBasicInfo 实例，或 None。
        """
        try:
            collection = self.db["users"]
            data = collection.find_one({"username": username})
            if data:
                concerns_raw = data.get("concerns") or ""
                interests_raw = data.get("interests") or ""
                data["concerns"] = concerns_raw.split(",") if concerns_raw else []
                data["interests"] = interests_raw.split(",") if interests_raw else []
                return UserBasicInfo(**data)
            return None
        except PyMongoError as e:
            logger.error(f"获取用户 {username} 失败: {e}")
            raise ConfigException("获取用户失败") from e

    def get_assessments(self, username: str) -> list[AssessmentResponse]:
        """获取用户测评结果列表。

        Args:
            username: 用户名。
        Returns:
            AssessmentResponse 列表。
        """
        try:
            assessments_col = self.db["assessments"]
            questions_col = self.db["questions"]
            assessment_ids = [
                doc["assessment_id"] for doc in assessments_col.find({"username": username})
            ]
            assessments = []
            for assessment_id in assessment_ids:
                questions_data = list(
                    questions_col.find({"username": username, "assessment_id": assessment_id})
                )
                questions = []
                for q in questions_data:
                    try:
                        answer_data = json.loads(q.get("answer_value", "null"))
                    except Exception as e:
                        logger.error(
                            "解析答案 JSON 失败 %s %s %s: %s",
                            username,
                            assessment_id,
                            q.get("question_id"),
                            e,
                        )
                        answer_data = None
                    answer = (
                        ChoiceAnswer(**answer_data)
                        if answer_data and q.get("answer_type") == "choice"
                        else (
                            TextAnswer(**answer_data)
                            if answer_data and q.get("answer_type") == "text"
                            else None
                        )
                    )
                    questions.append(QuestionResponse(question_id=q["question_id"], answer=answer))
                assessments.append(
                    AssessmentResponse(assessment_id=assessment_id, questions=questions)
                )
            return assessments
        except PyMongoError as e:
            logger.error(f"获取用户 {username} 的测评结果失败: {e}")
            raise ConfigException("获取测评结果失败") from e

    def get_label_mapping(self, assessment_id: str) -> dict[str, list[str]] | None:
        """获取测评的标签映射规则。

        Args:
            assessment_id: 测评 ID。
        Returns:
            分数范围到标签的字典，或 None。
        """
        try:
            collection = self.db["label_mappings"]
            docs = list(collection.find({"assessment_id": assessment_id}))
            if docs:
                return {doc["score_range"]: doc["labels"].split(",") for doc in docs}
            return None
        except PyMongoError as e:
            logger.error(f"获取测评 {assessment_id} 的标签映射失败: {e}")
            raise ConfigException("获取标签映射失败") from e

    def get_label_explanation(self, assessment_id: str, label: str) -> str | None:
        """获取测评标签解释。

        Args:
            assessment_id: 测评 ID。
            label: 标签名称。
        Returns:
            解释文本，或 None。
        """
        try:
            collection = self.db["labels"]
            doc = collection.find_one({"assessment_id": assessment_id, "label": label})
            return doc["explanation"] if doc else None
        except PyMongoError as e:
            logger.error(f"获取测评 {assessment_id} 的标签 {label} 解释失败: {e}")
            raise ConfigException("获取标签解释失败") from e

    def get_user_labels(self, username: str, assessment_id: str) -> list[str]:
        """获取用户测评标签。

        Args:
            username: 用户名。
            assessment_id: 测评 ID。
        Returns:
            标签列表。
        """
        try:
            collection = self.db["user_labels"]
            doc = collection.find_one({"username": username, "assessment_id": assessment_id})
            return doc["labels"].split(",") if doc and doc["labels"] else []
        except PyMongoError as e:
            logger.error(f"获取用户 {username} 的测评 {assessment_id} 标签失败: {e}")
            raise ConfigException("获取用户标签失败") from e
