from pydantic import BaseModel, Field, field_validator

from src.database import PyObjectId
from src.schema import Attachment


class ChoiceAnswer(BaseModel):
    """选择题答案模型。

    支持单选 (int) 或多选 (List[int])，表示选项索引。
    """

    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id", description="主键ID")
    indices: int | list[int] = Field(
        default_factory=list, description="选项索引，单选为 int，多选为 List[int]，空表示未选择"
    )

    @field_validator("indices")
    @classmethod
    def validate_indices(cls, v: int | list[int]) -> int | list[int]:
        if isinstance(v, list):
            if any(i < 0 for i in v):
                raise ValueError("选项索引不能为负数")
        elif isinstance(v, int) and v < 0:
            raise ValueError("选项索引不能为负数")
        return v


class TextAnswer(BaseModel):
    """填空题答案模型。

    支持单值 (str) 或多值 (List[str])，表示用户填写的内容。
    """

    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id", description="主键ID")
    values: str | list[str] = Field(
        default="", description="填空内容，单值为 str，多值为 List[str]，空表示未填写"
    )


class QuestionResponse(BaseModel):
    """单道题目答案模型。

    支持选择题或填空题答案，允许未完成 (None)。
    """

    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id", description="主键ID")
    question_id: str = Field(..., description="题目 ID")
    answer: ChoiceAnswer | TextAnswer | None = Field(
        default=None, description="答案，可为选择题、填空题或空，外键关联ChoiceAnswer/TextAnswer"
    )


class AssessmentResponse(BaseModel):
    """测评结果模型。

    支持多道题目，每个题目答案独立，支持混合类型（选择/填空），允许未完成（空 answer）。
    """

    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id", description="主键ID")
    assessment_id: str = Field(..., description="测评 ID")
    questions: list[QuestionResponse] = Field(
        default_factory=list,
        description="多道题目答案列表，允许部分未完成，外键关联QuestionResponse",
    )


class LabelSearchResult(BaseModel):
    """检索结果模型，包含 Attachment 和标签列表。"""

    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id", description="主键ID")
    attachment: Attachment = Field(..., description="附件，外键关联Attachment")
    labels: list[str]


class AssessmentLabelMapping(BaseModel):
    """测评标签映射模型。

    定义分数到标签的映射，支持多标签。
    """

    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id", description="主键ID")
    assessment_id: str = Field(..., description="测评 ID，外键关联AssessmentResponse.assessment_id")
    label_mapping: dict[str, list[str]] = Field(
        ..., description="分数范围到标签列表的映射，例如 '0-5': ['低兴趣']"
    )


class LabelExplanation(BaseModel):
    """标签解释模型。

    存储每个测评的标签及其解释。
    """

    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id", description="主键ID")
    assessment_id: str = Field(..., description="测评 ID，外键关联AssessmentResponse.assessment_id")
    label: str = Field(..., description="标签名称")
    explanation: str = Field(..., description="标签解释")
