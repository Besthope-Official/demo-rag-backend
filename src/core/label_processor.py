"""测评标签处理模块。

从数据表获取测评结果，映射到标签，存储到 user_labels 表，初始化标签映射和解释。
兼容 cache.py 和 chat.py 的风格，使用 typing 和 Loguru。
"""

from loguru import logger

from src.exceptions import ConfigException

from .assessment import (
    AssessmentLabelMapping,
    ChoiceAnswer,
    LabelExplanation,
    QuestionResponse,
    TextAnswer,
)
from .user import UserDAO


class LabelProcessor:
    """测评标签处理类。

    提供测评结果到标签的映射和标签映射/解释初始化功能。
    """

    def __init__(self, storage: UserDAO):
        """初始化处理器。

        Args:
            storage: UserDAO 实例。
        """
        self.storage = storage
        self.initialize_label_mappings_and_explanations()  # 确保初始化

    def _calculate_am_dictator_score(self, questions: list[QuestionResponse]) -> list[float]:
        """计算 AM-Dictator 测评的送出比例。

        Args:
            questions: 测评题目列表。
        Returns:
            送出比例列表 [0, 1]。
        """
        ratios = []
        for question in questions:
            if (
                question.answer
                and isinstance(question.answer, TextAnswer)
                and question.question_id.startswith("AM_Dictator_Sent_Ratio")
            ):
                values = question.answer.values
                if isinstance(values, list) and len(values) >= 2:
                    try:
                        sent = float(values[1])  # 送出的金额
                        total = float(values[0]) + sent  # 初始禀赋 + 送出
                        ratio = sent / total if total > 0 else 0.0
                        ratios.append(min(max(ratio, 0.0), 1.0))  # 限制 [0, 1]
                    except (ValueError, IndexError):
                        logger.warning(
                            f"AM-Dictator 测评 {question.question_id} 数据无效: {values}"
                        )
        return ratios

    def _calculate_trust_game_score(self, questions: list[QuestionResponse]) -> dict[str, float]:
        """计算 Trust Game 测评的信任和可信程度。

        Args:
            questions: 测评题目列表。
        Returns:
            字典包含 Trust_Offer_Ratio 和 Trust_Return_Ratio_Ave。
        """
        offer_ratio = 0.0
        return_ratios = []
        for question in questions:
            if question.answer:
                if question.question_id == "Trust_Offer" and isinstance(
                    question.answer, ChoiceAnswer
                ):
                    offer_ratio = (
                        float(question.answer.indices) / 20
                        if question.answer.indices is not None and question.answer.indices >= 0
                        else 0.0
                    )
                elif question.question_id.startswith("Trust_Return_Ratio_") and isinstance(
                    question.answer, TextAnswer
                ):
                    values = question.answer.values
                    if isinstance(values, str) and "/" in values:
                        try:
                            sent, returned = map(float, values.split("/"))
                            sent *= 3  # 假设返回金额基于 3 倍初始金额
                            ratio = returned / sent if sent > 0 else 0.0
                            return_ratios.append(min(max(ratio, 0.0), 1.0))
                        except (ValueError, IndexError):
                            logger.warning(
                                f"Trust Game 测评 {question.question_id} 数据无效: {values}"
                            )
        return {
            "Trust_Offer_Ratio": offer_ratio,
            "Trust_Return_Ratio_Ave": (
                sum(return_ratios) / len(return_ratios) if return_ratios else 0.0
            ),
        }

    def _calculate_ultimatum_score(self, questions: list[QuestionResponse]) -> dict[str, float]:
        """计算 Ultimatum Game 测评的 offer 和 MAO 比例。

        Args:
            questions: 测评题目列表。
        Returns:
            字典包含 Ultimatum_Offer_Ratio 和 Ultimatum_MAO_Ratio。
        """
        offer_ratio = 0.0
        mao_ratio = 0.0
        for question in questions:
            if question.answer and isinstance(question.answer, TextAnswer):
                values = question.answer.values
                if question.question_id == "Ultimatum_Offer" and isinstance(
                    values, (str | float | int)
                ):
                    try:
                        offer_ratio = float(values) if 0 <= float(values) <= 1 else 0.0
                    except (ValueError, TypeError):
                        logger.warning(f"Ultimatum_Offer 数据无效: {values}")
                elif question.question_id == "Ultimatum_MAO" and isinstance(
                    values, (str | float | int)
                ):
                    try:
                        mao_ratio = float(values) if 0 <= float(values) <= 1 else 0.0
                    except (ValueError, TypeError):
                        logger.warning(f"Ultimatum_MAO 数据无效: {values}")
        return {"Ultimatum_Offer_Ratio": offer_ratio, "Ultimatum_MAO_Ratio": mao_ratio}

    def _calculate_pgg_score(self, questions: list[QuestionResponse]) -> float:
        """计算 Public Goods Game 测评的投入比例。

        Args:
            questions: 测评题目列表。
        Returns:
            投入比例 [0, 1]。
        """
        ratio = 0.0
        for question in questions:
            if (
                question.answer
                and isinstance(question.answer, TextAnswer)
                and question.question_id == "PGG_Input"
            ):
                values = question.answer.values
                if isinstance(values, (str | float | int)):
                    try:
                        ratio = float(values) if 0 <= float(values) <= 1 else 0.0
                    except (ValueError, TypeError):
                        logger.warning(f"PGG_Input 数据无效: {values}")
        return ratio

    def _calculate_risk_preference_score(
        self, questions: list[QuestionResponse]
    ) -> dict[str, float]:
        """计算 Risk Preference 测评的分数。

        Args:
            questions: 测评题目列表。
        Returns:
            字典包含 Risk_Gain_Anumber, Risk_Loss_Anumber, Loss_Aversion。
        """
        gain_anumber = 0.0
        loss_anumber = 0.0
        for question in questions:
            if question.answer and isinstance(question.answer, ChoiceAnswer):
                if question.question_id == "Risk_Gain":
                    gain_anumber = (
                        float(question.answer.indices)
                        if question.answer.indices is not None
                        else 0.0
                    )
                elif question.question_id == "Risk_Loss":
                    loss_anumber = (
                        float(question.answer.indices)
                        if question.answer.indices is not None
                        else 0.0
                    )
        loss_aversion = loss_anumber - gain_anumber
        return {
            "Risk_Gain_Anumber": gain_anumber,
            "Risk_Loss_Anumber": loss_anumber,
            "Loss_Aversion": loss_aversion,
        }

    def _calculate_time_preference_score(
        self, questions: list[QuestionResponse]
    ) -> dict[str, float]:
        """计算 Time Preference 测评的分数。

        Args:
            questions: 测评题目列表。
        Returns:
            字典包含 Time_Recent_Anumber 和 Time_Present_Bias。
        """
        recent_anumber = 0.0
        future_anumber = 0.0
        for question in questions:
            if question.answer and isinstance(question.answer, ChoiceAnswer):
                if question.question_id == "Time_Recent":
                    recent_anumber = (
                        float(question.answer.indices)
                        if question.answer.indices is not None
                        else 0.0
                    )
                elif question.question_id == "Time_Future":
                    future_anumber = (
                        float(question.answer.indices)
                        if question.answer.indices is not None
                        else 0.0
                    )
        present_bias = recent_anumber - future_anumber
        return {"Time_Recent_Anumber": recent_anumber, "Time_Present_Bias": present_bias}

    def _calculate_optimism_score(self, questions: list[QuestionResponse]) -> dict[str, float]:
        """计算 乐观悲观测评的分数。

        Args:
            questions: 测评题目列表。
        Returns:
            字典包含 A_score, B_score, Optimistic_score。
        """
        a_score = 0
        b_score = 0
        for question in questions:
            if question.answer and isinstance(question.answer, ChoiceAnswer):
                if question.question_id in ["Q3", "Q4", "Q7", "Q9", "Q11", "Q12"]:
                    a_score += 1 if question.answer.indices == 0 else 0
                elif question.question_id in ["Q1", "Q2", "Q5", "Q6", "Q8", "Q10"]:
                    b_score += 1 if question.answer.indices == 1 else 0
        optimistic_score = b_score - a_score
        return {"A_score": a_score, "B_score": b_score, "Optimistic_score": optimistic_score}

    def _calculate_sociable_score(self, questions: list[QuestionResponse]) -> float:
        """计算 内向外向测评的分数。

        Args:
            questions: 测评题目列表。
        Returns:
            Sociable_score [0, 15]。
        """
        sociable_score = 0
        for question in questions:
            if (
                question.answer
                and isinstance(question.answer, ChoiceAnswer)
                and question.question_id.startswith("Q")
            ):
                sociable_score += 1 if question.answer.indices == 0 else 0
        return min(max(sociable_score, 0), 15)

    def _map_to_labels(
        self, assessment_id: str, score: dict[str, float] = None, single_score: float = None
    ) -> list[str]:
        """根据测评类型和分数映射到标签。

        Args:
            assessment_id: 测评 ID。
            score: 多个分数的字典（如 Trust Game）。
            single_score: 单个分数（如 PGG）。
        Returns:
            标签列表。
        """
        mapping = self.storage.get_label_mapping(assessment_id)
        if not mapping:
            logger.warning(f"测评 {assessment_id} 的标签映射未找到")
            return []
        labels = []
        if assessment_id == "assess_1":  # AM-Dictator
            ratios = score.get("AM_Dictator_Sent_Ratios", [])
            if ratios and all(r == 0 for r in ratios[:2]):
                labels.append("Selfish")
            elif (
                ratios
                and all(r == 1 for r in ratios[:2])
                and all(r == 0 for r in ratios[2:4])
                and 0.4 < ratios[4] < 0.6
            ):
                labels.append("Coasian")
            elif ratios and all(0.2 < r < 0.4 for r in ratios):
                labels.append("Rawlsian")
        elif assessment_id == "assess_2":  # Trust Game
            offer_ratio = score.get("Trust_Offer_Ratio", 0.0)
            return_ratio_ave = score.get("Trust_Return_Ratio_Ave", 0.0)
            if offer_ratio == 0:
                labels.append("Selfish_No-Trust")
            elif offer_ratio <= 0.3:
                labels.append("Low-Trust")
            else:
                labels.append("High-Trust")
            if return_ratio_ave == 0:
                labels.append("Opportunistic")
            elif return_ratio_ave < 0.3:
                labels.append("Conditional_Reciprocators")
            else:
                labels.append("Strong_Reciprocators")
        elif assessment_id == "assess_3":  # Ultimatum Game
            offer_ratio = score.get("Ultimatum_Offer_Ratio", 0.0)
            if offer_ratio < 0.2:
                labels.extend(["Selfish", "Purely_Selfish"])
            elif 0.2 <= offer_ratio <= 0.5:
                labels.extend(["Fair", "Conditional_Fair"])
            else:
                labels.extend(["Altruistic", "Strongly_Fair"])
        elif assessment_id == "assess_4":  # Public Goods Game
            if single_score == 0:
                labels.append("Free_Rider")
            elif single_score <= 0.4:
                labels.append("Low_Contributor")
            elif single_score <= 0.8:
                labels.append("High_Contributor")
            else:
                labels.append("Unconditional_Cooperator")
        elif assessment_id == "assess_5":  # Risk Preference
            gain_anumber = score.get("Risk_Gain_Anumber", 0.0)
            loss_anumber = score.get("Risk_Loss_Anumber", 0.0)
            loss_aversion = score.get("Loss_Aversion", 0.0)
            if 0 <= gain_anumber <= 5:
                labels.append("Risk_Averse_Gain")
            elif gain_anumber == 6:
                labels.append("Risk_Neutral_Gain")
            elif 7 <= gain_anumber <= 10:
                labels.append("Risk_Seeking_Gain")
            if 0 <= loss_anumber <= 2:
                labels.append("Risk_Averse_Loss")
            elif loss_anumber == 3:
                labels.append("Risk_Neutral_Loss")
            elif 4 <= loss_anumber <= 10:
                labels.append("Risk_Seeking_Loss")
            if loss_aversion > 0:
                labels.append("Loss_Averse")
        elif assessment_id == "assess_7":  # Time Preference
            recent_anumber = score.get("Time_Recent_Anumber", 0.0)
            present_bias = score.get("Time_Present_Bias", 0.0)
            if 0 <= recent_anumber <= 2:
                labels.append("Highly_Patient")
            elif 3 <= recent_anumber <= 4:
                labels.append("Patient")
            elif 5 <= recent_anumber <= 7:
                labels.append("Impatient")
            elif 8 <= recent_anumber <= 10:
                labels.append("Highly_Impatient")
            if present_bias > 0:
                labels.append("Present_Biased")
        elif assessment_id == "assess_8":  # Optimism
            optimistic_score = score.get("Optimistic_score", 0.0)
            if optimistic_score >= 2:
                labels.append("Optimistic")
            elif optimistic_score == 1:
                labels.append("Moderate")
            elif optimistic_score <= 0:
                labels.append("Pessimistic")
        elif assessment_id == "assess_9":  # Sociable
            sociable_score = score.get("Sociable_score", 0.0)
            if sociable_score > 10:
                labels.append("Introverted")
            elif sociable_score < 5:
                labels.append("Extraverted")
            else:
                labels.append("Ambivert")
        return list(set(labels))  # 去重

    def process_assessments(self, username: str) -> list[str]:
        """处理用户测评结果，映射到标签并存储。

        Args:
            username: 用户名。
        Returns:
            所有测评的标签列表。
        """
        storage = self.storage
        assessments = storage.get_assessments(username)
        all_labels = []
        for assess in assessments:
            if not assess.questions:  # 跳过未完成测评
                continue
            # 根据测评 ID 调用特定计算函数
            score = {}
            if assess.assessment_id == "assess_1":  # AM-Dictator
                ratios = self._calculate_am_dictator_score(assess.questions)
                score["AM_Dictator_Sent_Ratios"] = ratios
            elif assess.assessment_id == "assess_2":  # Trust Game
                score.update(self._calculate_trust_game_score(assess.questions))
            elif assess.assessment_id == "assess_3":  # Ultimatum Game
                score.update(self._calculate_ultimatum_score(assess.questions))
            elif assess.assessment_id == "assess_4":  # Public Goods Game
                score["PGG_Input_Ratio"] = self._calculate_pgg_score(assess.questions)
            elif assess.assessment_id == "assess_5":  # Risk Preference
                score.update(self._calculate_risk_preference_score(assess.questions))
            elif assess.assessment_id == "assess_7":  # Time Preference
                score.update(self._calculate_time_preference_score(assess.questions))
            elif assess.assessment_id == "assess_8":  # Optimism
                score.update(self._calculate_optimism_score(assess.questions))
            elif assess.assessment_id == "assess_9":  # Sociable
                score["Sociable_score"] = self._calculate_sociable_score(assess.questions)
            labels = self._map_to_labels(
                assess.assessment_id,
                score if score else None,
                score.get("PGG_Input_Ratio") if assess.assessment_id == "assess_4" else None,
            )
            if labels:
                storage.save_user_labels(username, assess.assessment_id, labels)
                all_labels.extend(labels)
        return list(set(all_labels))

    def initialize_label_mappings_and_explanations(self, assessment_ids: list[str] = None):
        """初始化标签映射和解释。

        Args:
            assessment_ids: 测评 ID 列表，若为 None 则使用默认示例。
        """
        try:
            if assessment_ids is None:
                assessment_ids = [f"assess_{i}" for i in range(1, 10)]
            mappings = []
            explanations = []
            for assess_id in assessment_ids:
                if assess_id == "assess_1":  # AM-Dictator
                    mappings.append(
                        AssessmentLabelMapping(
                            assessment_id=assess_id,
                            label_mapping={
                                "0-0.1": ["Selfish"],
                                "0.9-1.1": ["Coasian"],
                                "0.2-0.4": ["Rawlsian"],
                            },
                        )
                    )
                    explanations.extend(
                        [
                            LabelExplanation(
                                assessment_id=assess_id,
                                label="Selfish",
                                explanation="用户在 AM-Dictator 测评中完全自私。",
                            ),
                            LabelExplanation(
                                assessment_id=assess_id,
                                label="Coasian",
                                explanation="用户在 AM-Dictator 测评中效率导向。",
                            ),
                            LabelExplanation(
                                assessment_id=assess_id,
                                label="Rawlsian",
                                explanation="用户在 AM-Dictator 测评中公平导向。",
                            ),
                        ]
                    )
                elif assess_id == "assess_2":  # Trust Game
                    mappings.append(
                        AssessmentLabelMapping(
                            assessment_id=assess_id,
                            label_mapping={
                                "0-0.1": ["Selfish_No-Trust", "Opportunistic"],
                                "0.1-0.3": ["Low-Trust", "Conditional_Reciprocators"],
                                "0.3-1.1": ["High-Trust", "Strong_Reciprocators"],
                            },
                        )
                    )
                    explanations.extend(
                        [
                            LabelExplanation(
                                assessment_id=assess_id,
                                label="Selfish_No-Trust",
                                explanation="用户在 Trust Game 中不信任他人。",
                            ),
                            LabelExplanation(
                                assessment_id=assess_id,
                                label="Low-Trust",
                                explanation="用户在 Trust Game 中谨慎信任。",
                            ),
                            LabelExplanation(
                                assessment_id=assess_id,
                                label="High-Trust",
                                explanation="用户在 Trust Game 中高度信任。",
                            ),
                            LabelExplanation(
                                assessment_id=assess_id,
                                label="Opportunistic",
                                explanation="用户在 Trust Game 中完全自利。",
                            ),
                            LabelExplanation(
                                assessment_id=assess_id,
                                label="Conditional_Reciprocators",
                                explanation="用户在 Trust Game 中部分互惠。",
                            ),
                            LabelExplanation(
                                assessment_id=assess_id,
                                label="Strong_Reciprocators",
                                explanation="用户在 Trust Game 中强互惠。",
                            ),
                        ]
                    )
                elif assess_id == "assess_3":  # Ultimatum Game
                    mappings.append(
                        AssessmentLabelMapping(
                            assessment_id=assess_id,
                            label_mapping={
                                "0-0.2": ["Selfish", "Purely_Selfish"],
                                "0.2-0.5": ["Fair", "Conditional_Fair"],
                                "0.5-1.1": ["Altruistic", "Strongly_Fair"],
                            },
                        )
                    )
                    explanations.extend(
                        [
                            LabelExplanation(
                                assessment_id=assess_id,
                                label="Selfish",
                                explanation="用户在 Ultimatum Game 中自利型。",
                            ),
                            LabelExplanation(
                                assessment_id=assess_id,
                                label="Fair",
                                explanation="用户在 Ultimatum Game 中公平型。",
                            ),
                            LabelExplanation(
                                assessment_id=assess_id,
                                label="Altruistic",
                                explanation="用户在 Ultimatum Game 中利他型。",
                            ),
                            LabelExplanation(
                                assessment_id=assess_id,
                                label="Purely_Selfish",
                                explanation="用户在 Ultimatum Game 中完全自私。",
                            ),
                            LabelExplanation(
                                assessment_id=assess_id,
                                label="Conditional_Fair",
                                explanation="用户在 Ultimatum Game 中条件性公平。",
                            ),
                            LabelExplanation(
                                assessment_id=assess_id,
                                label="Strongly_Fair",
                                explanation="用户在 Ultimatum Game 中强烈公平。",
                            ),
                        ]
                    )
                elif assess_id == "assess_4":  # Public Goods Game
                    mappings.append(
                        AssessmentLabelMapping(
                            assessment_id=assess_id,
                            label_mapping={
                                "0-0.01": ["Free_Rider"],
                                "0.01-0.4": ["Low_Contributor"],
                                "0.4-0.8": ["High_Contributor"],
                                "0.8-1.1": ["Unconditional_Cooperator"],
                            },
                        )
                    )
                    explanations.extend(
                        [
                            LabelExplanation(
                                assessment_id=assess_id,
                                label="Free_Rider",
                                explanation="用户在 PGG 中搭便车。",
                            ),
                            LabelExplanation(
                                assessment_id=assess_id,
                                label="Low_Contributor",
                                explanation="用户在 PGG 中低合作。",
                            ),
                            LabelExplanation(
                                assessment_id=assess_id,
                                label="High_Contributor",
                                explanation="用户在 PGG 中高合作。",
                            ),
                            LabelExplanation(
                                assessment_id=assess_id,
                                label="Unconditional_Cooperator",
                                explanation="用户在 PGG 中无条件合作。",
                            ),
                        ]
                    )
                elif assess_id == "assess_5":  # Risk Preference
                    mappings.append(
                        AssessmentLabelMapping(
                            assessment_id=assess_id,
                            label_mapping={
                                "0-5": ["Risk_Averse_Gain"],
                                "6-6": ["Risk_Neutral_Gain"],
                                "7-10": ["Risk_Seeking_Gain"],
                                "0-2": ["Risk_Averse_Loss"],
                                "3-3": ["Risk_Neutral_Loss"],
                                "4-10": ["Risk_Seeking_Loss"],
                                "-10-0": ["No_Loss_Aversion"],
                                "0-10": ["Loss_Averse"],
                            },
                        )
                    )
                    explanations.extend(
                        [
                            LabelExplanation(
                                assessment_id=assess_id,
                                label="Risk_Averse_Gain",
                                explanation="用户在收益情景风险规避。",
                            ),
                            LabelExplanation(
                                assessment_id=assess_id,
                                label="Risk_Neutral_Gain",
                                explanation="用户在收益情景风险中性。",
                            ),
                            LabelExplanation(
                                assessment_id=assess_id,
                                label="Risk_Seeking_Gain",
                                explanation="用户在收益情景风险喜好。",
                            ),
                            LabelExplanation(
                                assessment_id=assess_id,
                                label="Risk_Averse_Loss",
                                explanation="用户在损失情景风险规避。",
                            ),
                            LabelExplanation(
                                assessment_id=assess_id,
                                label="Risk_Neutral_Loss",
                                explanation="用户在损失情景风险中性。",
                            ),
                            LabelExplanation(
                                assessment_id=assess_id,
                                label="Risk_Seeking_Loss",
                                explanation="用户在损失情景风险喜好。",
                            ),
                            LabelExplanation(
                                assessment_id=assess_id,
                                label="No_Loss_Aversion",
                                explanation="用户无损失厌恶。",
                            ),
                            LabelExplanation(
                                assessment_id=assess_id,
                                label="Loss_Averse",
                                explanation="用户存在损失厌恶。",
                            ),
                        ]
                    )
                elif assess_id == "assess_7":  # Time Preference
                    mappings.append(
                        AssessmentLabelMapping(
                            assessment_id=assess_id,
                            label_mapping={
                                "0-2": ["Highly_Patient"],
                                "3-4": ["Patient"],
                                "5-7": ["Impatient"],
                                "8-10": ["Highly_Impatient"],
                                "-10-0": ["No_Present_Bias"],
                                "0-10": ["Present_Biased"],
                            },
                        )
                    )
                    explanations.extend(
                        [
                            LabelExplanation(
                                assessment_id=assess_id,
                                label="Highly_Patient",
                                explanation="用户在时间偏好中高度耐心。",
                            ),
                            LabelExplanation(
                                assessment_id=assess_id,
                                label="Patient",
                                explanation="用户在时间偏好中较耐心。",
                            ),
                            LabelExplanation(
                                assessment_id=assess_id,
                                label="Impatient",
                                explanation="用户在时间偏好中较不耐心。",
                            ),
                            LabelExplanation(
                                assessment_id=assess_id,
                                label="Highly_Impatient",
                                explanation="用户在时间偏好中非常不耐心。",
                            ),
                            LabelExplanation(
                                assessment_id=assess_id,
                                label="No_Present_Bias",
                                explanation="用户无现时偏差。",
                            ),
                            LabelExplanation(
                                assessment_id=assess_id,
                                label="Present_Biased",
                                explanation="用户存在现时偏差。",
                            ),
                        ]
                    )
                elif assess_id == "assess_8":  # Optimism
                    mappings.append(
                        AssessmentLabelMapping(
                            assessment_id=assess_id,
                            label_mapping={
                                "-6-0": ["Pessimistic"],
                                "1-1": ["Moderate"],
                                "2-6": ["Optimistic"],
                            },
                        )
                    )
                    explanations.extend(
                        [
                            LabelExplanation(
                                assessment_id=assess_id,
                                label="Pessimistic",
                                explanation="用户在乐观悲观测评中悲观。",
                            ),
                            LabelExplanation(
                                assessment_id=assess_id,
                                label="Moderate",
                                explanation="用户在乐观悲观测评中中等。",
                            ),
                            LabelExplanation(
                                assessment_id=assess_id,
                                label="Optimistic",
                                explanation="用户在乐观悲观测评中乐观。",
                            ),
                        ]
                    )
                elif assess_id == "assess_9":  # Sociable
                    mappings.append(
                        AssessmentLabelMapping(
                            assessment_id=assess_id,
                            label_mapping={
                                "0-4.9": ["Extraverted"],
                                "5-10": ["Ambivert"],
                                "10.1-15": ["Introverted"],
                            },
                        )
                    )
                    explanations.extend(
                        [
                            LabelExplanation(
                                assessment_id=assess_id,
                                label="Extraverted",
                                explanation="用户在内向外向测评中外向。",
                            ),
                            LabelExplanation(
                                assessment_id=assess_id,
                                label="Ambivert",
                                explanation="用户在内向外向测评中中间型。",
                            ),
                            LabelExplanation(
                                assessment_id=assess_id,
                                label="Introverted",
                                explanation="用户在内向外向测评中内向。",
                            ),
                        ]
                    )
            for mapping in mappings:
                self.storage.save_label_mapping(mapping)
            for exp in explanations:
                self.storage.save_label_explanation(exp)
        except Exception as e:
            logger.error(f"初始化标签映射和解释失败: {e}")
            raise ConfigException(f"初始化标签映射和解释失败: {e}") from e
