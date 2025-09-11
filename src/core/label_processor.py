"""测评标签处理模块。

从数据表获取测评结果，映射到标签，存储到 user_labels 表，初始化标签映射和解释。
兼容 cache.py 和 chat.py 的风格，使用 typing 和 Loguru，优化调试日志。
"""

import numpy as np
from loguru import logger

from src.exceptions import ConfigException

from .assessment import (
    AssessmentLabelMapping,
    LabelExplanation,
    QuestionResponse,
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
        logger.info("初始化 LabelProcessor，调用标签映射和解释初始化")
        self.initialize_label_mappings_and_explanations()

    def _calculate_am_dictator_score(
        self, questions: list[QuestionResponse]
    ) -> dict[str, float | list[float]]:
        """计算 AM-Dictator 测评的分数。

        Args:
            questions: 测评题目列表。
        Returns:
            字典包含 AM_Dictator_Sent_Ratios 和 AM_Dictator_Sent_5。
        """
        ratios = [0.0] * 5
        sent_5 = 0.0
        question_map = {q.question_id: q for q in questions if q.answer}
        # logger.debug(f"AM-Dictator 输入问题: {list(question_map.keys())}, 答案类型: {[type(q.answer).__name__ for q in question_map.values()]}")

        for i, qid in enumerate(["q1", "q2", "q3", "q4", "q5"]):
            question = question_map.get(qid)
            # print("not question:", not question)
            # print("not isinstance(question.answer, TextAnswer):", not isinstance(question.answer, TextAnswer))
            # if not question or not isinstance(question.answer, TextAnswer):
            #     logger.warning(f"AM-Dictator 缺少问题 {qid} 或答案类型错误")
            #     continue
            values = question.answer.values
            logger.debug(f"AM-Dictator 问题 {qid} 原始答案: {values}")
            if not values or (isinstance(values, list) and len(values) < 2):
                logger.warning(f"AM-Dictator 问题 {qid} 数据无效: {values}")
                continue
            try:
                sent = float(values[1] if isinstance(values, list) else values)
                total = float({"q1": 160, "q2": 80, "q3": 160, "q4": 240, "q5": 120}[qid])
                ratio = min(max(sent / total, 0.0), 1.0)
                ratios[i] = ratio
                if qid == "q5":
                    sent_5 = min(max(sent, 0.0), 120.0)
                # logger.debug(
                #     f"AM-Dictator 问题 {qid}: sent={sent}, total={total}, ratio={ratio}")
            except (ValueError, IndexError, TypeError) as e:
                logger.warning(f"AM-Dictator 问题 {qid} 数据转换失败: {values}, 错误: {e}")

        result = {"AM_Dictator_Sent_Ratios": ratios, "AM_Dictator_Sent_5": sent_5}
        logger.info(f"AM-Dictator 分数: {result}")
        return result

    def _calculate_trust_game_score(self, questions: list[QuestionResponse]) -> dict[str, float]:
        """计算 Trust Game 测评的分数。

        Args:
            questions: 测评题目列表。
        Returns:
            字典包含 Trust_Offer, Trust_Return_1 到 Trust_Return_4。
        """
        scores = {
            "Trust_Offer": 0.0,
            "Trust_Return_1": 0.0,
            "Trust_Return_2": 0.0,
            "Trust_Return_3": 0.0,
            "Trust_Return_4": 0.0,
        }
        question_map = {q.question_id: q for q in questions if q.answer}
        # logger.debug(f"Trust_Game 输入问题: {list(question_map.keys())}, 答案类型: {[type(q.answer).__name__ for q in question_map.values()]}")

        for qid, max_value in [
            ("q1", 80.0),
            ("q2", 60.0),
            ("q3", 120.0),
            ("q4", 180.0),
            ("q5", 240.0),
        ]:
            question = question_map.get(qid)
            # if not question or not isinstance(question.answer, TextAnswer):
            #     logger.warning(f"Trust_Game 缺少问题 {qid} 或答案类型错误")
            #     continue
            values = question.answer.values
            # logger.debug(f"Trust_Game 问题 {qid} 原始答案: {values}")
            try:
                value = float(values if isinstance(values, str) else values[0])
                scores[
                    {
                        "q1": "Trust_Offer",
                        "q2": "Trust_Return_1",
                        "q3": "Trust_Return_2",
                        "q4": "Trust_Return_3",
                        "q5": "Trust_Return_4",
                    }[qid]
                ] = min(max(value, 0.0), max_value)
                # logger.debug(
                #     f"Trust_Game 问题 {qid}: value={value}, 限制范围=[0, {max_value}]")
            except (ValueError, IndexError, TypeError) as e:
                logger.warning(f"Trust_Game 问题 {qid} 数据转换失败: {values}, 错误: {e}")

        logger.info(f"Trust_Game 分数: {scores}")
        return scores

    def _calculate_ultimatum_game_score(
        self, questions: list[QuestionResponse]
    ) -> dict[str, float]:
        """计算 Ultimatum Game 测评的分数。

        Args:
            questions: 测评题目列表。
        Returns:
            字典包含 Ultimatum_Offer 和 Ultimatum_MAO。
        """
        scores = {"Ultimatum_Offer": 0.0, "Ultimatum_MAO": 0.0}
        question_map = {q.question_id: q for q in questions if q.answer}
        # logger.debug(
        #     f"Ultimatum_Game 输入问题: {list(question_map.keys())}, 答案类型: {[type(q.answer).__name__ for q in question_map.values()]}")

        for qid, key in [("q1", "Ultimatum_Offer"), ("q2", "Ultimatum_MAO")]:
            question = question_map.get(qid)
            # if not question or not isinstance(question.answer, TextAnswer):
            #     logger.warning(f"Ultimatum_Game 缺少问题 {qid} 或答案类型错误")
            #     continue
            values = question.answer.values
            # logger.debug(f"Ultimatum_Game 问题 {qid} 原始答案: {values}")
            try:
                value = float(values if isinstance(values, str) else values[0])
                max_value = 120.0 if qid == "q2" else float("inf")
                scores[key] = min(max(value, 0.0), max_value)
                # logger.debug(
                #     f"Ultimatum_Game 问题 {qid}: value={value}, 限制范围=[0, {max_value}]")
            except (ValueError, IndexError, TypeError) as e:
                logger.warning(f"Ultimatum_Game 问题 {qid} 数据转换失败: {values}, 错误: {e}")

        logger.info(f"Ultimatum_Game 分数: {scores}")
        return scores

    def _calculate_pgg_score(self, questions: list[QuestionResponse]) -> float:
        """计算 Public Goods Game 测评的投入值。

        Args:
            questions: 测评题目列表。
        Returns:
            投入值 [0, 80]。
        """
        for question in questions:
            if question.answer and question.question_id == "q1":
                values = question.answer.values
                # logger.debug(f"Public_Goods_Game 问题 q1 原始答案: {values}")
                try:
                    value = float(values)
                    result = min(max(value, 0.0), 80.0)
                    # logger.debug(
                    #     f"Public_Goods_Game 问题 q1: value={value}, 限制范围=[0, 80]")
                    logger.info(f"Public_Goods_Game 分数: {result}")
                    return result
                except (ValueError, IndexError, TypeError) as e:
                    logger.warning(f"Public_Goods_Game 问题 q1 数据转换失败: {values}, 错误: {e}")
        logger.warning("Public_Goods_Game 无有效问题数据")
        return 0.0

    def _calculate_risk_gain_score(self, questions: list[QuestionResponse]) -> float:
        """计算 Risk Gain 测评的分数。

        Args:
            questions: 测评题目列表。
        Returns:
            Risk_Gain_Anumber。
        """
        count = 0
        for question in questions:
            # if not question.answer or not isinstance(question.answer, ChoiceAnswer):
            #     logger.warning(f"Risk_Gain 问题 {question.question_id} 缺少答案或类型错误")
            #     continue
            indices = question.answer.indices
            # logger.debug(f"Risk_Gain 问题 {question.question_id} 原始答案: indices={indices}")
            try:
                count += 1 if int(indices) == 0 else 0
                # logger.debug(f"Risk_Gain 问题 {question.question_id}: indices={indices}, count={count}")
            except (ValueError, TypeError) as e:
                logger.warning(
                    f"Risk_Gain 问题 {question.question_id} 数据转换失败: {indices}, 错误: {e}"
                )
        result = min(max(count, 0), 10)
        logger.info(f"Risk_Gain 分数: {result}")
        return result

    def _calculate_risk_loss_score(self, questions: list[QuestionResponse]) -> float:
        """计算 Risk Loss 测评的分数。

        Args:
            questions: 测评题目列表。
        Returns:
            Risk_Loss_Anumber。
        """
        count = 0
        for question in questions:
            # if not question.answer or not isinstance(question.answer, ChoiceAnswer):
            #     logger.warning(f"Risk_Loss 问题 {question.question_id} 缺少答案或类型错误")
            #     continue
            indices = question.answer.indices
            # logger.debug(f"Risk_Loss 问题 {question.question_id} 原始答案: indices={indices}")
            try:
                count += 1 if int(indices) == 0 else 0
                # logger.debug(f"Risk_Loss 问题 {question.question_id}: indices={indices}, count={count}")
            except (ValueError, TypeError) as e:
                logger.warning(
                    f"Risk_Loss 问题 {question.question_id} 数据转换失败: {indices}, 错误: {e}"
                )
        result = min(max(count, 0), 10)
        logger.info(f"Risk_Loss 分数: {result}")
        return result

    def _calculate_risk_mixed_score(self, questions: list[QuestionResponse]) -> float:
        """计算 Risk Mixed 测评的分数。

        Args:
            questions: 测评题目列表。
        Returns:
            Risk_Mixed_Anumber。
        """
        count = 0
        for question in questions:
            # if not question.answer or not isinstance(question.answer, ChoiceAnswer):
            #     logger.warning(f"Risk_Mixed 问题 {question.question_id} 缺少答案或类型错误")
            #     continue
            indices = question.answer.indices
            # logger.debug(f"Risk_Mixed 问题 {question.question_id} 原始答案: indices={indices}")
            try:
                count += 1 if int(indices) == 0 else 0
                # logger.debug(f"Risk_Mixed 问题 {question.question_id}: indices={indices}, count={count}")
            except (ValueError, TypeError) as e:
                logger.warning(
                    f"Risk_Mixed 问题 {question.question_id} 数据转换失败: {indices}, 错误: {e}"
                )
        result = min(max(count, 0), 10)
        logger.info(f"Risk_Mixed 分数: {result}")
        return result

    def _calculate_time_preference_score(
        self, questions: list[QuestionResponse]
    ) -> dict[str, float]:
        """计算 Time Preference 测评的分数。

        Args:
            questions: 测评题目列表。
        Returns:
            字典包含 Time_Recent_Anumber 和 Time_Future_Anumber。
        """
        recent_count = 0
        future_count = 0
        for i, question in enumerate(questions):
            # if not question.answer or not isinstance(question.answer, ChoiceAnswer):
            #     logger.warning(f"Time_Preference 问题 {question.question_id} 缺少答案或类型错误")
            #     continue
            indices = question.answer.indices
            # logger.debug(f"Time_Preference 问题 {question.question_id} 原始答案: indices={indices}")
            try:
                count = 1 if int(indices) == 0 else 0
                if i < 10:  # q1-q10: Recent
                    recent_count += count
                else:  # q11-q20: Future
                    future_count += count
                # logger.debug(f"Time_Preference 问题 {question.question_id}: indices={indices}, recent_count={recent_count}, future_count={future_count}")
            except (ValueError, TypeError) as e:
                logger.warning(
                    f"Time_Preference 问题 {question.question_id} 数据转换失败: {indices}, 错误: {e}"
                )
        result = {
            "Time_Recent_Anumber": min(max(recent_count, 0), 10),
            "Time_Future_Anumber": min(max(future_count, 0), 10),
        }
        logger.info(f"Time_Preference 分数: {result}")
        return result

    def _calculate_optimistic_score(self, questions: list[QuestionResponse]) -> dict[str, float]:
        """计算 Optimistic 测评的分数。

        Args:
            questions: 测评题目列表。
        Returns:
            字典包含 A_score, B_score, Optimistic_score。
        """
        a_score = 0
        b_score = 0
        a_questions = ["q3", "q4", "q7", "q8", "q9", "q11", "q12"]
        b_questions = ["q1", "q2", "q5", "q6", "q10"]
        for question in questions:
            # if not question.answer or not isinstance(question.answer, ChoiceAnswer):
            #     logger.warning(f"Optimistic 问题 {question.question_id} 缺少答案或类型错误")
            #     continue
            qid = question.question_id
            indices = question.answer.indices
            # logger.debug(f"Optimistic 问题 {qid} 原始答案: indices={indices}")
            try:
                if qid in a_questions:
                    a_score += 1 if int(indices) == 0 else 0
                elif qid in b_questions:
                    b_score += 1 if int(indices) == 1 else 0
                # logger.debug(f"Optimistic 问题 {qid}: indices={indices}, a_score={a_score}, b_score={b_score}")
            except (ValueError, TypeError) as e:
                logger.warning(f"Optimistic 问题 {qid} 数据转换失败: {indices}, 错误: {e}")
        result = {"A_score": a_score, "B_score": b_score, "Optimistic_score": b_score - a_score}
        logger.info(f"Optimistic 分数: {result}")
        return result

    def _calculate_introverted_score(self, questions: list[QuestionResponse]) -> float:
        """计算 Introverted 测评的分数。

        Args:
            questions: 测评题目列表。
        Returns:
            Introverted_score。
        """
        count = 0
        for question in questions:
            # if not question.answer or not isinstance(question.answer, ChoiceAnswer):
            #     logger.warning(f"Introverted 问题 {question.question_id} 缺少答案或类型错误")
            #     continue
            indices = question.answer.indices
            # logger.debug(f"Introverted 问题 {question.question_id} 原始答案: indices={indices}")
            try:
                count += 1 if int(indices) == 0 else 0
                # logger.debug(f"Introverted 问题 {question.question_id}: indices={indices}, count={count}")
            except (ValueError, TypeError) as e:
                logger.warning(
                    f"Introverted 问题 {question.question_id} 数据转换失败: {indices}, 错误: {e}"
                )
        result = min(max(count, 0), 15)
        logger.info(f"Introverted 分数: {result}")
        return result

    def _map_to_labels(
        self,
        assessment_id: str,
        score: dict[str, float] | None = None,
        single_score: float | None = None,
    ) -> list[str]:
        """根据测评类型和分数映射到标签。

        Args:
            assessment_id: 测评 ID。
            score: 分数字典（如 AM_Dictator）。
            single_score: 单一分数（如 PGG）。
        Returns:
            标签列表。
        """
        labels = []
        # logger.debug(f"映射标签，测评 ID: {assessment_id}, 分数: {score}, 单分数: {single_score}")

        if assessment_id == "AM_Dictator":
            ratios = np.array(score.get("AM_Dictator_Sent_Ratios", [0.0] * 5))
            sent_5 = score.get("AM_Dictator_Sent_5", 0.0)
            # AM_Dictator_Type (基于欧式距离)
            distances = {
                "Selfish / Perfectly Selfish / 自私型": np.linalg.norm(
                    ratios - np.array([0, 0, 0, 0, 0])
                ),
                "Coasian / Perfect Substitutes / Efficiency-focused / Perfectly Selfless / 效率导向型": np.linalg.norm(
                    ratios - np.array([1, 1, 0, 0, 0.5])
                ),
                "Rawlsian / Leontief / Equality-focused / Equalitarians / 公平导向型": np.linalg.norm(
                    ratios - np.array([0.33, 0.25, 0.67, 0.75, 0.5])
                ),
            }
            am_dictator_type = min(distances, key=distances.get)
            labels.append(am_dictator_type)
            # logger.debug(f"AM_Dictator 距离: {distances}, 选择类型: {am_dictator_type}")
            # Dictator_Type (基于 sent_5)
            if sent_5 == 0:
                labels.append("自私者")
            elif 0 < sent_5 < 30:
                labels.append("比较自私")
            elif 30 <= sent_5 < 60:
                labels.append("比较利他")
            elif sent_5 == 60:
                labels.append("利他人群中的平等主义者")
            elif sent_5 > 60:
                labels.append("极端利他者")
            # logger.debug(f"AM_Dictator Dictator_Type: sent_5={sent_5}, 标签: {labels[-1]}")

        elif assessment_id == "Trust_Game":
            offer = score.get("Trust_Offer", 0.0)
            return_1 = score.get("Trust_Return_1", 0.0)
            return_2 = score.get("Trust_Return_2", 0.0)
            return_3 = score.get("Trust_Return_3", 0.0)
            return_4 = score.get("Trust_Return_4", 0.0)
            am_dictator_sent_2 = score.get("AM_Dictator_Sent_2", 0.0)
            am_dictator_sent_5 = score.get("AM_Dictator_Sent_5", 0.0)
            # Truster_Type
            if 0 <= offer <= 20:
                labels.append("信任度低类型")
            elif 20 < offer < 40:
                labels.append("信任度较低类型")
            elif 40 <= offer < 60:
                labels.append("信任度中等类型")
            elif 60 <= offer < 80:
                labels.append("信任度较高类型")
            elif offer == 80:
                labels.append("信任度极高类型")
            # logger.debug(f"Trust_Game Truster_Type: offer={offer}, 标签: {labels[-1] if labels else '无'}")
            # Trustee_1_Type
            if return_1 == 0:
                labels.append("当对方为信任度低类型，用户的值得信任程度为低")
            elif 0 < return_1 <= 10:
                labels.append("当对方为信任度低类型，用户值得信任程度为较低")
            elif 10 < return_1 <= 20:
                labels.append("当对方为信任度低类型，用户值得信任程度为中等")
            elif 20 < return_1 < 30:
                labels.append("当对方为信任度低类型，用户值得信任程度为较高")
            elif 30 <= return_1:
                labels.append("当对方为信任度低类型，用户值得信任程度为高")
            # logger.debug(f"Trust_Game Trustee_1_Type: return_1={return_1}, 标签: {labels[-1] if labels else '无'}")
            # Trustee_2_Type
            if 0 <= return_2 <= 10:
                labels.append("当对方为信任度中等类型，用户的值得信任程度为低")
            elif 10 < return_2 <= 20:
                labels.append("当对方为信任度中等类型，用户值得信任程度为较低")
            elif 20 < return_2 <= 40:
                labels.append("当对方为信任度中等类型，用户值得信任程度为中等")
            elif 40 < return_2 < 60:
                labels.append("当对方为信任度中等类型，用户值得信任程度为较高")
            elif 60 <= return_2:
                labels.append("当对方为信任度中等类型，用户值得信任程度为高")
            # logger.debug(f"Trust_Game Trustee_2_Type: return_2={return_2}, 标签: {labels[-1] if labels else '无'}")
            # Trustee_3_Type
            if 0 <= return_3 < 60:
                labels.append("当对方为信任度较高类型，用户的值得信任程度为低")
            elif 60 <= return_3 <= 70:
                labels.append("当对方为信任度较高类型，用户值得信任程度为较低")
            elif 70 < return_3 <= 80:
                labels.append("当对方为信任度较高类型，用户值得信任程度为中等")
            elif 80 < return_3 < 90:
                labels.append("当对方为信任度较高类型，用户值得信任程度为较高")
            elif 90 <= return_3:
                labels.append("当对方为信任度较高类型，用户值得信任程度为高")
            # logger.debug(f"Trust_Game Trustee_3_Type: return_3={return_3}, 标签: {labels[-1] if labels else '无'}")
            # Trustee_4_Type
            if return_4 == 0:
                labels.append("当对方为信任度极高类型，用户的值得信任程度为低")
            elif 0 < return_4 <= 80:
                labels.append("当对方为信任度极高类型，用户值得信任程度为较低")
            elif 80 < return_4 <= 100:
                labels.append("当对方为信任度极高类型，用户值得信任程度为中等")
            elif 100 < return_4 < 120:
                labels.append("当对方为信任度极高类型，用户值得信任程度为较高")
            elif 120 <= return_4:
                labels.append("当对方为信任度极高类型，用户值得信任程度为高")
            # logger.debug(f"Trust_Game Trustee_4_Type: return_4={return_4}, 标签: {labels[-1] if labels else '无'}")
            # Hope_Return_Type
            hope_return = offer - am_dictator_sent_2
            if hope_return < 0:
                labels.append("用户的期待回报程度为扰动型")
            elif 0 <= hope_return <= 10:
                labels.append("用户的期待回报程度为低")
            elif 10 < hope_return <= 20:
                labels.append("用户的期待回报程度为中等")
            elif hope_return > 20:
                labels.append("用户的期待回报程度为高")
            # logger.debug(f"Trust_Game Hope_Return_Type: hope_return={hope_return}, 标签: {labels[-1] if labels else '无'}")
            # Gratitude_Spite_Return_Type
            gratitude_spite = return_2 - am_dictator_sent_5
            if gratitude_spite >= 0:
                labels.append("用户在信任博弈中感恩对方")
            else:
                labels.append("用户在信任博弈中怨恨对方")
            # logger.debug(f"Trust_Game Gratitude_Spite_Return_Type: gratitude_spite={gratitude_spite}, 标签: {labels[-1] if labels else '无'}")

        elif assessment_id == "Ultimatum_Game":
            offer = score.get("Ultimatum_Offer", 0.0)
            mao = score.get("Ultimatum_MAO", 0.0)
            am_dictator_sent_5 = score.get("AM_Dictator_Sent_5", 0.0)
            offer_diff = offer - am_dictator_sent_5
            # Ultimatum_Offer_Type
            if am_dictator_sent_5 >= 30 and offer_diff <= 0:
                labels.append("利他非策略型")
            elif am_dictator_sent_5 >= 30 and offer_diff > 0:
                labels.append("利他策略型")
            elif am_dictator_sent_5 < 30 and offer_diff < 50:
                labels.append("自私非策略型")
            elif am_dictator_sent_5 < 30 and offer_diff >= 50:
                labels.append("自私策略型")
            # logger.debug(f"Ultimatum_Game Ultimatum_Offer_Type: sent_5={am_dictator_sent_5}, offer_diff={offer_diff}, 标签: {labels[-1] if labels else '无'}")
            # Ultimatum_MAO_Type
            if mao < 20:
                labels.append("极低公平需求者")
            elif 20 <= mao < 40:
                labels.append("低公平需求者")
            elif 40 <= mao < 60:
                labels.append("高公平需求者")
            elif 60 <= mao:
                labels.append("极高公平需求者")
            # logger.debug(f"Ultimatum_Game Ultimatum_MAO_Type: mao={mao}, 标签: {labels[-1] if labels else '无'}")

        elif assessment_id == "Public_Goods_Game":
            if single_score == 0:
                labels.append("纯搭便车者（合作意愿极低）")
            elif 0 < single_score <= 30:
                labels.append("合作意愿较低")
            elif 30 < single_score < 50:
                labels.append("合作意愿中等")
            elif 50 <= single_score < 80:
                labels.append("合作意愿较高")
            elif single_score == 80:
                labels.append("合作意愿极高")
            # logger.debug(f"Public_Goods_Game PGG_Type: single_score={single_score}, 标签: {labels[-1] if labels else '无'}")

        elif assessment_id == "Risk_Gain":
            if 0 <= single_score <= 2:
                labels.append("Highly_Risk_Averse_Gain")
            elif 3 <= single_score <= 5:
                labels.append("Risk_Averse_Gain")
            elif single_score == 6:
                labels.append("Risk_Neutral_Gain")
            elif 7 <= single_score <= 8:
                labels.append("Risk_Seeking_Gain")
            elif 9 <= single_score <= 10:
                labels.append("Highly_Risk_Seeking_Gain")
            # logger.debug(f"Risk_Gain Risk_Gain_Type: single_score={single_score}, 标签: {labels[-1] if labels else '无'}")

        elif assessment_id == "Risk_Loss":
            if 0 <= single_score <= 2:
                labels.append("Risk_Averse_Loss")
            elif single_score == 3:
                labels.append("Risk_Neutral_Loss")
            elif 4 <= single_score <= 5:
                labels.append("Risk_Seeking_Loss")
            elif 6 <= single_score <= 8:
                labels.append("Highly_Risk_Seeking_Loss")
            elif 9 <= single_score <= 10:
                labels.append("Extremely_Risk_Seeking_Loss")
            # logger.debug(f"Risk_Loss Risk_Loss_Type: single_score={single_score}, 标签: {labels[-1] if labels else '无'}")

        elif assessment_id == "Risk_Mixed":
            if 0 <= single_score <= 2:
                labels.append("Extremely_Loss_Aversion")
            elif 3 <= single_score <= 4:
                labels.append("Highly_Loss_Aversion")
            elif 5 <= single_score <= 6:
                labels.append("Loss_Aversion")
            elif 7 <= single_score <= 10:
                labels.append("No_Loss_Aversion")
            # logger.debug(f"Risk_Mixed Loss_Aversion_Type: single_score={single_score}, 标签: {labels[-1] if labels else '无'}")

        elif assessment_id == "Time_Preference":
            recent = score.get("Time_Recent_Anumber", 0.0)
            future = score.get("Time_Future_Anumber", 0.0)
            # Time_Type
            if 0 <= recent <= 2:
                labels.append("Highly_Patient")
            elif 3 <= recent <= 4:
                labels.append("Patient")
            elif 5 <= recent <= 6:
                labels.append("Impatient")
            elif 7 <= recent <= 10:
                labels.append("Highly_Impatient")
            # logger.debug(f"Time_Preference Time_Type: recent={recent}, 标签: {labels[-1] if labels else '无'}")
            # Time_Present_Bias_Type
            if recent - future > 0:
                labels.append("Present_Biased")
            else:
                labels.append("No_Present_Bias")
            # logger.debug(f"Time_Preference Time_Present_Bias_Type: recent-future={recent-future}, 标签: {labels[-1] if labels else '无'}")

        elif assessment_id == "Optimistic":
            optimistic_score = score.get("Optimistic_score", 0.0)
            if optimistic_score >= 2:
                labels.append("Optimist")
            elif optimistic_score == 1:
                labels.append("Moderate")
            elif optimistic_score <= 0:
                labels.append("Pessimist")
            # logger.debug(f"Optimistic Optimistic_Type: optimistic_score={optimistic_score}, 标签: {labels[-1] if labels else '无'}")

        elif assessment_id == "Introverted":
            if single_score > 10:
                labels.append("Introverted")
            elif single_score < 5:
                labels.append("Extroverted")
            else:
                labels.append("Ambivert")
            # logger.debug(f"Introverted Introverted_type: single_score={single_score}, 标签: {labels[-1] if labels else '无'}")

        logger.info(f"测评 {assessment_id} 标签: {labels}")
        return list(set(labels))

    def process_assessments(self, username: str) -> list[str]:
        """处理用户测评结果，映射到标签并存储。

        Args:
            username: 用户名。
        Returns:
            所有测评的标签列表。
        """
        logger.info(f"开始处理用户 {username} 的测评结果")
        storage = self.storage
        assessments = storage.get_assessments(username)
        logger.debug(f"获取用户 {username} 的测评: {[a.assessment_id for a in assessments]}")
        all_labels = []
        am_dictator_scores = {}

        for assess in assessments:
            if not assess.questions:
                logger.warning(f"测评 {assess.assessment_id} 无问题数据，跳过")
                continue
            # logger.debug(f"处理测评 {assess.assessment_id}, 问题数量: {len(assess.questions)}, 问题ID: {[q.question_id for q in assess.questions]}")
            score = {}
            single_score = None

            if assess.assessment_id == "AM_Dictator":
                am_dictator_scores = self._calculate_am_dictator_score(assess.questions)
                score = am_dictator_scores
            elif assess.assessment_id == "Trust_Game":
                score = self._calculate_trust_game_score(assess.questions)
                score["AM_Dictator_Sent_2"] = (
                    am_dictator_scores.get("AM_Dictator_Sent_Ratios", [0.0] * 5)[1] * 80
                )
                score["AM_Dictator_Sent_5"] = am_dictator_scores.get("AM_Dictator_Sent_5", 0.0)
                # logger.debug(f"Trust_Game 使用 AM_Dictator 分数: sent_2={score['AM_Dictator_Sent_2']}, sent_5={score['AM_Dictator_Sent_5']}")
            elif assess.assessment_id == "Ultimatum_Game":
                score = self._calculate_ultimatum_game_score(assess.questions)
                score["AM_Dictator_Sent_5"] = am_dictator_scores.get("AM_Dictator_Sent_5", 0.0)
                # logger.debug(f"Ultimatum_Game 使用 AM_Dictator 分数: sent_5={score['AM_Dictator_Sent_5']}")
            elif assess.assessment_id == "Public_Goods_Game":
                single_score = self._calculate_pgg_score(assess.questions)
            elif assess.assessment_id == "Risk_Gain":
                single_score = self._calculate_risk_gain_score(assess.questions)
            elif assess.assessment_id == "Risk_Loss":
                single_score = self._calculate_risk_loss_score(assess.questions)
            elif assess.assessment_id == "Risk_Mixed":
                single_score = self._calculate_risk_mixed_score(assess.questions)
            elif assess.assessment_id == "Time_Preference":
                score = self._calculate_time_preference_score(assess.questions)
            elif assess.assessment_id == "Optimistic":
                score = self._calculate_optimistic_score(assess.questions)
            elif assess.assessment_id == "Introverted":
                single_score = self._calculate_introverted_score(assess.questions)

            labels = self._map_to_labels(assess.assessment_id, score, single_score)
            if labels:
                self.storage.save_user_labels(username, assess.assessment_id, labels)
                all_labels.extend(labels)
                logger.info(f"用户 {username} 测评 {assess.assessment_id} 保存标签: {labels}")

        all_labels = list(set(all_labels))
        logger.info(f"用户 {username} 所有标签: {all_labels}")
        return all_labels

    def initialize_label_mappings_and_explanations(self, assessment_ids: list[str] = None):
        """初始化标签映射和解释。

        Args:
            assessment_ids: 测评 ID 列表，若为 None 则使用默认十个测评。
        """
        try:
            if assessment_ids is None:
                assessment_ids = [
                    "AM_Dictator",
                    "Trust_Game",
                    "Ultimatum_Game",
                    "Public_Goods_Game",
                    "Risk_Gain",
                    "Risk_Loss",
                    "Risk_Mixed",
                    "Time_Preference",
                    "Optimistic",
                    "Introverted",
                ]
            logger.info(f"初始化标签映射和解释，测评 ID: {assessment_ids}")
            mappings = []
            explanations = []
            for assess_id in assessment_ids:
                if assess_id == "AM_Dictator":
                    mappings.append(
                        AssessmentLabelMapping(
                            assessment_id=assess_id,
                            label_mapping={
                                "dictator_type_0-0": ["自私者"],
                                "dictator_type_0-30": ["比较自私"],
                                "dictator_type_30-60": ["比较利他"],
                                "dictator_type_60-60": ["利他人群中的平等主义者"],
                                "dictator_type_60-120": ["极端利他者"],
                            },
                        )
                    )
                    explanations.extend(
                        [
                            LabelExplanation(
                                assessment_id=assess_id,
                                label="Selfish / Perfectly Selfish / 自私型",
                                explanation="用户在AM独裁者博弈中，所有情景都追求自己的利益最大化。",
                            ),
                            LabelExplanation(
                                assessment_id=assess_id,
                                label="Coasian / Perfect Substitutes / Efficiency-focused / Perfectly Selfless / 效率导向型",
                                explanation="用户在AM独裁者博弈中，最大化社会蛋糕，同时忽略公平。",
                            ),
                            LabelExplanation(
                                assessment_id=assess_id,
                                label="Rawlsian / Leontief / Equality-focused / Equalitarians / 公平导向型",
                                explanation="用户在AM独裁者博弈中，通过以双方最终收入相等进行分配。",
                            ),
                            LabelExplanation(
                                assessment_id=assess_id,
                                label="自私者",
                                explanation="用户在标准独裁者博弈中，送出的值为0，非常自私，追求自己的利益最大化。",
                            ),
                            LabelExplanation(
                                assessment_id=assess_id,
                                label="比较自私",
                                explanation="用户在标准独裁者博弈中，送出的值低于25%，比较自私，更看重自己的利益。",
                            ),
                            LabelExplanation(
                                assessment_id=assess_id,
                                label="比较利他",
                                explanation="用户在标准独裁者博弈中，送出的值处于25%到50%之间，比较利他。",
                            ),
                            LabelExplanation(
                                assessment_id=assess_id,
                                label="利他人群中的平等主义者",
                                explanation="用户在标准独裁者博弈中，送出的值刚好为一半，利他的同时，也非常注重平等。",
                            ),
                            LabelExplanation(
                                assessment_id=assess_id,
                                label="极端利他者",
                                explanation="用户在标准独裁者博弈中，送出的值超过了一半，极端利他。",
                            ),
                        ]
                    )
                elif assess_id == "Trust_Game":
                    mappings.append(
                        AssessmentLabelMapping(
                            assessment_id=assess_id,
                            label_mapping={
                                "0-20": ["信任度低类型"],
                                "20-40": ["信任度较低类型"],
                                "40-60": ["信任度中等类型"],
                                "60-80": ["信任度较高类型"],
                                "80-80": ["信任度极高类型"],
                                "0-0": ["当对方为信任度低类型，用户的值得信任程度为低"],
                                "0-10": ["当对方为信任度低类型，用户值得信任程度为较低"],
                                "10-20": ["当对方为信任度低类型，用户值得信任程度为中等"],
                                "20-30": ["当对方为信任度低类型，用户值得信任程度为较高"],
                                "30-60": ["当对方为信任度低类型，用户值得信任程度为高"],
                                "0-10": ["当对方为信任度中等类型，用户的值得信任程度为低"],
                                "10-20": ["当对方为信任度中等类型，用户值得信任程度为较低"],
                                "20-40": ["当对方为信任度中等类型，用户值得信任程度为中等"],
                                "40-60": ["当对方为信任度中等类型，用户值得信任程度为较高"],
                                "60-120": ["当对方为信任度中等类型，用户值得信任程度为高"],
                                "0-60": ["当对方为信任度较高类型，用户的值得信任程度为低"],
                                "60-70": ["当对方为信任度较高类型，用户值得信任程度为较低"],
                                "70-80": ["当对方为信任度较高类型，用户值得信任程度为中等"],
                                "80-90": ["当对方为信任度较高类型，用户值得信任程度为较高"],
                                "90-180": ["当对方为信任度较高类型，用户值得信任程度为高"],
                                "0-0": ["当对方为信任度极高类型，用户的值得信任程度为低"],
                                "0-80": ["当对方为信任度极高类型，用户值得信任程度为较低"],
                                "80-100": ["当对方为信任度极高类型，用户值得信任程度为中等"],
                                "100-120": ["当对方为信任度极高类型，用户值得信任程度为较高"],
                                "120-240": ["当对方为信任度极高类型，用户值得信任程度为高"],
                                "-80-0": ["用户的期待回报程度为扰动项"],
                                "0-10": ["用户的期待回报程度为低"],
                                "10-20": ["用户的期待回报程度为中等"],
                                "20-80": ["用户的期待回报程度为高"],
                                "-120-0": ["用户在信任博弈中怨恨对方"],
                                "0-120": ["用户在信任博弈中感恩对方"],
                            },
                        )
                    )
                    explanations.extend(
                        [
                            LabelExplanation(
                                assessment_id=assess_id,
                                label="信任度低类型",
                                explanation="用户在信任博弈中，送出的值在整体分布中处于后20%。",
                            ),
                            LabelExplanation(
                                assessment_id=assess_id,
                                label="信任度较低类型",
                                explanation="用户在信任博弈中，送出的值在整体分布中处于后20%到后40%之间。",
                            ),
                            LabelExplanation(
                                assessment_id=assess_id,
                                label="信任度中等类型",
                                explanation="用户在信任博弈中，送出的值在整体分布中处于中等水平。",
                            ),
                            LabelExplanation(
                                assessment_id=assess_id,
                                label="信任度较高类型",
                                explanation="用户在信任博弈中，送出的值在整体分布中处于前40%到前20%之间。",
                            ),
                            LabelExplanation(
                                assessment_id=assess_id,
                                label="信任度极高类型",
                                explanation="用户在信任博弈中，送出了全部的金额，信任度极高。",
                            ),
                            LabelExplanation(
                                assessment_id=assess_id,
                                label="当对方为信任度低类型，用户的值得信任程度为低",
                                explanation="用户在信任博弈中，当对方为信任度低类型，返还的值为0。",
                            ),
                            LabelExplanation(
                                assessment_id=assess_id,
                                label="当对方为信任度低类型，用户值得信任程度为较低",
                                explanation="用户在信任博弈中，当对方为信任度低类型，用户返还的值较少。",
                            ),
                            LabelExplanation(
                                assessment_id=assess_id,
                                label="当对方为信任度低类型，用户值得信任程度为中等",
                                explanation="用户在信任博弈中，当对方为信任度低类型，用户返还的值处于整体分布的中间位置。",
                            ),
                            LabelExplanation(
                                assessment_id=assess_id,
                                label="当对方为信任度低类型，用户值得信任程度为较高",
                                explanation="用户在信任博弈中，当对方为信任度低类型，用户返还的值在整体分布中前40%到前20%之间。",
                            ),
                            LabelExplanation(
                                assessment_id=assess_id,
                                label="当对方为信任度低类型，用户值得信任程度为高",
                                explanation="用户在信任博弈中，当对方为信任度低类型，用户返还的值在整体分布中处于前20%。",
                            ),
                            LabelExplanation(
                                assessment_id=assess_id,
                                label="当对方为信任度中等类型，用户的值得信任程度为低",
                                explanation="用户在信任博弈中，当对方为信任度中等类型，返还的值在整体分布中处于后10%。",
                            ),
                            LabelExplanation(
                                assessment_id=assess_id,
                                label="当对方为信任度中等类型，用户值得信任程度为较低",
                                explanation="用户在信任博弈中，当对方为信任度中等类型，用户返还在整体分布中处于后10%到后20%之间。",
                            ),
                            LabelExplanation(
                                assessment_id=assess_id,
                                label="当对方为信任度中等类型，用户值得信任程度为中等",
                                explanation="用户在信任博弈中，当对方为信任度中等类型，用户返还的值处于整体分布的中间位置。",
                            ),
                            LabelExplanation(
                                assessment_id=assess_id,
                                label="当对方为信任度中等类型，用户值得信任程度为较高",
                                explanation="用户在信任博弈中，当对方为信任度中等类型，用户返还的值在整体分布中前40%到前20%之间。",
                            ),
                            LabelExplanation(
                                assessment_id=assess_id,
                                label="当对方为信任度中等类型，用户值得信任程度为高",
                                explanation="用户在信任博弈中，当对方为信任度中等类型，用户返还的值在整体分布中处于前20%。",
                            ),
                            LabelExplanation(
                                assessment_id=assess_id,
                                label="当对方为信任度较高类型，用户的值得信任程度为低",
                                explanation="用户在信任博弈中，当对方为信任度较高类型，返还的值在整体分布中处于后20%。",
                            ),
                            LabelExplanation(
                                assessment_id=assess_id,
                                label="当对方为信任度较高类型，用户值得信任程度为较低",
                                explanation="用户在信任博弈中，当对方为信任度较高类型，用户返还在整体分布中处于后20%到后40%之间。",
                            ),
                            LabelExplanation(
                                assessment_id=assess_id,
                                label="当对方为信任度较高类型，用户值得信任程度为中等",
                                explanation="用户在信任博弈中，当对方为信任度较高类型，用户返还的值处于整体分布的中间位置。",
                            ),
                            LabelExplanation(
                                assessment_id=assess_id,
                                label="当对方为信任度较高类型，用户值得信任程度为较高",
                                explanation="用户在信任博弈中，当对方为信任度较高类型，用户返还的值在整体分布中前40%到前20%之间。",
                            ),
                            LabelExplanation(
                                assessment_id=assess_id,
                                label="当对方为信任度较高类型，用户值得信任程度为高",
                                explanation="用户在信任博弈中，当对方为信任度较高类型，用户返还的值在整体分布中处于前20%。",
                            ),
                            LabelExplanation(
                                assessment_id=assess_id,
                                label="当对方为信任度极高类型，用户的值得信任程度为低",
                                explanation="用户在信任博弈中，当对方为信任度高类型，用户返还的值为0。",
                            ),
                            LabelExplanation(
                                assessment_id=assess_id,
                                label="当对方为信任度极高类型，用户值得信任程度为较低",
                                explanation="用户在信任博弈中，当对方为信任度高类型，用户返还的值在整体分布中处于后20%。",
                            ),
                            LabelExplanation(
                                assessment_id=assess_id,
                                label="当对方为信任度极高类型，用户值得信任程度为中等",
                                explanation="用户在信任博弈中，当对方为信任度高类型，用户返还的值处于中等水平。",
                            ),
                            LabelExplanation(
                                assessment_id=assess_id,
                                label="当对方为信任度极高类型，用户值得信任程度为较高",
                                explanation="用户在信任博弈中，当对方为信任度高类型，用户返还的值在处于较高水平。",
                            ),
                            LabelExplanation(
                                assessment_id=assess_id,
                                label="当对方为信任度极高类型，用户值得信任程度为高",
                                explanation="用户在信任博弈中，当对方为信任度高类型，用户返还的值等于或超过一半。",
                            ),
                            LabelExplanation(
                                assessment_id=assess_id,
                                label="用户的期待回报程度为扰动项",
                                explanation="用户在信任博弈中送出的值低于在3倍独裁者博弈中分出的值，用户并没有因为期待回报而在信任博弈中送出更多。",
                            ),
                            LabelExplanation(
                                assessment_id=assess_id,
                                label="用户的期待回报程度为低",
                                explanation="用户在信任博弈中送出的值高于在3倍独裁者博弈中分出的值，其差异值在整体分布中处于后30%。",
                            ),
                            LabelExplanation(
                                assessment_id=assess_id,
                                label="用户的期待回报程度为中等",
                                explanation="用户在信任博弈中送出的值高于在3倍独裁者博弈中分出的值，其差异值在整体分布中处于后60%到后30%之间。",
                            ),
                            LabelExplanation(
                                assessment_id=assess_id,
                                label="用户的期待回报程度为高",
                                explanation="用户在信任博弈中送出的值高于在3倍独裁者博弈中分出的值，其差异值在整体分布中处于前30%。",
                            ),
                            LabelExplanation(
                                assessment_id=assess_id,
                                label="用户在信任博弈中感恩对方",
                                explanation="用户在信任博弈中，当对方投入40时，返还的值高于或等于其在标准独裁者博弈中送出的值，用户饮水思源，希望满足对方的期待，避免自己内疚，同时也存在一定的感恩之心。",
                            ),
                            LabelExplanation(
                                assessment_id=assess_id,
                                label="用户在信任博弈中怨恨对方",
                                explanation="用户在信任博弈中，当对方投入40时，返还的值低于其在标准独裁者博弈中送出的值，用户感觉对方送出值太少了，因为怨恨而分配更少。",
                            ),
                        ]
                    )
                elif assess_id == "Ultimatum_Game":
                    mappings.append(
                        AssessmentLabelMapping(
                            assessment_id=assess_id,
                            label_mapping={
                                "30-0": ["利他非策略型"],
                                "30-120": ["利他策略型"],
                                "0-50": ["自私非策略型"],
                                "50-120": ["自私策略型"],
                                "0-20": ["极低公平需求者"],
                                "20-40": ["低公平需求者"],
                                "40-60": ["高公平需求者"],
                                "60-120": ["极高公平需求者"],
                            },
                        )
                    )
                    explanations.extend(
                        [
                            LabelExplanation(
                                assessment_id=assess_id,
                                label="利他非策略型",
                                explanation="用户在最后通牒博弈中认为社会对平等的需求小或需求大，由于用户在标准独裁者博弈中为利他型，已经分很多了，在最后通牒博弈中分配调整的范围不大。",
                            ),
                            LabelExplanation(
                                assessment_id=assess_id,
                                label="利他策略型",
                                explanation="用户在最后通牒博弈中认为社会对平等的需求极大，分配方案容易被拒绝，用户在标准独裁者博弈中为利他型，在最后通牒博弈中也分配更多。",
                            ),
                            LabelExplanation(
                                assessment_id=assess_id,
                                label="自私非策略型",
                                explanation="用户在最后通牒博弈中认为社会对平等的需求小，由于用户在标准独裁者博弈中为自私型，在最后通牒博弈中认为对方只要有一点收益就会接受，因此调整范围不大。",
                            ),
                            LabelExplanation(
                                assessment_id=assess_id,
                                label="自私策略型",
                                explanation="用户在最后通牒博弈中认为社会对平等的需求大，分配方案容易被拒绝，用户在标准独裁者博弈中为自私型，因此在最后通牒博弈中分配更多，调整的范围大。",
                            ),
                            LabelExplanation(
                                assessment_id=assess_id,
                                label="极低公平需求者",
                                explanation="用户在最后通牒博弈中最低可接受金额低，不公平的分配方案也会接受，希望自己有正的收益，较为理性。",
                            ),
                            LabelExplanation(
                                assessment_id=assess_id,
                                label="低公平需求者",
                                explanation="用户在最后通牒博弈中最低可接受金额较低，不太公平的分配方案也会接受。",
                            ),
                            LabelExplanation(
                                assessment_id=assess_id,
                                label="高公平需求者",
                                explanation="用户在最后通牒博弈中最低可接受金额较高，不太公平的分配方案都会被拒绝。",
                            ),
                            LabelExplanation(
                                assessment_id=assess_id,
                                label="极高公平需求者",
                                explanation="用户在最后通牒博弈中对公平有很高的需求，低于公平值的分配都拒绝。",
                            ),
                        ]
                    )
                elif assess_id == "Public_Goods_Game":
                    mappings.append(
                        AssessmentLabelMapping(
                            assessment_id=assess_id,
                            label_mapping={
                                "0-0": ["纯搭便车者（合作意愿极低）"],
                                "0-30": ["合作意愿较低"],
                                "30-50": ["合作意愿中等"],
                                "50-80": ["合作意愿较高"],
                                "80-80": ["合作意愿极高"],
                            },
                        )
                    )
                    explanations.extend(
                        [
                            LabelExplanation(
                                assessment_id=assess_id,
                                label="纯搭便车者（合作意愿极低）",
                                explanation="用户在公共品博弈中投入值为0，搭便车，合作意愿极低。",
                            ),
                            LabelExplanation(
                                assessment_id=assess_id,
                                label="合作意愿较低",
                                explanation="用户在公共品博弈中投入值处于整体分布的后25%，合作意愿极低。",
                            ),
                            LabelExplanation(
                                assessment_id=assess_id,
                                label="合作意愿中等",
                                explanation="用户在公共品博弈中投入值处于整体分布的后50%到后25%，合作意愿中等。",
                            ),
                            LabelExplanation(
                                assessment_id=assess_id,
                                label="合作意愿较高",
                                explanation="用户在公共品博弈中投入处于整体分布前50%到前30%，合作意愿较高。",
                            ),
                            LabelExplanation(
                                assessment_id=assess_id,
                                label="合作意愿极高",
                                explanation="用户在公共品博弈中投入所有的金额，合作意愿极高。",
                            ),
                        ]
                    )
                elif assess_id == "Risk_Gain":
                    mappings.append(
                        AssessmentLabelMapping(
                            assessment_id=assess_id,
                            label_mapping={
                                "0-2": ["Highly_Risk_Averse_Gain"],
                                "3-5": ["Risk_Averse_Gain"],
                                "6-6": ["Risk_Neutral_Gain"],
                                "7-8": ["Risk_Seeking_Gain"],
                                "9-10": ["Highly_Risk_Seeking_Gain"],
                            },
                        )
                    )
                    explanations.extend(
                        [
                            LabelExplanation(
                                assessment_id=assess_id,
                                label="Highly_Risk_Averse_Gain",
                                explanation="用户在收益情景中非常风险厌恶（风险规避 / Risk Averse），在风险厌恶的人群中属于前50%。",
                            ),
                            LabelExplanation(
                                assessment_id=assess_id,
                                label="Risk_Averse_Gain",
                                explanation="用户在收益情景中比较风险厌恶（风险规避 / Risk Averse），在风险厌恶的人群中属于后50%。",
                            ),
                            LabelExplanation(
                                assessment_id=assess_id,
                                label="Risk_Neutral_Gain",
                                explanation="用户在收益情景中属于风险中性类型，较为符合期望效用模型中的结果。",
                            ),
                            LabelExplanation(
                                assessment_id=assess_id,
                                label="Risk_Seeking_Gain",
                                explanation="用户在收益情景中比较风险喜好（风险寻求 / Risk Seeking / Risk Loving），在风险喜好的人群中属于后50%。",
                            ),
                            LabelExplanation(
                                assessment_id=assess_id,
                                label="Highly_Risk_Seeking_Gain",
                                explanation="用户在收益情景中非常风险喜好（风险寻求 / Risk Seeking / Risk Loving），在风险喜好的人群中属于前50%。",
                            ),
                        ]
                    )
                elif assess_id == "Risk_Loss":
                    mappings.append(
                        AssessmentLabelMapping(
                            assessment_id=assess_id,
                            label_mapping={
                                "0-2": ["Risk_Averse_Loss"],
                                "3-3": ["Risk_Neutral_Loss"],
                                "4-5": ["Risk_Seeking_Loss"],
                                "6-8": ["Highly_Risk_Seeking_Loss"],
                                "9-10": ["Extremely_Risk_Seeking_Loss"],
                            },
                        )
                    )
                    explanations.extend(
                        [
                            LabelExplanation(
                                assessment_id=assess_id,
                                label="Risk_Averse_Loss",
                                explanation="用户在损失情景中风险厌恶（风险规避 / Risk Averse）。",
                            ),
                            LabelExplanation(
                                assessment_id=assess_id,
                                label="Risk_Neutral_Loss",
                                explanation="用户在损失情景中属于风险中性类型，较为符合期望效用模型中的结果。",
                            ),
                            LabelExplanation(
                                assessment_id=assess_id,
                                label="Risk_Seeking_Loss",
                                explanation="用户在损失情景中风险喜好（风险寻求 / Risk Seeking / Risk Loving），在风险喜好的人群中属于后30%。",
                            ),
                            LabelExplanation(
                                assessment_id=assess_id,
                                label="Highly_Risk_Seeking_Loss",
                                explanation="用户在损失情景中非常风险喜好（风险寻求 / Risk Seeking / Risk Loving），在风险喜好的人群中属于后30%到前30%之间。",
                            ),
                            LabelExplanation(
                                assessment_id=assess_id,
                                label="Extremely_Risk_Seeking_Loss",
                                explanation="用户在损失情景中极端风险喜好（风险寻求 / Risk Seeking / Risk Loving），在风险喜好的人群中属于前30%。",
                            ),
                        ]
                    )
                elif assess_id == "Risk_Mixed":
                    mappings.append(
                        AssessmentLabelMapping(
                            assessment_id=assess_id,
                            label_mapping={
                                "0-2": ["Extremely_Loss_Aversion"],
                                "3-4": ["Highly_Loss_Aversion"],
                                "5-6": ["Loss_Aversion"],
                                "7-10": ["No_Loss_Aversion"],
                            },
                        )
                    )
                    explanations.extend(
                        [
                            LabelExplanation(
                                assessment_id=assess_id,
                                label="Extremely_Loss_Aversion",
                                explanation="用户损失厌恶程度极高。",
                            ),
                            LabelExplanation(
                                assessment_id=assess_id,
                                label="Highly_Loss_Aversion",
                                explanation="用户损失厌恶程度较高。",
                            ),
                            LabelExplanation(
                                assessment_id=assess_id,
                                label="Loss_Aversion",
                                explanation="用户损失厌恶程度高。",
                            ),
                            LabelExplanation(
                                assessment_id=assess_id,
                                label="No_Loss_Aversion",
                                explanation="用户不存在明显的损失厌恶。",
                            ),
                        ]
                    )
                elif assess_id == "Time_Preference":
                    mappings.append(
                        AssessmentLabelMapping(
                            assessment_id=assess_id,
                            label_mapping={
                                "0-2": ["Highly_Patient"],
                                "3-4": ["Patient"],
                                "5-6": ["Impatient"],
                                "7-10": ["Highly_Impatient"],
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
                                explanation="用户在时间偏好测评中非常有耐心，耐心程度在人群中属于前25%。",
                            ),
                            LabelExplanation(
                                assessment_id=assess_id,
                                label="Patient",
                                explanation="用户在时间偏好测评中比较有耐心，耐心程度在人群中属于前50%到前25%。",
                            ),
                            LabelExplanation(
                                assessment_id=assess_id,
                                label="Impatient",
                                explanation="用户在时间偏好测评中比较没有耐心，耐心程度在人群中属于后50%到后25%。",
                            ),
                            LabelExplanation(
                                assessment_id=assess_id,
                                label="Highly_Impatient",
                                explanation="用户在时间偏好测评中非常没有耐心，耐心程度在人群中属于后25%。",
                            ),
                            LabelExplanation(
                                assessment_id=assess_id,
                                label="Present_Biased",
                                explanation="用户存在一定的现时偏差 / 现时导向型（Present Bias）。在距离你更近，马上要实现的决策情景中更不耐心，而在距离更远，在未来才会实现的决策情景中更加耐心。",
                            ),
                            LabelExplanation(
                                assessment_id=assess_id,
                                label="No_Present_Bias",
                                explanation="用户没有明显的现时偏差，在当下和未来的跨期决策中，行为较一致。",
                            ),
                        ]
                    )
                elif assess_id == "Optimistic":
                    mappings.append(
                        AssessmentLabelMapping(
                            assessment_id=assess_id,
                            label_mapping={
                                "2-6": ["Optimist"],
                                "1-1": ["Moderate"],
                                "-6-0": ["Pessimist"],
                            },
                        )
                    )
                    explanations.extend(
                        [
                            LabelExplanation(
                                assessment_id=assess_id,
                                label="Optimist",
                                explanation="用户在马丁·塞利格曼的乐观悲观测评中，根据选择以及赋分模型，判定为'乐观'类型 (Optimist)。",
                            ),
                            LabelExplanation(
                                assessment_id=assess_id,
                                label="Moderate",
                                explanation="用户在马丁·塞利格曼的乐观悲观测评中，根据选择以及赋分模型，判定为介于'乐观'和'悲观'的'中等'类型 (Moderate)。",
                            ),
                            LabelExplanation(
                                assessment_id=assess_id,
                                label="Pessimist",
                                explanation="用户在马丁·塞利格曼的乐观悲观测评中，根据选择以及赋分模型，判定为'悲观'类型 (Pessimist)。",
                            ),
                        ]
                    )
                elif assess_id == "Introverted":
                    mappings.append(
                        AssessmentLabelMapping(
                            assessment_id=assess_id,
                            label_mapping={
                                "11-15": ["Introverted"],
                                "0-4": ["Extroverted"],
                                "5-10": ["Ambivert"],
                            },
                        )
                    )
                    explanations.extend(
                        [
                            LabelExplanation(
                                assessment_id=assess_id,
                                label="Introverted",
                                explanation="用户在内向外向测评中，根据选择以及赋分模型，判定为内向类型。",
                            ),
                            LabelExplanation(
                                assessment_id=assess_id,
                                label="Extroverted",
                                explanation="用户在内向外向测评中，根据选择以及赋分模型，判定为外向类型。",
                            ),
                            LabelExplanation(
                                assessment_id=assess_id,
                                label="Ambivert",
                                explanation="用户在内向外向测评中，根据选择以及赋分模型，判定为介于'内向'和'外向'中间的类型 / 混合型。",
                            ),
                        ]
                    )
            logger.info(f"开始保存标签映射和解释，测评数量: {len(mappings)}")
            for mapping in mappings:
                self.storage.save_label_mapping(mapping)
            for exp in explanations:
                self.storage.save_label_explanation(exp)
            logger.info("标签映射和解释初始化完成")
        except Exception as e:
            logger.error(f"初始化标签映射和解释失败: {e}")
            raise ConfigException(f"初始化标签映射和解释失败: {e}") from e
