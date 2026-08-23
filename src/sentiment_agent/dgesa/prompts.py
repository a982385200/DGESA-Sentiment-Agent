from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from sentiment_agent.dgesa.models import PatternExperience, SampleExperience


class AppendixPromptBuilder:
    def prediction(self, *, text: str, sample_experiences: Sequence[SampleExperience],
                   pattern_experiences: Sequence[PatternExperience]) -> list[BaseMessage]:
        system = (
            "你是一个多语言情感分析助手。请根据输入文本以及提供的外部经验完成情感分类任务。"
            "样本级经验用于提供局部语境线索和错误修正依据；模式级经验用于提供可跨样本复用的情感判断规律。"
            "请选择适用经验并避免使用不匹配的信息。仅输出结构化字段：language、sentiment、reason；"
            "sentiment 仅可为 positive、neutral 或 negative。"
        )
        payload = {
            "text": text,
            "sample_level_experiences": [item.experience for item in sample_experiences],
            "pattern_level_experiences": [item.text for item in pattern_experiences],
        }
        return _messages(system, payload)

    def generation(self, *, text: str, predicted_language: str,
                   predicted_sentiment: str, gold_sentiment: str,
                   prediction_reason: str) -> list[BaseMessage]:
        system = (
            "你是一个情感分析经验构建助手。根据误分类反馈分别生成样本级经验和模式级经验。"
            "样本级经验保留关键语境和局部误判线索；模式级经验抽象可跨样本复用的判断规律，"
            "不得包含具体实体、样本编号或一次性场景。仅输出 sample_experience、"
            "pattern_experience、pattern_label；pattern_label 仅可为 positive、neutral 或 negative。"
        )
        return _messages(system, {
            "text": text, "predicted_language": predicted_language,
            "predicted_sentiment": predicted_sentiment,
            "gold_sentiment": gold_sentiment, "prediction_reason": prediction_reason,
        })

    def admission(self, *, text: str, predicted_sentiment: str, gold_sentiment: str,
                  current_experience: str, candidates: Sequence[str]) -> list[BaseMessage]:
        system = (
            "你是一个样本级情感经验筛选助手。综合情感类别、误判方向、局部语境线索和候选经验覆盖情况，"
            "判断新经验是否提供新的局部判别信息。仅输出 admission，值为 informative 或 redundant。"
        )
        return _messages(system, {
            "text": text, "predicted_sentiment": predicted_sentiment,
            "gold_sentiment": gold_sentiment,
            "current_sample_level_experience": current_experience,
            "candidate_sample_level_experiences": list(candidates),
        })

    def alignment(self, *, current_pattern: str, pattern_label: str,
                  candidates: Mapping[str, str]) -> list[BaseMessage]:
        system = (
            "你是一个模式级情感经验匹配助手。根据情感类别、判断条件和误判触发因素判断模式是否关联，"
            "而非仅看表面相似度。仅输出 alignment：可归入时为 align(candidate_id)，否则为 new。"
        )
        return _messages(system, {
            "current_pattern_level_experience": current_pattern,
            "pattern_label": pattern_label,
            "candidate_pattern_level_experiences": dict(candidates),
        })

    def abstraction(self, *, existing_pattern: str,
                    new_pattern: str) -> list[BaseMessage]:
        system = (
            "你是一个模式级情感经验抽象助手。保持原情感类别和核心判断条件，融合新经验的有效信息，"
            "避免具体实体、样本编号和一次性场景。仅输出 updated_pattern_experience。"
        )
        return _messages(system, {
            "existing_pattern_level_experience": existing_pattern,
            "new_pattern_level_experience": new_pattern,
        })


def _messages(system: str, payload: dict) -> list[BaseMessage]:
    return [SystemMessage(content=system),
            HumanMessage(content=json.dumps(payload, ensure_ascii=False))]

