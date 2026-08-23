import json

import pytest

from sentiment_agent.dgesa.llm import LangChainAppendixLLM
from sentiment_agent.dgesa.models import PatternExperience, SampleExperience
from sentiment_agent.dgesa.models import SampleAdmissionPayload
from sentiment_agent.dgesa.prompts import AppendixPromptBuilder


def test_prediction_prompt_separates_sample_and_pattern_experiences() -> None:
    builder = AppendixPromptBuilder()
    messages = builder.prediction(
        text="không tốt",
        sample_experiences=[SampleExperience(id="s", text="x", experience="local", sentiment="negative", language="vi", source="r", source_sample_id="1", created_batch=1)],
        pattern_experiences=[PatternExperience(id="p", text="abstract", sentiment="negative", source_language="vi", created_batch=1, last_updated_batch=1)],
    )
    payload = json.loads(messages[1].content)
    assert payload == {"text": "không tốt", "sample_level_experiences": ["local"],
                       "pattern_level_experiences": ["abstract"]}
    assert "sentiment" in messages[0].content and "domain" not in messages[0].content


def test_generation_prompt_contains_all_feedback_fields() -> None:
    messages = AppendixPromptBuilder().generation(
        text="bad", predicted_language="English", predicted_sentiment="positive",
        gold_sentiment="negative", prediction_reason="praise token")
    assert json.loads(messages[1].content) == {
        "text": "bad", "predicted_language": "English",
        "predicted_sentiment": "positive", "gold_sentiment": "negative",
        "prediction_reason": "praise token",
    }


def test_admission_alignment_and_abstraction_prompts_have_paper_payloads() -> None:
    builder = AppendixPromptBuilder()
    admission = json.loads(builder.admission(
        text="x", predicted_sentiment="positive", gold_sentiment="negative",
        current_experience="new", candidates=["old"])[1].content)
    alignment = json.loads(builder.alignment(
        current_pattern="new", pattern_label="negative",
        candidates={"p1": "old"})[1].content)
    abstraction = json.loads(builder.abstraction(
        existing_pattern="old", new_pattern="new")[1].content)
    assert admission["current_sample_level_experience"] == "new"
    assert alignment["candidate_pattern_level_experiences"] == {"p1": "old"}
    assert abstraction == {"existing_pattern_level_experience": "old",
                           "new_pattern_level_experience": "new"}


@pytest.mark.anyio
async def test_langchain_appendix_llm_validates_requested_schema() -> None:
    class Runnable:
        async def ainvoke(self, messages):
            return {"admission": "informative"}

    class ChatModel:
        def with_structured_output(self, schema):
            assert schema is SampleAdmissionPayload
            return Runnable()

    result = await LangChainAppendixLLM(ChatModel()).complete(
        AppendixPromptBuilder().admission(
            text="x", predicted_sentiment="positive", gold_sentiment="negative",
            current_experience="new", candidates=[]),
        SampleAdmissionPayload,
    )
    assert result.admission == "informative"
