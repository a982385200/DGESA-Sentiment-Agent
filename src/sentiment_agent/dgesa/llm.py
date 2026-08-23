from __future__ import annotations

from typing import Protocol, TypeVar

from langchain_core.messages import BaseMessage

from sentiment_agent.schemas import StrictModel

PayloadT = TypeVar("PayloadT", bound=StrictModel)


class AppendixLLM(Protocol):
    async def complete(self, messages: list[BaseMessage], schema: type[PayloadT]) -> PayloadT: ...


class LangChainAppendixLLM:
    def __init__(self, chat_model) -> None:
        self.chat_model = chat_model

    async def complete(self, messages: list[BaseMessage], schema: type[PayloadT]) -> PayloadT:
        runnable = self.chat_model.with_structured_output(schema)
        response = await runnable.ainvoke(messages)
        return schema.model_validate(response)
