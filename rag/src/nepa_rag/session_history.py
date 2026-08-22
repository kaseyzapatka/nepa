from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ChatTurn:
    question: str
    answer: str
    model: str | None
    source_count: int


@dataclass
class SessionHistory:
    turns: list[ChatTurn] = field(default_factory=list)

    def add(self, question: str, answer: str, model: str | None, source_count: int) -> None:
        self.turns.append(ChatTurn(question=question, answer=answer, model=model, source_count=source_count))

    def clear(self) -> None:
        self.turns.clear()
