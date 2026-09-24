from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CanonicalPassage:
    text: str
    english_text: str | None = None
    translated_text: str | None = None
    selected: bool = False


@dataclass
class CanonicalDocument:
    id: str
    query: str
    answer: str
    english_query: str | None
    english_answer: str | None
    language: str
    source_language: str | None
    target_language: str | None
    query_type: str | None
    passages: list[CanonicalPassage] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def normalize_record(record: dict[str, Any]) -> CanonicalDocument:
    """
    Convert one MSMARCO-XI record into the internal VĀDA format.

    This function intentionally preserves source information instead
    of transforming away potentially useful retrieval metadata.
    """

    passages_data = record.get("passages") or {}

    english_passages = passages_data.get("English_passages") or []
    translated_passages = passages_data.get("Translated_passages") or []
    selected_values = passages_data.get("is_selected") or []

    passages: list[CanonicalPassage] = []

    max_length = max(
        len(english_passages),
        len(translated_passages),
        len(selected_values),
        0,
    )

    for index in range(max_length):
        english_text = (
            english_passages[index]
            if index < len(english_passages)
            else None
        )

        translated_text = (
            translated_passages[index]
            if index < len(translated_passages)
            else None
        )

        selected = bool(
            selected_values[index]
            if index < len(selected_values)
            else False
        )

        text = english_text or translated_text or ""

        passages.append(
            CanonicalPassage(
                text=text,
                english_text=english_text,
                translated_text=translated_text,
                selected=selected,
            )
        )

    target_language = record.get("target_lang")
    language = (
        target_language.split("_")[-1]
        if isinstance(target_language, str)
        else "unknown"
    )

    metadata = {
    "dataset": "ai4bharat/MSMARCO-XI",
    "query_id": record.get("query_id"),
    "query_type": record.get("query_type"),
    "source_lang": record.get("source_lang"),
    "target_lang": record.get("target_lang"),
    **(record.get("meta") or {}),
}

    return CanonicalDocument(
        id=str(record.get("query_id")),
        query=record.get("query", ""),
        answer=record.get("Answer", ""),
        english_query=record.get("Eng_Query"),
        english_answer=record.get("Eng_Answer"),
        language=language,
        source_language=record.get("source_lang"),
        target_language=target_language,
        query_type=record.get("query_type"),
        passages=passages,
        metadata=metadata,
    )
