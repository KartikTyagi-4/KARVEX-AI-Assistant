import re

from backend.chunking.models import Chunk


def chunk_document(
    document,
    chunk_size: int = 500,
    overlap: int = 50,
    strategy: str = "fixed_window",
) -> list[Chunk]:
    """
    Multi-strategy chunking for KARVEX's RAG pipeline.

    Supported strategies:
        - fixed_window
        - sentence
        - paragraph
        - auto

    The default remains fixed_window for backward compatibility.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if overlap < 0:
        raise ValueError("overlap cannot be negative")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    valid_strategies = {
        "fixed_window",
        "sentence",
        "paragraph",
        "auto",
    }

    if strategy not in valid_strategies:
        raise ValueError(
            f"Unknown chunking strategy: {strategy}. "
            f"Choose from {sorted(valid_strategies)}."
        )

    # Automatically choose a strategy based on document structure.
    if strategy == "auto":
        strategy = _choose_strategy(document)

    chunks: list[Chunk] = []

    for passage_index, passage in enumerate(document.passages):
        text = (passage.text or "").strip()

        if not text:
            continue

        if strategy == "fixed_window":
            passage_chunks = _fixed_window_chunks(
                text=text,
                chunk_size=chunk_size,
                overlap=overlap,
            )

        elif strategy == "sentence":
            passage_chunks = _sentence_chunks(
                text=text,
                chunk_size=chunk_size,
                overlap=overlap,
            )

        elif strategy == "paragraph":
            passage_chunks = _paragraph_chunks(
                text=text,
                chunk_size=chunk_size,
                overlap=overlap,
            )

        else:
            raise ValueError(f"Unsupported strategy: {strategy}")

        for chunk_index, item in enumerate(passage_chunks):

            chunk_text = item["text"]
            start = item["start"]
            end = item["end"]

            chunk_id = (
                f"{document.id}:"
                f"{passage_index}:"
                f"{strategy}:"
                f"{chunk_index}"
            )

            metadata = {
                **document.metadata,

                # Dataset provenance
                "dataset": document.metadata.get("dataset"),
                "query_id": document.metadata.get("query_id"),
                "query_type": document.query_type,

                # Chunking provenance
                "chunking_strategy": strategy,
                "chunk_size": chunk_size,
                "overlap": overlap,
                "passage_index": passage_index,
                "chunk_index": chunk_index,
            }

            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    document_id=document.id,
                    text=chunk_text,
                    passage_id=str(passage_index),
                    language=document.language,
                    source_language=document.source_language,
                    target_language=document.target_language,
                    strategy=strategy,
                    chunk_index=chunk_index,
                    start_char=start,
                    end_char=end,
                    metadata=metadata,
                )
            )

    return chunks


# ============================================================
# STRATEGY SELECTION
# ============================================================


def _choose_strategy(document) -> str:
    """
    Metadata-aware automatic strategy selection.

    Structured / multi-paragraph passages benefit from paragraph
    chunking, while long continuous passages benefit from sentence
    chunking.
    """

    passages = getattr(document, "passages", [])

    if not passages:
        return "fixed_window"

    total_text = " ".join(
        (passage.text or "").strip()
        for passage in passages
    )

    if not total_text:
        return "fixed_window"

    paragraph_count = len(
        [
            p
            for p in re.split(r"\n\s*\n", total_text)
            if p.strip()
        ]
    )

    sentence_count = len(
        re.findall(
            r"(?<=[.!?])\s+",
            total_text,
        )
    )

    if paragraph_count >= 3:
        return "paragraph"

    if sentence_count >= 5:
        return "sentence"

    return "fixed_window"


# ============================================================
# FIXED WINDOW
# ============================================================


def _fixed_window_chunks(
    text: str,
    chunk_size: int,
    overlap: int,
) -> list[dict]:
    """
    Deterministic character-window chunking with overlap.
    """

    chunks = []

    start = 0

    while start < len(text):
        end = min(
            start + chunk_size,
            len(text),
        )

        chunk_text = text[start:end].strip()

        if chunk_text:
            chunks.append(
                {
                    "text": chunk_text,
                    "start": start,
                    "end": end,
                }
            )

        if end >= len(text):
            break

        start = end - overlap

    return chunks


# ============================================================
# SENTENCE-AWARE
# ============================================================


def _sentence_chunks(
    text: str,
    chunk_size: int,
    overlap: int,
) -> list[dict]:
    """
    Sentence-aware chunking.

    Sentences are grouped together until the target chunk size
    is reached. A small sentence overlap is retained between
    neighboring chunks.
    """

    sentences = _split_sentences(text)

    if not sentences:
        return _fixed_window_chunks(
            text,
            chunk_size,
            overlap,
        )

    chunks = []

    current_sentences = []
    current_length = 0

    sentence_cursor = 0

    while sentence_cursor < len(sentences):

        sentence = sentences[sentence_cursor]

        proposed_length = (
            current_length
            + len(sentence)
            + (1 if current_sentences else 0)
        )

        if (
            current_sentences
            and proposed_length > chunk_size
        ):
            chunk_text = " ".join(
                current_sentences
            )

            start = text.find(
                current_sentences[0]
            )

            end = start + len(chunk_text)

            chunks.append(
                {
                    "text": chunk_text,
                    "start": max(start, 0),
                    "end": min(end, len(text)),
                }
            )

            # Sentence-level overlap.
            overlap_text = []
            overlap_length = 0

            for previous in reversed(
                current_sentences
            ):
                if overlap_length + len(previous) > overlap:
                    break

                overlap_text.insert(
                    0,
                    previous,
                )

                overlap_length += len(previous) + 1

            current_sentences = overlap_text
            current_length = overlap_length

        else:
            current_sentences.append(sentence)
            current_length = proposed_length
            sentence_cursor += 1

    if current_sentences:
        chunk_text = " ".join(
            current_sentences
        )

        start = text.find(
            current_sentences[0]
        )

        end = min(
            start + len(chunk_text),
            len(text),
        )

        chunks.append(
            {
                "text": chunk_text,
                "start": max(start, 0),
                "end": end,
            }
        )

    return chunks


def _split_sentences(text: str) -> list[str]:
    """
    Lightweight multilingual-friendly sentence splitter.

    Handles common English and Indian-language punctuation.
    """

    parts = re.split(
        r"(?<=[.!?।])\s+",
        text,
    )

    return [
        part.strip()
        for part in parts
        if part.strip()
    ]


# ============================================================
# PARAGRAPH-AWARE
# ============================================================


def _paragraph_chunks(
    text: str,
    chunk_size: int,
    overlap: int,
) -> list[dict]:
    """
    Paragraph-aware chunking.

    Keeps natural paragraph boundaries whenever possible.
    Long paragraphs fall back to fixed-window splitting.
    """

    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(
            r"\n\s*\n",
            text,
        )
        if paragraph.strip()
    ]

    if not paragraphs:
        return _fixed_window_chunks(
            text,
            chunk_size,
            overlap,
        )

    chunks = []

    current = []
    current_length = 0

    for paragraph in paragraphs:

        # Very large paragraph:
        # split it independently.
        if len(paragraph) > chunk_size:

            if current:
                chunks.append(
                    _make_chunk_record(
                        text,
                        "\n\n".join(current),
                    )
                )

                current = []
                current_length = 0

            chunks.extend(
                _fixed_window_chunks(
                    paragraph,
                    chunk_size,
                    overlap,
                )
            )

            continue

        proposed_length = (
            current_length
            + len(paragraph)
            + (2 if current else 0)
        )

        if (
            current
            and proposed_length > chunk_size
        ):
            chunk_text = "\n\n".join(current)

            chunks.append(
                _make_chunk_record(
                    text,
                    chunk_text,
                )
            )

            # Keep the last paragraph as contextual overlap.
            current = [current[-1]]
            current_length = len(current[0])

        current.append(paragraph)
        current_length += len(paragraph) + (
            2 if len(current) > 1 else 0
        )

    if current:
        chunks.append(
            _make_chunk_record(
                text,
                "\n\n".join(current),
            )
        )

    return chunks


def _make_chunk_record(
    original_text: str,
    chunk_text: str,
) -> dict:
    """
    Create a chunk record while preserving character provenance.
    """

    start = original_text.find(chunk_text)

    if start < 0:
        start = 0

    end = min(
        start + len(chunk_text),
        len(original_text),
    )

    return {
        "text": chunk_text,
        "start": start,
        "end": end,
    }