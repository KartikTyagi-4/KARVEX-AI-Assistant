import json

from backend.ingestion.dataset_adapter import normalize_record
from backend.chunking.chunker import chunk_document


def load_sample_document():
    with open("data/sample/dev.json", encoding="utf-8") as f:
        record = json.load(f)[0]

    return normalize_record(record)


def test_chunk_document_creates_chunks():
    document = load_sample_document()

    chunks = chunk_document(
        document,
        chunk_size=20,
        overlap=5,
    )

    assert len(chunks) > 0


def test_chunk_provenance_is_preserved():
    document = load_sample_document()

    chunks = chunk_document(
        document,
        chunk_size=20,
        overlap=5,
    )

    for chunk in chunks:
        assert chunk.document_id == document.id
        assert chunk.passage_id is not None
        assert chunk.strategy == "fixed_window"


def test_chunk_offsets_are_valid():
    document = load_sample_document()

    chunks = chunk_document(
        document,
        chunk_size=20,
        overlap=5,
    )

    for chunk in chunks:
        assert chunk.start_char >= 0
        assert chunk.end_char > chunk.start_char
        assert chunk.end_char - chunk.start_char == len(chunk.text)


def test_chunk_overlap_is_correct():
    document = load_sample_document()

    chunks = chunk_document(
        document,
        chunk_size=20,
        overlap=5,
    )

    for previous, current in zip(chunks, chunks[1:]):
        if previous.passage_id == current.passage_id:
            assert current.start_char == previous.end_char - 5