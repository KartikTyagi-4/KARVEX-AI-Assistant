from backend.generation.context_builder import build_context
from backend.retrieval.models import VectorRecord
from backend.retrieval.retriever import RetrievalResult


def test_build_context():
    record_a = VectorRecord(
        chunk_id="a",
        document_id="d1",
        text="Solar energy is renewable.",
        embedding=[1.0, 0.0, 0.0],
        language="en",
        source_language="eng_Latn",
        target_language="eng_Latn",
    )

    record_b = VectorRecord(
        chunk_id="b",
        document_id="d1",
        text="Solar panels convert sunlight into electricity.",
        embedding=[0.0, 1.0, 0.0],
        language="en",
        source_language="eng_Latn",
        target_language="eng_Latn",
    )

    results = [
        RetrievalResult(record=record_a, score=0.99),
        RetrievalResult(record=record_b, score=0.80),
    ]

    request = build_context(
        query="What are the benefits of solar energy?",
        results=results,
    )

    assert request.query == "What are the benefits of solar energy?"
    assert "Solar energy is renewable." in request.context
    assert "Solar panels convert sunlight into electricity." in request.context
    assert "[Source: a]" in request.context
    assert "[Source: b]" in request.context
    