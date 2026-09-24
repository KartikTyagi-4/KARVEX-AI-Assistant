from backend.generation.models import GenerationRequest
from backend.retrieval.retriever import RetrievalResult


MAX_CONTEXT_CHARS = 1800
MAX_CHUNKS = 2


def build_context(
    query: str,
    results: list[RetrievalResult],
) -> GenerationRequest:
    """
    Build a compact, high-signal generation context.

    Keeps only the strongest retrieved chunks and limits the
    total context size to reduce LLM latency.
    """

    if not results:
        return GenerationRequest(
            query=query,
            context="",
            metadata={
                "results_count": 0,
                "context_chunks": 0,
                "context_chars": 0,
            },
        )

    # Highest scoring results first.
    ranked_results = sorted(
        results,
        key=lambda result: result.score,
        reverse=True,
    )

    selected = ranked_results[:MAX_CHUNKS]

    context_parts = []
    total_chars = 0

    for result in selected:

        text = (
            result.record.text or ""
        ).strip()

        if not text:
            continue

        remaining = (
            MAX_CONTEXT_CHARS
            - total_chars
        )

        if remaining <= 0:
            break

        # Leave room for a small source marker.
        text = text[:remaining]

        context_parts.append(text)

        total_chars += len(text)

    context = "\n\n".join(
        context_parts
    )

    return GenerationRequest(
        query=query,
        context=context,
        metadata={
            "results_count": len(results),
            "context_chunks": len(context_parts),
            "context_chars": len(context),
            "max_context_chars": MAX_CONTEXT_CHARS,
        },
    )