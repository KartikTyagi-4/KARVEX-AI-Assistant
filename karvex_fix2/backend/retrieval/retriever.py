from dataclasses import dataclass
from time import perf_counter

from backend.embeddings.embedder import Embedder
from backend.retrieval.models import VectorRecord
from backend.retrieval.vector_store import InMemoryVectorStore


@dataclass(frozen=True)
class RetrievalResult:
    record: VectorRecord
    score: float


class Retriever:
    """
    Fast retrieval layer.

    Responsibilities:
    - Embed the user's query once.
    - Retrieve the highest-scoring records.
    - Preserve similarity scores.
    - Expose lightweight latency information.
    """

    def __init__(
        self,
        store: InMemoryVectorStore,
        embedder: Embedder,
    ):
        self.store = store
        self.embedder = embedder

        # Simple in-memory query cache.
        # This makes repeated/similar test queries much faster.
        self._query_cache: dict[str, list[float]] = {}
        self._max_cache_size = 256

        self.last_latency_ms = 0.0
        self.last_embedding_ms = 0.0
        self.last_search_ms = 0.0

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievalResult]:

        started = perf_counter()

        query = query.strip()

        if not query:
            self.last_latency_ms = 0.0
            self.last_embedding_ms = 0.0
            self.last_search_ms = 0.0
            return []

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        # ========================================================
        # Query embedding
        # ========================================================

        embedding_start = perf_counter()

        query_vector = self._get_query_embedding(query)

        self.last_embedding_ms = (
            perf_counter() - embedding_start
        ) * 1000

        # ========================================================
        # Vector search
        # ========================================================

        search_start = perf_counter()

        results = self.store.search(
            query_vector,
            top_k=top_k,
            min_score=0.45,
        )     

        self.last_search_ms = (
            perf_counter() - search_start
        ) * 1000

        # ========================================================
        # Convert to structured results
        # ========================================================

        retrieval_results = [
            RetrievalResult(
                record=record,
                score=float(score),
            )
            for record, score in results
        ]

        # Make ordering deterministic.
        retrieval_results.sort(
            key=lambda item: item.score,
            reverse=True,
        )

        self.last_latency_ms = (
            perf_counter() - started
        ) * 1000

        return retrieval_results

    # ============================================================
    # QUERY EMBEDDING CACHE
    # ============================================================

    def _get_query_embedding(
        self,
        query: str,
    ) -> list[float]:

        cache_key = query.casefold()

        cached = self._query_cache.get(cache_key)

        if cached is not None:
            return cached

        vector = self.embedder.embed_texts(
            [query]
        )[0]

        # Keep memory bounded.
        if len(self._query_cache) >= self._max_cache_size:
            oldest_key = next(
                iter(self._query_cache)
            )
            del self._query_cache[oldest_key]

        self._query_cache[cache_key] = vector

        return vector

    # ============================================================
    # CACHE CONTROL
    # ============================================================

    def clear_cache(self):
        """
        Clear cached query embeddings.
        """

        self._query_cache.clear()

    # ============================================================
    # LATENCY INFORMATION
    # ============================================================

    def latency_report(self) -> dict:
        """
        Return retrieval timing for debugging and benchmarking.
        """

        return {
            "retrieval_ms": round(
                self.last_latency_ms,
                2,
            ),
            "embedding_ms": round(
                self.last_embedding_ms,
                2,
            ),
            "vector_search_ms": round(
                self.last_search_ms,
                2,
            ),
        }