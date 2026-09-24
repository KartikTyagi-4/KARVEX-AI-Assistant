from math import sqrt

from backend.retrieval.models import VectorRecord


class InMemoryVectorStore:
    """
    Local vector store using cosine similarity.

    Supports:
    - top-k retrieval
    - similarity thresholds
    - deterministic ranking
    """

    def __init__(self) -> None:
        self._records: list[VectorRecord] = []

    def add(
        self,
        records: list[VectorRecord],
    ) -> None:
        """Add vector records to the store."""

        self._records.extend(records)

    def count(self) -> int:
        """Return the number of stored records."""

        return len(self._records)

    @staticmethod
    def _cosine_similarity(
        a: list[float],
        b: list[float],
    ) -> float:
        """Calculate cosine similarity between two vectors."""

        if len(a) != len(b):
            raise ValueError(
                "Vectors must have the same dimensions."
            )

        dot = sum(
            x * y
            for x, y in zip(a, b)
        )

        norm_a = sqrt(
            sum(x * x for x in a)
        )

        norm_b = sqrt(
            sum(y * y for y in b)
        )

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot / (
            norm_a * norm_b
        )

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> list[tuple[VectorRecord, float]]:
        """
        Return the top-k records above the similarity threshold.

        min_score prevents unrelated queries from being treated
        as grounded knowledge.
        """

        if top_k <= 0:
            return []

        if not -1.0 <= min_score <= 1.0:
            raise ValueError(
                "min_score must be between -1.0 and 1.0."
            )

        scored = []

        for record in self._records:
            score = self._cosine_similarity(
                query_embedding,
                record.embedding,
            )

            if score >= min_score:
                scored.append(
                    (
                        record,
                        score,
                    )
                )

        scored.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        return scored[:top_k]