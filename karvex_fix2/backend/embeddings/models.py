from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Embedding:
    """
    A vector representation of a chunk.

    The original chunk identity and provenance are preserved so
    retrieval results can always be traced back to the source.
    """

    chunk_id: str
    vector: list[float]

    # Embedding metadata
    model: str
    dimensions: int

    # Original chunk information
    document_id: str
    passage_id: str
    text: str

    # Additional metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "chunk_id": self.chunk_id,
            "vector": self.vector,
            "model": self.model,
            "dimensions": self.dimensions,
            "document_id": self.document_id,
            "passage_id": self.passage_id,
            "text": self.text,
            "metadata": self.metadata,
        }