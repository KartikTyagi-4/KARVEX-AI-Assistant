from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class VectorRecord:
    """
    A chunk together with its embedding and retrieval metadata.
    """

    chunk_id: str
    document_id: str
    text: str
    embedding: list[float]

    language: str
    source_language: str
    target_language: str

    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "text": self.text,
            "embedding": self.embedding,
            "language": self.language,
            "source_language": self.source_language,
            "target_language": self.target_language,
            "metadata": self.metadata,
        }