from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Chunk:
    """
    A normalized retrieval unit produced by any chunking strategy.

    Every chunk keeps enough provenance to trace retrieved evidence
    back to the original MSMARCO-XI document and passage.
    """

    chunk_id: str
    document_id: str
    text: str

    # Provenance
    passage_id: str
    language: str
    source_language: str
    target_language: str

    # Chunking metadata
    strategy: str
    chunk_index: int
    start_char: int
    end_char: int

    # Retrieval metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "text": self.text,
            "passage_id": self.passage_id,
            "language": self.language,
            "source_language": self.source_language,
            "target_language": self.target_language,
            "strategy": self.strategy,
            "chunk_index": self.chunk_index,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "metadata": self.metadata,
        }