from typing import Any

from backend.chunking.chunker import chunk_document
from backend.embeddings.embedder import Embedder
from backend.ingestion.dataset_adapter import normalize_record
from backend.retrieval.models import VectorRecord
from backend.retrieval.vector_store import InMemoryVectorStore


class Indexer:
    """
    Converts dataset records into searchable vector records
    and stores them in the vector store.
    """

    def __init__(
        self,
        embedder: Embedder,
        vector_store: InMemoryVectorStore,
        chunk_size: int = 500,
        overlap: int = 50,
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.chunk_size = chunk_size
        self.overlap = overlap

    def index_records(
        self,
        records: list[dict[str, Any]],
    ) -> int:
        """
        Normalize, chunk, embed, and store dataset records.

        Returns the number of indexed chunks.
        """

        all_chunks = []

        for record in records:
            document = normalize_record(record)

            chunks = chunk_document(
                document,
                chunk_size=self.chunk_size,
                overlap=self.overlap,
            )

            all_chunks.extend(chunks)

        if not all_chunks:
            return 0

        embeddings = self.embedder.embed_chunks(all_chunks)

        vector_records = []

        for chunk, embedding in zip(
            all_chunks,
            embeddings,
            strict=True,
        ):
            vector_records.append(
                VectorRecord(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    text=chunk.text,
                    embedding=embedding,
                    language=chunk.language,
                    source_language=chunk.source_language,
                    target_language=chunk.target_language,
                    metadata=chunk.metadata,
                )
            )

        self.vector_store.add(vector_records)

        return len(vector_records)