import json

from backend.chunking.chunker import chunk_document
from backend.embeddings.embedder import Embedder
from backend.ingestion.dataset_adapter import normalize_record
from backend.retrieval.models import VectorRecord
from backend.retrieval.retriever import Retriever
from backend.retrieval.vector_store import InMemoryVectorStore


def test_real_retrieval_pipeline():
    # Load sample dataset record
    record = json.load(
        open("data/sample/dev.json", encoding="utf-8")
    )[0]

    # Normalize
    document = normalize_record(record)

    # Chunk
    chunks = chunk_document(
        document,
        chunk_size=20,
        overlap=5,
    )

    # Embed chunks
    embedder = Embedder()
    vectors = embedder.embed_chunks(chunks)

    # Build vector records
    records = [
        VectorRecord(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            text=chunk.text,
            embedding=vector,
            language=chunk.language,
            source_language=chunk.source_language,
            target_language=chunk.target_language,
            metadata=chunk.metadata,
        )
        for chunk, vector in zip(chunks, vectors)
    ]

    # Store
    store = InMemoryVectorStore()
    store.add(records)

    # Embed the Hindi query
    query = "भारत में सौर ऊर्जा के क्या लाभ हैं?"
    query_vector = embedder.embed_texts([query])[0]

    # Retrieve
    results = store.search(query_vector, top_k=3)

    assert len(results) == 3
    assert results[0][1] >= results[1][1]

        # Test Retriever abstraction
    retriever = Retriever(
        store=store,
        embedder=embedder,
    )

    retrieval_results = retriever.search(
        query,
        top_k=3,
    )

    assert len(retrieval_results) == 3
    assert retrieval_results[0].score >= retrieval_results[1].score
    

    print("\nTop retrieval results:")
    for record, score in results:
        print(f"{score:.4f} | {record.chunk_id} | {record.text}")


if __name__ == "__main__":
    test_real_retrieval_pipeline()