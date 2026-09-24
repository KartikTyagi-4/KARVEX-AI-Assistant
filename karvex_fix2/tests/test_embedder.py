from backend.embeddings.embedder import Embedder
from backend.embeddings.models import Embedding


def test_embed_texts_returns_vectors():
    embedder = Embedder()

    vectors = embedder.embed_texts(
        [
            "What is artificial intelligence?",
            "What is machine learning?",
        ]
    )

    assert len(vectors) == 2
    assert len(vectors[0]) > 0
    assert len(vectors[1]) == len(vectors[0])


def test_embed_texts_preserves_order():
    embedder = Embedder()

    texts = [
        "The sky is blue.",
        "Python is a programming language.",
    ]

    vectors = embedder.embed_texts(texts)

    assert len(vectors) == len(texts)
    assert all(isinstance(value, float) for value in vectors[0])


def test_embedding_to_dict():
    embedding = Embedding(
        chunk_id="doc1:0:0",
        vector=[0.1, 0.2, 0.3],
        model="test-model",
        dimensions=3,
        document_id="doc1",
        passage_id="0",
        text="Hello world",
    )

    data = embedding.to_dict()

    assert data["chunk_id"] == "doc1:0:0"
    assert data["vector"] == [0.1, 0.2, 0.3]
    assert data["model"] == "test-model"
    assert data["dimensions"] == 3
    assert data["document_id"] == "doc1"
    assert data["passage_id"] == "0"
    assert data["text"] == "Hello world"


def test_embedding_metadata_defaults_to_empty_dict():
    embedding = Embedding(
        chunk_id="doc1:0:0",
        vector=[0.1, 0.2],
        model="test-model",
        dimensions=2,
        document_id="doc1",
        passage_id="0",
        text="Hello",
    )

    assert embedding.metadata == {}