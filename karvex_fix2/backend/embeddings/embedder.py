import os
import requests

from backend.chunking.models import Chunk


class Embedder:
    """
    Remote multilingual embedding layer for KARVEX.

    Uses Jina AI instead of loading SentenceTransformer/PyTorch
    inside the Vercel serverless function.
    """

    def __init__(
        self,
        model_name: str = "jina-embeddings-v3",
    ):
        self.model_name = model_name
        self.api_key = os.getenv("JINA_API_KEY")

        if not self.api_key:
            raise RuntimeError("JINA_API_KEY environment variable is not set")

        self.url = "https://api.jina.ai/v1/embeddings"

    def embed_texts(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        if not texts:
            return []

        cleaned = [
            str(text).strip()
            for text in texts
        ]

        response = requests.post(
            self.url,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            json={
                "model": self.model_name,
                "task": "text-matching",
                "normalized": True,
                "input": cleaned,
            },
            timeout=60,
        )

        response.raise_for_status()

        data = response.json()["data"]

        data.sort(key=lambda item: item["index"])

        return [
            item["embedding"]
            for item in data
        ]

    def embed_chunks(
        self,
        chunks: list[Chunk],
    ) -> list[list[float]]:
        if not chunks:
            return []

        return self.embed_texts(
            [
                chunk.text
                for chunk in chunks
            ]
        )