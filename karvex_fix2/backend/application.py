from backend.embeddings.embedder import Embedder
from backend.generation.generator import Generator
from backend.indexing.indexer import Indexer
from backend.retrieval.retriever import Retriever
from backend.retrieval.vector_store import InMemoryVectorStore
from backend.orchestration.pipeline import RAGPipeline
from backend.service import RAGService


class Application:
    def __init__(self):
        # Infrastructure
        self.embedder = Embedder()
        self.vector_store = InMemoryVectorStore()

        # Indexing
        self.indexer = Indexer(
            embedder=self.embedder,
            vector_store=self.vector_store,
        )

        # Retrieval
        self.retriever = Retriever(
            store=self.vector_store,
            embedder=self.embedder,
        )

        # Generation
        self.generator = Generator()

        # Orchestration
        self.pipeline = RAGPipeline(
            retriever=self.retriever,
            generator=self.generator,
        )

        # Service
        self.service = RAGService(
            pipeline=self.pipeline,
        )

    def index_records(self, records):
        return self.indexer.index_records(records)


def create_application() -> Application:
    return Application()

 