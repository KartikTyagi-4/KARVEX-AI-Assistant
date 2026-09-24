from backend.orchestration.pipeline import RAGPipeline


class RAGService:
    def __init__(self, pipeline: RAGPipeline):
        self.pipeline = pipeline

    def ask(self, query: str):
        return self.pipeline.run(query)