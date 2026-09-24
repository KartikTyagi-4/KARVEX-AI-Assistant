from backend.service import RAGService


class FakePipeline:
    def run(self, query):
        return f"Result for: {query}"


def test_service_delegates_to_pipeline():
    pipeline = FakePipeline()
    service = RAGService(pipeline)

    result = service.ask("What is solar energy?")

    assert result == "Result for: What is solar energy?"