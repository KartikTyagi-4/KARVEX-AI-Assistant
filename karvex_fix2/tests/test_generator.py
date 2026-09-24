from backend.generation.models import GenerationResponse
from backend.orchestration.pipeline import RAGPipeline


class FakeRetriever:
    def search(self, query, top_k=3):
        return [
            ("chunk-1", 0.95),
            ("chunk-2", 0.80),
        ]


class FakeGenerator:
    def generate(self, request):
        return GenerationResponse(
            answer=f"Answer for: {request.query}",
        )


def test_pipeline_runs_retrieval_and_generation():
    pipeline = RAGPipeline(
        retriever=FakeRetriever(),
        generator=FakeGenerator(),
    )

    result = pipeline.run("What is solar energy?")

    assert result.success is True
    assert result.query == "What is solar energy?"
    assert result.answer == "Answer for: What is solar energy?"

    assert len(result.stages) == 2
    assert result.stages[0].stage == "retrieval"
    assert result.stages[1].stage == "generation"

    assert result.stages[0].success is True
    assert result.stages[1].success is True


class FailingRetriever:
    def search(self, query, top_k=3):
        raise RuntimeError("retrieval failed")


def test_pipeline_handles_retrieval_failure():
    pipeline = RAGPipeline(
        retriever=FailingRetriever(),
        generator=FakeGenerator(),
    )

    result = pipeline.run("What is solar energy?")

    assert result.success is False
    assert result.answer == ""
    assert result.error == "retrieval failed"

    assert len(result.stages) == 1
    assert result.stages[0].stage == "retrieval"
    assert result.stages[0].success is False


class FailingGenerator:
    def generate(self, request):
        raise RuntimeError("generation failed")


def test_pipeline_handles_generation_failure():
    pipeline = RAGPipeline(
        retriever=FakeRetriever(),
        generator=FailingGenerator(),
    )

    result = pipeline.run("What is solar energy?")

    assert result.success is False
    assert result.answer == ""
    assert result.error == "generation failed"

    assert len(result.stages) == 2

    assert result.stages[0].stage == "retrieval"
    assert result.stages[0].success is True

    assert result.stages[1].stage == "generation"
    assert result.stages[1].success is False