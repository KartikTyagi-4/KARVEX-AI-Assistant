from backend.generation.generator import Generator 
from backend.generation.models import (
    GenerationRequest,
    GenerationResponse,
)


def test_generation_models():
    request = GenerationRequest(
        query="What are the benefits of solar energy?",
        context="Solar energy is a renewable source of energy.",
    )

    response = GenerationResponse(
        answer="Solar energy is a renewable source of energy.",
        sources=["1:0:0"],
    )

    assert request.query == "What are the benefits of solar energy?"
    assert request.context != ""

    assert response.answer != ""
    assert response.sources == ["1:0:0"]
def test_generator_generates_from_context():
    generator = Generator()

    request = GenerationRequest(
        query="What are the benefits of solar energy?",
        context="Solar energy is a renewable source of energy.",
    )

    response = generator.generate(request)

    assert response.answer != ""
    assert "solar energy" in response.answer.lower()


def test_generator_handles_empty_context():
    generator = Generator()

    request = GenerationRequest(
        query="What are the benefits of solar energy?",
        context="",
    )

    response = generator.generate(request)

    assert response.answer == (
        "I don't have enough context to answer the question."
    )