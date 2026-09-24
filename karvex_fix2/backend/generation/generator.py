import ollama

from backend.generation.models import (
    GenerationRequest,
    GenerationResponse,
)


class Generator:
    """
    Fast local LLM generation layer using Ollama.

    Optimized for KARVEX:
    - short spoken answers
    - low temperature
    - bounded output
    - persistent model in memory
    """

    def __init__(self, model: str = "qwen2.5:0.5b-instruct"):
        self.model = model

    def generate(
        self,
        request: GenerationRequest,
    ) -> GenerationResponse:

        # --------------------------------------------------------
        # No context
        # --------------------------------------------------------

        if not request.context.strip():
            return GenerationResponse(
                answer=(
                    "I don't have enough information "
                    "to answer that reliably."
                ),
            )

        # --------------------------------------------------------
        # Keep the prompt compact
        # --------------------------------------------------------

        prompt = f"""You are KARVEX, a voice-first AI assistant.

Answer the question using the provided context.

Rules:
- Answer directly.
- Be concise.
- Use 1 to 3 short sentences.
- Do not repeat the question.
- Do not mention the context.
- Do not add unnecessary explanations.
- If the context does not contain enough information, say so.

Context:
{request.context}

Question:
{request.query}

Answer:"""

        # --------------------------------------------------------
        # Fast Ollama generation
        # --------------------------------------------------------

        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
               ],
               options={
                   "temperature": 0.2,
                   "num_predict": 44,
               },
               keep_alive=-1,
            )

            answer = (
                response["message"]["content"]
                .strip()
            )

            if not answer:
                answer = (
                    "I couldn't generate an answer "
                    "right now."
                )

            return GenerationResponse(
                answer=answer,
            )

        except Exception as exc:
            return GenerationResponse(
                answer=(
                    "I couldn't process that request "
                    "right now."
                ),
            )