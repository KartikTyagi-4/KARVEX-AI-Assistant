import inspect
import statistics
import time
from collections import deque

from backend.generation.context_builder import build_context
from backend.generation.models import GenerationRequest
from backend.orchestration.models import PipelineResult, StageResult


class RAGPipeline:
    """
    KARVEX RAG orchestration layer.

    Tracks:
    - retrieval latency
    - context-building latency
    - generation latency
    - total pipeline latency
    - recent latency distribution
    """

    def __init__(
        self,
        retriever,
        generator,
        history_size: int = 1000,
    ):
        self.retriever = retriever
        self.generator = generator

        self._latencies_ms = deque(
            maxlen=history_size
        )

    # ============================================================
    # MAIN PIPELINE
    # ============================================================

    def run(
        self,
        query: str,
        top_k: int = 3,
    ) -> PipelineResult:

        pipeline_start = time.perf_counter()

        stages = []

        # ========================================================
        # Retrieval
        # ========================================================

        retrieval_start = time.perf_counter()

        try:
            results = self.retriever.search(
                query,
                top_k=top_k,
            )

            retrieval_latency = (
                time.perf_counter()
                - retrieval_start
            ) * 1000

            stages.append(
                StageResult(
                    stage="retrieval",
                    success=True,
                    latency_ms=retrieval_latency,
                    metadata={
                        "results_count": len(results),
                        "embedding_ms": getattr(
                            self.retriever,
                            "last_embedding_ms",
                            None,
                        ),
                        "vector_search_ms": getattr(
                            self.retriever,
                            "last_search_ms",
                            None,
                        ),
                    },
                )
            )

            # ====================================================
            # Grounding Guardrail
            # ====================================================

            if not results:
                total_latency = (
                    time.perf_counter()
                    - pipeline_start
                ) * 1000

                self._record_latency(
                    total_latency
                )

                stages.append(
                    StageResult(
                        stage="guardrail",
                        success=True,
                        latency_ms=0.0,
                        metadata={
                            "grounded": False,
                            "reason": "no_relevant_results",
                        },
                    )
                )

                stages.append(
                    StageResult(
                        stage="total",
                        success=True,
                        latency_ms=total_latency,
                        metadata={
                            "top_k": top_k,
                            "results_count": 0,
                            "grounded": False,
                        },
                    )
                )

                return PipelineResult(
                    query=query,
                    answer=(
                        "I don't have enough information "
                        "in my knowledge base to answer "
                        "that reliably."
                    ),
                    stages=stages,
                    success=True,
                )

        except Exception as exc:

            retrieval_latency = (
                time.perf_counter()
                - retrieval_start
            ) * 1000

            stages.append(
                StageResult(
                    stage="retrieval",
                    success=False,
                    latency_ms=retrieval_latency,
                    error=str(exc),
                )
            )

            return PipelineResult(
                query=query,
                answer="",
                stages=stages,
                success=False,
                error=str(exc),
            )

        # ========================================================
        # Context construction
        # ========================================================

        context_start = time.perf_counter()

        try:

            # New RetrievalResult interface.
            if results and hasattr(
                results[0],
                "record",
            ):

                generation_request = build_context(
                    query=query,
                    results=results,
                )

            # Legacy/test retrieval interface.
            else:

                context = "\n\n".join(
                    str(result[0])
                    for result in results
                )

                generation_request = GenerationRequest(
                    query=query,
                    context=context,
                    metadata={
                        "results_count": len(results),
                    },
                )

            context_latency = (
                time.perf_counter()
                - context_start
            ) * 1000

            stages.append(
                StageResult(
                    stage="context",
                    success=True,
                    latency_ms=context_latency,
                    metadata={
                        "results_count": len(results),
                    },
                )
            )

        except Exception as exc:

            context_latency = (
                time.perf_counter()
                - context_start
            ) * 1000

            stages.append(
                StageResult(
                    stage="context",
                    success=False,
                    latency_ms=context_latency,
                    error=str(exc),
                )
            )

            total_latency = (
                time.perf_counter()
                - pipeline_start
            ) * 1000

            self._record_latency(
                total_latency
            )

            return PipelineResult(
                query=query,
                answer="",
                stages=stages,
                success=False,
                error=str(exc),
            )

        # ========================================================
        # Generation
        # ========================================================

        generation_start = time.perf_counter()

        try:

            generate_method = (
                self.generator.generate
            )

            parameters = inspect.signature(
                generate_method
            ).parameters

            # New interface:
            # generate(request)
            if len(parameters) == 1:

                generation_response = (
                    generate_method(
                        generation_request
                    )
                )

            # Legacy interface:
            # generate(query, results)
            else:

                generation_response = (
                    generate_method(
                        query,
                        results,
                    )
                )

            answer = (
                generation_response.answer
            )

            generation_latency = (
                time.perf_counter()
                - generation_start
            ) * 1000

            stages.append(
                StageResult(
                    stage="generation",
                    success=True,
                    latency_ms=generation_latency,
                )
            )

        except Exception as exc:

            generation_latency = (
                time.perf_counter()
                - generation_start
            ) * 1000

            stages.append(
                StageResult(
                    stage="generation",
                    success=False,
                    latency_ms=generation_latency,
                    error=str(exc),
                )
            )

            total_latency = (
                time.perf_counter()
                - pipeline_start
            ) * 1000

            self._record_latency(
                total_latency
            )

            return PipelineResult(
                query=query,
                answer="",
                stages=stages,
                success=False,
                error=str(exc),
            )

        # ========================================================
        # Total latency
        # ========================================================

        total_latency = (
            time.perf_counter()
            - pipeline_start
        ) * 1000

        self._record_latency(
            total_latency
        )

        stages.append(
            StageResult(
                stage="total",
                success=True,
                latency_ms=total_latency,
                metadata={
                    "top_k": top_k,
                    "results_count": len(results),
                    "grounded": True,
                },
            )
        )

        return PipelineResult(
            query=query,
            answer=answer,
            stages=stages,
            success=True,
        )

    # ============================================================
    # LATENCY HISTORY
    # ============================================================

    def _record_latency(
        self,
        latency_ms: float,
    ):
        self._latencies_ms.append(
            float(latency_ms)
        )

    # ============================================================
    # LATENCY REPORT
    # ============================================================

    def latency_report(self) -> dict:
        """
        Return P50/P70/P100 latency statistics.
        """

        if not self._latencies_ms:
            return {
                "samples": 0,
                "p50_ms": None,
                "p70_ms": None,
                "p100_ms": None,
            }

        values = sorted(
            self._latencies_ms
        )

        return {
            "samples": len(values),
            "p50_ms": round(
                self._percentile(
                    values,
                    50,
                ),
                2,
            ),
            "p70_ms": round(
                self._percentile(
                    values,
                    70,
                ),
                2,
            ),
            "p100_ms": round(
                max(values),
                2,
            ),
            "min_ms": round(
                min(values),
                2,
            ),
            "mean_ms": round(
                statistics.mean(values),
                2,
            ),
        }

    # ============================================================
    # PERCENTILE
    # ============================================================

    @staticmethod
    def _percentile(
        values: list[float],
        percentile: float,
    ) -> float:

        if not values:
            return 0.0

        if len(values) == 1:
            return values[0]

        index = (
            (len(values) - 1)
            * percentile
            / 100
        )

        lower = int(index)

        upper = min(
            lower + 1,
            len(values) - 1,
        )

        weight = index - lower

        return (
            values[lower]
            + (
                values[upper]
                - values[lower]
            )
            * weight
        )