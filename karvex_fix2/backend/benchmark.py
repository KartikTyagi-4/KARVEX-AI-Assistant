import statistics
import time

import ollama

from backend.application import create_application
from backend.ingestion.load_dataset import load_msmarco_xi_json


QUERIES = [
    "What is solar energy?",
    "What are solar panels?",
    "How does photovoltaic energy work?",
    "What is renewable energy?",
    "What are the advantages of solar power?",
    "How does solar radiation affect energy production?",
    "What is the difference between renewable and non renewable energy?",
    "How is solar electricity generated?",
    "What are the main components of a solar power system?",
    "Why is solar energy important?",
]


def percentile(values, p):
    if not values:
        return 0.0

    values = sorted(values)

    if len(values) == 1:
        return values[0]

    index = (len(values) - 1) * p / 100

    lower = int(index)
    upper = min(lower + 1, len(values) - 1)

    weight = index - lower

    return (
        values[lower]
        + (values[upper] - values[lower]) * weight
    )


def benchmark_ollama():
    print("\n" + "=" * 60)
    print("OLLAMA GENERATION CHECK")
    print("=" * 60)

    start = time.perf_counter()

    response = ollama.chat(
        model="qwen2.5:3b",
        messages=[
            {
                "role": "user",
                "content": (
                    "Answer in one short sentence: "
                    "What is solar energy?"
                ),
            }
        ],
        options={
            "temperature": 0.2,
            "num_predict": 44,
        },
        keep_alive=-1,
    )

    latency = (
        time.perf_counter() - start
    ) * 1000

    answer = response["message"]["content"].strip()

    print(f"Generation latency: {latency:.2f} ms")
    print(f"Answer: {answer}")

    return latency


def main():
    print("=" * 60)
    print("KARVEX RAG LATENCY BENCHMARK")
    print("=" * 60)

    # ============================================================
    # CREATE APPLICATION
    # ============================================================

    print("\nLoading application...")

    application = create_application()

    print("Application loaded.")

    # ============================================================
    # LOAD DATASET
    # ============================================================

    print("\nLoading MS MARCO-XI dataset...")

    records = load_msmarco_xi_json(
        "data/sample/dev20.json",
        limit=20,
    )

    print(
        f"Loaded {len(records)} records."
    )

    # ============================================================
    # INDEX DATASET
    # ============================================================

    print("\nIndexing records...")

    index_start = time.perf_counter()

    application.index_records(records)

    index_latency = (
        time.perf_counter() - index_start
    ) * 1000

    print(
        f"Indexing complete: "
        f"{index_latency:.2f} ms"
    )

    # ============================================================
    # DIRECT OLLAMA TEST
    # ============================================================

    ollama_latency = benchmark_ollama()

    # ============================================================
    # WARMUP
    # ============================================================

    print("\n" + "=" * 60)
    print("RAG PIPELINE BENCHMARK")
    print("=" * 60)

    print("\nWarming up...")

    for query in QUERIES[:2]:
        application.pipeline.run(
            query,
            top_k=3,
        )

    print("Warm-up complete.\n")

    # ============================================================
    # BENCHMARK
    # ============================================================

    latencies = []
    retrieval_latencies = []
    generation_latencies = []

    for index, query in enumerate(
        QUERIES,
        start=1,
    ):
        print(
            f"[{index}/{len(QUERIES)}] "
            f"{query}"
        )

        start = time.perf_counter()

        result = application.pipeline.run(
            query,
            top_k=3,
        )

        latency_ms = (
            time.perf_counter() - start
        ) * 1000

        latencies.append(latency_ms)

        print(
            f"  Success: {result.success}"
        )

        print(
            f"  End-to-end: "
            f"{latency_ms:.2f} ms"
        )

        for stage in result.stages:

            print(
                f"  {stage.stage}: "
                f"{stage.latency_ms:.2f} ms"
            )

            if stage.stage == "retrieval":
                retrieval_latencies.append(
                    stage.latency_ms
                )

            elif stage.stage == "generation":
                generation_latencies.append(
                    stage.latency_ms
                )

        # Important diagnostic
        if not result.answer:
            print(
                "  WARNING: Empty answer returned."
            )

        print()

    # ============================================================
    # RESULTS
    # ============================================================

    print("=" * 60)
    print("RESULTS")
    print("=" * 60)

    print(
        f"Samples: {len(latencies)}"
    )

    print(
        f"P50:  "
        f"{percentile(latencies, 50):.2f} ms"
    )

    print(
        f"P70:  "
        f"{percentile(latencies, 70):.2f} ms"
    )

    print(
        f"P100: "
        f"{percentile(latencies, 100):.2f} ms"
    )

    print(
        f"Mean:  "
        f"{statistics.mean(latencies):.2f} ms"
    )

    print(
        f"Min:   "
        f"{min(latencies):.2f} ms"
    )

    print(
        f"Max:   "
        f"{max(latencies):.2f} ms"
    )

    # ============================================================
    # RETRIEVAL RESULTS
    # ============================================================

    if retrieval_latencies:

        print("\nRetrieval:")

        print(
            f"  P50:  "
            f"{percentile(retrieval_latencies, 50):.2f} ms"
        )

        print(
            f"  P100: "
            f"{percentile(retrieval_latencies, 100):.2f} ms"
        )

    # ============================================================
    # GENERATION RESULTS
    # ============================================================

    if generation_latencies:

        print("\nGeneration:")

        print(
            f"  P50:  "
            f"{percentile(generation_latencies, 50):.2f} ms"
        )

        print(
            f"  P100: "
            f"{percentile(generation_latencies, 100):.2f} ms"
        )

    # ============================================================
    # DIRECT OLLAMA
    # ============================================================

    print("\nDirect Ollama check:")

    print(
        f"  {ollama_latency:.2f} ms"
    )

    # ============================================================
    # INDEXING
    # ============================================================

    print("\nIndexing:")

    print(
        f"  {index_latency:.2f} ms"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()