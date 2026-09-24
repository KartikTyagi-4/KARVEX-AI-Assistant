from backend.application import create_application
from backend.ingestion.load_dataset import load_msmarco_xi_json


QUERIES = [
    # Relevant
    "What is solar energy?",
    "How do solar panels work?",
    "What are the advantages of solar power?",

    # Clearly unrelated
    "What is the capital of France?",
    "Who won the FIFA World Cup?",
    "How do I cook pasta?",
    "What is the latest iPhone?",
]


def main():
    print("=" * 70)
    print("KARVEX RETRIEVAL SCORE CALIBRATION")
    print("=" * 70)

    application = create_application()

    records = load_msmarco_xi_json(
        "data/sample/dev20.json",
        limit=20,
    )

    application.index_records(records)

    for query in QUERIES:
        print("\n" + "-" * 70)
        print(f"QUERY: {query}")

        results = application.retriever.search(
            query,
            top_k=3,
        )

        if not results:
            print("NO RESULTS")
            continue

        for index, result in enumerate(
            results,
            start=1,
        ):
            print(
                f"{index}. "
                f"score={result.score:.4f} "
                f"chunk={result.record.chunk_id}"
            )

            text = (
                result.record.text
                .replace("\n", " ")
                .strip()
            )

            print(
                f"   {text[:180]}"
            )


if __name__ == "__main__":
    main()