import json
import pyarrow.parquet as pq


def load_msmarco_xi_json(
    path: str,
    limit: int = 100,
):
    """
    Load a small number of MSMARCO-XI records
    from a local JSON file.
    """

    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    return data[:limit]


def load_msmarco_xi(
    path: str,
    limit: int = 100,
):
    """
    Load a small number of MSMARCO-XI records directly
    from the local Parquet file.
    """

    parquet_file = pq.ParquetFile(path)

    columns = [
        "query_id",
        "source_lang",
        "target_lang",
        "query",
        "Eng_Query",
        "Answer",
        "Eng_Answer",
        "passages",
    ]

    batch = next(
        parquet_file.iter_batches(
            batch_size=limit,
            columns=columns,
        )
    )

    data = batch.to_pydict()

    records = []

    for i in range(len(data["query_id"])):
        records.append(
            {
                "query_id": data["query_id"][i],
                "source_lang": data["source_lang"][i],
                "target_lang": data["target_lang"][i],
                "query": data["query"][i],
                "Eng_Query": data["Eng_Query"][i],
                "Answer": data["Answer"][i],
                "Eng_Answer": data["Eng_Answer"][i],
                "passages": data["passages"][i],
            }
        )

    return records