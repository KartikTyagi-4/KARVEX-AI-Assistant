from backend.application import Application


def test_application_builds_rag_stack():
    application = Application()

    assert application.embedder is not None
    assert application.vector_store is not None
    assert application.retriever is not None
    assert application.generator is not None
    assert application.pipeline is not None
    assert application.service is not None

    assert application.pipeline.retriever is application.retriever
    assert application.pipeline.generator is application.generator

    assert application.service.pipeline is application.pipeline
def test_application_service_handles_query():
    application = Application()

    result = application.service.ask(
        "What is solar energy?"
    )

    assert result.query == "What is solar energy?"
    assert result.success is True
    assert result.answer != ""
def test_application_indexes_and_answers():
    application = Application()

    records = [
        {
            "query_id": "q1",
            "query": "What is solar energy?",
            "Answer": "Solar energy comes from the sun.",
            "Eng_Query": "What is solar energy?",
            "Eng_Answer": "Solar energy comes from the sun.",
            "source_lang": "eng_Latn",
            "target_lang": "eng_Latn",
            "query_type": "factoid",
            "passages": {
                "English_passages": [
                    "Solar energy is renewable and comes from sunlight."
                ],
                "Translated_passages": [],
                "is_selected": [1],
            },
            "meta": {},
        }
    ]

    count = application.index_records(records)

    assert count == 1

    result = application.service.ask(
        "What is solar energy?"
    )

    assert result.success is True
    assert result.query == "What is solar energy?"
    assert result.answer != ""