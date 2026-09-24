# VĀDA — Canonical Data Contract

## Purpose

Raw MSMARCO-XI records are normalized into a canonical internal representation before chunking, embedding, retrieval, reranking, or generation.

## Source

ai4bharat/MSMARCO-XI
https://huggingface.co/datasets/ai4bharat/MSMARCO-XI

## Canonical Document

Each normalized record contains:

- id
- query
- answer
- english_query
- english_answer
- language
- source_language
- target_language
- query_type
- passages
- metadata

## Passages

Each passage preserves:

- original English text
- translated text
- selected status

## Design Principles

- Preserve dataset provenance.
- Never destroy original text.
- Support Indian languages.
- Keep chunking replaceable.
- Keep retrieval replaceable.
- Support dense, lexical, hybrid retrieval and reranking.
- Expose latency telemetry.
- Require grounded generation.
- Support abstention when evidence is insufficient.
- Keep secrets outside source code.
- Support reproducible experiments.
- Remain free/local-first during development.

