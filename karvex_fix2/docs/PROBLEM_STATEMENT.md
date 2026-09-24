# HH Goa 2026 — Task 2
# VĀDA: Voice-Enabled Retrieval-Augmented Generation System

## 1. Problem

Build a voice-enabled Retrieval-Augmented Generation (RAG) system over the
AI4Bharat MSMARCO-XI dataset.

The user speaks a question.

The system must:

1. Capture voice input.
2. Transcribe the speech using Sarvam or ElevenLabs.
3. Process and normalize the query.
4. Retrieve relevant evidence from MSMARCO-XI.
5. Use a sophisticated chunking/indexing strategy.
6. Generate an answer grounded in retrieved evidence.
7. Refuse to answer when sufficient evidence is unavailable.
8. Handle unsafe, irrelevant, malformed, and adversarial inputs.
9. Return the result through a robust orchestration harness.
10. Measure and report pipeline latency.

## 2. Mandatory Requirements

### Speech-to-text
Use Sarvam Saaras v3.

### Dataset
Use:
https://huggingface.co/datasets/ai4bharat/MSMARCO-XI

The dataset must be accessed responsibly without requiring the entire
dataset to be downloaded locally.

### Chunking
The system must implement and evaluate multiple chunking strategies rather
than relying on a single naive fixed-size splitter.

Candidate strategies include:

- fixed-size
- overlapping fixed-size
- sentence-aware
- paragraph-aware
- hierarchical
- semantic
- adaptive

The final strategy must be selected based on measurable retrieval quality
and latency.

### Retrieval

The retrieval architecture should support:

- dense vector retrieval
- sparse/BM25 retrieval
- metadata filtering
- candidate fusion
- reranking

### Generation

Answers must be generated from retrieved evidence and must not rely on
unsupported external knowledge.

### Guardrails

The system must handle:

- off-topic questions
- insufficient evidence
- hallucination risk
- unsafe inputs
- prompt injection attempts
- malformed requests
- service/API failures

### Harness

The pipeline must use structured orchestration including:

- typed inputs/outputs
- retries
- timeouts
- error handling
- request IDs
- stage-level telemetry
- deterministic failure behavior

### Latency

Measure the actual pipeline across a meaningful benchmark set.

Report:

- P50
- P70
- P100

Do not fabricate performance numbers.

Separate:

1. voice/STT latency
2. RAG latency
3. complete end-to-end latency

## 3. Performance Objective

Target a sub-200ms RAG pipeline where technically achievable.

Because external speech and LLM APIs introduce network latency, the system
must report latency transparently rather than falsely claiming that every
voice-to-answer interaction completes under 200ms.

## 4. Memory

Implement three layers:

### Working memory
Current request and retrieved context.

### Session memory
Recent conversation turns and query context.

### Semantic memory
Only useful long-lived information should be retained.

Memory must be bounded and controllable.

## 5. Product Objective

The final product should not feel like a generic chatbot.

It should feel like a voice-first research and evidence engine.

The user should be able to:

- speak naturally
- see the transcript
- see retrieval status
- inspect sources
- see confidence/grounding information
- see latency
- receive a concise answer
- optionally hear the answer

## 6. Engineering Principles

- correctness before complexity
- measurable performance
- modular architecture
- graceful degradation
- reproducibility
- zero-secret exposure
- free/local-first development
- minimal external dependencies
- asynchronous I/O where beneficial
- caching where beneficial
- no unnecessary database calls
- no unnecessary LLM calls

## 7. Definition of Done

The final system must:

- accept voice input
- transcribe it
- retrieve evidence
- generate a grounded answer
- refuse unsupported questions
- expose sources
- expose latency telemetry
- survive common failure cases
- provide benchmark results
- have a polished responsive UI
- have a public live deployment
- have reproducible setup instructions