from dataclasses import dataclass, field
from typing import Any


@dataclass
class StageResult:
    stage: str
    success: bool
    latency_ms: float
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineResult:
    query: str
    answer: str
    stages: list[StageResult] = field(default_factory=list)
    success: bool = True
    error: str | None = None
    