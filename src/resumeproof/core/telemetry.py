from dataclasses import dataclass, field
from datetime import datetime, timezone
import logging
from typing import Optional

logger = logging.getLogger("resumeproof.telemetry")

@dataclass
class CallMetrics:
    model: str
    endpoint: str
    structured_path_used: str  # "json_schema" | "native_format" | "fake_recorded"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    status: str = "success"
    error: Optional[str] = None

class TelemetryCollector:
    def __init__(self):
        self.records: list[CallMetrics] = []

    def log_call(self, metric: CallMetrics) -> None:
        self.records.append(metric)
        logger.info(
            f"LLM Call: model={metric.model} path={metric.structured_path_used} "
            f"tokens={metric.total_tokens} latency={metric.latency_ms:.2f}ms status={metric.status}"
        )

    def summary(self) -> dict:
        total_tokens = sum(r.total_tokens for r in self.records)
        total_latency = sum(r.latency_ms for r in self.records)
        return {
            "total_calls": len(self.records),
            "total_tokens": total_tokens,
            "total_latency_ms": total_latency,
            "avg_latency_ms": total_latency / max(1, len(self.records)),
            "paths_used": list(set(r.structured_path_used for r in self.records))
        }

telemetry = TelemetryCollector()
