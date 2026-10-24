"""LLM client abstractions, OpenAI/Ollama client with schema fallback, and offline FakeLLMClient."""

from abc import ABC, abstractmethod
import json
import logging
import time
from typing import Any, Optional, Type
import httpx
from pydantic import BaseModel
from resumeproof.core.telemetry import telemetry, CallMetrics

logger = logging.getLogger("resumeproof.llm")

def _clean_json_str(raw: str) -> str:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    return cleaned

class LLMClient(ABC):
    """Abstract LLM Client interface."""

    @abstractmethod
    def chat_completion(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float = 0.0,
        timeout: float = 60.0,
    ) -> str:
        """Standard text completion."""
        pass

    @abstractmethod
    def structured_completion(
        self,
        messages: list[dict[str, str]],
        schema: Type[BaseModel] | dict[str, Any],
        model: str,
        temperature: float = 0.0,
        timeout: float = 60.0,
    ) -> dict[str, Any]:
        """Structured completion enforcing a JSON schema."""
        pass

class OpenAICompatibleClient(LLMClient):
    """Client for Ollama or OpenAI-compatible endpoints with json_schema fallback."""

    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url.rstrip("/")
        self.v1_endpoint = f"{self.base_url}/v1/chat/completions"
        self.ollama_endpoint = f"{self.base_url}/api/chat"
        self._tested_json_schema_support: Optional[bool] = None

    def chat_completion(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float = 0.0,
        timeout: float = 60.0,
    ) -> str:
        start_time = time.perf_counter()
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(self.v1_endpoint, json=payload)
            resp.raise_for_status()
            data = resp.json()
            latency = (time.perf_counter() - start_time) * 1000
            usage = data.get("usage", {})
            telemetry.log_call(CallMetrics(
                model=model,
                endpoint=self.v1_endpoint,
                structured_path_used="standard_chat",
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
                latency_ms=latency,
            ))
            return data["choices"][0]["message"]["content"]

    def structured_completion(
        self,
        messages: list[dict[str, str]],
        schema: Type[BaseModel] | dict[str, Any],
        model: str,
        temperature: float = 0.0,
        timeout: float = 60.0,
    ) -> dict[str, Any]:
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            schema_dict = schema.model_json_schema()
            schema_name = schema.__name__
        else:
            schema_dict = schema
            schema_name = "OutputSchema"

        start_time = time.perf_counter()

        # Step 1: Try OpenAI response_format json_schema if not already known to fail
        if self._tested_json_schema_support is not False:
            v1_payload = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema_name,
                        "strict": True,
                        "schema": schema_dict,
                    },
                },
            }
            try:
                with httpx.Client(timeout=timeout) as client:
                    resp = client.post(self.v1_endpoint, json=v1_payload)
                    if resp.status_code == 200:
                        self._tested_json_schema_support = True
                        data = resp.json()
                        latency = (time.perf_counter() - start_time) * 1000
                        usage = data.get("usage", {})
                        content = data["choices"][0]["message"]["content"]
                        telemetry.log_call(CallMetrics(
                            model=model,
                            endpoint=self.v1_endpoint,
                            structured_path_used="json_schema",
                            prompt_tokens=usage.get("prompt_tokens", 0),
                            completion_tokens=usage.get("completion_tokens", 0),
                            total_tokens=usage.get("total_tokens", 0),
                            latency_ms=latency,
                        ))
                        return json.loads(_clean_json_str(content))
                    else:
                        logger.warning(
                            f"/v1/chat/completions json_schema returned status {resp.status_code}: {resp.text}. Falling back to native Ollama format parameter."
                        )
                        self._tested_json_schema_support = False
            except Exception as e:
                logger.warning(f"Failed calling /v1 json_schema ({e}); falling back to native Ollama format.")
                self._tested_json_schema_support = False

        # Step 2: Fallback to Ollama native /api/chat with format parameter
        ollama_payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "format": schema_dict,
            "options": {"temperature": temperature},
        }
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(self.ollama_endpoint, json=ollama_payload)
            resp.raise_for_status()
            data = resp.json()
            latency = (time.perf_counter() - start_time) * 1000
            content = data["message"]["content"]
            telemetry.log_call(CallMetrics(
                model=model,
                endpoint=self.ollama_endpoint,
                structured_path_used="native_format",
                prompt_tokens=data.get("prompt_eval_count", 0),
                completion_tokens=data.get("eval_count", 0),
                total_tokens=data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
                latency_ms=latency,
            ))
            return json.loads(_clean_json_str(content))

class FakeLLMClient(LLMClient):
    """Offline, deterministic LLM client for tests and CI with recorded fixtures."""

    def __init__(self, recorded_fixtures: Optional[dict[str, Any]] = None):
        self.recorded_fixtures = recorded_fixtures or {}
        self.call_history: list[dict[str, Any]] = []

    def set_fixture(self, key: str, response: Any) -> None:
        self.recorded_fixtures[key] = response

    def chat_completion(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float = 0.0,
        timeout: float = 60.0,
    ) -> str:
        self.call_history.append({"messages": messages, "model": model, "type": "chat"})
        telemetry.log_call(CallMetrics(
            model=model,
            endpoint="fake://local",
            structured_path_used="fake_recorded",
            prompt_tokens=10,
            completion_tokens=10,
            total_tokens=20,
            latency_ms=1.0,
        ))
        # Look for fixture match
        msg_str = " ".join(m.get("content", "") for m in messages)
        for k, v in self.recorded_fixtures.items():
            if k in msg_str:
                return v if isinstance(v, str) else json.dumps(v)
        return "Deterministic response from FakeLLMClient."

    def structured_completion(
        self,
        messages: list[dict[str, str]],
        schema: Type[BaseModel] | dict[str, Any],
        model: str,
        temperature: float = 0.0,
        timeout: float = 60.0,
    ) -> dict[str, Any]:
        self.call_history.append({"messages": messages, "model": model, "schema": str(schema), "type": "structured"})
        telemetry.log_call(CallMetrics(
            model=model,
            endpoint="fake://local",
            structured_path_used="fake_recorded",
            prompt_tokens=15,
            completion_tokens=15,
            total_tokens=30,
            latency_ms=1.0,
        ))
        msg_str = " ".join(m.get("content", "") for m in messages)

        # Check explicit fixture keys
        for k, v in self.recorded_fixtures.items():
            if k.lower() in msg_str.lower():
                if isinstance(v, dict):
                    return v
                elif isinstance(v, str):
                    return json.loads(v)

        # Automatic detection of demo personas in resume text
        from resumeproof.fixtures.demo_fixtures import DEMO_FIXTURES
        msg_lower = msg_str.lower()
        if "alex rivera" in msg_lower or "alex_rivera" in msg_lower:
            return DEMO_FIXTURES["alex_rivera"]
        if "priya nair" in msg_lower or "priya_nair" in msg_lower:
            return DEMO_FIXTURES["priya_nair"]
        if "sam okafor" in msg_lower or "sam_okafor" in msg_lower:
            return DEMO_FIXTURES["sam_okafor"]

        # Default fallback structure based on schema
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            schema_name = schema.__name__
        else:
            schema_name = str(schema.get("title", ""))

        if "ProfileFact" in schema_name or "profile" in schema_name.lower():
            return {
                "profile_facts": [
                    {"fact_type": "name", "value": "Test Candidate", "source_line": 1},
                    {"fact_type": "email", "value": "candidate@example.com", "source_line": 1},
                    {"fact_type": "phone", "value": "(555) 000-1111", "source_line": 1},
                    {"fact_type": "education_institution", "value": "Tech University", "source_line": 3},
                    {"fact_type": "cgpa", "value": "3.8 / 4.0", "source_line": 3},
                    {"fact_type": "graduation_date", "value": "May 2022", "source_line": 3},
                ],
                "claims": []
            }

        return {
            "profile_facts": [],
            "claims": [
                {
                    "claim_type": "skill",
                    "text": "Python programming and REST APIs",
                    "tools": ["Python", "FastAPI"],
                    "metric_values": [],
                    "source_line": 10
                }
            ]
        }
