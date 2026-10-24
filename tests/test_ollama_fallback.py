"""Tests for Ollama / OpenAI structured output json_schema support and native format fallback."""

import json
from unittest.mock import patch, MagicMock
from pydantic import BaseModel
import httpx

from resumeproof.core.llm import OpenAICompatibleClient
from resumeproof.core.telemetry import telemetry

class SampleTestSchema(BaseModel):
    summary: str
    points: list[str]

def test_v1_json_schema_success():
    client = OpenAICompatibleClient(base_url="http://localhost:11434")

    # Mock response from /v1/chat/completions
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": json.dumps({"summary": "Success v1", "points": ["p1", "p2"]})
                }
            }
        ],
        "usage": {"prompt_tokens": 20, "completion_tokens": 15, "total_tokens": 35}
    }

    with patch("httpx.Client.post", return_value=mock_resp) as mock_post:
        result = client.structured_completion(
            messages=[{"role": "user", "content": "Extract"}],
            schema=SampleTestSchema,
            model="qwen2.5:7b"
        )

        assert result["summary"] == "Success v1"
        assert result["points"] == ["p1", "p2"]
        # Verify it called /v1/chat/completions
        assert mock_post.call_args[0][0] == "http://localhost:11434/v1/chat/completions"
        payload = mock_post.call_args[1]["json"]
        assert "response_format" in payload
        assert payload["response_format"]["type"] == "json_schema"

        # Check telemetry logged json_schema
        last_metric = telemetry.records[-1]
        assert last_metric.structured_path_used == "json_schema"

def test_v1_json_schema_fails_and_falls_back_to_native_ollama():
    client = OpenAICompatibleClient(base_url="http://localhost:11434")

    # 1st call to /v1 returns 400 Bad Request (unsupported response_format)
    resp_v1_fail = MagicMock()
    resp_v1_fail.status_code = 400
    resp_v1_fail.text = "json_schema not supported by this Ollama version"

    # 2nd call to /api/chat returns 200 OK
    resp_native_ok = MagicMock()
    resp_native_ok.status_code = 200
    resp_native_ok.json.return_value = {
        "message": {
            "content": json.dumps({"summary": "Fallback native", "points": ["f1"]})
        },
        "prompt_eval_count": 25,
        "eval_count": 10
    }

    with patch("httpx.Client.post", side_effect=[resp_v1_fail, resp_native_ok]) as mock_post:
        result = client.structured_completion(
            messages=[{"role": "user", "content": "Extract"}],
            schema=SampleTestSchema,
            model="qwen2.5:7b"
        )

        assert result["summary"] == "Fallback native"
        assert result["points"] == ["f1"]

        # Verify second call was to Ollama native /api/chat with format parameter
        assert mock_post.call_count == 2
        second_call_url = mock_post.call_args_list[1][0][0]
        second_call_json = mock_post.call_args_list[1][1]["json"]

        assert second_call_url == "http://localhost:11434/api/chat"
        assert "format" in second_call_json
        assert second_call_json["format"] == SampleTestSchema.model_json_schema()

        # Check telemetry logged native_format
        last_metric = telemetry.records[-1]
        assert last_metric.structured_path_used == "native_format"

def test_structured_completion_handles_markdown_code_fences():
    client = OpenAICompatibleClient(base_url="http://localhost:11434")

    # Mock response where LLM wraps JSON in ```json\n...\n```
    fenced_content = "```json\n" + json.dumps({"summary": "Fenced output", "points": ["m1", "m2"]}) + "\n```"
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": fenced_content
                }
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}
    }

    with patch("httpx.Client.post", return_value=mock_resp):
        result = client.structured_completion(
            messages=[{"role": "user", "content": "Extract"}],
            schema=SampleTestSchema,
            model="qwen2.5:7b"
        )
        assert result["summary"] == "Fenced output"
        assert result["points"] == ["m1", "m2"]
