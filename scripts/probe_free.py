import os
import httpx

key = os.environ.get("OPENROUTER_API_KEY")
headers = {"Authorization": f"Bearer {key}"}

candidates = [
    "poolside/laguna-xs-2.1:free",
    "cohere/north-mini-code:free",
    "thinkingmachines/inkling-small:free",
    "recursive/flint-v1.0:free",
    "liquid/lfm-2.5-2.6b:free",
]

payload = {
    "messages": [{"role": "user", "content": "Respond with JSON: {\"status\": \"ok\"}"}],
    "max_tokens": 100,
    "response_format": {"type": "json_object"}
}

for m in candidates:
    payload["model"] = m
    try:
        r = httpx.post("https://openrouter.ai/api/v1/chat/completions", json=payload, headers=headers, timeout=10.0)
        print(f"Model: {m} -> Status: {r.status_code}")
        if r.status_code == 200:
            print("  Output:", r.json()["choices"][0]["message"]["content"])
    except Exception as e:
        print(f"Model: {m} -> Error: {e}")
