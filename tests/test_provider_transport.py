"""Exercise the real pinned SDKs against in-memory HTTP, without API calls."""

import json

import httpx
import pytest

from tools.demo_support import load_preset, load_sample
from tools import model_provider
from workflows.draft_writers_room import write_draft_scene


@pytest.mark.parametrize("provider", ["cloudflare", "groq"])
def test_real_provider_sdk_serializes_and_parses_both_draft_stages(monkeypatch, provider):
    monkeypatch.setenv("MODEL_PROVIDER", provider)
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "offline-token")
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "offline-account")
    monkeypatch.setenv("GROQ_API_KEY", "offline-token")
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.delenv(name, raising=False)
    sample = load_sample()
    requests = []

    def handle(request):
        body = json.loads(request.content)
        requests.append((str(request.url), body))
        output = sample["character_notes"] if len(requests) == 1 else sample["scene"]
        return httpx.Response(200, json={
            "id": "offline-completion", "object": "chat.completion", "created": 1,
            "model": body["model"],
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": json.dumps(output)}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
        })

    class_name = "ChatOpenAI" if provider == "cloudflare" else "ChatGroq"
    constructor = getattr(model_provider, class_name)
    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        def build(**kwargs):
            assert kwargs["max_retries"] == 0
            return constructor(http_client=client, **kwargs)
        monkeypatch.setattr(model_provider, class_name, build)
        trace = []
        result = write_draft_scene(load_preset("shadows_scene_request.json"), trace=trace)
    assert result.assurance == "unverified"
    assert len(requests) == 2
    assert all(entry["provider_request_attempted"] for entry in trace)
    for index, (url, body) in enumerate(requests):
        assert url.endswith("/chat/completions")
        assert body["response_format"] == {"type": "json_object"}
        assert body["reasoning_effort"] == "low"
        assert body.get("max_completion_tokens", body.get("max_tokens")) == [1800, 3200][index]
        assert body["model"] == ("@cf/openai/gpt-oss-120b" if provider == "cloudflare" else "openai/gpt-oss-120b")
