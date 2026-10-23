import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest
from streamlit.testing.v1 import AppTest

from tools.demo_support import (
    MAX_UPLOAD_BYTES, live_configuration_status, load_preset, load_sample,
    parse_reference_upload, public_error, valid_access_code,
)
from workflows import draft_writers_room as D

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def clean_config(monkeypatch):
    for name in ("GROQ_API_KEY", "CLOUDFLARE_API_TOKEN", "CLOUDFLARE_ACCOUNT_ID", "WRITERS_ROOM_ACCESS_CODE", "MODEL_PROVIDER"):
        monkeypatch.delenv(name, raising=False)


def app():
    return AppTest.from_file(str(ROOT / "app.py"), default_timeout=15).run()


def live_app(monkeypatch):
    monkeypatch.setenv("MODEL_PROVIDER", "cloudflare")
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "test-token-not-a-secret")
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "test-account")
    monkeypatch.setenv("WRITERS_ROOM_ACCESS_CODE", "test-access-code")
    at = app()
    at.radio[0].set_value("Write a scene").run()
    return at


def field(at, label):
    return next(x for x in at.text_input if x.label == label)


def button(at, label):
    return next(x for x in at.button if x.label.startswith(label))


def fake_transport(stage, schema, messages, entry):
    entry["provider_request_attempted"] = True
    sample = load_sample()
    return schema.model_validate(sample["character_notes"] if stage == "character_draft" else sample["scene"])


def test_sample_opens_without_credentials_or_provider_imports():
    with patch.object(D, "_configured_invoke", side_effect=AssertionError("Unexpected model call")):
        at = app()
    assert not at.exception
    assert any("not a recorded model run" in x.value for x in at.info)
    assert at.metric[0].value == "0"
    assert "GARAK" in at.code[0].value


def test_live_disabled_without_host_configuration():
    at = app()
    at.radio[0].set_value("Write a scene").run()
    assert not at.exception
    assert button(at, "Generate scene").disabled
    button(at, "Check brief").click().run()
    assert not at.exception
    assert "zero provider calls" in at.success[0].value


def test_empty_brief_and_duplicate_characters_do_not_invoke(monkeypatch):
    at = live_app(monkeypatch)
    with patch.object(D, "_configured_invoke") as invoke:
        field(at, "Story era").set_value("   ")
        button(at, "Check brief").click().run()
        assert at.error
        field(at, "Story era").set_value("DS9")
        next(x for x in at.text_area if x.label.startswith("Characters")).set_value("Garak\ngarak")
        button(at, "Generate scene").click().run()
        assert at.error
        invoke.assert_not_called()


def test_access_and_acknowledgment_checked_before_generation(monkeypatch):
    at = live_app(monkeypatch)
    with patch.object(D, "_configured_invoke") as invoke:
        button(at, "Generate scene").click().run()
        assert "acknowledgment" in at.error[0].value
        at.checkbox[0].check()
        field(at, "Demo access code").set_value("wrong")
        button(at, "Generate scene").click().run()
        assert "access code" in at.error[0].value
        invoke.assert_not_called()


def test_complete_workflow_renders_downloads_and_rerun_does_not_regenerate(monkeypatch):
    at = live_app(monkeypatch)
    at.checkbox[0].check()
    field(at, "Demo access code").set_value("test-access-code")
    with patch.object(D, "_configured_invoke", side_effect=fake_transport) as invoke:
        button(at, "Generate scene").click().run()
        assert not at.exception
        assert not at.error
        assert invoke.call_count == 2
        assert len(at.get("download_button")) == 2
        saved = D.DraftSceneArtifact.model_validate(at.session_state["draft_artifact"])
        assert saved.requires_manual_review and not saved.semantic_audits_run
        assert saved.assurance == "unverified"
        assert at.metric[0].value == "2"
        at.run()
        assert invoke.call_count == 2
        button(at, "Generate scene").click().run()
        assert any("30 seconds" in x.value for x in at.warning)
        assert invoke.call_count == 2


def test_failed_generation_keeps_previous_scene_and_never_displays_raw_error(monkeypatch):
    at = live_app(monkeypatch)
    at.checkbox[0].check()
    field(at, "Demo access code").set_value("test-access-code")
    with patch.object(D, "_configured_invoke", side_effect=fake_transport):
        button(at, "Generate scene").click().run()
    previous = at.session_state["draft_artifact"]
    at.session_state["last_attempt"] = float("-inf")

    def fail(stage, schema, messages, entry):
        if stage == "scene_draft":
            entry["provider_request_attempted"] = True
            raise ValueError("provider echoed test-token-not-a-secret")
        return fake_transport(stage, schema, messages, entry)

    with patch.object(D, "_configured_invoke", side_effect=fail) as invoke:
        button(at, "Generate scene").click().run()
        assert invoke.call_count == 2
    assert not at.exception
    assert "test-token-not-a-secret" not in str([x.value for x in at.error])
    assert at.session_state["draft_artifact"] == previous
    assert at.session_state["draft_failure"]["calls_attempted"] == 2


def test_presets_validate_and_sample_provenance_is_explicit():
    for filename in ("shadows_scene_request.json", "shadows_scene_2_request.json", "shadows_scene_3_request.json"):
        assert load_preset(filename).characters
    sample = load_sample()
    assert sample["origin"] == "illustrative_sample" and sample["provider_calls"] == 0
    with pytest.raises(ValueError):
        load_preset("../../.env")


def test_upload_boundaries_and_invalid_source_containers():
    for payload in (b"x" * (MAX_UPLOAD_BYTES + 1), b"not json", b"[]", b'{"summary": "generated prose"}'):
        with pytest.raises(ValueError):
            parse_reference_upload(payload)


def test_access_code_needs_explicit_host_configuration(monkeypatch):
    assert not valid_access_code("")
    assert not live_configuration_status()[0]
    monkeypatch.setenv("MODEL_PROVIDER", "cloudflare")
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "test")
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "test")
    assert not live_configuration_status()[0]
    monkeypatch.setenv("WRITERS_ROOM_ACCESS_CODE", "different-code")
    assert live_configuration_status()[0]
    assert valid_access_code("different-code")
    assert not valid_access_code("wrong")


def test_public_error_does_not_echo_provider_payload():
    error = RuntimeError("sensitive-token-and-prompt")
    error.status_code = 429
    assert "quota" in public_error(error)
    assert "sensitive" not in public_error(error)


def test_separate_sessions_do_not_share_drafts(monkeypatch):
    first = live_app(monkeypatch)
    first.checkbox[0].check()
    field(first, "Demo access code").set_value("test-access-code")
    with patch.object(D, "_configured_invoke", side_effect=fake_transport):
        button(first, "Generate scene").click().run()
    second = app()
    second.radio[0].set_value("Write a scene").run()
    assert "draft_artifact" not in second.session_state
    assert not second.get("download_button")
