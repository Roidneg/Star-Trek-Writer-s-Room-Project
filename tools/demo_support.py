"""Small, offline helpers for the Streamlit drafting interface."""

from __future__ import annotations

import hmac
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from workflows.draft_writers_room import (
    DraftCharacterNotes, DraftScene, DraftSceneRequest, load_draft_references,
)

ROOT = Path(__file__).resolve().parents[1]
MAX_UPLOAD_BYTES = 256 * 1024


def lines(value: str) -> list[str]:
    return [line.strip() for line in value.splitlines() if line.strip()]


def load_preset(filename: str) -> DraftSceneRequest:
    # Filenames are selected by the application, never supplied as paths by users.
    allowed = {"shadows_scene_request.json", "shadows_scene_2_request.json", "shadows_scene_3_request.json"}
    if filename not in allowed:
        raise ValueError("Unknown scene preset.")
    return DraftSceneRequest.model_validate_json((ROOT / "drafts" / filename).read_text(encoding="utf-8-sig"))


def parse_reference_upload(payload: bytes):
    if len(payload) > MAX_UPLOAD_BYTES:
        raise ValueError("Source upload must be 256 KB or smaller.")
    # Fixed basename and a private temporary directory prevent user-controlled
    # paths and cross-session source sharing. The original package hash remains.
    with TemporaryDirectory(prefix="writers_room_sources_") as directory:
        path = Path(directory) / "uploaded_sources.json"
        path.write_bytes(payload)
        return load_draft_references(path)


def load_sample() -> dict:
    payload = json.loads((ROOT / "examples" / "sample_scene.json").read_text(encoding="utf-8"))
    if payload["origin"] != "illustrative_sample" or payload["provider_calls"] != 0:
        raise ValueError("The sample must be explicitly identified as illustrative.")
    DraftSceneRequest.model_validate(payload["request"])
    DraftCharacterNotes.model_validate(payload["character_notes"])
    DraftScene.model_validate(payload["scene"])
    return payload


def live_configuration_status() -> tuple[bool, str]:
    provider = os.getenv("MODEL_PROVIDER", "groq").strip().casefold()
    required = {"cloudflare": ("CLOUDFLARE_API_TOKEN", "CLOUDFLARE_ACCOUNT_ID"), "groq": ("GROQ_API_KEY",)}
    if provider not in required:
        return False, "The host must configure Cloudflare or Groq."
    if any(not os.getenv(name, "").strip() for name in required[provider]):
        return False, "Live drafting is unavailable until the host adds provider credentials."
    if not os.getenv("WRITERS_ROOM_ACCESS_CODE", "").strip():
        return False, "Live drafting is unavailable until the host sets a demo access code."
    return True, "Live drafting is available with the host's demo access code."


def valid_access_code(value: str) -> bool:
    expected = os.getenv("WRITERS_ROOM_ACCESS_CODE", "")
    return bool(expected.strip()) and hmac.compare_digest(value.encode(), expected.encode())


def public_error(error: Exception) -> str:
    """Never publish provider exception text, which can contain secrets/prompts."""
    name = type(error).__name__
    status = getattr(error, "status_code", None)
    if status == 429 or name == "RateLimitError":
        return "The provider's rate limit or quota was reached. No automatic retry was made. Try again after the limit resets."
    if status in (401, 403) or name in ("AuthenticationError", "PermissionDeniedError"):
        return "The provider rejected the host's credentials or permissions. The host needs to check the configuration."
    if name in ("APITimeoutError", "TimeoutError", "APIConnectionError"):
        return "The provider could not be reached in time. No automatic retry was made."
    if isinstance(error, ValueError):
        return "The response did not meet the draft's structure or completion checks. No automatic repair was made."
    return "The draft could not be completed. No automatic retry was made. The host can use the error type below to investigate."
