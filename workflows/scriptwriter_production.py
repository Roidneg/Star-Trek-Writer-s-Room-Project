from datetime import datetime, timezone
import json
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel

from agents.scriptwriter_agent import (
    ScriptScene,
    ScriptSceneAudit,
    ScriptwriterInput,
    write_scene,
    enumerate_script_scene_segments,
)


# ---------------------------------------------------------
# Scriptwriter Production Result
# ---------------------------------------------------------


class ScriptwriterProductionResult(BaseModel):
    """
    Complete Scriptwriter production package for downstream
    Writer's Room orchestration.

    The frozen Scriptwriter Agent owns scene generation,
    validation, semantic grounding, one-pass repair, and
    hard-failure behavior.

    This adapter preserves the input, returned scene, and
    successful grounding status without re-running the
    semantic audit.
    """

    script_input: ScriptwriterInput

    scene: ScriptScene

    audit: ScriptSceneAudit


# ---------------------------------------------------------
# Production Adapter
# ---------------------------------------------------------


DIAGNOSTICS_DIRECTORY = (
    Path(__file__).resolve().parent.parent / "runtime" / "diagnostics"
)


def save_scriptwriter_failure(
    script_input: ScriptwriterInput,
    error: Exception,
) -> Path:
    """Write a separate, atomic failure report; never alter checkpoints."""
    payload = dict(getattr(error, "scriptwriter_diagnostics", {}) or {})
    payload.setdefault("script_input", script_input.model_dump(mode="json"))
    payload.setdefault("schema_version", 1)
    payload["status"] = "failed"
    payload["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    payload["error"] = {
        "type": type(error).__name__,
        "message": str(error),
    }
    # Include exact segment text alongside audit IDs for offline inspection.
    for key in ("generated_scene", "repaired_scene"):
        if key in payload:
            scene = ScriptScene.model_validate(payload[key])
            payload[key + "_segments"] = [
                segment.model_dump(mode="json")
                for segment in enumerate_script_scene_segments(scene)
            ]

    report_text = json.dumps(payload, indent=2, ensure_ascii=False)
    DIAGNOSTICS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    report_path = DIAGNOSTICS_DIRECTORY / (
        "scriptwriter_failure_" + uuid4().hex + ".json"
    )
    temporary_path = report_path.with_suffix(".tmp")
    try:
        temporary_path.write_text(report_text, encoding="utf-8")
        temporary_path.replace(report_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return report_path


def write_scene_for_production(
    script_input: ScriptwriterInput,
) -> ScriptwriterProductionResult:
    """
    Run the frozen Scriptwriter production path and package
    its successful result for downstream orchestration.

    write_scene() has a frozen contract:

    - validate input
    - generate scene
    - validate generated output
    - semantically audit scene
    - return immediately if grounded
    - otherwise repair once
    - validate repaired output
    - independently re-audit repaired scene
    - return only if grounded
    - otherwise raise ValueError

    Therefore, a successfully returned ScriptScene has already
    passed the frozen Scriptwriter grounding gate.

    This adapter does not perform another semantic audit.
    """

    try:
        scene = write_scene(script_input)
    except Exception as error:
        try:
            report_path = save_scriptwriter_failure(script_input, error)
        except Exception as diagnostic_error:
            # Disk/serialization errors must not replace the production error.
            error.add_note(
                "Scriptwriter diagnostics could not be saved: "
                + str(diagnostic_error)
            )
        else:
            error.add_note(
                "Scriptwriter failure diagnostics saved to: "
                + str(report_path)
            )
        raise

    # -----------------------------------------------------
    # Frozen Contract Receipt
    #
    # write_scene() cannot successfully return an
    # ungrounded ScriptScene. Its internal control flow
    # raises instead.
    #
    # We therefore preserve that authoritative production
    # outcome without spending another LLM audit call.
    # -----------------------------------------------------

    audit = ScriptSceneAudit(
        grounded=True,
        issues=[],
        reason=(
            "Frozen Scriptwriter write_scene() returned "
            "successfully. Its production contract returns "
            "a ScriptScene only after the scene passes its "
            "semantic grounding gate."
        ),
    )

    return ScriptwriterProductionResult(
        script_input=script_input,
        scene=scene,
        audit=audit,
    )