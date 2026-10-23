import json
import re
import sys
from pathlib import Path


# ---------------------------------------------------------
# Project Import Path
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------
# Scriptwriter Imports
# ---------------------------------------------------------

from agents.scriptwriter_agent import (
    ScriptScene,
    ScriptwriterInput,
    audit_script_scene,
    build_script_scene_audit,
    enumerate_script_scene_segments,
    validate_script_scene_output,
    validate_scriptwriter_input,
)


# ---------------------------------------------------------
# Saved Production Failure
# ---------------------------------------------------------

DIAGNOSTIC_PATH = (
    PROJECT_ROOT
    / "runtime"
    / "diagnostics"
    / "scriptwriter_failure_e08190355ed44fcab94ad127b156f539.json"
)


def main() -> None:
    print(
        "\n--- Scriptwriter Saved Repaired Scene Re-Audit ---"
    )

    # -----------------------------------------------------
    # Load the exact failed production artifact.
    # -----------------------------------------------------

    if not DIAGNOSTIC_PATH.exists():
        raise FileNotFoundError(
            "Saved Scriptwriter diagnostic was not found:\n"
            f"{DIAGNOSTIC_PATH}"
        )

    payload = json.loads(
        DIAGNOSTIC_PATH.read_text(
            encoding="utf-8"
        )
    )

    print(
        "Diagnostic:",
        DIAGNOSTIC_PATH,
    )

    print(
        "Saved failure stage:",
        payload.get("stage"),
    )

    if payload.get("stage") != "repaired_audit":
        raise ValueError(
            "Expected a failure captured at repaired_audit, "
            f"received: {payload.get('stage')!r}"
        )

    if "script_input" not in payload:
        raise ValueError(
            "Diagnostic does not contain script_input."
        )

    if "repaired_scene" not in payload:
        raise ValueError(
            "Diagnostic does not contain repaired_scene."
        )

    # -----------------------------------------------------
    # Reconstruct the exact production objects.
    # -----------------------------------------------------

    script_input = ScriptwriterInput.model_validate(
        payload["script_input"]
    )

    repaired_scene = ScriptScene.model_validate(
        payload["repaired_scene"]
    )

    # -----------------------------------------------------
    # Apply the single deterministic repair identified by
    # the previous semantic audit.
    #
    # Preserve the original diagnostic artifact unchanged.
    # Only the in-memory candidate is modified.
    #
    # Remove the entire physical screenplay line rather
    # than replacing only its text with an empty string.
    # Leaving a blank line here would terminate the active
    # speaker block and could misclassify the courier's
    # following dialogue as action.
    # -----------------------------------------------------

    unsupported_sentence = (
        "The transmission was garbled, the rod is cracked."
    )

    # The saved screenplay physically wraps this sentence:
    #
    #     The transmission was
    #     garbled, the rod is cracked. But if it is true...
    #
    # Match semantic whitespace rather than requiring the
    # sentence to exist on one physical screenplay line.
    unsupported_pattern = re.compile(
        r"The transmission was\s+"
        r"garbled, the rod is cracked\.\s+"
        r"(?=But if it is true)"
    )

    repaired_script_text, replacement_count = (
        unsupported_pattern.subn(
            "",
            repaired_scene.script_text,
            count=1,
        )
    )

    if replacement_count != 1:
        raise ValueError(
            "Expected exactly one wrapped unsupported "
            "sentence in the saved repaired ScriptScene, "
            f"but replaced {replacement_count}."
        )

    repaired_scene = repaired_scene.model_copy(
        update={
            "script_text": repaired_script_text
        }
    )

    # Verify semantically, collapsing screenplay whitespace.
    normalized_repaired_text = " ".join(
        repaired_scene.script_text.split()
    )

    if unsupported_sentence in normalized_repaired_text:
        raise AssertionError(
            "Unsupported sentence remained in the "
            "in-memory repaired ScriptScene."
        )

    if "But if it is true" not in normalized_repaired_text:
        raise AssertionError(
            "Deterministic cleanup accidentally removed "
            "the following courier dialogue."
        )

    print(
        "Known unsupported sentence removed "
        "from in-memory candidate: PASS"
    )

    print("Saved ScriptwriterInput restored: PASS")
    print("Saved repaired ScriptScene restored: PASS")

    # -----------------------------------------------------
    # Deterministic validation first.
    #
    # No model calls occur before these checks pass.
    # -----------------------------------------------------

    validate_scriptwriter_input(
        script_input
    )

    validate_script_scene_output(
        script_input,
        repaired_scene,
    )

    print("ScriptwriterInput validation: PASS")
    print("Repaired ScriptScene validation: PASS")

    # -----------------------------------------------------
    # Show how the NEW parser sees the saved scene.
    # -----------------------------------------------------

    segments = enumerate_script_scene_segments(
        repaired_scene
    )

    print(
        "\nReparsed segment count:",
        len(segments),
    )

    print("\n--- REPARSED SAVED SCENE ---")

    for segment in segments:
        print(
            segment.segment_id,
            "|",
            segment.segment_type,
            "|",
            segment.speaker,
            "|",
            segment.text,
        )

    print("--- END REPARSED SAVED SCENE ---")

    # -----------------------------------------------------
    # Guard specifically against the production parser
    # defects we just fixed.
    # -----------------------------------------------------

    malformed_continuation = [
        segment
        for segment in segments
        if (
            segment.text.casefold()
            == "garak (cont.)"
            and segment.segment_type
            != "speaker_cue"
        )
    ]

    if malformed_continuation:
        raise AssertionError(
            "GARAK (cont.) is still being "
            "misclassified."
        )

    expected_continuation_dialogue = {
        "I will examine the fragment.",
        (
            "If the cipher yields anything coherent, "
            "I will decide whether Sisko needs to hear it."
        ),
        "Until then, keep this between us.",
    }

    continuation_dialogue = {
        segment.text
        for segment in segments
        if (
            segment.segment_type == "dialogue"
            and segment.speaker is not None
            and segment.speaker.casefold()
            == "garak (cont.)"
        )
    }

    missing_dialogue = (
        expected_continuation_dialogue
        - continuation_dialogue
    )

    if missing_dialogue:
        raise AssertionError(
            "Saved repaired scene still contains "
            "misparsed continuation dialogue:\n"
            + "\n".join(
                sorted(missing_dialogue)
            )
        )

    # -----------------------------------------------------
    # Guard against physical action-line wrapping becoming
    # artificial continuity statements.
    # -----------------------------------------------------

    forbidden_fragment_actions = {
        "a scarred metal cylinder.",
        "Garak",
        (
            "lifts the rod, his fingers brushing "
            "the worn surface, and"
        ),
        (
            "turns the holo-screen toward a "
            "flickering data analysis program."
        ),
        (
            "begins to decode the corrupted files."
        ),
    }

    fragmented_actions = [
        segment.text
        for segment in segments
        if (
            segment.segment_type == "action"
            and segment.text in forbidden_fragment_actions
        )
    ]

    if fragmented_actions:
        raise AssertionError(
            "Saved repaired scene still contains "
            "physically wrapped action fragments:\n"
            + "\n".join(fragmented_actions)
        )

    print(
        "\nSaved production parser recovery: PASS"
    )

    # -----------------------------------------------------
    # IMPORTANT:
    #
    # This is the ONLY model-driven stage in this test.
    #
    # We do NOT:
    # - regenerate the scene
    # - rerun Story Architect
    # - rerun Lore
    # - rerun Character
    # - invoke Scriptwriter repair
    #
    # We only perform the repaired-scene audit that was
    # interrupted by the rate limit.
    # -----------------------------------------------------

    print(
        "\nRunning repaired-scene semantic audit..."
    )

    audit_batch = audit_script_scene(
        script_input,
        repaired_scene,
    )

    scene_audit = build_script_scene_audit(
        script_input,
        repaired_scene,
        audit_batch,
    )

    print(
        "\n--- REPAIRED SCENE AUDIT ---"
    )

    print(
        scene_audit.model_dump_json(
            indent=2
        )
    )

    print(
        "--- END REPAIRED SCENE AUDIT ---"
    )

    if not scene_audit.grounded:
        print(
            "\nRESULT: REPAIRED SCENE STILL NEEDS WORK"
        )

        for issue in scene_audit.issues:
            print(
                "-",
                issue.segment_id,
                "|",
                issue.verdict,
                "|",
                issue.text,
                "|",
                issue.reason,
            )

        raise AssertionError(
            "Saved repaired ScriptScene did not "
            "pass its fresh grounding audit."
        )

    print(
        "\n============================================"
    )
    print(
        "SAVED REPAIRED SCRIPTSCENE RE-AUDIT: PASS"
    )
    print(
        "============================================"
    )


if __name__ == "__main__":
    main()