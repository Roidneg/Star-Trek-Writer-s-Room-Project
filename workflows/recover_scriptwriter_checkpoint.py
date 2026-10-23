"""Targeted Scriptwriter checkpoint recovery for the first live Writer's Room flight.

Place this file at:

    workflows/recover_scriptwriter_checkpoint.py

Run from the project root.

Default behavior:
    Preview only. No provider calls. No checkpoint writes.

Apply behavior:
    python -m workflows.recover_scriptwriter_checkpoint --apply

On --apply this utility:

1. verifies that the current checkpoint is still exactly pre-Scriptwriter;
2. verifies that the saved Scriptwriter diagnostic belongs to the fixed
   live-smoke scenario;
3. reconstructs the saved repaired scene;
4. removes the one known unsupported sentence deterministically;
5. verifies the diagnostic ScriptwriterInput against current authoritative
   Writer's Room Lore and Character artifacts;
6. validates the recovered ScriptScene with the current Scriptwriter parser;
7. performs one fresh Scriptwriter semantic audit;
8. refuses recovery unless that audit is grounded;
9. builds a real ScriptwriterProductionResult from that fresh audit;
10. backs up the original checkpoint;
11. writes the recovered checkpoint atomically;
12. leaves Continuity, Evaluation, and the story-state ledger untouched.

This is a targeted recovery tool. It does not regenerate the scene and does
not modify the frozen Scriptwriter generation/repair workflow.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "runtime"
    / "checkpoints"
    / "writers_room_live_smoke.json"
)

DIAGNOSTIC_PATH = (
    PROJECT_ROOT
    / "runtime"
    / "diagnostics"
    / "scriptwriter_failure_e08190355ed44fcab94ad127b156f539.json"
)


EXPECTED_STORY_IDEA = (
    "During Star Trek: Deep Space Nine, after Garak has "
    "already encountered the Founders but before the "
    "Dominion War begins, a fictional Cardassian courier "
    "secretly approaches him on Deep Space Nine. "
    "The courier possesses a damaged data rod allegedly "
    "containing partial Obsidian Order records related to "
    "possible Founder activity. Garak must decide whether "
    "the information is credible enough to bring to "
    "Benjamin Sisko. Do not assume Garak has current "
    "contacts inside Cardassian Central Command or the "
    "Obsidian Order."
)

EXPECTED_STORY_CONTEXT = (
    "A fictional Cardassian courier privately brings Garak a "
    "damaged data rod aboard Deep Space Nine. The courier claims "
    "it contains partial Obsidian Order records concerning "
    "possible Founder activity. The claim and the rod's contents "
    "remain unverified. Garak must decide whether further "
    "investigation is warranted and whether to approach Sisko. "
    "The dramatic stakes are the risk of acting on misinformation "
    "and the risk of overlooking a potentially useful warning."
)

EXPECTED_SCENE_OBJECTIVE = (
    "The fictional Cardassian courier privately "
    "presents the damaged data rod and its allegation "
    "to Garak. Garak must assess whether the claim "
    "deserves further investigation without the scene "
    "assuming facts not established by supplied canon "
    "or Character evidence."
)

EXPECTED_STORY_ERA = (
    "Deep Space Nine, before the official start of the Dominion "
    "War. No prior personal encounter between Garak and the "
    "Founders is established for this scene."
)

EXPECTED_ERA_STYLE_CONSTRAINTS = [
    (
        "Use restrained dramatic pacing and "
        "screenplay conventions appropriate to "
        "Star Trek: Deep Space Nine."
    ),
    (
        "Keep the scene focused on conversation, "
        "uncertainty, and character tension rather "
        "than action spectacle."
    ),
]

EXPECTED_SCENE_REQUIREMENTS = [
    (
        "The Cardassian Courier must present "
        "the damaged data rod to Garak."
    ),
    (
        "The courier must claim that the rod "
        "contains partial Obsidian Order records "
        "related to possible Founder activity."
    ),
    (
        "Garak must decide whether the information "
        "deserves further investigation."
    ),
]

EXPECTED_PROHIBITED_ASSUMPTIONS = [
    (
        "Garak has personally encountered the Founders before "
        "this scene, or a particular episode establishes his "
        "first direct encounter with them."
    ),
    (
        "Garak has current contacts inside "
        "Cardassian Central Command."
    ),
    (
        "Garak has current access to the "
        "Obsidian Order."
    ),
    (
        "The courier's allegation is proven true."
    ),
    (
        "The data rod's contents have already "
        "been independently verified."
    ),
]

UNSUPPORTED_SENTENCE = (
    "The transmission was garbled, the rod is cracked."
)

# The saved screenplay physically wraps this sentence:
#
#     The transmission was
#     garbled, the rod is cracked. But if it is true...
#
# Match semantic whitespace and preserve the following sentence.
UNSUPPORTED_PATTERN = re.compile(
    r"The transmission was\s+"
    r"garbled, the rod is cracked\.\s+"
    r"(?=But if it is true)"
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_json_bytes(path: Path) -> tuple[bytes, dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"Required file was not found:\n{path}"
        )

    raw = path.read_bytes()

    try:
        payload = json.loads(
            raw.decode("utf-8")
        )
    except Exception as error:
        raise ValueError(
            f"Could not parse JSON file:\n{path}"
        ) from error

    if not isinstance(payload, dict):
        raise ValueError(
            f"Expected top-level JSON object:\n{path}"
        )

    return raw, payload


def assert_exact(
    actual,
    expected,
    label: str,
) -> None:
    if actual != expected:
        raise ValueError(
            f"{label} does not match the targeted live-smoke scenario.\n"
            f"Expected:\n{expected!r}\n\n"
            f"Received:\n{actual!r}"
        )


def validate_checkpoint_shape(
    checkpoint_data: dict,
) -> None:
    """Fail closed unless recovery starts after Character and before Scriptwriter."""

    assert_exact(
        checkpoint_data.get("story_idea"),
        EXPECTED_STORY_IDEA,
        "checkpoint story_idea",
    )

    if checkpoint_data.get("story_plan") is None:
        raise ValueError(
            "Recovery requires a completed Story Architect stage."
        )

    lore_results = checkpoint_data.get("lore_results")

    if not isinstance(lore_results, list) or not lore_results:
        raise ValueError(
            "Recovery requires completed Lore artifacts."
        )

    character_results = checkpoint_data.get(
        "character_results"
    )

    if (
        not isinstance(character_results, list)
        or len(character_results) != 1
    ):
        raise ValueError(
            "Recovery expects exactly one Character artifact."
        )

    character_result = character_results[0]

    if (
        character_result.get("character_id")
        != "character_000"
        or character_result.get("character_name")
        != "Elim Garak"
    ):
        raise ValueError(
            "Recovery found an unexpected Character identity."
        )

    character_audit = character_result.get("audit", {})

    if (
        character_audit.get("grounded") is not True
        or character_audit.get("issues")
    ):
        raise ValueError(
            "Recovery requires the checkpointed Garak "
            "Character artifact to be grounded."
        )

    if checkpoint_data.get("script_result") is not None:
        raise ValueError(
            "Checkpoint already contains a Scriptwriter result. "
            "Targeted recovery is not applicable."
        )

    if checkpoint_data.get("continuity_result") is not None:
        raise ValueError(
            "Checkpoint already contains a Continuity result. "
            "Targeted recovery is not applicable."
        )

    if checkpoint_data.get("evaluation_result") is not None:
        raise ValueError(
            "Checkpoint already contains an Evaluation result. "
            "Targeted recovery is not applicable."
        )

    ledger = checkpoint_data.get(
        "story_state_ledger",
        {},
    )

    if ledger.get("entries"):
        raise ValueError(
            "Checkpoint already contains persistent story state. "
            "Targeted pre-Continuity recovery is not applicable."
        )


def validate_diagnostic_shape(
    diagnostic: dict,
) -> None:
    """Verify this is the specific interrupted repaired-audit artifact."""

    if diagnostic.get("schema_version") != 1:
        raise ValueError(
            "Unexpected Scriptwriter diagnostic schema version."
        )

    if diagnostic.get("stage") != "repaired_audit":
        raise ValueError(
            "Recovery requires a Scriptwriter failure captured "
            "at stage='repaired_audit'."
        )

    if diagnostic.get("status") != "failed":
        raise ValueError(
            "Recovery requires a failed Scriptwriter diagnostic."
        )

    error_data = diagnostic.get("error", {})

    if error_data.get("type") != "RateLimitError":
        raise ValueError(
            "Target diagnostic did not fail because of RateLimitError."
        )

    if "script_input" not in diagnostic:
        raise ValueError(
            "Diagnostic is missing script_input."
        )

    if "repaired_scene" not in diagnostic:
        raise ValueError(
            "Diagnostic is missing repaired_scene."
        )

    script_input = diagnostic["script_input"]

    assert_exact(
        script_input.get("story_context"),
        EXPECTED_STORY_CONTEXT,
        "diagnostic story_context",
    )

    assert_exact(
        script_input.get("scene_objective"),
        EXPECTED_SCENE_OBJECTIVE,
        "diagnostic scene_objective",
    )

    assert_exact(
        script_input.get("story_era"),
        EXPECTED_STORY_ERA,
        "diagnostic story_era",
    )

    assert_exact(
        script_input.get(
            "era_style_constraints"
        ),
        EXPECTED_ERA_STYLE_CONSTRAINTS,
        "diagnostic era_style_constraints",
    )

    assert_exact(
        script_input.get("scene_requirements"),
        EXPECTED_SCENE_REQUIREMENTS,
        "diagnostic scene_requirements",
    )

    assert_exact(
        script_input.get(
            "prohibited_assumptions"
        ),
        EXPECTED_PROHIBITED_ASSUMPTIONS,
        "diagnostic prohibited_assumptions",
    )

    characters = script_input.get("characters")

    if (
        not isinstance(characters, list)
        or len(characters) != 2
    ):
        raise ValueError(
            "Diagnostic ScriptwriterInput must contain "
            "exactly two characters."
        )

    assert_exact(
        characters[0].get("character_name"),
        "Elim Garak",
        "diagnostic character[0]",
    )

    assert_exact(
        characters[1].get("character_name"),
        "Cardassian Courier",
        "diagnostic character[1]",
    )

    if (
        characters[1].get(
            "grounded_character_guidance"
        )
        or characters[1].get(
            "grounded_voice_guidance"
        )
    ):
        raise ValueError(
            "Fictional Cardassian Courier unexpectedly "
            "contains grounded Character guidance."
        )


def deterministically_clean_scene_text(
    script_text: str,
) -> str:
    """
    Remove exactly the one unsupported sentence found by the
    post-parser semantic audit.

    The following courier sentence must remain intact.
    """

    cleaned_text, replacement_count = (
        UNSUPPORTED_PATTERN.subn(
            "",
            script_text,
            count=1,
        )
    )

    if replacement_count != 1:
        raise ValueError(
            "Expected exactly one wrapped unsupported sentence "
            f"in the saved repaired scene, but replaced "
            f"{replacement_count}."
        )

    normalized_text = " ".join(
        cleaned_text.split()
    )

    if UNSUPPORTED_SENTENCE in normalized_text:
        raise AssertionError(
            "Unsupported sentence remained after "
            "deterministic cleanup."
        )

    if (
        "But if it is true—if the Founders are being "
        "discussed within the Order—then it could be "
        "valuable to the station."
        not in normalized_text
    ):
        raise AssertionError(
            "Deterministic cleanup damaged the courier's "
            "following dialogue."
        )

    return cleaned_text


def prepare_preview(
    checkpoint_data: dict,
    diagnostic: dict,
) -> dict:
    """
    Perform all cheap structural checks and build a preview candidate.

    No project modules and no provider-backed code are invoked here.
    """

    validate_checkpoint_shape(
        checkpoint_data
    )

    validate_diagnostic_shape(
        diagnostic
    )

    repaired_scene = diagnostic[
        "repaired_scene"
    ]

    script_text = repaired_scene.get(
        "script_text"
    )

    if not isinstance(script_text, str):
        raise ValueError(
            "Diagnostic repaired_scene.script_text "
            "is missing or invalid."
        )

    cleaned_text = (
        deterministically_clean_scene_text(
            script_text
        )
    )

    return {
        "diagnostic_stage": diagnostic.get(
            "stage"
        ),
        "character_id": (
            checkpoint_data[
                "character_results"
            ][0]["character_id"]
        ),
        "verified_lore_count": sum(
            1
            for result in checkpoint_data[
                "lore_results"
            ]
            if (
                result.get("response", {})
                .get("evidence_supported")
                is True
            )
        ),
        "original_script_characters": [
            character.get("character_name")
            for character
            in diagnostic[
                "script_input"
            ]["characters"]
        ],
        "unsupported_sentence_removed": (
            UNSUPPORTED_SENTENCE
        ),
        "original_script_text_length": len(
            script_text
        ),
        "candidate_script_text_length": len(
            cleaned_text
        ),
    }


def verify_authoritative_routing(
    state,
    script_input,
) -> None:
    """
    Ensure the saved diagnostic input still equals what the current
    checkpoint authoritatively establishes for Lore and Character routing.
    """

    from workflows.writers_room import (
        WritersRoomScriptCharacterRequest,
        build_script_character_context,
    )

    expected_verified_canon_facts = [
        result.response.summary
        for result in state.lore_results
        if result.response.evidence_supported
    ]

    if (
        script_input.verified_canon_facts
        != expected_verified_canon_facts
    ):
        raise ValueError(
            "Diagnostic verified_canon_facts no longer match "
            "the current checkpoint's verified Lore artifacts."
        )

    garak_result = state.character_results[0]

    expected_characters = [
        build_script_character_context(
            state=state,
            request=(
                WritersRoomScriptCharacterRequest(
                    character_name="Elim Garak",
                    character_result_id=(
                        garak_result.character_id
                    ),
                )
            ),
        ),
        build_script_character_context(
            state=state,
            request=(
                WritersRoomScriptCharacterRequest(
                    character_name=(
                        "Cardassian Courier"
                    ),
                    character_result_id=None,
                )
            ),
        ),
    ]

    actual_character_payload = [
        character.model_dump(
            mode="json"
        )
        for character
        in script_input.characters
    ]

    expected_character_payload = [
        character.model_dump(
            mode="json"
        )
        for character
        in expected_characters
    ]

    if (
        actual_character_payload
        != expected_character_payload
    ):
        raise ValueError(
            "Diagnostic Scriptwriter character guidance no "
            "longer matches the current authoritative "
            "Writer's Room Character routing."
        )


def verify_parser_recovery(
    scene,
) -> None:
    """
    Verify that the current parser retains the production fixes
    before any semantic audit is allowed to promote the scene.
    """

    from agents.scriptwriter_agent import (
        enumerate_script_scene_segments,
    )

    segments = enumerate_script_scene_segments(
        scene
    )

    if not segments:
        raise ValueError(
            "Recovered scene produced no auditable segments."
        )

    continuation_cues = [
        segment
        for segment in segments
        if (
            segment.text.casefold()
            == "garak (cont.)"
        )
    ]

    if (
        len(continuation_cues) != 1
        or continuation_cues[0].segment_type
        != "speaker_cue"
    ):
        raise AssertionError(
            "Current parser does not correctly preserve "
            "GARAK (cont.) as a speaker cue."
        )

    expected_continuation_dialogue = {
        "I will examine the fragment.",
        (
            "If the cipher yields anything coherent, "
            "I will decide whether Sisko needs to hear it."
        ),
        "Until then, keep this between us.",
    }

    parsed_continuation_dialogue = {
        segment.text
        for segment in segments
        if (
            segment.segment_type
            == "dialogue"
            and segment.speaker is not None
            and segment.speaker.casefold()
            == "garak (cont.)"
        )
    }

    if (
        not expected_continuation_dialogue
        .issubset(
            parsed_continuation_dialogue
        )
    ):
        raise AssertionError(
            "Current parser does not correctly preserve "
            "the continuation dialogue block."
        )

    forbidden_action_fragments = {
        "Garak",
        "a scarred metal cylinder.",
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

    bad_fragments = [
        segment.text
        for segment in segments
        if (
            segment.segment_type == "action"
            and segment.text
            in forbidden_action_fragments
        )
    ]

    if bad_fragments:
        raise AssertionError(
            "Current parser still exposes physical "
            "action-line fragments:\n- "
            + "\n- ".join(
                bad_fragments
            )
        )

    normalized_scene_text = " ".join(
        scene.script_text.split()
    )

    if (
        UNSUPPORTED_SENTENCE
        in normalized_scene_text
    ):
        raise AssertionError(
            "Known unsupported sentence is still "
            "present in the candidate scene."
        )

    print(
        "Current parser recovery checks: PASS"
    )

    print(
        "Auditable segment count:",
        len(segments),
    )


def atomic_write_checkpoint(
    checkpoint_path: Path,
    state,
    original_bytes: bytes,
) -> tuple[Path, Path]:
    """
    Back up the exact original checkpoint and atomically replace it.

    Returns:
        (backup_path, checkpoint_path)
    """

    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if checkpoint_path.read_bytes() != original_bytes:
        raise ValueError(
            "Checkpoint changed during Scriptwriter audit. "
            "Recovery was not saved."
        )

    with tempfile.NamedTemporaryFile(
        mode="wb",
        prefix=(
            "writers_room_before_"
            "scriptwriter_recovery_"
        ),
        suffix=".json",
        dir=checkpoint_path.parent,
        delete=False,
    ) as backup_file:
        backup_file.write(
            original_bytes
        )
        backup_path = Path(
            backup_file.name
        )

    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=(
                "scriptwriter_recovery_"
            ),
            suffix=".tmp",
            dir=checkpoint_path.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(
                temporary_file.name
            )

            temporary_file.write(
                state.model_dump_json(
                    indent=2
                )
            )

            temporary_file.write(
                "\n"
            )

        temporary_path.replace(
            checkpoint_path
        )

    finally:
        if temporary_path is not None:
            temporary_path.unlink(
                missing_ok=True
            )

    return (
        backup_path,
        checkpoint_path,
    )


def run_apply(
    checkpoint_bytes: bytes,
    checkpoint_data: dict,
    diagnostic: dict,
) -> None:
    """
    Run the one model-driven recovery audit and save only on success.
    """

    from dotenv import load_dotenv

    load_dotenv()

    from agents.scriptwriter_agent import (
        ScriptScene,
        ScriptwriterInput,
        audit_script_scene,
        build_script_scene_audit,
        validate_script_scene_output,
        validate_scriptwriter_input,
    )

    from workflows.scriptwriter_production import (
        ScriptwriterProductionResult,
    )

    from workflows.writers_room import (
        WritersRoomState,
    )

    # -----------------------------------------------------
    # Reconstruct authoritative current checkpoint state.
    # -----------------------------------------------------

    state = (
        WritersRoomState
        .model_validate_json(
            checkpoint_bytes.decode(
                "utf-8"
            )
        )
    )

    if state.script_result is not None:
        raise ValueError(
            "Scriptwriter stage has already completed."
        )

    if (
        state.continuity_result is not None
        or state.evaluation_result is not None
        or state.story_state_ledger.entries
    ):
        raise ValueError(
            "Downstream Writer's Room state exists. "
            "Targeted Scriptwriter recovery is not applicable."
        )

    # -----------------------------------------------------
    # Reconstruct the exact saved input and repaired scene.
    # -----------------------------------------------------

    script_input = (
        ScriptwriterInput.model_validate(
            diagnostic[
                "script_input"
            ]
        )
    )

    repaired_scene = (
        ScriptScene.model_validate(
            diagnostic[
                "repaired_scene"
            ]
        )
    )

    validate_scriptwriter_input(
        script_input
    )

    verify_authoritative_routing(
        state=state,
        script_input=script_input,
    )

    cleaned_script_text = (
        deterministically_clean_scene_text(
            repaired_scene.script_text
        )
    )

    recovered_scene = (
        repaired_scene.model_copy(
            update={
                "script_text": (
                    cleaned_script_text
                )
            }
        )
    )

    validate_script_scene_output(
        script_input,
        recovered_scene,
    )

    verify_parser_recovery(
        recovered_scene
    )

    print(
        "ScriptwriterInput authoritative routing: PASS"
    )

    print(
        "Recovered ScriptScene deterministic validation: PASS"
    )

    # -----------------------------------------------------
    # One fresh semantic audit.
    #
    # This is the only provider-backed work performed by
    # this recovery utility.
    # -----------------------------------------------------

    print(
        "\nRunning one fresh recovered-scene "
        "Scriptwriter semantic audit..."
    )

    audit_batch = audit_script_scene(
        script_input,
        recovered_scene,
    )

    scene_audit = (
        build_script_scene_audit(
            script_input,
            recovered_scene,
            audit_batch,
        )
    )

    print(
        "\n--- RECOVERY SCRIPTWRITER AUDIT ---"
    )

    print(
        scene_audit.model_dump_json(
            indent=2
        )
    )

    print(
        "--- END RECOVERY SCRIPTWRITER AUDIT ---"
    )

    if not scene_audit.grounded:
        raise ValueError(
            "Recovered ScriptScene did not pass the "
            "fresh Scriptwriter grounding audit. "
            "Original checkpoint preserved."
        )

    if scene_audit.issues:
        raise ValueError(
            "Grounded Scriptwriter recovery audit "
            "unexpectedly contains issues."
        )

    # -----------------------------------------------------
    # Build the exact downstream package expected by
    # WritersRoomState.
    #
    # Unlike the normal adapter's success receipt, this
    # recovery stores the fresh grounded audit we just ran.
    # -----------------------------------------------------

    production_result = (
        ScriptwriterProductionResult(
            script_input=script_input,
            scene=recovered_scene,
            audit=scene_audit,
        )
    )

    recovered_state = state.model_copy(
        update={
            "script_result": (
                production_result
            )
        }
    )

    # Full schema validation before disk mutation.
    recovered_payload = (
        recovered_state.model_dump(
            mode="json"
        )
    )

    validated_recovered_state = (
        WritersRoomState.model_validate(
            recovered_payload
        )
    )

    if (
        validated_recovered_state
        .script_result
        is None
    ):
        raise AssertionError(
            "Recovered state lost Scriptwriter result."
        )

    if not (
        validated_recovered_state
        .script_result
        .audit
        .grounded
    ):
        raise AssertionError(
            "Recovered state contains an ungrounded "
            "Scriptwriter result."
        )

    if (
        validated_recovered_state
        .continuity_result
        is not None
        or validated_recovered_state
        .evaluation_result
        is not None
        or validated_recovered_state
        .story_state_ledger
        .entries
    ):
        raise AssertionError(
            "Scriptwriter recovery changed downstream "
            "Writer's Room state."
        )

    # -----------------------------------------------------
    # Final concurrency check + backup + atomic write.
    # -----------------------------------------------------

    backup_path, _ = (
        atomic_write_checkpoint(
            checkpoint_path=(
                CHECKPOINT_PATH
            ),
            state=(
                validated_recovered_state
            ),
            original_bytes=(
                checkpoint_bytes
            ),
        )
    )

    # -----------------------------------------------------
    # Post-write validation.
    # -----------------------------------------------------

    saved_state = (
        WritersRoomState
        .model_validate_json(
            CHECKPOINT_PATH.read_text(
                encoding="utf-8"
            )
        )
    )

    if (
        saved_state.script_result is None
        or not saved_state
        .script_result
        .audit
        .grounded
    ):
        raise AssertionError(
            "Saved checkpoint failed post-write "
            "Scriptwriter validation."
        )

    if (
        saved_state.continuity_result is not None
        or saved_state.evaluation_result is not None
        or saved_state.story_state_ledger.entries
    ):
        raise AssertionError(
            "Saved recovery checkpoint unexpectedly "
            "contains downstream artifacts."
        )

    print(
        "\n============================================"
    )
    print(
        "SCRIPTWRITER CHECKPOINT RECOVERY: PASS"
    )
    print(
        "============================================"
    )

    print(
        "Backup:",
        backup_path,
    )

    print(
        "Recovered checkpoint:",
        CHECKPOINT_PATH,
    )

    print(
        "Scriptwriter grounding:",
        saved_state
        .script_result
        .audit
        .grounded,
    )

    print(
        "Continuity result:",
        (
            "PRESENT"
            if saved_state.continuity_result
            else "MISSING"
        ),
    )

    print(
        "Evaluation result:",
        (
            "PRESENT"
            if saved_state.evaluation_result
            else "MISSING"
        ),
    )

    print(
        "\nNext:"
    )

    print(
        "python -m "
        "workflows.writers_room_live_smoke_test"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--apply",
        action="store_true",
        help=(
            "Run one fresh Scriptwriter semantic audit "
            "and save the recovered checkpoint only if "
            "the audit passes."
        ),
    )

    args = parser.parse_args()

    checkpoint_bytes, checkpoint_data = (
        load_json_bytes(
            CHECKPOINT_PATH
        )
    )

    diagnostic_bytes, diagnostic = (
        load_json_bytes(
            DIAGNOSTIC_PATH
        )
    )

    preview = prepare_preview(
        checkpoint_data=checkpoint_data,
        diagnostic=diagnostic,
    )

    print(
        "\n--- Scriptwriter Checkpoint Recovery ---"
    )

    print(
        "Checkpoint:",
        CHECKPOINT_PATH,
    )

    print(
        "Checkpoint SHA256:",
        sha256_bytes(
            checkpoint_bytes
        ),
    )

    print(
        "Diagnostic:",
        DIAGNOSTIC_PATH,
    )

    print(
        "Diagnostic SHA256:",
        sha256_bytes(
            diagnostic_bytes
        ),
    )

    print(
        "Diagnostic stage:",
        preview[
            "diagnostic_stage"
        ],
    )

    print(
        "Character:",
        preview[
            "character_id"
        ],
    )

    print(
        "Verified Lore artifacts:",
        preview[
            "verified_lore_count"
        ],
    )

    print(
        "Script characters:",
        preview[
            "original_script_characters"
        ],
    )

    print(
        "Known unsupported sentence:",
        preview[
            "unsupported_sentence_removed"
        ],
    )

    print(
        "Original script-text length:",
        preview[
            "original_script_text_length"
        ],
    )

    print(
        "Candidate script-text length:",
        preview[
            "candidate_script_text_length"
        ],
    )

    if not args.apply:
        print(
            "\nPREVIEW ONLY: PASS"
        )

        print(
            "No provider calls were made."
        )

        print(
            "No checkpoint files were changed."
        )

        print(
            "\nTo run the audited recovery:"
        )

        print(
            "python -m "
            "workflows.recover_scriptwriter_checkpoint "
            "--apply"
        )

        return

    run_apply(
        checkpoint_bytes=(
            checkpoint_bytes
        ),
        checkpoint_data=(
            checkpoint_data
        ),
        diagnostic=diagnostic,
    )


if __name__ == "__main__":
    main()
