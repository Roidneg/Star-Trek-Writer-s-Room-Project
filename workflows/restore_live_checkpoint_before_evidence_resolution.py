"""
One-use rollback utility for the live Writer's Room checkpoint.

Purpose
-------
Restore the exact pre-evidence-resolution checkpoint created during the
failed live evidence-resolution flight where refreshed Garak guidance
degraded from grounded -> ungrounded.

Safety behavior
---------------
1. Validate both the current checkpoint and rollback source as
   WritersRoomState.
2. Verify the rollback source contains a grounded Elim Garak artifact.
3. Preserve the current checkpoint as a unique safety backup.
4. Restore the requested rollback checkpoint atomically.
5. Reload and validate the restored state.
6. Confirm Garak remains grounded after restoration.

No provider calls are made.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from workflows.writers_room import WritersRoomState


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "runtime"
    / "checkpoints"
    / "writers_room_live_smoke.json"
)

ROLLBACK_SOURCE = (
    PROJECT_ROOT
    / "runtime"
    / "checkpoints"
    / "writers_room_before_evidence_resolution_9yo3akz5.json"
)


def load_state(path: Path) -> WritersRoomState:
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint file does not exist:\n{path}"
        )

    try:
        return WritersRoomState.model_validate_json(
            path.read_text(
                encoding="utf-8"
            )
        )
    except Exception as error:
        raise ValueError(
            f"Checkpoint failed WritersRoomState validation:\n{path}"
        ) from error


def find_garak(state: WritersRoomState):
    matches = [
        result
        for result in state.character_results
        if result.character_name == "Elim Garak"
    ]

    if len(matches) != 1:
        raise ValueError(
            "Expected exactly one Elim Garak Character artifact "
            f"in rollback checkpoint; found {len(matches)}."
        )

    return matches[0]


def main() -> None:
    print(
        "\n--- Writer's Room Live Checkpoint Rollback ---"
    )

    current_state = load_state(
        CHECKPOINT_PATH
    )

    rollback_state = load_state(
        ROLLBACK_SOURCE
    )

    rollback_garak = find_garak(
        rollback_state
    )

    if not rollback_garak.audit.grounded:
        raise ValueError(
            "Rollback source is not safe: Elim Garak is "
            "not grounded in the pre-resolution checkpoint."
        )

    if (
        rollback_state.story_idea
        != current_state.story_idea
    ):
        raise ValueError(
            "Rollback source belongs to a different story idea."
        )

    current_bytes = CHECKPOINT_PATH.read_bytes()
    rollback_bytes = ROLLBACK_SOURCE.read_bytes()

    with tempfile.NamedTemporaryFile(
        mode="wb",
        prefix="writers_room_before_manual_rollback_",
        suffix=".json",
        dir=CHECKPOINT_PATH.parent,
        delete=False,
    ) as safety_file:
        safety_file.write(
            current_bytes
        )
        safety_backup = Path(
            safety_file.name
        )

    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix="writers_room_rollback_",
            suffix=".tmp",
            dir=CHECKPOINT_PATH.parent,
            delete=False,
        ) as temp_file:
            temp_file.write(
                rollback_bytes
            )
            temporary_path = Path(
                temp_file.name
            )

        if temporary_path.read_bytes() != rollback_bytes:
            raise IOError(
                "Rollback temporary file does not match "
                "the source checkpoint bytes."
            )

        temporary_path.replace(
            CHECKPOINT_PATH
        )

    finally:
        if temporary_path is not None:
            temporary_path.unlink(
                missing_ok=True
            )

    restored_state = load_state(
        CHECKPOINT_PATH
    )

    restored_garak = find_garak(
        restored_state
    )

    if not restored_garak.audit.grounded:
        raise AssertionError(
            "Rollback completed but Garak is not grounded."
        )

    if (
        restored_state.model_dump(
            mode="json"
        )
        != rollback_state.model_dump(
            mode="json"
        )
    ):
        raise AssertionError(
            "Restored checkpoint does not semantically match "
            "the rollback source."
        )

    print(
        "Rollback source:",
        ROLLBACK_SOURCE,
    )

    print(
        "Current-state safety backup:",
        safety_backup,
    )

    print(
        "Restored checkpoint:",
        CHECKPOINT_PATH,
    )

    print(
        "Garak grounded:",
        restored_garak.audit.grounded,
    )

    print(
        "Evaluation present:",
        restored_state.evaluation_result is not None,
    )

    if restored_state.evaluation_result is not None:
        assessment = (
            restored_state
            .evaluation_result
            .production_review
            .review
            .assessment
        )

        print(
            "Evaluation status:",
            assessment.status,
        )

        print(
            "Blocking criteria:",
            assessment.blocking_criterion_ids,
        )

    print(
        "\n============================================"
    )

    print(
        "LIVE CHECKPOINT ROLLBACK: PASS"
    )

    print(
        "============================================"
    )


if __name__ == "__main__":
    main()
