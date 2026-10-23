"""
Pure-Python regression coverage for workflows.evidence_refresh_checkpoint.

Run from the project root:

    python -m workflows.evidence_refresh_checkpoint_regression_test

No provider calls are made.
The real Writer's Room checkpoint is never touched.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from pydantic import BaseModel, Field

from workflows.evaluation_resolution import (
    EvaluationResolutionTarget,
)
from workflows.evidence_refresh import (
    refresh_lore_evidence,
)
from workflows.evidence_refresh_checkpoint import (
    persist_lore_refresh_commit,
    sha256_bytes,
)
from workflows.evidence_refresh_commit import (
    commit_lore_refresh_to_state,
)


# ---------------------------------------------------------
# Lightweight Models
# ---------------------------------------------------------


class FakeLoreResponse(BaseModel):
    premise_supported: bool
    evidence_supported: bool
    summary: str


class FakeLoreResult(BaseModel):
    research_id: str = Field(
        min_length=1,
    )

    research_question: str = Field(
        min_length=1,
    )

    response: FakeLoreResponse


class FakeCharacterResult(BaseModel):
    character_id: str
    character_name: str
    marker: str


class FakeScriptResult(BaseModel):
    marker: str


class FakeContinuityResult(BaseModel):
    marker: str


class FakeEvaluationResult(BaseModel):
    marker: str


class FakeLedger(BaseModel):
    entries: list[str] = Field(
        default_factory=list
    )


class FakeWriterRoomState(BaseModel):
    story_plan: str

    lore_results: list[
        FakeLoreResult
    ]

    character_results: list[
        FakeCharacterResult
    ]

    script_result: (
        FakeScriptResult
        | None
    )

    story_state_ledger: FakeLedger

    continuity_result: (
        FakeContinuityResult
        | None
    )

    evaluation_result: (
        FakeEvaluationResult
        | None
    )


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------


def assert_true(
    condition: bool,
    message: str,
) -> None:
    if not condition:
        raise AssertionError(
            message
        )


def assert_raises(
    expected_exception: type[Exception],
    callable_object,
    expected_message_fragment: str,
) -> None:
    try:
        callable_object()

    except expected_exception as error:
        if (
            expected_message_fragment
            not in str(error)
        ):
            raise AssertionError(
                "Exception message did not contain "
                f"{expected_message_fragment!r}.\n\n"
                f"Received:\n{error}"
            ) from error

        return

    except Exception as error:
        raise AssertionError(
            "Expected "
            f"{expected_exception.__name__}, "
            f"received {type(error).__name__}: "
            f"{error}"
        ) from error

    raise AssertionError(
        "Expected "
        f"{expected_exception.__name__} "
        "but no exception was raised."
    )


def lore_target() -> EvaluationResolutionTarget:
    return EvaluationResolutionTarget(
        criterion_id=(
            "lore_agent_evidence_lore_000"
        ),
        category="canon_grounding",
        owner="lore",
        action="refresh_lore_evidence",
        artifact_id="lore_000",
        subject_name=None,
        reason=(
            "Regression Lore blocker."
        ),
    )


def build_state() -> FakeWriterRoomState:
    return FakeWriterRoomState(
        story_plan="StoryPlan marker",
        lore_results=[
            FakeLoreResult(
                research_id="lore_000",
                research_question=(
                    "When does Garak directly encounter "
                    "the Founders?"
                ),
                response=FakeLoreResponse(
                    premise_supported=True,
                    evidence_supported=False,
                    summary=(
                        "The available evidence does not "
                        "establish a direct encounter."
                    ),
                ),
            ),
            FakeLoreResult(
                research_id="lore_001",
                research_question=(
                    "When does the Dominion War begin?"
                ),
                response=FakeLoreResponse(
                    premise_supported=True,
                    evidence_supported=True,
                    summary=(
                        "Call to Arms marks the official "
                        "start of the Dominion War."
                    ),
                ),
            ),
        ],
        character_results=[
            FakeCharacterResult(
                character_id="character_000",
                character_name="Elim Garak",
                marker="grounded-character",
            )
        ],
        script_result=(
            FakeScriptResult(
                marker="script"
            )
        ),
        story_state_ledger=FakeLedger(
            entries=[
                "door open",
                "rod on table",
            ]
        ),
        continuity_result=(
            FakeContinuityResult(
                marker="continuity"
            )
        ),
        evaluation_result=(
            FakeEvaluationResult(
                marker="evaluation"
            )
        ),
    )


def serialize_state(
    state: FakeWriterRoomState,
) -> bytes:
    return (
        state.model_dump_json(
            indent=2
        )
        + "\n"
    ).encode(
        "utf-8"
    )


def build_unchanged_commit(
    state: FakeWriterRoomState,
):
    def fake_runner(
        *,
        query: str,
        debug: bool,
    ) -> FakeLoreResponse:
        return FakeLoreResponse(
            premise_supported=True,
            evidence_supported=False,
            summary=(
                "Retrieved material still does not "
                "establish a direct encounter."
            ),
        )

    refresh_result = (
        refresh_lore_evidence(
            target=lore_target(),
            current_lore_result=(
                state.lore_results[0]
            ),
            lore_runner=fake_runner,
        )
    )

    return commit_lore_refresh_to_state(
        state=state,
        refresh_result=(
            refresh_result
        ),
    )


def build_material_commit(
    state: FakeWriterRoomState,
):
    def fake_runner(
        *,
        query: str,
        debug: bool,
    ) -> FakeLoreResponse:
        return FakeLoreResponse(
            premise_supported=True,
            evidence_supported=True,
            summary=(
                "New evidence establishes the "
                "requested canon fact."
            ),
        )

    refresh_result = (
        refresh_lore_evidence(
            target=lore_target(),
            current_lore_result=(
                state.lore_results[0]
            ),
            lore_runner=fake_runner,
        )
    )

    return commit_lore_refresh_to_state(
        state=state,
        refresh_result=(
            refresh_result
        ),
    )


def write_checkpoint(
    checkpoint_path: Path,
    state: FakeWriterRoomState,
) -> bytes:
    checkpoint_bytes = serialize_state(
        state
    )

    checkpoint_path.write_bytes(
        checkpoint_bytes
    )

    return checkpoint_bytes


# ---------------------------------------------------------
# Regression Cases
# ---------------------------------------------------------


def test_unchanged_commit_persists_atomically() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(
            temp_dir
        )

        checkpoint_path = (
            temp_path
            / "writers_room.json"
        )

        state = build_state()

        original_bytes = write_checkpoint(
            checkpoint_path,
            state,
        )

        commit_result = (
            build_unchanged_commit(
                state
            )
        )

        persistence = (
            persist_lore_refresh_commit(
                checkpoint_path=(
                    checkpoint_path
                ),
                original_checkpoint_bytes=(
                    original_bytes
                ),
                commit_result=(
                    commit_result
                ),
            )
        )

        backup_path = Path(
            persistence.backup_path
        )

        assert_true(
            persistence.persistence_status
            == "saved",
            "Unchanged commit was not persisted.",
        )

        assert_true(
            persistence.commit_status
            == (
                "committed_without_invalidation"
            ),
            "Unchanged persistence lost commit status.",
        )

        assert_true(
            persistence.invalidated_artifacts
            == [],
            "Unchanged persistence incorrectly "
            "recorded invalidations.",
        )

        assert_true(
            backup_path.exists(),
            "Checkpoint backup was not created.",
        )

        assert_true(
            backup_path.read_bytes()
            == original_bytes,
            "Checkpoint backup does not preserve "
            "the exact original bytes.",
        )

        saved_state = (
            FakeWriterRoomState
            .model_validate_json(
                checkpoint_path.read_text(
                    encoding="utf-8"
                )
            )
        )

        assert_true(
            saved_state.lore_results[0]
            .response
            .summary
            == (
                "Retrieved material still does not "
                "establish a direct encounter."
            ),
            "Updated Lore provenance was not saved.",
        )

        assert_true(
            saved_state.script_result
            == state.script_result,
            "Unchanged Lore persistence changed "
            "script_result.",
        )

        assert_true(
            saved_state.continuity_result
            == state.continuity_result,
            "Unchanged Lore persistence changed "
            "continuity_result.",
        )

        assert_true(
            saved_state.story_state_ledger
            == state.story_state_ledger,
            "Unchanged Lore persistence changed "
            "the story-state ledger.",
        )

        assert_true(
            saved_state.evaluation_result
            == state.evaluation_result,
            "Unchanged Lore persistence changed "
            "evaluation_result.",
        )

        assert_true(
            persistence.original_sha256
            == sha256_bytes(
                original_bytes
            ),
            "Original checkpoint hash is wrong.",
        )

        assert_true(
            persistence.saved_sha256
            == sha256_bytes(
                checkpoint_path.read_bytes()
            ),
            "Saved checkpoint hash is wrong.",
        )

    print(
        "Unchanged checkpoint persistence: PASS"
    )


def test_material_commit_persists_invalidations() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(
            temp_dir
        )

        checkpoint_path = (
            temp_path
            / "writers_room.json"
        )

        state = build_state()

        original_bytes = write_checkpoint(
            checkpoint_path,
            state,
        )

        commit_result = (
            build_material_commit(
                state
            )
        )

        persistence = (
            persist_lore_refresh_commit(
                checkpoint_path=(
                    checkpoint_path
                ),
                original_checkpoint_bytes=(
                    original_bytes
                ),
                commit_result=(
                    commit_result
                ),
            )
        )

        saved_state = (
            persistence.saved_state
        )

        assert_true(
            persistence.commit_status
            == "committed_with_invalidation",
            "Material persistence lost commit status.",
        )

        assert_true(
            persistence.invalidated_artifacts
            == [
                "script_result",
                "continuity_result",
                "story_state_ledger",
                "evaluation_result",
            ],
            "Material persistence recorded the "
            "wrong invalidation set.",
        )

        assert_true(
            saved_state.lore_results[0]
            .response
            .evidence_supported
            is True,
            "Material Lore candidate was not saved.",
        )

        assert_true(
            saved_state.script_result is None,
            "Material persistence did not clear "
            "script_result.",
        )

        assert_true(
            saved_state.continuity_result is None,
            "Material persistence did not clear "
            "continuity_result.",
        )

        assert_true(
            saved_state.evaluation_result is None,
            "Material persistence did not clear "
            "evaluation_result.",
        )

        assert_true(
            saved_state.story_state_ledger.entries
            == [],
            "Material persistence did not reset "
            "story-state ledger.",
        )

        assert_true(
            saved_state.character_results
            == state.character_results,
            "Material Lore persistence changed "
            "Character artifacts.",
        )

        assert_true(
            saved_state.story_plan
            == state.story_plan,
            "Material Lore persistence changed "
            "StoryPlan.",
        )

        assert_true(
            saved_state.lore_results[1]
            == state.lore_results[1],
            "Material Lore persistence changed "
            "sibling Lore artifact.",
        )

        backup_path = Path(
            persistence.backup_path
        )

        assert_true(
            backup_path.read_bytes()
            == original_bytes,
            "Material persistence backup does not "
            "match original checkpoint bytes.",
        )

    print(
        "Material checkpoint persistence: PASS"
    )


def test_stale_checkpoint_fails_closed() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(
            temp_dir
        )

        checkpoint_path = (
            temp_path
            / "writers_room.json"
        )

        state = build_state()

        original_bytes = write_checkpoint(
            checkpoint_path,
            state,
        )

        commit_result = (
            build_material_commit(
                state
            )
        )

        # Simulate a newer checkpoint being written after
        # refresh execution began.
        newer_state = state.model_copy(
            update={
                "story_plan": (
                    "Newer StoryPlan state"
                )
            }
        )

        checkpoint_path.write_bytes(
            serialize_state(
                newer_state
            )
        )

        assert_raises(
            ValueError,
            lambda: persist_lore_refresh_commit(
                checkpoint_path=(
                    checkpoint_path
                ),
                original_checkpoint_bytes=(
                    original_bytes
                ),
                commit_result=(
                    commit_result
                ),
            ),
            "changed after the evidence refresh began",
        )

        assert_true(
            checkpoint_path.read_bytes()
            == serialize_state(
                newer_state
            ),
            "Stale persistence attempt overwrote "
            "newer checkpoint state.",
        )

    print(
        "Stale checkpoint fail-closed: PASS"
    )


def test_missing_checkpoint_fails_closed() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(
            temp_dir
        )

        checkpoint_path = (
            temp_path
            / "missing.json"
        )

        state = build_state()
        original_bytes = serialize_state(
            state
        )

        commit_result = (
            build_material_commit(
                state
            )
        )

        assert_raises(
            FileNotFoundError,
            lambda: persist_lore_refresh_commit(
                checkpoint_path=(
                    checkpoint_path
                ),
                original_checkpoint_bytes=(
                    original_bytes
                ),
                commit_result=(
                    commit_result
                ),
            ),
            "checkpoint disappeared",
        )

    print(
        "Missing checkpoint fail-closed: PASS"
    )


def test_non_path_checkpoint_fails_closed() -> None:
    state = build_state()

    commit_result = (
        build_material_commit(
            state
        )
    )

    assert_raises(
        TypeError,
        lambda: persist_lore_refresh_commit(
            checkpoint_path="writers_room.json",
            original_checkpoint_bytes=(
                serialize_state(
                    state
                )
            ),
            commit_result=(
                commit_result
            ),
        ),
        "pathlib.Path",
    )

    print(
        "Non-Path checkpoint fail-closed: PASS"
    )


def test_empty_original_bytes_fail_closed() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(
            temp_dir
        )

        checkpoint_path = (
            temp_path
            / "writers_room.json"
        )

        state = build_state()

        write_checkpoint(
            checkpoint_path,
            state,
        )

        commit_result = (
            build_material_commit(
                state
            )
        )

        assert_raises(
            ValueError,
            lambda: persist_lore_refresh_commit(
                checkpoint_path=(
                    checkpoint_path
                ),
                original_checkpoint_bytes=b"",
                commit_result=(
                    commit_result
                ),
            ),
            "cannot be empty",
        )

    print(
        "Empty checkpoint bytes fail-closed: PASS"
    )


def test_wrong_commit_result_type_fails_closed() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(
            temp_dir
        )

        checkpoint_path = (
            temp_path
            / "writers_room.json"
        )

        state = build_state()

        original_bytes = write_checkpoint(
            checkpoint_path,
            state,
        )

        assert_raises(
            TypeError,
            lambda: persist_lore_refresh_commit(
                checkpoint_path=(
                    checkpoint_path
                ),
                original_checkpoint_bytes=(
                    original_bytes
                ),
                commit_result=object(),
            ),
            "LoreRefreshCommitResult",
        )

    print(
        "Wrong commit-result type fail-closed: PASS"
    )


def test_tampered_commit_contract_fails_closed() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(
            temp_dir
        )

        checkpoint_path = (
            temp_path
            / "writers_room.json"
        )

        state = build_state()

        original_bytes = write_checkpoint(
            checkpoint_path,
            state,
        )

        commit_result = (
            build_material_commit(
                state
            )
        )

        tampered = (
            commit_result.model_copy(
                update={
                    "invalidated_artifacts": [
                        "script_result",
                    ]
                }
            )
        )

        assert_raises(
            ValueError,
            lambda: persist_lore_refresh_commit(
                checkpoint_path=(
                    checkpoint_path
                ),
                original_checkpoint_bytes=(
                    original_bytes
                ),
                commit_result=(
                    tampered
                ),
            ),
            "unexpected invalidation set",
        )

        assert_true(
            checkpoint_path.read_bytes()
            == original_bytes,
            "Tampered commit contract changed "
            "checkpoint.",
        )

    print(
        "Tampered commit contract fail-closed: PASS"
    )


def test_saved_hash_changes_for_material_commit() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(
            temp_dir
        )

        checkpoint_path = (
            temp_path
            / "writers_room.json"
        )

        state = build_state()

        original_bytes = write_checkpoint(
            checkpoint_path,
            state,
        )

        commit_result = (
            build_material_commit(
                state
            )
        )

        persistence = (
            persist_lore_refresh_commit(
                checkpoint_path=(
                    checkpoint_path
                ),
                original_checkpoint_bytes=(
                    original_bytes
                ),
                commit_result=(
                    commit_result
                ),
            )
        )

        assert_true(
            persistence.original_sha256
            != persistence.saved_sha256,
            "Material commit should change "
            "checkpoint content hash.",
        )

    print(
        "Material checkpoint hash change: PASS"
    )


def main() -> None:
    print(
        "\n--- Evidence Refresh Checkpoint Regression Test ---"
    )

    test_unchanged_commit_persists_atomically()
    test_material_commit_persists_invalidations()
    test_stale_checkpoint_fails_closed()
    test_missing_checkpoint_fails_closed()
    test_non_path_checkpoint_fails_closed()
    test_empty_original_bytes_fail_closed()
    test_wrong_commit_result_type_fails_closed()
    test_tampered_commit_contract_fails_closed()
    test_saved_hash_changes_for_material_commit()

    print(
        "\n============================================"
    )
    print(
        "EVIDENCE REFRESH CHECKPOINT REGRESSION: PASS"
    )
    print(
        "============================================"
    )


if __name__ == "__main__":
    main()
