"""
Pure-Python regression coverage for workflows.evidence_refresh_commit.

Run from the project root:

    python -m workflows.evidence_refresh_commit_regression_test

No provider calls are made.
No checkpoint files are changed.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from workflows.evaluation_resolution import (
    EvaluationResolutionTarget,
)
from workflows.evidence_refresh import (
    LoreEvidenceRefreshResult,
    refresh_lore_evidence,
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
    lore_results: list[FakeLoreResult]
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
        raise AssertionError(message)


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
        reason="Regression Lore blocker.",
    )


def build_state(
    *,
    lore_summary: str = (
        "The available evidence does not "
        "establish a direct encounter."
    ),
    lore_evidence_supported: bool = False,
) -> FakeWriterRoomState:
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
                    evidence_supported=(
                        lore_evidence_supported
                    ),
                    summary=lore_summary,
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
                        "Call to Arms marks the "
                        "official start."
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
        script_result=FakeScriptResult(
            marker="script"
        ),
        story_state_ledger=FakeLedger(
            entries=[
                "door open",
                "rod on table",
            ]
        ),
        continuity_result=FakeContinuityResult(
            marker="continuity"
        ),
        evaluation_result=FakeEvaluationResult(
            marker="evaluation"
        ),
    )


def build_unchanged_refresh(
    state: FakeWriterRoomState,
) -> LoreEvidenceRefreshResult:
    def fake_runner(
        *,
        query: str,
        debug: bool,
    ) -> FakeLoreResponse:
        return FakeLoreResponse(
            premise_supported=True,
            evidence_supported=False,
            summary=(
                "Retrieved material still does "
                "not establish a direct encounter."
            ),
        )

    return refresh_lore_evidence(
        target=lore_target(),
        current_lore_result=(
            state.lore_results[0]
        ),
        lore_runner=fake_runner,
    )


def build_material_refresh(
    state: FakeWriterRoomState,
) -> LoreEvidenceRefreshResult:
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

    return refresh_lore_evidence(
        target=lore_target(),
        current_lore_result=(
            state.lore_results[0]
        ),
        lore_runner=fake_runner,
    )


# ---------------------------------------------------------
# Regression Cases
# ---------------------------------------------------------


def test_unchanged_commit_preserves_downstream() -> None:
    state = build_state()
    refresh = build_unchanged_refresh(
        state
    )

    original_dump = state.model_dump(
        mode="json"
    )

    result = commit_lore_refresh_to_state(
        state=state,
        refresh_result=refresh,
    )

    assert_true(
        result.commit_status
        == "committed_without_invalidation",
        "Unchanged Lore refresh used wrong "
        "commit status.",
    )

    assert_true(
        result.invalidated_artifacts == [],
        "Unchanged Lore refresh invalidated "
        "downstream artifacts.",
    )

    committed = result.committed_state

    assert_true(
        committed.lore_results[0].response.summary
        == (
            "Retrieved material still does "
            "not establish a direct encounter."
        ),
        "Refreshed Lore provenance was not "
        "committed.",
    )

    assert_true(
        committed.script_result
        == state.script_result,
        "Unchanged Lore refresh changed "
        "script_result.",
    )

    assert_true(
        committed.continuity_result
        == state.continuity_result,
        "Unchanged Lore refresh changed "
        "continuity_result.",
    )

    assert_true(
        committed.story_state_ledger
        == state.story_state_ledger,
        "Unchanged Lore refresh changed "
        "story_state_ledger.",
    )

    assert_true(
        committed.evaluation_result
        == state.evaluation_result,
        "Unchanged Lore refresh changed "
        "evaluation_result.",
    )

    assert_true(
        committed.character_results
        == state.character_results,
        "Unchanged Lore refresh changed "
        "Character artifacts.",
    )

    assert_true(
        state.model_dump(
            mode="json"
        )
        == original_dump,
        "Commit mutated supplied state in place.",
    )

    print(
        "Unchanged Lore commit preservation: PASS"
    )


def test_material_commit_invalidates_exact_dependents() -> None:
    state = build_state()
    refresh = build_material_refresh(
        state
    )

    original_dump = state.model_dump(
        mode="json"
    )

    result = commit_lore_refresh_to_state(
        state=state,
        refresh_result=refresh,
    )

    assert_true(
        result.commit_status
        == "committed_with_invalidation",
        "Material Lore refresh used wrong "
        "commit status.",
    )

    assert_true(
        result.invalidated_artifacts
        == [
            "script_result",
            "continuity_result",
            "story_state_ledger",
            "evaluation_result",
        ],
        "Material Lore commit invalidated "
        "the wrong artifact set.",
    )

    committed = result.committed_state

    assert_true(
        committed.lore_results[0]
        .response
        .evidence_supported
        is True,
        "Material Lore candidate was not committed.",
    )

    assert_true(
        committed.script_result is None,
        "Material Lore change did not clear "
        "script_result.",
    )

    assert_true(
        committed.continuity_result is None,
        "Material Lore change did not clear "
        "continuity_result.",
    )

    assert_true(
        committed.evaluation_result is None,
        "Material Lore change did not clear "
        "evaluation_result.",
    )

    assert_true(
        committed.story_state_ledger.entries
        == [],
        "Material Lore change did not reset "
        "the story-state ledger.",
    )

    assert_true(
        committed.character_results
        == state.character_results,
        "Material Lore change should preserve "
        "Character artifacts.",
    )

    assert_true(
        committed.story_plan
        == state.story_plan,
        "Material Lore change should preserve "
        "StoryPlan.",
    )

    assert_true(
        committed.lore_results[1]
        == state.lore_results[1],
        "Material Lore change unexpectedly "
        "changed sibling Lore artifact.",
    )

    assert_true(
        state.model_dump(
            mode="json"
        )
        == original_dump,
        "Material commit mutated supplied "
        "state in place.",
    )

    print(
        "Material Lore commit invalidation: PASS"
    )


def test_stale_refresh_commit_fails_closed() -> None:
    state = build_state()
    refresh = build_material_refresh(
        state
    )

    changed_state = state.model_copy(
        update={
            "lore_results": [
                state.lore_results[0].model_copy(
                    update={
                        "response": (
                            FakeLoreResponse(
                                premise_supported=True,
                                evidence_supported=True,
                                summary=(
                                    "State changed after "
                                    "refresh began."
                                ),
                            )
                        )
                    }
                ),
                state.lore_results[1],
            ]
        }
    )

    assert_raises(
        ValueError,
        lambda: commit_lore_refresh_to_state(
            state=changed_state,
            refresh_result=refresh,
        ),
        "Refusing stale refresh commit",
    )

    print(
        "Stale Lore refresh fail-closed: PASS"
    )


def test_missing_target_artifact_fails_closed() -> None:
    state = build_state()
    refresh = build_material_refresh(
        state
    )

    missing_state = state.model_copy(
        update={
            "lore_results": [
                state.lore_results[1]
            ]
        }
    )

    assert_raises(
        ValueError,
        lambda: commit_lore_refresh_to_state(
            state=missing_state,
            refresh_result=refresh,
        ),
        "does not contain Lore artifact",
    )

    print(
        "Missing Lore artifact fail-closed: PASS"
    )


def test_duplicate_target_artifact_fails_closed() -> None:
    state = build_state()
    refresh = build_material_refresh(
        state
    )

    duplicate_state = state.model_copy(
        update={
            "lore_results": [
                state.lore_results[0],
                state.lore_results[0],
                state.lore_results[1],
            ]
        }
    )

    assert_raises(
        ValueError,
        lambda: commit_lore_refresh_to_state(
            state=duplicate_state,
            refresh_result=refresh,
        ),
        "duplicate Lore artifact ID",
    )

    print(
        "Duplicate Lore artifact fail-closed: PASS"
    )


def test_tampered_candidate_fails_closed() -> None:
    state = build_state()
    refresh = build_material_refresh(
        state
    )

    tampered_candidate = (
        refresh.candidate_artifact.model_copy(
            update={
                "response": FakeLoreResponse(
                    premise_supported=True,
                    evidence_supported=True,
                    summary=(
                        "Tampered candidate summary."
                    ),
                )
            }
        )
    )

    tampered_refresh = refresh.model_copy(
        update={
            "candidate_artifact": (
                tampered_candidate
            )
        }
    )

    assert_raises(
        ValueError,
        lambda: commit_lore_refresh_to_state(
            state=state,
            refresh_result=tampered_refresh,
        ),
        "candidate no longer matches",
    )

    print(
        "Tampered candidate fail-closed: PASS"
    )


def test_unchanged_refresh_with_invalidation_plan_fails_closed() -> None:
    state = build_state()
    unchanged = build_unchanged_refresh(
        state
    )

    material = build_material_refresh(
        state
    )

    bad_refresh = unchanged.model_copy(
        update={
            "invalidation_plan": (
                material.invalidation_plan
            )
        }
    )

    assert_raises(
        ValueError,
        lambda: commit_lore_refresh_to_state(
            state=state,
            refresh_result=bad_refresh,
        ),
        "must not carry an invalidation plan",
    )

    print(
        "Unchanged+invalidation inconsistency "
        "fail-closed: PASS"
    )


def test_material_refresh_without_invalidation_plan_fails_closed() -> None:
    state = build_state()
    material = build_material_refresh(
        state
    )

    bad_refresh = material.model_copy(
        update={
            "invalidation_plan": None
        }
    )

    assert_raises(
        ValueError,
        lambda: commit_lore_refresh_to_state(
            state=state,
            refresh_result=bad_refresh,
        ),
        "requires an invalidation plan",
    )

    print(
        "Material-without-plan inconsistency "
        "fail-closed: PASS"
    )


def test_non_pydantic_state_fails_closed() -> None:
    state = build_state()
    refresh = build_material_refresh(
        state
    )

    assert_raises(
        TypeError,
        lambda: commit_lore_refresh_to_state(
            state={
                "lore_results": []
            },
            refresh_result=refresh,
        ),
        "Pydantic BaseModel",
    )

    print(
        "Non-Pydantic state fail-closed: PASS"
    )


def test_bad_ledger_default_fails_closed() -> None:
    class BadLedger(BaseModel):
        entries: list[str]

    class BadLedgerState(BaseModel):
        story_plan: str
        lore_results: list[FakeLoreResult]
        character_results: list[
            FakeCharacterResult
        ]
        script_result: (
            FakeScriptResult
            | None
        )
        story_state_ledger: BadLedger
        continuity_result: (
            FakeContinuityResult
            | None
        )
        evaluation_result: (
            FakeEvaluationResult
            | None
        )

    base = build_state()

    bad_state = BadLedgerState(
        story_plan=base.story_plan,
        lore_results=base.lore_results,
        character_results=(
            base.character_results
        ),
        script_result=base.script_result,
        story_state_ledger=BadLedger(
            entries=["existing"]
        ),
        continuity_result=(
            base.continuity_result
        ),
        evaluation_result=(
            base.evaluation_result
        ),
    )

    refresh = build_material_refresh(
        base
    )

    assert_raises(
        TypeError,
        lambda: commit_lore_refresh_to_state(
            state=bad_state,
            refresh_result=refresh,
        ),
        "cannot be reconstructed",
    )

    print(
        "Non-default ledger fail-closed: PASS"
    )


def main() -> None:
    print(
        "\n--- Evidence Refresh Commit Regression Test ---"
    )

    test_unchanged_commit_preserves_downstream()
    test_material_commit_invalidates_exact_dependents()
    test_stale_refresh_commit_fails_closed()
    test_missing_target_artifact_fails_closed()
    test_duplicate_target_artifact_fails_closed()
    test_tampered_candidate_fails_closed()
    test_unchanged_refresh_with_invalidation_plan_fails_closed()
    test_material_refresh_without_invalidation_plan_fails_closed()
    test_non_pydantic_state_fails_closed()
    test_bad_ledger_default_fails_closed()

    print(
        "\n============================================"
    )
    print(
        "EVIDENCE REFRESH COMMIT REGRESSION: PASS"
    )
    print(
        "============================================"
    )


if __name__ == "__main__":
    main()
