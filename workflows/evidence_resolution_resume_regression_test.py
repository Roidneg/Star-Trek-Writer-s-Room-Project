"""
Pure-Python regression coverage for workflows.evidence_resolution_resume.

Run from the project root:

    python -m workflows.evidence_resolution_resume_regression_test

No provider calls are made.
No checkpoint files are changed.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from workflows.evidence_resolution_orchestrator import (
    EvidenceResolutionOrchestrationResult,
)
from workflows.evidence_resolution_resume import (
    resume_after_evidence_resolution,
)


# ---------------------------------------------------------
# Lightweight State Models
# ---------------------------------------------------------


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

    lore_results: list[str]

    character_results: list[str]

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


class FakeScriptRequest(BaseModel):
    marker: str


# ---------------------------------------------------------
# Assertions
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


# ---------------------------------------------------------
# Fixtures
# ---------------------------------------------------------


def base_upstream() -> dict[str, object]:
    return {
        "story_plan": (
            "Shadows of the Obsidian Order"
        ),
        "lore_results": [
            "lore_000",
            "lore_001",
        ],
        "character_results": [
            "character_000",
        ],
    }


def state_for_scriptwriter_resume() -> FakeWriterRoomState:
    upstream = base_upstream()

    return FakeWriterRoomState(
        story_plan=upstream["story_plan"],
        lore_results=upstream["lore_results"],
        character_results=upstream["character_results"],
        script_result=None,
        story_state_ledger=FakeLedger(
            entries=[]
        ),
        continuity_result=None,
        evaluation_result=None,
    )


def state_for_continuity_resume() -> FakeWriterRoomState:
    upstream = base_upstream()

    return FakeWriterRoomState(
        story_plan=upstream["story_plan"],
        lore_results=upstream["lore_results"],
        character_results=upstream["character_results"],
        script_result=FakeScriptResult(
            marker="existing-script"
        ),
        story_state_ledger=FakeLedger(
            entries=[]
        ),
        continuity_result=None,
        evaluation_result=None,
    )


def state_for_evaluation_resume() -> FakeWriterRoomState:
    upstream = base_upstream()

    return FakeWriterRoomState(
        story_plan=upstream["story_plan"],
        lore_results=upstream["lore_results"],
        character_results=upstream["character_results"],
        script_result=FakeScriptResult(
            marker="existing-script"
        ),
        story_state_ledger=FakeLedger(
            entries=[
                "state-one"
            ]
        ),
        continuity_result=FakeContinuityResult(
            marker="existing-continuity"
        ),
        evaluation_result=None,
    )


def orchestration_result(
    *,
    state: FakeWriterRoomState,
    earliest_stale_stage: str | None,
    material_change_detected: bool,
) -> EvidenceResolutionOrchestrationResult:
    invalidated = []

    if earliest_stale_stage == "scriptwriter":
        invalidated = [
            "script_result",
            "continuity_result",
            "story_state_ledger",
            "evaluation_result",
        ]

    elif earliest_stale_stage == "continuity":
        invalidated = [
            "continuity_result",
            "story_state_ledger",
            "evaluation_result",
        ]

    elif earliest_stale_stage == "evaluation":
        invalidated = [
            "evaluation_result",
        ]

    return EvidenceResolutionOrchestrationResult(
        run_status="resolved",
        evaluation_status="needs_evidence",
        processed_targets=[
            {
                "criterion_id": "fake_criterion",
                "owner": "lore",
                "action": "refresh_lore_evidence",
                "artifact_id": "lore_000",
                "subject_name": None,
                "refresh_status": (
                    "material_change"
                    if material_change_detected
                    else "unchanged"
                ),
                "commit_status": (
                    "committed_with_invalidation"
                    if material_change_detected
                    else "committed_without_invalidation"
                ),
                "material_change": (
                    material_change_detected
                ),
                "invalidated_artifacts": (
                    invalidated
                ),
                "reason": "Regression record.",
            }
        ],
        material_change_detected=(
            material_change_detected
        ),
        cumulative_invalidated_artifacts=(
            invalidated
        ),
        earliest_stale_stage=(
            earliest_stale_stage
        ),
        final_state=state,
        reason="Regression orchestration.",
    )


# ---------------------------------------------------------
# Regression Cases
# ---------------------------------------------------------


def test_scriptwriter_suffix_resume() -> None:
    state = state_for_scriptwriter_resume()
    original_dump = state.model_dump(
        mode="json"
    )

    call_order: list[str] = []

    def fake_scriptwriter(
        *,
        state: FakeWriterRoomState,
        request: FakeScriptRequest,
    ) -> FakeWriterRoomState:
        call_order.append(
            "scriptwriter"
        )

        assert_true(
            request.marker == "request",
            "Scriptwriter request was not routed.",
        )

        return state.model_copy(
            update={
                "script_result": (
                    FakeScriptResult(
                        marker="new-script"
                    )
                )
            }
        )

    def fake_continuity(
        *,
        state: FakeWriterRoomState,
        scene_index: int,
    ) -> FakeWriterRoomState:
        call_order.append(
            "continuity"
        )

        assert_true(
            scene_index == 0,
            "Continuity scene_index was not routed.",
        )

        return state.model_copy(
            update={
                "story_state_ledger": (
                    FakeLedger(
                        entries=[
                            "new-state"
                        ]
                    )
                ),
                "continuity_result": (
                    FakeContinuityResult(
                        marker="new-continuity"
                    )
                ),
            }
        )

    def fake_evaluation(
        state: FakeWriterRoomState,
    ) -> FakeWriterRoomState:
        call_order.append(
            "evaluation"
        )

        return state.model_copy(
            update={
                "evaluation_result": (
                    FakeEvaluationResult(
                        marker="new-evaluation"
                    )
                )
            }
        )

    result = resume_after_evidence_resolution(
        orchestration_result=(
            orchestration_result(
                state=state,
                earliest_stale_stage=(
                    "scriptwriter"
                ),
                material_change_detected=True,
            )
        ),
        script_request=FakeScriptRequest(
            marker="request"
        ),
        scriptwriter_runner=fake_scriptwriter,
        continuity_runner=fake_continuity,
        evaluation_runner=fake_evaluation,
        scene_index=0,
    )

    assert_true(
        result.run_status == "resumed",
        "Scriptwriter suffix did not resume.",
    )

    assert_true(
        result.started_at == "scriptwriter",
        "Resume start stage is wrong.",
    )

    assert_true(
        result.completed_stages
        == [
            "scriptwriter",
            "continuity",
            "evaluation",
        ],
        "Scriptwriter suffix completed wrong stages.",
    )

    assert_true(
        call_order
        == [
            "scriptwriter",
            "continuity",
            "evaluation",
        ],
        "Stage call order is wrong.",
    )

    assert_true(
        result.final_state.script_result.marker
        == "new-script",
        "Scriptwriter result missing.",
    )

    assert_true(
        result.final_state
        .continuity_result
        .marker
        == "new-continuity",
        "Continuity result missing.",
    )

    assert_true(
        result.final_state
        .evaluation_result
        .marker
        == "new-evaluation",
        "Evaluation result missing.",
    )

    assert_true(
        result.final_state
        .story_state_ledger
        .entries
        == [
            "new-state"
        ],
        "Continuity ledger was not rebuilt.",
    )

    assert_true(
        state.model_dump(
            mode="json"
        )
        == original_dump,
        "Resume mutated caller-owned state.",
    )

    print(
        "Scriptwriter suffix resume: PASS"
    )


def test_continuity_suffix_resume() -> None:
    state = state_for_continuity_resume()
    call_order: list[str] = []

    def fake_continuity(
        *,
        state: FakeWriterRoomState,
        scene_index: int,
    ) -> FakeWriterRoomState:
        call_order.append(
            "continuity"
        )

        return state.model_copy(
            update={
                "story_state_ledger": (
                    FakeLedger(
                        entries=[
                            "rebuilt-state"
                        ]
                    )
                ),
                "continuity_result": (
                    FakeContinuityResult(
                        marker="new-continuity"
                    )
                ),
            }
        )

    def fake_evaluation(
        state: FakeWriterRoomState,
    ) -> FakeWriterRoomState:
        call_order.append(
            "evaluation"
        )

        return state.model_copy(
            update={
                "evaluation_result": (
                    FakeEvaluationResult(
                        marker="new-evaluation"
                    )
                )
            }
        )

    result = resume_after_evidence_resolution(
        orchestration_result=(
            orchestration_result(
                state=state,
                earliest_stale_stage=(
                    "continuity"
                ),
                material_change_detected=True,
            )
        ),
        continuity_runner=fake_continuity,
        evaluation_runner=fake_evaluation,
    )

    assert_true(
        result.completed_stages
        == [
            "continuity",
            "evaluation",
        ],
        "Continuity suffix completed wrong stages.",
    )

    assert_true(
        call_order
        == [
            "continuity",
            "evaluation",
        ],
        "Continuity suffix call order is wrong.",
    )

    assert_true(
        result.final_state.script_result.marker
        == "existing-script",
        "Continuity resume should preserve Scriptwriter.",
    )

    print(
        "Continuity suffix resume: PASS"
    )


def test_evaluation_only_resume() -> None:
    state = state_for_evaluation_resume()
    call_order: list[str] = []

    def fake_evaluation(
        state: FakeWriterRoomState,
    ) -> FakeWriterRoomState:
        call_order.append(
            "evaluation"
        )

        return state.model_copy(
            update={
                "evaluation_result": (
                    FakeEvaluationResult(
                        marker="new-evaluation"
                    )
                )
            }
        )

    result = resume_after_evidence_resolution(
        orchestration_result=(
            orchestration_result(
                state=state,
                earliest_stale_stage=(
                    "evaluation"
                ),
                material_change_detected=True,
            )
        ),
        evaluation_runner=fake_evaluation,
    )

    assert_true(
        result.completed_stages
        == [
            "evaluation"
        ],
        "Evaluation-only resume completed wrong stages.",
    )

    assert_true(
        call_order == [
            "evaluation"
        ],
        "Evaluation-only runner order is wrong.",
    )

    assert_true(
        result.final_state.script_result.marker
        == "existing-script",
        "Evaluation-only resume changed Scriptwriter.",
    )

    assert_true(
        result.final_state
        .continuity_result
        .marker
        == "existing-continuity",
        "Evaluation-only resume changed Continuity.",
    )

    print(
        "Evaluation-only resume: PASS"
    )


def test_no_resume_when_authority_unchanged() -> None:
    state = state_for_evaluation_resume()

    result = resume_after_evidence_resolution(
        orchestration_result=(
            orchestration_result(
                state=state,
                earliest_stale_stage=None,
                material_change_detected=False,
            )
        ),
    )

    assert_true(
        result.run_status == "no_resume",
        "Unchanged authority should not resume.",
    )

    assert_true(
        result.completed_stages == [],
        "Unchanged authority should run no stages.",
    )

    print(
        "No-resume unchanged authority: PASS"
    )


def test_missing_script_request_fails_closed() -> None:
    state = state_for_scriptwriter_resume()

    assert_raises(
        ValueError,
        lambda: resume_after_evidence_resolution(
            orchestration_result=(
                orchestration_result(
                    state=state,
                    earliest_stale_stage=(
                        "scriptwriter"
                    ),
                    material_change_detected=True,
                )
            ),
        ),
        "Scriptwriter resume requires the "
        "WritersRoomScriptRequest explicitly",
    )

    print(
        "Missing Scriptwriter request fail-closed: PASS"
    )


def test_nonempty_ledger_at_scriptwriter_resume_fails_closed() -> None:
    state = state_for_scriptwriter_resume()

    state = state.model_copy(
        update={
            "story_state_ledger": (
                FakeLedger(
                    entries=[
                        "stale-entry"
                    ]
                )
            )
        }
    )

    assert_raises(
        ValueError,
        lambda: resume_after_evidence_resolution(
            orchestration_result=(
                orchestration_result(
                    state=state,
                    earliest_stale_stage=(
                        "scriptwriter"
                    ),
                    material_change_detected=True,
                )
            ),
            script_request=FakeScriptRequest(
                marker="request"
            ),
        ),
        "story-state ledger to be empty",
    )

    print(
        "Non-empty Scriptwriter ledger fail-closed: PASS"
    )


def test_existing_script_at_scriptwriter_resume_fails_closed() -> None:
    state = state_for_scriptwriter_resume()

    state = state.model_copy(
        update={
            "script_result": (
                FakeScriptResult(
                    marker="should-be-stale"
                )
            )
        }
    )

    assert_raises(
        ValueError,
        lambda: resume_after_evidence_resolution(
            orchestration_result=(
                orchestration_result(
                    state=state,
                    earliest_stale_stage=(
                        "scriptwriter"
                    ),
                    material_change_detected=True,
                )
            ),
            script_request=FakeScriptRequest(
                marker="request"
            ),
        ),
        "script_result=None",
    )

    print(
        "Existing Scriptwriter artifact fail-closed: PASS"
    )


def test_stage_cannot_change_upstream_authority() -> None:
    state = state_for_scriptwriter_resume()

    def bad_scriptwriter(
        *,
        state: FakeWriterRoomState,
        request: FakeScriptRequest,
    ) -> FakeWriterRoomState:
        return state.model_copy(
            update={
                "story_plan": (
                    "Illegally changed story plan"
                ),
                "script_result": (
                    FakeScriptResult(
                        marker="new-script"
                    )
                ),
            }
        )

    assert_raises(
        AssertionError,
        lambda: resume_after_evidence_resolution(
            orchestration_result=(
                orchestration_result(
                    state=state,
                    earliest_stale_stage=(
                        "scriptwriter"
                    ),
                    material_change_detected=True,
                )
            ),
            script_request=FakeScriptRequest(
                marker="request"
            ),
            scriptwriter_runner=bad_scriptwriter,
        ),
        "changed upstream",
    )

    print(
        "Upstream authority mutation fail-closed: PASS"
    )


def test_scriptwriter_runner_must_produce_script() -> None:
    state = state_for_scriptwriter_resume()

    def bad_scriptwriter(
        *,
        state: FakeWriterRoomState,
        request: FakeScriptRequest,
    ) -> FakeWriterRoomState:
        return state

    assert_raises(
        ValueError,
        lambda: resume_after_evidence_resolution(
            orchestration_result=(
                orchestration_result(
                    state=state,
                    earliest_stale_stage=(
                        "scriptwriter"
                    ),
                    material_change_detected=True,
                )
            ),
            script_request=FakeScriptRequest(
                marker="request"
            ),
            scriptwriter_runner=bad_scriptwriter,
        ),
        "without producing script_result",
    )

    print(
        "Missing Scriptwriter output fail-closed: PASS"
    )


def test_continuity_runner_must_produce_continuity() -> None:
    state = state_for_continuity_resume()

    def bad_continuity(
        *,
        state: FakeWriterRoomState,
        scene_index: int,
    ) -> FakeWriterRoomState:
        return state

    assert_raises(
        ValueError,
        lambda: resume_after_evidence_resolution(
            orchestration_result=(
                orchestration_result(
                    state=state,
                    earliest_stale_stage=(
                        "continuity"
                    ),
                    material_change_detected=True,
                )
            ),
            continuity_runner=bad_continuity,
        ),
        "without producing continuity_result",
    )

    print(
        "Missing Continuity output fail-closed: PASS"
    )


def test_evaluation_runner_must_produce_evaluation() -> None:
    state = state_for_evaluation_resume()

    def bad_evaluation(
        state: FakeWriterRoomState,
    ) -> FakeWriterRoomState:
        return state

    assert_raises(
        ValueError,
        lambda: resume_after_evidence_resolution(
            orchestration_result=(
                orchestration_result(
                    state=state,
                    earliest_stale_stage=(
                        "evaluation"
                    ),
                    material_change_detected=True,
                )
            ),
            evaluation_runner=bad_evaluation,
        ),
        "without producing evaluation_result",
    )

    print(
        "Missing Evaluation output fail-closed: PASS"
    )


def test_negative_scene_index_fails_closed() -> None:
    state = state_for_continuity_resume()

    assert_raises(
        ValueError,
        lambda: resume_after_evidence_resolution(
            orchestration_result=(
                orchestration_result(
                    state=state,
                    earliest_stale_stage=(
                        "continuity"
                    ),
                    material_change_detected=True,
                )
            ),
            scene_index=-1,
        ),
        "cannot be negative",
    )

    print(
        "Negative scene_index fail-closed: PASS"
    )


def main() -> None:
    print(
        "\n--- Evidence Resolution Resume "
        "Regression Test ---"
    )

    test_scriptwriter_suffix_resume()
    test_continuity_suffix_resume()
    test_evaluation_only_resume()
    test_no_resume_when_authority_unchanged()

    test_missing_script_request_fails_closed()
    test_nonempty_ledger_at_scriptwriter_resume_fails_closed()
    test_existing_script_at_scriptwriter_resume_fails_closed()
    test_stage_cannot_change_upstream_authority()

    test_scriptwriter_runner_must_produce_script()
    test_continuity_runner_must_produce_continuity()
    test_evaluation_runner_must_produce_evaluation()
    test_negative_scene_index_fails_closed()

    print(
        "\n============================================"
    )
    print(
        "EVIDENCE RESOLUTION RESUME "
        "REGRESSION: PASS"
    )
    print(
        "============================================"
    )


if __name__ == "__main__":
    main()
