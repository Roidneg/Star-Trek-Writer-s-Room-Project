"""
Pure-Python regression coverage for
workflows.evidence_resolution_live_bridge.

Run from the project root:

    python -m workflows.evidence_resolution_live_bridge_regression_test

No provider calls are made.
The real Writer's Room checkpoint is never touched.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from pydantic import BaseModel, Field

from workflows.evidence_resolution_live_bridge import (
    resolve_live_evaluation_checkpoint,
)


# ---------------------------------------------------------
# Minimal Evaluation Models
# ---------------------------------------------------------


class FakeCriterionResult(BaseModel):
    criterion_id: str
    category: str
    verdict: str


class FakeEvaluationSuite(BaseModel):
    results: list[FakeCriterionResult]


class FakeEvaluationAssessment(BaseModel):
    status: str
    blocking_criterion_ids: list[str] = Field(
        default_factory=list
    )


class FakeEvaluationReview(BaseModel):
    assessment: FakeEvaluationAssessment
    suite: FakeEvaluationSuite


class FakeProductionReview(BaseModel):
    review: FakeEvaluationReview


class FakeWriterRoomEvaluationResult(BaseModel):
    production_review: FakeProductionReview

    lore_research_ids: list[str] = Field(
        default_factory=list
    )

    character_result_ids: list[str] = Field(
        default_factory=list
    )


# ---------------------------------------------------------
# Lore Models
# ---------------------------------------------------------


class FakeLoreResponse(BaseModel):
    premise_supported: bool
    evidence_supported: bool
    summary: str


class FakeLoreResult(BaseModel):
    research_id: str
    research_question: str
    response: FakeLoreResponse


# ---------------------------------------------------------
# Character Models
# ---------------------------------------------------------


class FakeResearchPlan(BaseModel):
    character_name: str
    story_context: str
    story_era: str

    canon_questions: list[str] = Field(
        default_factory=list
    )

    voice_questions: list[str] = Field(
        default_factory=list
    )


class FakeCharacterAnalysis(BaseModel):
    character_name: str
    story_role: str = ""

    motivations: list[str] = Field(
        default_factory=list
    )

    likely_behavior: list[str] = Field(
        default_factory=list
    )

    unlikely_behavior: list[str] = Field(
        default_factory=list
    )

    relationships: list[str] = Field(
        default_factory=list
    )

    dramatic_tension: str = ""

    speech_patterns: list[str] = Field(
        default_factory=list
    )

    preferred_vocabulary: list[str] = Field(
        default_factory=list
    )

    communication_style: list[str] = Field(
        default_factory=list
    )

    behavioral_constraints: list[str] = Field(
        default_factory=list
    )

    era_language_notes: list[str] = Field(
        default_factory=list
    )

    out_of_character_risks: list[str] = Field(
        default_factory=list
    )


class FakeClaimAudit(BaseModel):
    marker: str


class FakeCharacterAudit(BaseModel):
    grounded: bool
    reason: str = ""


class FakeCharacterResult(BaseModel):
    character_id: str
    character_name: str
    research_plan: FakeResearchPlan

    canon_research: list[str] = Field(
        default_factory=list
    )

    voice_research: list[str] = Field(
        default_factory=list
    )

    analysis: FakeCharacterAnalysis
    claim_audit: FakeClaimAudit
    audit: FakeCharacterAudit


class FakeCharacterProductionResult(BaseModel):
    research_plan: FakeResearchPlan

    combined_canon_research: list[str] = Field(
        default_factory=list
    )

    voice_research: list[str] = Field(
        default_factory=list
    )

    character_analysis: FakeCharacterAnalysis
    claim_audit: FakeClaimAudit
    analysis_audit: FakeCharacterAudit


# ---------------------------------------------------------
# Downstream Models / State
# ---------------------------------------------------------


class FakeScriptResult(BaseModel):
    marker: str


class FakeContinuityResult(BaseModel):
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
        FakeWriterRoomEvaluationResult
        | None
    )


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


def base_garak_analysis(
    *,
    likely_behavior: list[str] | None = None,
) -> FakeCharacterAnalysis:
    return FakeCharacterAnalysis(
        character_name="Elim Garak",
        story_role="Information broker",
        motivations=[
            "Protect his interests.",
        ],
        likely_behavior=(
            likely_behavior
            if likely_behavior is not None
            else [
                (
                    "Evaluate uncertain information "
                    "before acting."
                )
            ]
        ),
        unlikely_behavior=[
            (
                "Treat an allegation as proven "
                "without corroboration."
            )
        ],
        relationships=[
            "Maintains layered trust with allies.",
        ],
        dramatic_tension=(
            "He must judge whether uncertain "
            "information deserves action."
        ),
        speech_patterns=[
            "Uses implication and understatement.",
        ],
        preferred_vocabulary=[
            "my dear doctor",
        ],
        communication_style=[
            "Indirect but precise.",
        ],
        behavioral_constraints=[
            (
                "Do not make him needlessly "
                "transparent."
            )
        ],
        era_language_notes=[
            "Keep dialogue consistent with DS9.",
        ],
        out_of_character_risks=[
            "Blunt exposition without subtext.",
        ],
    )


def needs_evidence_evaluation() -> FakeWriterRoomEvaluationResult:
    lore_id = (
        "lore_agent_evidence_lore_000"
    )

    character_id = (
        "character_agent_guidance_elim_garak"
    )

    return FakeWriterRoomEvaluationResult(
        production_review=FakeProductionReview(
            review=FakeEvaluationReview(
                assessment=FakeEvaluationAssessment(
                    status="needs_evidence",
                    blocking_criterion_ids=[
                        lore_id,
                        character_id,
                    ],
                ),
                suite=FakeEvaluationSuite(
                    results=[
                        FakeCriterionResult(
                            criterion_id=lore_id,
                            category="canon_grounding",
                            verdict=(
                                "insufficient_evidence"
                            ),
                        ),
                        FakeCriterionResult(
                            criterion_id=character_id,
                            category=(
                                "character_consistency"
                            ),
                            verdict=(
                                "insufficient_evidence"
                            ),
                        ),
                    ]
                ),
            )
        ),
        lore_research_ids=[
            "lore_000",
            "lore_001",
        ],
        character_result_ids=[
            "character_000",
        ],
    )


def approved_evaluation() -> FakeWriterRoomEvaluationResult:
    return FakeWriterRoomEvaluationResult(
        production_review=FakeProductionReview(
            review=FakeEvaluationReview(
                assessment=FakeEvaluationAssessment(
                    status="approved",
                    blocking_criterion_ids=[],
                ),
                suite=FakeEvaluationSuite(
                    results=[
                        FakeCriterionResult(
                            criterion_id=(
                                "lore_agent_evidence_"
                                "lore_000"
                            ),
                            category=(
                                "canon_grounding"
                            ),
                            verdict="pass",
                        )
                    ]
                ),
            )
        ),
        lore_research_ids=[
            "lore_000",
            "lore_001",
        ],
        character_result_ids=[
            "character_000",
        ],
    )


def rejected_evaluation() -> FakeWriterRoomEvaluationResult:
    fail_id = "scriptwriter_grounding_review"

    return FakeWriterRoomEvaluationResult(
        production_review=FakeProductionReview(
            review=FakeEvaluationReview(
                assessment=FakeEvaluationAssessment(
                    status="rejected",
                    blocking_criterion_ids=[
                        fail_id
                    ],
                ),
                suite=FakeEvaluationSuite(
                    results=[
                        FakeCriterionResult(
                            criterion_id=fail_id,
                            category=(
                                "upstream_constraint"
                            ),
                            verdict="fail",
                        )
                    ]
                ),
            )
        ),
        lore_research_ids=[
            "lore_000",
            "lore_001",
        ],
        character_result_ids=[
            "character_000",
        ],
    )


def build_state(
    *,
    evaluation_result: (
        FakeWriterRoomEvaluationResult
        | None
    ) = None,
) -> FakeWriterRoomState:
    return FakeWriterRoomState(
        story_plan=(
            "Shadows of the Obsidian Order"
        ),
        lore_results=[
            FakeLoreResult(
                research_id="lore_000",
                research_question=(
                    "In which DS9 episode does Garak "
                    "first directly encounter the "
                    "Founders?"
                ),
                response=FakeLoreResponse(
                    premise_supported=True,
                    evidence_supported=False,
                    summary=(
                        "Current evidence does not "
                        "establish the requested direct "
                        "Founder encounter."
                    ),
                ),
            ),
            FakeLoreResult(
                research_id="lore_001",
                research_question=(
                    "When does the Dominion War "
                    "officially begin?"
                ),
                response=FakeLoreResponse(
                    premise_supported=True,
                    evidence_supported=True,
                    summary=(
                        "Call to Arms marks the official "
                        "start in 2373."
                    ),
                ),
            ),
        ],
        character_results=[
            FakeCharacterResult(
                character_id="character_000",
                character_name="Elim Garak",
                research_plan=FakeResearchPlan(
                    character_name="Elim Garak",
                    story_context=(
                        "Garak evaluates a damaged data "
                        "rod containing an uncertain "
                        "warning."
                    ),
                    story_era=(
                        "Deep Space Nine, "
                        "pre-Dominion War"
                    ),
                    canon_questions=[
                        "Existing canon question.",
                    ],
                    voice_questions=[
                        "Existing voice question.",
                    ],
                ),
                canon_research=[
                    "Existing canon research.",
                ],
                voice_research=[
                    "Existing voice research.",
                ],
                analysis=base_garak_analysis(),
                claim_audit=FakeClaimAudit(
                    marker="old-claim-audit"
                ),
                audit=FakeCharacterAudit(
                    grounded=True,
                    reason="Existing grounded audit.",
                ),
            )
        ],
        script_result=FakeScriptResult(
            marker="script"
        ),
        story_state_ledger=FakeLedger(
            entries=[
                "The door is open.",
                "The rod is on the table.",
                "The door is sealed.",
                "Garak possesses the rod.",
            ]
        ),
        continuity_result=FakeContinuityResult(
            marker="continuity"
        ),
        evaluation_result=(
            evaluation_result
            if evaluation_result is not None
            else needs_evidence_evaluation()
        ),
    )


def character_production(
    *,
    story_context: str,
    story_era: str,
    analysis: FakeCharacterAnalysis,
) -> FakeCharacterProductionResult:
    return FakeCharacterProductionResult(
        research_plan=FakeResearchPlan(
            character_name="Elim Garak",
            story_context=story_context,
            story_era=story_era,
            canon_questions=[
                "Fresh canon question.",
            ],
            voice_questions=[
                "Fresh voice question.",
            ],
        ),
        combined_canon_research=[
            "Fresh canon research.",
        ],
        voice_research=[
            "Fresh voice research.",
        ],
        character_analysis=analysis,
        claim_audit=FakeClaimAudit(
            marker="fresh-claim-audit"
        ),
        analysis_audit=FakeCharacterAudit(
            grounded=True,
            reason="Fresh grounded audit.",
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


# ---------------------------------------------------------
# Regression Cases
# ---------------------------------------------------------


def test_live_material_bridge_combines_and_persists_once() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(
            temp_dir
        )

        checkpoint_path = (
            temp_path
            / "writers_room.json"
        )

        state = build_state()

        original_bytes = serialize_state(
            state
        )

        checkpoint_path.write_bytes(
            original_bytes
        )

        call_order: list[str] = []

        def fake_lore_runner(
            *,
            query: str,
            debug: bool,
        ) -> FakeLoreResponse:
            call_order.append(
                "lore"
            )

            return FakeLoreResponse(
                premise_supported=True,
                evidence_supported=True,
                summary=(
                    "Fresh evidence now establishes "
                    "the requested canon point."
                ),
            )

        changed_analysis = (
            base_garak_analysis(
                likely_behavior=[
                    (
                        "Evaluate uncertain information "
                        "before acting."
                    ),
                    (
                        "Seek corroboration before "
                        "accepting a strategically "
                        "important warning."
                    ),
                ]
            )
        )

        def fake_character_runner(
            *,
            character_name: str,
            story_context: str,
            story_era: str,
            debug: bool,
        ) -> FakeCharacterProductionResult:
            call_order.append(
                "character"
            )

            return character_production(
                story_context=story_context,
                story_era=story_era,
                analysis=changed_analysis,
            )

        result = (
            resolve_live_evaluation_checkpoint(
                state=state,
                checkpoint_path=(
                    checkpoint_path
                ),
                lore_runner=fake_lore_runner,
                character_runner=(
                    fake_character_runner
                ),
            )
        )

        assert_true(
            result.outcome
            == "resolved_material_change",
            "Material live bridge used wrong outcome.",
        )

        assert_true(
            call_order
            == [
                "lore",
                "character",
            ],
            "Live bridge did not preserve deterministic "
            "Lore-before-Character execution.",
        )

        assert_true(
            result.earliest_stale_stage
            == "scriptwriter",
            "Material live bridge should resume from "
            "Scriptwriter.",
        )

        assert_true(
            result.checkpoint_changed
            is True,
            "Material live bridge did not report "
            "checkpoint change.",
        )

        assert_true(
            result.persistence_result
            is not None,
            "Material live bridge did not persist.",
        )

        assert_true(
            result.persistence_result
            .processed_target_count
            == 2,
            "Material live bridge did not persist both "
            "resolution targets.",
        )

        final_state = result.final_state

        assert_true(
            final_state.lore_results[0]
            .response
            .evidence_supported
            is True,
            "Resolved Lore evidence was not saved.",
        )

        assert_true(
            len(
                final_state.character_results[0]
                .analysis
                .likely_behavior
            )
            == 2,
            "Resolved Character guidance was not saved.",
        )

        assert_true(
            final_state.script_result is None,
            "Material bridge did not invalidate "
            "Scriptwriter.",
        )

        assert_true(
            final_state.continuity_result is None,
            "Material bridge did not invalidate "
            "Continuity.",
        )

        assert_true(
            final_state.story_state_ledger.entries
            == [],
            "Material bridge did not reset ledger.",
        )

        assert_true(
            final_state.evaluation_result is None,
            "Material bridge did not invalidate "
            "Evaluation.",
        )

        backup_files = list(
            temp_path.glob(
                "writers_room_before_"
                "evidence_resolution_*.json"
            )
        )

        assert_true(
            len(backup_files) == 1,
            "Live multi-blocker bridge should create "
            "exactly one combined backup.",
        )

        assert_true(
            backup_files[0].read_bytes()
            == original_bytes,
            "Live bridge backup does not preserve "
            "the exact original checkpoint.",
        )

        assert_true(
            checkpoint_path.read_bytes()
            != original_bytes,
            "Material live bridge did not replace "
            "checkpoint content.",
        )

    print(
        "Live material bridge combined persistence: PASS"
    )


def test_live_unchanged_bridge_persists_provenance_no_resume() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(
            temp_dir
        )

        checkpoint_path = (
            temp_path
            / "writers_room.json"
        )

        state = build_state()
        original_bytes = serialize_state(
            state
        )

        checkpoint_path.write_bytes(
            original_bytes
        )

        def fake_lore_runner(
            *,
            query: str,
            debug: bool,
        ) -> FakeLoreResponse:
            return FakeLoreResponse(
                premise_supported=True,
                evidence_supported=False,
                summary=(
                    "Fresh retrieval still does not "
                    "establish the requested encounter."
                ),
            )

        def fake_character_runner(
            *,
            character_name: str,
            story_context: str,
            story_era: str,
            debug: bool,
        ) -> FakeCharacterProductionResult:
            return character_production(
                story_context=story_context,
                story_era=story_era,
                analysis=base_garak_analysis(),
            )

        result = (
            resolve_live_evaluation_checkpoint(
                state=state,
                checkpoint_path=(
                    checkpoint_path
                ),
                lore_runner=fake_lore_runner,
                character_runner=(
                    fake_character_runner
                ),
            )
        )

        assert_true(
            result.outcome
            == "resolved_no_material_change",
            "Unchanged live bridge used wrong outcome.",
        )

        assert_true(
            result.earliest_stale_stage is None,
            "Unchanged live bridge should request no "
            "downstream resume.",
        )

        assert_true(
            result.checkpoint_changed
            is True,
            "Refreshed provenance should still be "
            "persisted.",
        )

        final_state = result.final_state

        assert_true(
            final_state.script_result
            == state.script_result,
            "Unchanged live bridge changed Scriptwriter.",
        )

        assert_true(
            final_state.continuity_result
            == state.continuity_result,
            "Unchanged live bridge changed Continuity.",
        )

        assert_true(
            final_state.story_state_ledger
            == state.story_state_ledger,
            "Unchanged live bridge changed ledger.",
        )

        assert_true(
            final_state.evaluation_result
            == state.evaluation_result,
            "Unchanged live bridge changed Evaluation.",
        )

        assert_true(
            final_state.character_results[0]
            .research_plan
            .canon_questions
            == [
                "Fresh canon question.",
            ],
            "Unchanged live bridge did not persist "
            "fresh Character provenance.",
        )

    print(
        "Live unchanged bridge no-resume: PASS"
    )


def test_approved_bridge_does_not_touch_checkpoint() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_path = (
            Path(temp_dir)
            / "writers_room.json"
        )

        state = build_state(
            evaluation_result=(
                approved_evaluation()
            )
        )

        # The no-action path must not even require a
        # checkpoint to exist.
        assert_true(
            not checkpoint_path.exists(),
            "Regression setup unexpectedly created "
            "checkpoint.",
        )

        result = (
            resolve_live_evaluation_checkpoint(
                state=state,
                checkpoint_path=(
                    checkpoint_path
                ),
            )
        )

        assert_true(
            result.outcome == "no_action",
            "Approved bridge should be no_action.",
        )

        assert_true(
            result.checkpoint_changed
            is False,
            "Approved bridge should not report a "
            "checkpoint write.",
        )

        assert_true(
            result.persistence_result is None,
            "Approved bridge should not persist.",
        )

        assert_true(
            not checkpoint_path.exists(),
            "Approved bridge touched checkpoint path.",
        )

    print(
        "Approved live bridge no checkpoint touch: PASS"
    )


def test_rejected_bridge_does_not_touch_checkpoint() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_path = (
            Path(temp_dir)
            / "writers_room.json"
        )

        state = build_state(
            evaluation_result=(
                rejected_evaluation()
            )
        )

        result = (
            resolve_live_evaluation_checkpoint(
                state=state,
                checkpoint_path=(
                    checkpoint_path
                ),
            )
        )

        assert_true(
            result.outcome
            == "not_applicable",
            "Rejected bridge should be "
            "not_applicable.",
        )

        assert_true(
            result.checkpoint_changed
            is False,
            "Rejected bridge should not write checkpoint.",
        )

        assert_true(
            not checkpoint_path.exists(),
            "Rejected bridge touched checkpoint path.",
        )

    print(
        "Rejected live bridge no checkpoint touch: PASS"
    )


def test_divergent_checkpoint_fails_closed() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_path = (
            Path(temp_dir)
            / "writers_room.json"
        )

        state = build_state()

        divergent = state.model_copy(
            update={
                "story_plan": (
                    "Different checkpoint state"
                )
            }
        )

        checkpoint_path.write_bytes(
            serialize_state(
                divergent
            )
        )

        assert_raises(
            ValueError,
            lambda: resolve_live_evaluation_checkpoint(
                state=state,
                checkpoint_path=(
                    checkpoint_path
                ),
                lore_runner=(
                    lambda **kwargs: FakeLoreResponse(
                        premise_supported=True,
                        evidence_supported=False,
                        summary="Unused.",
                    )
                ),
                character_runner=(
                    lambda **kwargs: (
                        character_production(
                            story_context=(
                                kwargs[
                                    "story_context"
                                ]
                            ),
                            story_era=(
                                kwargs[
                                    "story_era"
                                ]
                            ),
                            analysis=(
                                base_garak_analysis()
                            ),
                        )
                    )
                ),
            ),
            "does not match the current checkpoint",
        )

    print(
        "Divergent live checkpoint fail-closed: PASS"
    )


def test_missing_evaluation_result_fails_closed() -> None:
    state = build_state()
    state = state.model_copy(
        update={
            "evaluation_result": None
        }
    )

    assert_raises(
        ValueError,
        lambda: resolve_live_evaluation_checkpoint(
            state=state,
            checkpoint_path=Path(
                "unused.json"
            ),
        ),
        "requires a completed Evaluation result",
    )

    print(
        "Missing live Evaluation fail-closed: PASS"
    )


def test_unknown_evaluation_lore_source_fails_closed() -> None:
    evaluation = needs_evidence_evaluation()

    evaluation = evaluation.model_copy(
        update={
            "lore_research_ids": [
                "lore_999",
            ]
        }
    )

    state = build_state(
        evaluation_result=evaluation
    )

    assert_raises(
        ValueError,
        lambda: resolve_live_evaluation_checkpoint(
            state=state,
            checkpoint_path=Path(
                "unused.json"
            ),
        ),
        "unknown Writer's Room Lore artifact ID",
    )

    print(
        "Unknown Evaluation Lore source "
        "fail-closed: PASS"
    )


def test_duplicate_evaluation_character_source_fails_closed() -> None:
    evaluation = needs_evidence_evaluation()

    evaluation = evaluation.model_copy(
        update={
            "character_result_ids": [
                "character_000",
                "character_000",
            ]
        }
    )

    state = build_state(
        evaluation_result=evaluation
    )

    assert_raises(
        ValueError,
        lambda: resolve_live_evaluation_checkpoint(
            state=state,
            checkpoint_path=Path(
                "unused.json"
            ),
        ),
        "duplicate Character result IDs",
    )

    print(
        "Duplicate Evaluation Character routing "
        "fail-closed: PASS"
    )


def test_unsupported_era_style_blocker_fails_closed() -> None:
    era_id = "era_style_review"

    evaluation = FakeWriterRoomEvaluationResult(
        production_review=FakeProductionReview(
            review=FakeEvaluationReview(
                assessment=FakeEvaluationAssessment(
                    status="needs_evidence",
                    blocking_criterion_ids=[
                        era_id
                    ],
                ),
                suite=FakeEvaluationSuite(
                    results=[
                        FakeCriterionResult(
                            criterion_id=era_id,
                            category="era_style",
                            verdict=(
                                "insufficient_evidence"
                            ),
                        )
                    ]
                ),
            )
        ),
        lore_research_ids=[
            "lore_000",
            "lore_001",
        ],
        character_result_ids=[
            "character_000",
        ],
    )

    state = build_state(
        evaluation_result=evaluation
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_path = (
            Path(temp_dir)
            / "writers_room.json"
        )

        checkpoint_path.write_bytes(
            serialize_state(
                state
            )
        )

        assert_raises(
            ValueError,
            lambda: resolve_live_evaluation_checkpoint(
                state=state,
                checkpoint_path=(
                    checkpoint_path
                ),
            ),
            "supports only Lore and Character targets",
        )

    print(
        "Unsupported live era/style blocker "
        "fail-closed: PASS"
    )


def main() -> None:
    print(
        "\n--- Evidence Resolution Live Bridge "
        "Regression Test ---"
    )

    test_live_material_bridge_combines_and_persists_once()
    test_live_unchanged_bridge_persists_provenance_no_resume()
    test_approved_bridge_does_not_touch_checkpoint()
    test_rejected_bridge_does_not_touch_checkpoint()

    test_divergent_checkpoint_fails_closed()
    test_missing_evaluation_result_fails_closed()
    test_unknown_evaluation_lore_source_fails_closed()
    test_duplicate_evaluation_character_source_fails_closed()
    test_unsupported_era_style_blocker_fails_closed()

    print(
        "\n============================================"
    )
    print(
        "EVIDENCE RESOLUTION LIVE BRIDGE "
        "REGRESSION: PASS"
    )
    print(
        "============================================"
    )


if __name__ == "__main__":
    main()
