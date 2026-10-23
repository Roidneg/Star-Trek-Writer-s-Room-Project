"""
Pure-Python regression coverage for
workflows.evidence_resolution_checkpoint.

Run from the project root:

    python -m workflows.evidence_resolution_checkpoint_regression_test

No provider calls are made.
The real Writer's Room checkpoint is never touched.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from pydantic import BaseModel, Field

from workflows.evaluation_resolution import (
    EvaluationResolutionPlan,
    EvaluationResolutionTarget,
)
from workflows.evidence_resolution_checkpoint import (
    persist_evidence_resolution,
    sha256_bytes,
)
from workflows.evidence_resolution_orchestrator import (
    orchestrate_evidence_resolution,
)


# ---------------------------------------------------------
# Lightweight Lore Models
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
# Lightweight Character Models
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
# Downstream Models
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
# Assertion Helpers
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


def build_state() -> FakeWriterRoomState:
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
                        "Call to Arms marks the "
                        "official start in 2373."
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
        evaluation_result=FakeEvaluationResult(
            marker="needs-evidence"
        ),
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
        reason="Lore blocker.",
    )


def character_target() -> EvaluationResolutionTarget:
    return EvaluationResolutionTarget(
        criterion_id=(
            "character_agent_guidance_elim_garak"
        ),
        category="character_consistency",
        owner="character",
        action="refresh_character_evidence",
        artifact_id="character_000",
        subject_name="Elim Garak",
        reason="Character blocker.",
    )


def two_blocker_plan() -> EvaluationResolutionPlan:
    return EvaluationResolutionPlan(
        evaluation_status="needs_evidence",
        resolution_status="needs_resolution",
        targets=[
            # Intentionally Character first. The orchestrator
            # must still execute Lore first.
            character_target(),
            lore_target(),
        ],
        reason="First live-flight blocker pair.",
    )


def approved_plan() -> EvaluationResolutionPlan:
    return EvaluationResolutionPlan(
        evaluation_status="approved",
        resolution_status="no_action",
        targets=[],
        reason="Approved.",
    )


def rejected_plan() -> EvaluationResolutionPlan:
    return EvaluationResolutionPlan(
        evaluation_status="rejected",
        resolution_status="not_applicable",
        targets=[],
        reason="Rejected.",
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
# Orchestration Builders
# ---------------------------------------------------------


def build_material_orchestration(
    state: FakeWriterRoomState,
):
    def fake_lore_runner(
        *,
        query: str,
        debug: bool,
    ) -> FakeLoreResponse:
        return FakeLoreResponse(
            premise_supported=True,
            evidence_supported=True,
            summary=(
                "Fresh evidence now establishes "
                "the requested canon point."
            ),
        )

    changed_analysis = base_garak_analysis(
        likely_behavior=[
            (
                "Evaluate uncertain information "
                "before acting."
            ),
            (
                "Seek corroboration before accepting "
                "a strategically important warning."
            ),
        ]
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
            analysis=changed_analysis,
        )

    return orchestrate_evidence_resolution(
        resolution_plan=two_blocker_plan(),
        state=state,
        lore_runner=fake_lore_runner,
        character_runner=fake_character_runner,
    )


def build_unchanged_orchestration(
    state: FakeWriterRoomState,
):
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

    return orchestrate_evidence_resolution(
        resolution_plan=two_blocker_plan(),
        state=state,
        lore_runner=fake_lore_runner,
        character_runner=fake_character_runner,
    )


# ---------------------------------------------------------
# Regression Cases
# ---------------------------------------------------------


def test_live_flight_material_transaction_persists_once() -> None:
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

        orchestration = (
            build_material_orchestration(
                state
            )
        )

        persistence = (
            persist_evidence_resolution(
                checkpoint_path=checkpoint_path,
                original_checkpoint_bytes=(
                    original_bytes
                ),
                orchestration_result=(
                    orchestration
                ),
            )
        )

        assert_true(
            persistence.persistence_status
            == "saved",
            "Material orchestration was not persisted.",
        )

        assert_true(
            persistence.orchestration_status
            == "resolved",
            "Material orchestration lost run status.",
        )

        assert_true(
            persistence.evaluation_status
            == "needs_evidence",
            "Material orchestration lost Evaluation status.",
        )

        assert_true(
            persistence.processed_target_count
            == 2,
            "Live-flight transaction should persist "
            "two processed targets.",
        )

        assert_true(
            persistence.material_change_detected
            is True,
            "Material transaction lost material-change flag.",
        )

        assert_true(
            persistence.cumulative_invalidated_artifacts
            == [
                "script_result",
                "continuity_result",
                "story_state_ledger",
                "evaluation_result",
            ],
            "Material transaction persisted wrong "
            "invalidation set.",
        )

        assert_true(
            persistence.earliest_stale_stage
            == "scriptwriter",
            "Material transaction should resume from "
            "Scriptwriter.",
        )

        backup_path = Path(
            persistence.backup_path
        )

        assert_true(
            backup_path.exists(),
            "Combined transaction backup was not created.",
        )

        assert_true(
            backup_path.read_bytes()
            == original_bytes,
            "Combined transaction backup does not preserve "
            "exact original checkpoint bytes.",
        )

        saved_state = persistence.saved_state

        assert_true(
            saved_state.lore_results[0]
            .response
            .evidence_supported
            is True,
            "Resolved Lore artifact was not persisted.",
        )

        assert_true(
            len(
                saved_state.character_results[0]
                .analysis
                .likely_behavior
            )
            == 2,
            "Resolved Character guidance was not persisted.",
        )

        assert_true(
            saved_state.script_result is None,
            "Persisted transaction should leave "
            "Scriptwriter stale.",
        )

        assert_true(
            saved_state.continuity_result is None,
            "Persisted transaction should leave "
            "Continuity stale.",
        )

        assert_true(
            saved_state.story_state_ledger.entries
            == [],
            "Persisted transaction should reset "
            "Continuity ledger.",
        )

        assert_true(
            saved_state.evaluation_result is None,
            "Persisted transaction should leave "
            "Evaluation stale.",
        )

        assert_true(
            persistence.original_sha256
            == sha256_bytes(
                original_bytes
            ),
            "Original transaction hash is wrong.",
        )

        assert_true(
            persistence.saved_sha256
            == sha256_bytes(
                checkpoint_path.read_bytes()
            ),
            "Saved transaction hash is wrong.",
        )

        # One transaction backup should be enough for this
        # single persistence call.
        backup_files = list(
            temp_path.glob(
                "writers_room_before_evidence_resolution_*.json"
            )
        )

        assert_true(
            len(backup_files) == 1,
            "Combined evidence resolution should create "
            "exactly one transaction backup.",
        )

    print(
        "Live-flight combined persistence: PASS"
    )


def test_all_unchanged_transaction_persists_provenance() -> None:
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

        orchestration = (
            build_unchanged_orchestration(
                state
            )
        )

        persistence = (
            persist_evidence_resolution(
                checkpoint_path=checkpoint_path,
                original_checkpoint_bytes=(
                    original_bytes
                ),
                orchestration_result=(
                    orchestration
                ),
            )
        )

        assert_true(
            persistence.material_change_detected
            is False,
            "Unchanged transaction incorrectly persisted "
            "material change.",
        )

        assert_true(
            persistence.cumulative_invalidated_artifacts
            == [],
            "Unchanged transaction persisted invalidations.",
        )

        assert_true(
            persistence.earliest_stale_stage is None,
            "Unchanged transaction should not persist a "
            "resume stage.",
        )

        saved_state = persistence.saved_state

        assert_true(
            saved_state.script_result
            == state.script_result,
            "Unchanged transaction changed Scriptwriter.",
        )

        assert_true(
            saved_state.continuity_result
            == state.continuity_result,
            "Unchanged transaction changed Continuity.",
        )

        assert_true(
            saved_state.story_state_ledger
            == state.story_state_ledger,
            "Unchanged transaction changed ledger.",
        )

        assert_true(
            saved_state.evaluation_result
            == state.evaluation_result,
            "Unchanged transaction changed Evaluation.",
        )

        # Research provenance should still have refreshed.
        assert_true(
            saved_state.character_results[0]
            .research_plan
            .canon_questions
            == [
                "Fresh canon question.",
            ],
            "Unchanged transaction did not persist "
            "refreshed Character provenance.",
        )

    print(
        "All-unchanged combined persistence: PASS"
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

        orchestration = (
            build_material_orchestration(
                state
            )
        )

        newer_state = state.model_copy(
            update={
                "story_plan": "Newer story plan state"
            }
        )

        newer_bytes = serialize_state(
            newer_state
        )

        checkpoint_path.write_bytes(
            newer_bytes
        )

        assert_raises(
            ValueError,
            lambda: persist_evidence_resolution(
                checkpoint_path=checkpoint_path,
                original_checkpoint_bytes=(
                    original_bytes
                ),
                orchestration_result=(
                    orchestration
                ),
            ),
            "changed after the evidence-resolution "
            "transaction began",
        )

        assert_true(
            checkpoint_path.read_bytes()
            == newer_bytes,
            "Stale transaction overwrote newer checkpoint.",
        )

    print(
        "Combined stale-checkpoint fail-closed: PASS"
    )


def test_approved_noop_persistence_contract() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_path = (
            Path(temp_dir)
            / "writers_room.json"
        )

        state = build_state()

        original_bytes = write_checkpoint(
            checkpoint_path,
            state,
        )

        orchestration = (
            orchestrate_evidence_resolution(
                resolution_plan=approved_plan(),
                state=state,
            )
        )

        persistence = (
            persist_evidence_resolution(
                checkpoint_path=checkpoint_path,
                original_checkpoint_bytes=(
                    original_bytes
                ),
                orchestration_result=orchestration,
            )
        )

        assert_true(
            persistence.orchestration_status
            == "no_action",
            "Approved persistence should preserve "
            "no_action status.",
        )

        assert_true(
            persistence.processed_target_count
            == 0,
            "Approved persistence should have zero "
            "processed targets.",
        )

        assert_true(
            persistence.saved_state
            == state,
            "Approved no-op persistence changed state.",
        )

    print(
        "Approved no-op persistence: PASS"
    )


def test_rejected_not_applicable_persistence_contract() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_path = (
            Path(temp_dir)
            / "writers_room.json"
        )

        state = build_state()

        original_bytes = write_checkpoint(
            checkpoint_path,
            state,
        )

        orchestration = (
            orchestrate_evidence_resolution(
                resolution_plan=rejected_plan(),
                state=state,
            )
        )

        persistence = (
            persist_evidence_resolution(
                checkpoint_path=checkpoint_path,
                original_checkpoint_bytes=(
                    original_bytes
                ),
                orchestration_result=orchestration,
            )
        )

        assert_true(
            persistence.orchestration_status
            == "not_applicable",
            "Rejected persistence should preserve "
            "not_applicable status.",
        )

        assert_true(
            persistence.saved_state
            == state,
            "Rejected persistence changed state.",
        )

    print(
        "Rejected persistence exclusion: PASS"
    )


def test_missing_checkpoint_fails_closed() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_path = (
            Path(temp_dir)
            / "missing.json"
        )

        state = build_state()

        orchestration = (
            build_material_orchestration(
                state
            )
        )

        assert_raises(
            FileNotFoundError,
            lambda: persist_evidence_resolution(
                checkpoint_path=checkpoint_path,
                original_checkpoint_bytes=(
                    serialize_state(
                        state
                    )
                ),
                orchestration_result=orchestration,
            ),
            "checkpoint disappeared",
        )

    print(
        "Missing combined checkpoint fail-closed: PASS"
    )


def test_wrong_result_type_fails_closed() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_path = (
            Path(temp_dir)
            / "writers_room.json"
        )

        state = build_state()

        original_bytes = write_checkpoint(
            checkpoint_path,
            state,
        )

        assert_raises(
            TypeError,
            lambda: persist_evidence_resolution(
                checkpoint_path=checkpoint_path,
                original_checkpoint_bytes=(
                    original_bytes
                ),
                orchestration_result=object(),
            ),
            "EvidenceResolutionOrchestrationResult",
        )

    print(
        "Wrong orchestration-result type "
        "fail-closed: PASS"
    )


def test_tampered_material_contract_fails_closed() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_path = (
            Path(temp_dir)
            / "writers_room.json"
        )

        state = build_state()

        original_bytes = write_checkpoint(
            checkpoint_path,
            state,
        )

        orchestration = (
            build_material_orchestration(
                state
            )
        )

        tampered = orchestration.model_copy(
            update={
                "cumulative_invalidated_artifacts": [],
                "earliest_stale_stage": None,
            }
        )

        assert_raises(
            ValueError,
            lambda: persist_evidence_resolution(
                checkpoint_path=checkpoint_path,
                original_checkpoint_bytes=(
                    original_bytes
                ),
                orchestration_result=tampered,
            ),
            "Material orchestration must contain "
            "invalidated artifacts",
        )

        assert_true(
            checkpoint_path.read_bytes()
            == original_bytes,
            "Tampered transaction changed checkpoint.",
        )

    print(
        "Tampered orchestration contract "
        "fail-closed: PASS"
    )


def test_material_saved_hash_changes() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_path = (
            Path(temp_dir)
            / "writers_room.json"
        )

        state = build_state()

        original_bytes = write_checkpoint(
            checkpoint_path,
            state,
        )

        orchestration = (
            build_material_orchestration(
                state
            )
        )

        persistence = (
            persist_evidence_resolution(
                checkpoint_path=checkpoint_path,
                original_checkpoint_bytes=(
                    original_bytes
                ),
                orchestration_result=orchestration,
            )
        )

        assert_true(
            persistence.original_sha256
            != persistence.saved_sha256,
            "Material combined transaction should "
            "change checkpoint hash.",
        )

    print(
        "Combined material hash change: PASS"
    )


def main() -> None:
    print(
        "\n--- Evidence Resolution Checkpoint "
        "Regression Test ---"
    )

    test_live_flight_material_transaction_persists_once()
    test_all_unchanged_transaction_persists_provenance()
    test_stale_checkpoint_fails_closed()
    test_approved_noop_persistence_contract()
    test_rejected_not_applicable_persistence_contract()
    test_missing_checkpoint_fails_closed()
    test_wrong_result_type_fails_closed()
    test_tampered_material_contract_fails_closed()
    test_material_saved_hash_changes()

    print(
        "\n============================================"
    )
    print(
        "EVIDENCE RESOLUTION CHECKPOINT "
        "REGRESSION: PASS"
    )
    print(
        "============================================"
    )


if __name__ == "__main__":
    main()
