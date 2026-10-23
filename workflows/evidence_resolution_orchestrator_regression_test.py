"""
Pure-Python regression coverage for
workflows.evidence_resolution_orchestrator.

Run from the project root:

    python -m workflows.evidence_resolution_orchestrator_regression_test

No provider calls are made.
No checkpoint files are changed.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from workflows.evaluation_resolution import (
    EvaluationResolutionPlan,
    EvaluationResolutionTarget,
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
# Downstream State Models
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
# Test Fixtures
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
        reason=(
            "Lore evidence remains unresolved."
        ),
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
        reason=(
            "Character evidence requires refresh."
        ),
    )


def two_blocker_plan(
    *,
    reverse_order: bool = False,
) -> EvaluationResolutionPlan:
    targets = [
        lore_target(),
        character_target(),
    ]

    if reverse_order:
        targets.reverse()

    return EvaluationResolutionPlan(
        evaluation_status="needs_evidence",
        resolution_status="needs_resolution",
        targets=targets,
        reason=(
            "First live-flight blocker pair."
        ),
    )


def approved_plan() -> EvaluationResolutionPlan:
    return EvaluationResolutionPlan(
        evaluation_status="approved",
        resolution_status="no_action",
        targets=[],
        reason="Evaluation approved.",
    )


def rejected_plan() -> EvaluationResolutionPlan:
    return EvaluationResolutionPlan(
        evaluation_status="rejected",
        resolution_status="not_applicable",
        targets=[],
        reason="Evaluation rejected.",
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


# ---------------------------------------------------------
# Regression Cases
# ---------------------------------------------------------


def test_live_flight_two_blocker_material_resolution() -> None:
    state = build_state()
    original_dump = state.model_dump(
        mode="json"
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
        call_order.append(
            "character"
        )

        return character_production(
            story_context=story_context,
            story_era=story_era,
            analysis=changed_analysis,
        )

    result = orchestrate_evidence_resolution(
        resolution_plan=(
            two_blocker_plan(
                reverse_order=True
            )
        ),
        state=state,
        lore_runner=fake_lore_runner,
        character_runner=(
            fake_character_runner
        ),
    )

    assert_true(
        call_order
        == [
            "lore",
            "character",
        ],
        "Orchestrator did not enforce deterministic "
        "Lore-before-Character target ordering.",
    )

    assert_true(
        result.run_status
        == "resolved",
        "Two-blocker plan did not resolve.",
    )

    assert_true(
        result.material_change_detected
        is True,
        "Material upstream changes were not "
        "reported.",
    )

    assert_true(
        result.cumulative_invalidated_artifacts
        == [
            "script_result",
            "continuity_result",
            "story_state_ledger",
            "evaluation_result",
        ],
        "Combined invalidations were incorrect.",
    )

    assert_true(
        result.earliest_stale_stage
        == "scriptwriter",
        "Earliest stale stage should be Scriptwriter.",
    )

    assert_true(
        len(
            result.processed_targets
        )
        == 2,
        "Both live-flight blockers were not processed.",
    )

    assert_true(
        result.processed_targets[0].owner
        == "lore",
        "First execution record should be Lore.",
    )

    assert_true(
        result.processed_targets[1].owner
        == "character",
        "Second execution record should be Character.",
    )

    final_state = result.final_state

    assert_true(
        final_state.lore_results[0]
        .response
        .evidence_supported
        is True,
        "Lore candidate was not carried into final "
        "orchestration state.",
    )

    assert_true(
        final_state.character_results[0]
        .analysis
        .likely_behavior
        == changed_analysis.likely_behavior,
        "Character candidate was not carried into "
        "final orchestration state.",
    )

    assert_true(
        final_state.script_result is None,
        "Material evidence resolution did not leave "
        "Scriptwriter stale.",
    )

    assert_true(
        final_state.continuity_result is None,
        "Material evidence resolution did not leave "
        "Continuity stale.",
    )

    assert_true(
        final_state.story_state_ledger.entries
        == [],
        "Material evidence resolution did not reset "
        "the Continuity ledger.",
    )

    assert_true(
        final_state.evaluation_result is None,
        "Material evidence resolution did not leave "
        "Evaluation stale.",
    )

    assert_true(
        state.model_dump(
            mode="json"
        )
        == original_dump,
        "Orchestrator mutated caller-owned state.",
    )

    print(
        "Live-flight two-blocker material "
        "resolution: PASS"
    )


def test_all_unchanged_preserves_downstream() -> None:
    state = build_state()

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

    result = orchestrate_evidence_resolution(
        resolution_plan=(
            two_blocker_plan()
        ),
        state=state,
        lore_runner=fake_lore_runner,
        character_runner=(
            fake_character_runner
        ),
    )

    assert_true(
        result.material_change_detected
        is False,
        "Unchanged authority should not report "
        "material change.",
    )

    assert_true(
        result.cumulative_invalidated_artifacts
        == [],
        "Unchanged authority should not accumulate "
        "invalidations.",
    )

    assert_true(
        result.earliest_stale_stage is None,
        "Unchanged authority should not create a "
        "resume stage.",
    )

    final_state = result.final_state

    assert_true(
        final_state.script_result
        == state.script_result,
        "All-unchanged resolution changed "
        "Scriptwriter.",
    )

    assert_true(
        final_state.continuity_result
        == state.continuity_result,
        "All-unchanged resolution changed "
        "Continuity.",
    )

    assert_true(
        final_state.story_state_ledger
        == state.story_state_ledger,
        "All-unchanged resolution changed ledger.",
    )

    assert_true(
        final_state.evaluation_result
        == state.evaluation_result,
        "All-unchanged resolution changed "
        "Evaluation.",
    )

    print(
        "All-unchanged multi-target preservation: PASS"
    )


def test_lore_material_character_unchanged() -> None:
    state = build_state()

    def fake_lore_runner(
        *,
        query: str,
        debug: bool,
    ) -> FakeLoreResponse:
        return FakeLoreResponse(
            premise_supported=True,
            evidence_supported=True,
            summary=(
                "Fresh supported Lore authority."
            ),
        )

    character_saw_stale_downstream = {
        "checked": False,
    }

    def fake_character_runner(
        *,
        character_name: str,
        story_context: str,
        story_era: str,
        debug: bool,
    ) -> FakeCharacterProductionResult:
        # The Character runner itself receives only its task
        # contract, so the state-carrying assertion is made
        # through the final record/state behavior.
        character_saw_stale_downstream[
            "checked"
        ] = True

        return character_production(
            story_context=story_context,
            story_era=story_era,
            analysis=base_garak_analysis(),
        )

    result = orchestrate_evidence_resolution(
        resolution_plan=(
            two_blocker_plan()
        ),
        state=state,
        lore_runner=fake_lore_runner,
        character_runner=(
            fake_character_runner
        ),
    )

    assert_true(
        character_saw_stale_downstream[
            "checked"
        ],
        "Character target was not processed after "
        "material Lore refresh.",
    )

    assert_true(
        result.processed_targets[0]
        .material_change
        is True,
        "Lore target should be material.",
    )

    assert_true(
        result.processed_targets[1]
        .material_change
        is False,
        "Character target should remain unchanged.",
    )

    assert_true(
        result.earliest_stale_stage
        == "scriptwriter",
        "Any material upstream authority change "
        "should resume from Scriptwriter.",
    )

    assert_true(
        result.final_state
        .character_results[0]
        .research_plan
        .canon_questions
        == [
            "Fresh canon question.",
        ],
        "Character provenance refresh was not "
        "committed after Lore invalidation.",
    )

    assert_true(
        result.final_state.script_result
        is None,
        "Later unchanged Character refresh should "
        "not resurrect stale Scriptwriter output.",
    )

    print(
        "Lore-material + Character-unchanged "
        "sequencing: PASS"
    )


def test_lore_unchanged_character_material() -> None:
    state = build_state()

    def fake_lore_runner(
        *,
        query: str,
        debug: bool,
    ) -> FakeLoreResponse:
        return FakeLoreResponse(
            premise_supported=True,
            evidence_supported=False,
            summary=(
                "Still unresolved."
            ),
        )

    changed_analysis = base_garak_analysis(
        likely_behavior=[
            (
                "Seek independent corroboration "
                "before acting."
            )
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

    result = orchestrate_evidence_resolution(
        resolution_plan=(
            two_blocker_plan()
        ),
        state=state,
        lore_runner=fake_lore_runner,
        character_runner=(
            fake_character_runner
        ),
    )

    assert_true(
        result.processed_targets[0]
        .material_change
        is False,
        "Lore target should remain unchanged.",
    )

    assert_true(
        result.processed_targets[1]
        .material_change
        is True,
        "Character target should be material.",
    )

    assert_true(
        result.earliest_stale_stage
        == "scriptwriter",
        "Material Character change should resume "
        "from Scriptwriter.",
    )

    print(
        "Lore-unchanged + Character-material "
        "sequencing: PASS"
    )


def test_approved_plan_is_noop() -> None:
    state = build_state()
    original_dump = state.model_dump(
        mode="json"
    )

    result = orchestrate_evidence_resolution(
        resolution_plan=(
            approved_plan()
        ),
        state=state,
    )

    assert_true(
        result.run_status
        == "no_action",
        "Approved plan should be no_action.",
    )

    assert_true(
        result.processed_targets
        == [],
        "Approved plan should process no targets.",
    )

    assert_true(
        result.earliest_stale_stage is None,
        "Approved plan should have no stale stage.",
    )

    assert_true(
        result.final_state.model_dump(
            mode="json"
        )
        == original_dump,
        "Approved no-op changed state.",
    )

    print(
        "Approved orchestration no-op: PASS"
    )


def test_rejected_plan_is_not_applicable() -> None:
    state = build_state()

    result = orchestrate_evidence_resolution(
        resolution_plan=(
            rejected_plan()
        ),
        state=state,
    )

    assert_true(
        result.run_status
        == "not_applicable",
        "Rejected plan should be outside the "
        "evidence-refresh path.",
    )

    assert_true(
        result.processed_targets
        == [],
        "Rejected plan should process no targets.",
    )

    assert_true(
        result.earliest_stale_stage is None,
        "Rejected plan should not create a resume "
        "stage.",
    )

    print(
        "Rejected orchestration exclusion: PASS"
    )


def test_unsupported_era_style_target_fails_closed() -> None:
    state = build_state()

    unsupported = EvaluationResolutionTarget(
        criterion_id="era_style_review",
        category="era_style",
        owner="scriptwriter_input",
        action="supply_era_style_constraints",
        artifact_id=None,
        subject_name=None,
        reason="Era/style evidence missing.",
    )

    plan = EvaluationResolutionPlan(
        evaluation_status="needs_evidence",
        resolution_status="needs_resolution",
        targets=[
            unsupported
        ],
        reason="Unsupported v0.1 target.",
    )

    assert_raises(
        ValueError,
        lambda: orchestrate_evidence_resolution(
            resolution_plan=plan,
            state=state,
        ),
        "supports only Lore and Character targets",
    )

    print(
        "Unsupported era/style target fail-closed: PASS"
    )


def test_duplicate_same_artifact_fails_closed() -> None:
    state = build_state()

    duplicate = EvaluationResolutionTarget(
        criterion_id=(
            "lore_agent_evidence_duplicate"
        ),
        category="canon_grounding",
        owner="lore",
        action="refresh_lore_evidence",
        artifact_id="lore_000",
        subject_name=None,
        reason="Duplicate artifact target.",
    )

    plan = EvaluationResolutionPlan(
        evaluation_status="needs_evidence",
        resolution_status="needs_resolution",
        targets=[
            lore_target(),
            duplicate,
        ],
        reason="Duplicate upstream artifact.",
    )

    assert_raises(
        ValueError,
        lambda: orchestrate_evidence_resolution(
            resolution_plan=plan,
            state=state,
        ),
        "same upstream artifact",
    )

    print(
        "Duplicate artifact target fail-closed: PASS"
    )


def test_missing_lore_artifact_fails_closed() -> None:
    state = build_state()

    missing_target = EvaluationResolutionTarget(
        criterion_id=(
            "lore_agent_evidence_lore_999"
        ),
        category="canon_grounding",
        owner="lore",
        action="refresh_lore_evidence",
        artifact_id="lore_999",
        subject_name=None,
        reason="Missing artifact.",
    )

    plan = EvaluationResolutionPlan(
        evaluation_status="needs_evidence",
        resolution_status="needs_resolution",
        targets=[
            missing_target
        ],
        reason="Missing Lore artifact.",
    )

    assert_raises(
        ValueError,
        lambda: orchestrate_evidence_resolution(
            resolution_plan=plan,
            state=state,
            lore_runner=(
                lambda **kwargs: FakeLoreResponse(
                    premise_supported=True,
                    evidence_supported=False,
                    summary="Unused.",
                )
            ),
        ),
        "does not contain artifact",
    )

    print(
        "Missing Lore artifact fail-closed: PASS"
    )


def test_non_pydantic_state_fails_closed() -> None:
    assert_raises(
        TypeError,
        lambda: orchestrate_evidence_resolution(
            resolution_plan=(
                approved_plan()
            ),
            state={
                "lore_results": [],
            },
        ),
        "Pydantic BaseModel",
    )

    print(
        "Non-Pydantic orchestrator state "
        "fail-closed: PASS"
    )


def main() -> None:
    print(
        "\n--- Evidence Resolution Orchestrator "
        "Regression Test ---"
    )

    test_live_flight_two_blocker_material_resolution()
    test_all_unchanged_preserves_downstream()
    test_lore_material_character_unchanged()
    test_lore_unchanged_character_material()
    test_approved_plan_is_noop()
    test_rejected_plan_is_not_applicable()
    test_unsupported_era_style_target_fails_closed()
    test_duplicate_same_artifact_fails_closed()
    test_missing_lore_artifact_fails_closed()
    test_non_pydantic_state_fails_closed()

    print(
        "\n============================================"
    )
    print(
        "EVIDENCE RESOLUTION ORCHESTRATOR "
        "REGRESSION: PASS"
    )
    print(
        "============================================"
    )


if __name__ == "__main__":
    main()
