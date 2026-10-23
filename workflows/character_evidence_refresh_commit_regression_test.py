"""
Pure-Python regression coverage for
workflows.character_evidence_refresh_commit.

Run from the project root:

    python -m workflows.character_evidence_refresh_commit_regression_test

No provider calls are made.
No checkpoint files are changed.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from workflows.character_evidence_refresh import (
    refresh_character_evidence,
)
from workflows.character_evidence_refresh_commit import (
    commit_character_refresh_to_state,
)
from workflows.evaluation_resolution import (
    EvaluationResolutionTarget,
)


# ---------------------------------------------------------
# Lightweight Models
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


class FakeWriterRoomCharacterResult(BaseModel):
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


class FakeLoreResult(BaseModel):
    research_id: str
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
        FakeWriterRoomCharacterResult
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


def character_target() -> EvaluationResolutionTarget:
    return EvaluationResolutionTarget(
        criterion_id=(
            "character_agent_guidance_"
            "elim_garak"
        ),
        category="character_consistency",
        owner="character",
        action="refresh_character_evidence",
        artifact_id="character_000",
        subject_name="Elim Garak",
        reason=(
            "Regression Character blocker."
        ),
    )


def base_analysis(
    *,
    character_name: str = "Elim Garak",
    likely_behavior: list[str] | None = None,
) -> FakeCharacterAnalysis:
    return FakeCharacterAnalysis(
        character_name=character_name,
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


def character_artifact(
    *,
    character_id: str,
    character_name: str,
    story_context: str,
    story_era: str,
    grounded: bool = True,
    analysis: FakeCharacterAnalysis | None = None,
    marker_suffix: str = "existing",
) -> FakeWriterRoomCharacterResult:
    return FakeWriterRoomCharacterResult(
        character_id=character_id,
        character_name=character_name,
        research_plan=FakeResearchPlan(
            character_name=character_name,
            story_context=story_context,
            story_era=story_era,
            canon_questions=[
                f"{marker_suffix} canon question",
            ],
            voice_questions=[
                f"{marker_suffix} voice question",
            ],
        ),
        canon_research=[
            f"{marker_suffix} canon research",
        ],
        voice_research=[
            f"{marker_suffix} voice research",
        ],
        analysis=(
            analysis
            if analysis is not None
            else base_analysis(
                character_name=character_name
            )
        ),
        claim_audit=FakeClaimAudit(
            marker=(
                f"{marker_suffix}-claim-audit"
            )
        ),
        audit=FakeCharacterAudit(
            grounded=grounded,
            reason=(
                f"{marker_suffix} aggregate audit"
            ),
        ),
    )


def build_state() -> FakeWriterRoomState:
    garak_context = (
        "Garak evaluates a damaged data rod "
        "containing an uncertain warning."
    )

    era = (
        "Deep Space Nine, pre-Dominion War"
    )

    return FakeWriterRoomState(
        story_plan="StoryPlan marker",
        lore_results=[
            FakeLoreResult(
                research_id="lore_000",
                marker="lore-zero",
            ),
            FakeLoreResult(
                research_id="lore_001",
                marker="lore-one",
            ),
        ],
        character_results=[
            character_artifact(
                character_id="character_000",
                character_name="Elim Garak",
                story_context=garak_context,
                story_era=era,
            ),
            character_artifact(
                character_id="character_001",
                character_name="Benjamin Sisko",
                story_context=(
                    "Sisko evaluates station security."
                ),
                story_era=era,
                analysis=base_analysis(
                    character_name=(
                        "Benjamin Sisko"
                    )
                ),
                marker_suffix="sibling",
            ),
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


def production_result(
    *,
    character_name: str,
    story_context: str,
    story_era: str,
    grounded: bool,
    analysis: FakeCharacterAnalysis,
) -> FakeCharacterProductionResult:
    return FakeCharacterProductionResult(
        research_plan=FakeResearchPlan(
            character_name=character_name,
            story_context=story_context,
            story_era=story_era,
            canon_questions=[
                (
                    "Fresh canon question about "
                    "uncertain intelligence."
                )
            ],
            voice_questions=[
                (
                    "Fresh voice question about "
                    "signaling doubt."
                )
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
            grounded=grounded,
            reason="Fresh aggregate audit.",
        ),
    )


def build_unchanged_refresh(
    state: FakeWriterRoomState,
):
    current = state.character_results[0]

    def fake_runner(
        *,
        character_name: str,
        story_context: str,
        story_era: str,
        debug: bool,
    ) -> FakeCharacterProductionResult:
        return production_result(
            character_name=character_name,
            story_context=story_context,
            story_era=story_era,
            grounded=True,
            analysis=base_analysis(
                character_name=character_name
            ),
        )

    return refresh_character_evidence(
        target=character_target(),
        current_character_result=current,
        character_runner=fake_runner,
    )


def build_material_refresh(
    state: FakeWriterRoomState,
):
    current = state.character_results[0]

    changed_analysis = base_analysis(
        character_name=(
            current.character_name
        ),
        likely_behavior=[
            (
                "Evaluate uncertain information "
                "before acting."
            ),
            (
                "Seek corroboration before trusting "
                "a strategically important warning."
            ),
        ],
    )

    def fake_runner(
        *,
        character_name: str,
        story_context: str,
        story_era: str,
        debug: bool,
    ) -> FakeCharacterProductionResult:
        return production_result(
            character_name=character_name,
            story_context=story_context,
            story_era=story_era,
            grounded=True,
            analysis=changed_analysis,
        )

    return refresh_character_evidence(
        target=character_target(),
        current_character_result=current,
        character_runner=fake_runner,
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

    result = (
        commit_character_refresh_to_state(
            state=state,
            refresh_result=refresh,
        )
    )

    assert_true(
        result.commit_status
        == "committed_without_invalidation",
        "Unchanged Character refresh used wrong "
        "commit status.",
    )

    assert_true(
        result.invalidated_artifacts == [],
        "Unchanged Character refresh invalidated "
        "downstream artifacts.",
    )

    committed = result.committed_state

    assert_true(
        committed.character_results[0]
        .research_plan
        .canon_questions
        != state.character_results[0]
        .research_plan
        .canon_questions,
        "Refreshed Character provenance was not "
        "committed.",
    )

    assert_true(
        committed.lore_results
        == state.lore_results,
        "Unchanged Character refresh changed Lore.",
    )

    assert_true(
        committed.script_result
        == state.script_result,
        "Unchanged Character refresh changed "
        "script_result.",
    )

    assert_true(
        committed.continuity_result
        == state.continuity_result,
        "Unchanged Character refresh changed "
        "continuity_result.",
    )

    assert_true(
        committed.story_state_ledger
        == state.story_state_ledger,
        "Unchanged Character refresh changed ledger.",
    )

    assert_true(
        committed.evaluation_result
        == state.evaluation_result,
        "Unchanged Character refresh changed "
        "evaluation_result.",
    )

    assert_true(
        committed.character_results[1]
        == state.character_results[1],
        "Unchanged Character refresh changed sibling "
        "Character artifact.",
    )

    assert_true(
        state.model_dump(
            mode="json"
        )
        == original_dump,
        "Character commit mutated supplied state "
        "in place.",
    )

    print(
        "Unchanged Character commit preservation: PASS"
    )


def test_material_commit_invalidates_exact_dependents() -> None:
    state = build_state()
    refresh = build_material_refresh(
        state
    )

    original_dump = state.model_dump(
        mode="json"
    )

    result = (
        commit_character_refresh_to_state(
            state=state,
            refresh_result=refresh,
        )
    )

    assert_true(
        result.commit_status
        == "committed_with_invalidation",
        "Material Character refresh used wrong "
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
        "Material Character commit invalidated "
        "the wrong artifact set.",
    )

    committed = result.committed_state

    assert_true(
        committed.character_results[0]
        .analysis
        .likely_behavior
        != state.character_results[0]
        .analysis
        .likely_behavior,
        "Material Character candidate was not "
        "committed.",
    )

    assert_true(
        committed.script_result is None,
        "Material Character change did not clear "
        "script_result.",
    )

    assert_true(
        committed.continuity_result is None,
        "Material Character change did not clear "
        "continuity_result.",
    )

    assert_true(
        committed.evaluation_result is None,
        "Material Character change did not clear "
        "evaluation_result.",
    )

    assert_true(
        committed.story_state_ledger.entries
        == [],
        "Material Character change did not reset "
        "story-state ledger.",
    )

    assert_true(
        committed.lore_results
        == state.lore_results,
        "Material Character change should preserve "
        "Lore artifacts.",
    )

    assert_true(
        committed.story_plan
        == state.story_plan,
        "Material Character change should preserve "
        "StoryPlan.",
    )

    assert_true(
        committed.character_results[1]
        == state.character_results[1],
        "Material Character change unexpectedly "
        "changed sibling Character artifact.",
    )

    assert_true(
        state.model_dump(
            mode="json"
        )
        == original_dump,
        "Material Character commit mutated supplied "
        "state in place.",
    )

    print(
        "Material Character commit invalidation: PASS"
    )


def test_stale_guidance_refresh_fails_closed() -> None:
    state = build_state()
    refresh = build_material_refresh(
        state
    )

    changed_character = (
        state.character_results[0]
        .model_copy(
            update={
                "analysis": (
                    base_analysis(
                        likely_behavior=[
                            (
                                "State guidance changed "
                                "after refresh began."
                            )
                        ]
                    )
                )
            }
        )
    )

    changed_state = state.model_copy(
        update={
            "character_results": [
                changed_character,
                state.character_results[1],
            ]
        }
    )

    assert_raises(
        ValueError,
        lambda: commit_character_refresh_to_state(
            state=changed_state,
            refresh_result=refresh,
        ),
        "Refusing stale refresh commit",
    )

    print(
        "Stale Character guidance fail-closed: PASS"
    )


def test_stale_story_context_fails_closed() -> None:
    state = build_state()
    refresh = build_material_refresh(
        state
    )

    current = state.character_results[0]

    changed_plan = (
        current.research_plan.model_copy(
            update={
                "story_context": (
                    "Changed story context."
                )
            }
        )
    )

    changed_character = (
        current.model_copy(
            update={
                "research_plan": (
                    changed_plan
                )
            }
        )
    )

    changed_state = state.model_copy(
        update={
            "character_results": [
                changed_character,
                state.character_results[1],
            ]
        }
    )

    assert_raises(
        ValueError,
        lambda: commit_character_refresh_to_state(
            state=changed_state,
            refresh_result=refresh,
        ),
        "story_context changed",
    )

    print(
        "Stale Character story_context fail-closed: PASS"
    )


def test_stale_story_era_fails_closed() -> None:
    state = build_state()
    refresh = build_material_refresh(
        state
    )

    current = state.character_results[0]

    changed_plan = (
        current.research_plan.model_copy(
            update={
                "story_era": (
                    "Different story era"
                )
            }
        )
    )

    changed_character = (
        current.model_copy(
            update={
                "research_plan": (
                    changed_plan
                )
            }
        )
    )

    changed_state = state.model_copy(
        update={
            "character_results": [
                changed_character,
                state.character_results[1],
            ]
        }
    )

    assert_raises(
        ValueError,
        lambda: commit_character_refresh_to_state(
            state=changed_state,
            refresh_result=refresh,
        ),
        "story_era changed",
    )

    print(
        "Stale Character story_era fail-closed: PASS"
    )


def test_missing_target_character_fails_closed() -> None:
    state = build_state()
    refresh = build_material_refresh(
        state
    )

    missing_state = state.model_copy(
        update={
            "character_results": [
                state.character_results[1]
            ]
        }
    )

    assert_raises(
        ValueError,
        lambda: commit_character_refresh_to_state(
            state=missing_state,
            refresh_result=refresh,
        ),
        "does not contain Character artifact",
    )

    print(
        "Missing Character artifact fail-closed: PASS"
    )


def test_duplicate_target_character_fails_closed() -> None:
    state = build_state()
    refresh = build_material_refresh(
        state
    )

    duplicate_state = state.model_copy(
        update={
            "character_results": [
                state.character_results[0],
                state.character_results[0],
                state.character_results[1],
            ]
        }
    )

    assert_raises(
        ValueError,
        lambda: commit_character_refresh_to_state(
            state=duplicate_state,
            refresh_result=refresh,
        ),
        "duplicate Character artifact ID",
    )

    print(
        "Duplicate Character artifact fail-closed: PASS"
    )


def test_tampered_candidate_fails_closed() -> None:
    state = build_state()
    refresh = build_material_refresh(
        state
    )

    tampered_analysis = base_analysis(
        likely_behavior=[
            "Tampered behavior.",
        ]
    )

    tampered_candidate = (
        refresh.candidate_artifact.model_copy(
            update={
                "analysis": (
                    tampered_analysis
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
        lambda: commit_character_refresh_to_state(
            state=state,
            refresh_result=tampered_refresh,
        ),
        "candidate no longer matches",
    )

    print(
        "Tampered Character candidate fail-closed: PASS"
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
        lambda: commit_character_refresh_to_state(
            state=state,
            refresh_result=bad_refresh,
        ),
        "must not carry an invalidation plan",
    )

    print(
        "Unchanged Character+plan inconsistency "
        "fail-closed: PASS"
    )


def test_material_refresh_without_plan_fails_closed() -> None:
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
        lambda: commit_character_refresh_to_state(
            state=state,
            refresh_result=bad_refresh,
        ),
        "requires an invalidation plan",
    )

    print(
        "Material Character-without-plan "
        "fail-closed: PASS"
    )


def test_non_pydantic_state_fails_closed() -> None:
    state = build_state()
    refresh = build_material_refresh(
        state
    )

    assert_raises(
        TypeError,
        lambda: commit_character_refresh_to_state(
            state={
                "character_results": []
            },
            refresh_result=refresh,
        ),
        "Pydantic BaseModel",
    )

    print(
        "Non-Pydantic Character state fail-closed: PASS"
    )


def test_bad_ledger_default_fails_closed() -> None:
    class BadLedger(BaseModel):
        entries: list[str]

    class BadLedgerState(BaseModel):
        story_plan: str

        lore_results: list[
            FakeLoreResult
        ]

        character_results: list[
            FakeWriterRoomCharacterResult
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
        script_result=(
            base.script_result
        ),
        story_state_ledger=BadLedger(
            entries=[
                "existing"
            ]
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
        lambda: commit_character_refresh_to_state(
            state=bad_state,
            refresh_result=refresh,
        ),
        "cannot be reconstructed",
    )

    print(
        "Non-default Character ledger "
        "fail-closed: PASS"
    )


def main() -> None:
    print(
        "\n--- Character Evidence Refresh Commit "
        "Regression Test ---"
    )

    test_unchanged_commit_preserves_downstream()
    test_material_commit_invalidates_exact_dependents()
    test_stale_guidance_refresh_fails_closed()
    test_stale_story_context_fails_closed()
    test_stale_story_era_fails_closed()
    test_missing_target_character_fails_closed()
    test_duplicate_target_character_fails_closed()
    test_tampered_candidate_fails_closed()
    test_unchanged_refresh_with_invalidation_plan_fails_closed()
    test_material_refresh_without_plan_fails_closed()
    test_non_pydantic_state_fails_closed()
    test_bad_ledger_default_fails_closed()

    print(
        "\n============================================"
    )
    print(
        "CHARACTER EVIDENCE REFRESH COMMIT "
        "REGRESSION: PASS"
    )
    print(
        "============================================"
    )


if __name__ == "__main__":
    main()
