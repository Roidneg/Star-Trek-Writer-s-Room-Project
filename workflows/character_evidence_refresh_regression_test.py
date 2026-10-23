"""
Pure-Python regression coverage for workflows.character_evidence_refresh.

Run from the project root:

    python -m workflows.character_evidence_refresh_regression_test

No provider calls are made.
No checkpoint files are changed.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from workflows.character_evidence_refresh import (
    refresh_character_evidence,
)
from workflows.evaluation_resolution import (
    EvaluationResolutionTarget,
)


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


class FakeWriterRoomCharacterResult(BaseModel):
    character_id: str = Field(
        min_length=1,
    )

    character_name: str = Field(
        min_length=1,
    )

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


def character_target(
    *,
    artifact_id: str = "character_000",
    subject_name: str = "Elim Garak",
) -> EvaluationResolutionTarget:
    return EvaluationResolutionTarget(
        criterion_id=(
            "character_agent_guidance_"
            "elim_garak"
        ),
        category="character_consistency",
        owner="character",
        action="refresh_character_evidence",
        artifact_id=artifact_id,
        subject_name=subject_name,
        reason=(
            "Regression Character evidence blocker."
        ),
    )


def base_analysis(
    *,
    character_name: str = "Elim Garak",
    likely_behavior: list[str] | None = None,
    communication_style: list[str] | None = None,
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
        communication_style=(
            communication_style
            if communication_style is not None
            else [
                "Indirect but precise.",
            ]
        ),
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


def current_character_result(
    *,
    grounded: bool = True,
    analysis: FakeCharacterAnalysis | None = None,
) -> FakeWriterRoomCharacterResult:
    return FakeWriterRoomCharacterResult(
        character_id="character_000",
        character_name="Elim Garak",
        research_plan=FakeResearchPlan(
            character_name="Elim Garak",
            story_context=(
                "Garak evaluates a damaged data rod "
                "containing an uncertain warning."
            ),
            story_era=(
                "Deep Space Nine, pre-Dominion War"
            ),
            canon_questions=[
                "How does Garak evaluate uncertain intelligence?",
            ],
            voice_questions=[
                "How does Garak speak when withholding certainty?",
            ],
        ),
        canon_research=[
            "Existing canon research.",
        ],
        voice_research=[
            "Existing voice research.",
        ],
        analysis=(
            analysis
            if analysis is not None
            else base_analysis()
        ),
        claim_audit=FakeClaimAudit(
            marker="existing-claim-audit"
        ),
        audit=FakeCharacterAudit(
            grounded=grounded,
            reason="Existing aggregate audit.",
        ),
    )


def production_result(
    *,
    grounded: bool = True,
    analysis: FakeCharacterAnalysis | None = None,
    character_name: str = "Elim Garak",
    story_context: str = (
        "Garak evaluates a damaged data rod "
        "containing an uncertain warning."
    ),
    story_era: str = (
        "Deep Space Nine, pre-Dominion War"
    ),
) -> FakeCharacterProductionResult:
    return FakeCharacterProductionResult(
        research_plan=FakeResearchPlan(
            character_name=character_name,
            story_context=story_context,
            story_era=story_era,
            # Deliberately refreshed research plan.
            canon_questions=[
                (
                    "What canonical examples show "
                    "Garak handling uncertain intelligence?"
                )
            ],
            voice_questions=[
                (
                    "What speech patterns does Garak "
                    "use when signaling doubt?"
                )
            ],
        ),
        combined_canon_research=[
            "Fresh canon research.",
        ],
        voice_research=[
            "Fresh voice research.",
        ],
        character_analysis=(
            analysis
            if analysis is not None
            else base_analysis(
                character_name=character_name
            )
        ),
        claim_audit=FakeClaimAudit(
            marker="fresh-claim-audit"
        ),
        analysis_audit=FakeCharacterAudit(
            grounded=grounded,
            reason="Fresh aggregate audit.",
        ),
    )


# ---------------------------------------------------------
# Regression Cases
# ---------------------------------------------------------


def test_unchanged_grounded_guidance_preserves_downstream() -> None:
    current = current_character_result(
        grounded=True
    )

    calls: list[
        tuple[
            str,
            str,
            str,
            bool,
        ]
    ] = []

    def fake_runner(
        *,
        character_name: str,
        story_context: str,
        story_era: str,
        debug: bool,
    ) -> FakeCharacterProductionResult:
        calls.append(
            (
                character_name,
                story_context,
                story_era,
                debug,
            )
        )

        return production_result(
            grounded=True,
            analysis=base_analysis(),
        )

    result = refresh_character_evidence(
        target=character_target(),
        current_character_result=current,
        character_runner=fake_runner,
        debug=False,
    )

    assert_true(
        len(calls) == 1,
        "Character refresh runner should be called "
        "exactly once.",
    )

    assert_true(
        calls[0][0]
        == current.character_name,
        "Character refresh changed character_name "
        "routing.",
    )

    assert_true(
        calls[0][1]
        == current.research_plan.story_context,
        "Character refresh changed story_context "
        "routing.",
    )

    assert_true(
        calls[0][2]
        == current.research_plan.story_era,
        "Character refresh changed story_era routing.",
    )

    assert_true(
        calls[0][3] is False,
        "Character refresh changed debug routing.",
    )

    assert_true(
        result.refresh_status
        == "unchanged",
        "Fresh research with identical grounded "
        "guidance should be unchanged.",
    )

    assert_true(
        result.material_change.material_change
        is False,
        "Research-only Character differences should "
        "not be material.",
    )

    assert_true(
        result.invalidation_plan is None,
        "Unchanged Character refresh should not "
        "produce an invalidation plan.",
    )

    assert_true(
        result.candidate_artifact
        .research_plan
        .canon_questions
        != current
        .research_plan
        .canon_questions,
        "Regression did not actually refresh "
        "Character research plan.",
    )

    assert_true(
        result.candidate_artifact
        .canon_research
        != current.canon_research,
        "Regression did not actually refresh "
        "Character canon research.",
    )

    print(
        "Unchanged grounded Character guidance: PASS"
    )


def test_grounded_guidance_change_builds_invalidation_plan() -> None:
    current = current_character_result(
        grounded=True
    )

    changed_analysis = base_analysis(
        likely_behavior=[
            (
                "Evaluate uncertain information "
                "before acting."
            ),
            (
                "Seek corroboration before trusting "
                "a strategically important warning."
            ),
        ]
    )

    def fake_runner(
        *,
        character_name: str,
        story_context: str,
        story_era: str,
        debug: bool,
    ) -> FakeCharacterProductionResult:
        return production_result(
            grounded=True,
            analysis=changed_analysis,
        )

    result = refresh_character_evidence(
        target=character_target(),
        current_character_result=current,
        character_runner=fake_runner,
    )

    assert_true(
        result.refresh_status
        == "material_change",
        "Changed grounded Character guidance should "
        "be material.",
    )

    assert_true(
        result.material_change.material_change
        is True,
        "Grounded Character guidance change was not "
        "reported as material.",
    )

    assert_true(
        "likely_behavior"
        in result.material_change.changed_fields,
        "Changed Character behavior guidance was "
        "not reported.",
    )

    assert_true(
        result.invalidation_plan is not None,
        "Material Character refresh should attach "
        "an invalidation plan.",
    )

    assert_true(
        result.invalidation_plan
        .invalidated_artifacts
        == [
            "script_result",
            "continuity_result",
            "story_state_ledger",
            "evaluation_result",
        ],
        "Material Character refresh produced the "
        "wrong invalidation set.",
    )

    assert_true(
        result.invalidation_plan
        .requires_ledger_rebuild
        is True,
        "Material Character refresh should require "
        "ledger rebuild.",
    )

    print(
        "Material Character guidance invalidation: PASS"
    )


def test_character_grounding_transition_is_material() -> None:
    current = current_character_result(
        grounded=False,
        analysis=base_analysis(
            likely_behavior=[
                "Old speculative behavior.",
            ]
        ),
    )

    def fake_runner(
        *,
        character_name: str,
        story_context: str,
        story_era: str,
        debug: bool,
    ) -> FakeCharacterProductionResult:
        return production_result(
            grounded=True,
            analysis=base_analysis(),
        )

    result = refresh_character_evidence(
        target=character_target(),
        current_character_result=current,
        character_runner=fake_runner,
    )

    assert_true(
        result.refresh_status
        == "material_change",
        "Character ungrounded→grounded transition "
        "must be material.",
    )

    assert_true(
        "grounded"
        in result.material_change.changed_fields,
        "Grounding transition was not reported.",
    )

    print(
        "Character grounding transition: PASS"
    )


def test_grounded_to_ungrounded_refresh_fails_closed() -> None:
    current = current_character_result(
        grounded=True,
        analysis=base_analysis(),
    )

    def fake_runner(
        *,
        character_name: str,
        story_context: str,
        story_era: str,
        debug: bool,
    ) -> FakeCharacterProductionResult:
        return production_result(
            grounded=False,
            analysis=base_analysis(
                likely_behavior=[
                    "Speculative degraded behavior.",
                ]
            ),
        )

    assert_raises(
        ValueError,
        lambda: refresh_character_evidence(
            target=character_target(),
            current_character_result=current,
            character_runner=fake_runner,
        ),
        "degraded an already-grounded authoritative artifact",
    )

    print(
        "Grounded→ungrounded Character degradation "
        "fail-closed: PASS"
    )


def test_ungrounded_analysis_change_is_not_material() -> None:
    current = current_character_result(
        grounded=False,
        analysis=base_analysis(
            likely_behavior=[
                "Old speculative behavior.",
            ]
        ),
    )

    def fake_runner(
        *,
        character_name: str,
        story_context: str,
        story_era: str,
        debug: bool,
    ) -> FakeCharacterProductionResult:
        return production_result(
            grounded=False,
            analysis=base_analysis(
                likely_behavior=[
                    "Different speculative behavior.",
                ],
                communication_style=[
                    "Different speculative voice.",
                ],
            ),
        )

    result = refresh_character_evidence(
        target=character_target(),
        current_character_result=current,
        character_runner=fake_runner,
    )

    assert_true(
        result.refresh_status
        == "unchanged",
        "Ungrounded Character analysis differences "
        "should not be material.",
    )

    assert_true(
        result.invalidation_plan is None,
        "Ungrounded analysis-only changes should "
        "not invalidate downstream artifacts.",
    )

    print(
        "Ungrounded Character wording exclusion: PASS"
    )


def test_target_artifact_mismatch_fails_closed() -> None:
    current = current_character_result()

    assert_raises(
        ValueError,
        lambda: refresh_character_evidence(
            target=character_target(
                artifact_id="character_999"
            ),
            current_character_result=current,
            character_runner=(
                lambda **kwargs: production_result()
            ),
        ),
        "does not match the supplied "
        "Writer's Room Character artifact",
    )

    print(
        "Character target/artifact mismatch "
        "fail-closed: PASS"
    )


def test_target_subject_mismatch_fails_closed() -> None:
    current = current_character_result()

    assert_raises(
        ValueError,
        lambda: refresh_character_evidence(
            target=character_target(
                subject_name="Benjamin Sisko"
            ),
            current_character_result=current,
            character_runner=(
                lambda **kwargs: production_result()
            ),
        ),
        "subject_name does not match",
    )

    print(
        "Character target subject mismatch "
        "fail-closed: PASS"
    )


def test_non_character_target_fails_closed() -> None:
    bad_target = EvaluationResolutionTarget(
        criterion_id=(
            "lore_agent_evidence_lore_000"
        ),
        category="canon_grounding",
        owner="lore",
        action="refresh_lore_evidence",
        artifact_id="lore_000",
        subject_name=None,
        reason="Wrong refresh family.",
    )

    assert_raises(
        ValueError,
        lambda: refresh_character_evidence(
            target=bad_target,
            current_character_result=(
                current_character_result()
            ),
            character_runner=(
                lambda **kwargs: production_result()
            ),
        ),
        "category='character_consistency'",
    )

    print(
        "Non-Character target fail-closed: PASS"
    )


def test_none_runner_result_fails_closed() -> None:
    current = current_character_result()

    def fake_runner(
        *,
        character_name: str,
        story_context: str,
        story_era: str,
        debug: bool,
    ):
        return None

    assert_raises(
        ValueError,
        lambda: refresh_character_evidence(
            target=character_target(),
            current_character_result=current,
            character_runner=fake_runner,
        ),
        "returned None",
    )

    print(
        "None Character production result "
        "fail-closed: PASS"
    )


def test_wrong_runner_signature_fails_closed() -> None:
    current = current_character_result()

    def bad_runner(
        character_name: str,
    ) -> FakeCharacterProductionResult:
        return production_result()

    assert_raises(
        TypeError,
        lambda: refresh_character_evidence(
            target=character_target(),
            current_character_result=current,
            character_runner=bad_runner,
        ),
        "frozen signature",
    )

    print(
        "Character runner signature fail-closed: PASS"
    )


def test_changed_story_context_fails_closed() -> None:
    current = current_character_result()

    def fake_runner(
        *,
        character_name: str,
        story_context: str,
        story_era: str,
        debug: bool,
    ) -> FakeCharacterProductionResult:
        return production_result(
            story_context=(
                "Different story context."
            )
        )

    assert_raises(
        ValueError,
        lambda: refresh_character_evidence(
            target=character_target(),
            current_character_result=current,
            character_runner=fake_runner,
        ),
        "changed the exact story_context",
    )

    print(
        "Changed Character story_context "
        "fail-closed: PASS"
    )


def test_changed_story_era_fails_closed() -> None:
    current = current_character_result()

    def fake_runner(
        *,
        character_name: str,
        story_context: str,
        story_era: str,
        debug: bool,
    ) -> FakeCharacterProductionResult:
        return production_result(
            story_era=(
                "The Next Generation"
            )
        )

    assert_raises(
        ValueError,
        lambda: refresh_character_evidence(
            target=character_target(),
            current_character_result=current,
            character_runner=fake_runner,
        ),
        "changed the exact story_era",
    )

    print(
        "Changed Character story_era "
        "fail-closed: PASS"
    )


def test_changed_character_name_fails_closed() -> None:
    current = current_character_result()

    def fake_runner(
        *,
        character_name: str,
        story_context: str,
        story_era: str,
        debug: bool,
    ) -> FakeCharacterProductionResult:
        return production_result(
            character_name="Benjamin Sisko",
            analysis=base_analysis(
                character_name="Benjamin Sisko"
            ),
        )

    assert_raises(
        ValueError,
        lambda: refresh_character_evidence(
            target=character_target(),
            current_character_result=current,
            character_runner=fake_runner,
        ),
        "changed the exact character_name",
    )

    print(
        "Changed Character name fail-closed: PASS"
    )


def test_malformed_production_package_fails_closed() -> None:
    current = current_character_result()

    class IncompleteProductionResult(BaseModel):
        research_plan: FakeResearchPlan

    def fake_runner(
        *,
        character_name: str,
        story_context: str,
        story_era: str,
        debug: bool,
    ) -> IncompleteProductionResult:
        return IncompleteProductionResult(
            research_plan=FakeResearchPlan(
                character_name=character_name,
                story_context=story_context,
                story_era=story_era,
            )
        )

    assert_raises(
        TypeError,
        lambda: refresh_character_evidence(
            target=character_target(),
            current_character_result=current,
            character_runner=fake_runner,
        ),
        "missing required attribute",
    )

    print(
        "Malformed Character production package "
        "fail-closed: PASS"
    )


def test_non_pydantic_character_artifact_fails_closed() -> None:
    class PlainCharacterArtifact:
        def __init__(self) -> None:
            self.character_id = "character_000"
            self.character_name = "Elim Garak"
            self.research_plan = FakeResearchPlan(
                character_name="Elim Garak",
                story_context=(
                    "Garak evaluates a damaged data rod "
                    "containing an uncertain warning."
                ),
                story_era=(
                    "Deep Space Nine, pre-Dominion War"
                ),
            )
            self.analysis = base_analysis()
            self.audit = FakeCharacterAudit(
                grounded=True
            )

    plain = PlainCharacterArtifact()

    def fake_runner(
        *,
        character_name: str,
        story_context: str,
        story_era: str,
        debug: bool,
    ) -> FakeCharacterProductionResult:
        return production_result()

    assert_raises(
        TypeError,
        lambda: refresh_character_evidence(
            target=character_target(),
            current_character_result=plain,
            character_runner=fake_runner,
        ),
        "Pydantic BaseModel",
    )

    print(
        "Non-Pydantic Character artifact "
        "fail-closed: PASS"
    )


def main() -> None:
    print(
        "\n--- Character Evidence Refresh Regression Test ---"
    )

    test_unchanged_grounded_guidance_preserves_downstream()
    test_grounded_guidance_change_builds_invalidation_plan()
    test_character_grounding_transition_is_material()
    test_grounded_to_ungrounded_refresh_fails_closed()
    test_ungrounded_analysis_change_is_not_material()

    test_target_artifact_mismatch_fails_closed()
    test_target_subject_mismatch_fails_closed()
    test_non_character_target_fails_closed()
    test_none_runner_result_fails_closed()
    test_wrong_runner_signature_fails_closed()

    test_changed_story_context_fails_closed()
    test_changed_story_era_fails_closed()
    test_changed_character_name_fails_closed()

    test_malformed_production_package_fails_closed()
    test_non_pydantic_character_artifact_fails_closed()

    print(
        "\n============================================"
    )
    print(
        "CHARACTER EVIDENCE REFRESH REGRESSION: PASS"
    )
    print(
        "============================================"
    )


if __name__ == "__main__":
    main()
