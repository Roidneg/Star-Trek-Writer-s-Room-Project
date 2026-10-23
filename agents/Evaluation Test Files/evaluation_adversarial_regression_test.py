from agents.evaluation_agent import (
    CharacterEvaluationSource,
    LoreEvaluationSource,
    review_production_subject,
)

from agents.character_agent import (
    CharacterAnalysis,
    CharacterAnalysisAudit,
)

from agents.continuity_agent import (
    ContinuitySceneAssessment,
    ContinuitySceneIssue,
    ContinuityScriptSceneAudit,
    ContinuitySceneReview,
)

from agents.lore_agent import (
    LoreResponseModel,
)

from agents.scriptwriter_agent import (
    ScriptCharacterContext,
    ScriptSceneAudit,
    ScriptSceneAuditIssue,
    ScriptwriterInput,
)


def build_script_input() -> ScriptwriterInput:
    return ScriptwriterInput(
        story_context=(
            "A Cardassian defector meets Garak "
            "with sensitive intelligence."
        ),
        scene_objective=(
            "Garak evaluates the intelligence "
            "without assuming unsupported access."
        ),
        story_era=(
            "Deep Space Nine, shortly before "
            "the Dominion War."
        ),
        characters=[
            ScriptCharacterContext(
                character_name="Elim Garak",
                grounded_character_guidance=[],
                grounded_voice_guidance=[],
            ),
        ],
        verified_canon_facts=[],
        era_style_constraints=[
            (
                "Use restrained, professional "
                "DS9-era dialogue without modern "
                "internet slang."
            ),
            (
                "Favor concise dramatic exchanges "
                "appropriate to a 1990s-era "
                "Star Trek television scene."
            ),
        ],
        scene_requirements=[],
        prohibited_assumptions=[
            (
                "Do not assume Garak currently "
                "retains contacts inside "
                "Cardassian Central Command."
            ),
        ],
    )


def build_character_source(
) -> CharacterEvaluationSource:
    analysis = CharacterAnalysis(
        character_name="Elim Garak",
        story_role=(
            "Evaluate the defector's claim "
            "before committing to action."
        ),
        motivations=[],
        likely_behavior=[
            (
                "Evaluate uncertain claims "
                "before acting on them."
            ),
        ],
        unlikely_behavior=[],
        relationships=[],
        dramatic_tension="",
        remaining_canon_questions=[
            (
                "Does Garak currently retain "
                "contacts inside Cardassian "
                "Central Command?"
            ),
        ],
        remaining_voice_questions=[],
        speech_patterns=[
            (
                "Uses indirect and carefully "
                "qualified phrasing."
            ),
        ],
        preferred_vocabulary=[],
        communication_style=[
            (
                "Often communicates through "
                "implication rather than blunt "
                "declaration."
            ),
        ],
        behavioral_constraints=[
            (
                "Do not assume current access "
                "to historical contacts."
            ),
        ],
        era_language_notes=[],
        out_of_character_risks=[],
    )

    audit = CharacterAnalysisAudit(
        grounded=True,
        issues=[],
        reason=(
            "All CharacterAnalysis claims "
            "are grounded."
        ),
    )

    return CharacterEvaluationSource(
        character_analysis=analysis,
        audit=audit,
    )


def build_unresolved_lore_source(
) -> LoreEvaluationSource:
    response = LoreResponseModel(
        topic="Elim Garak",
        summary=(
            "The retrieved evidence does not "
            "establish that Garak currently "
            "retains Central Command contacts."
        ),
        sources=[],
        tools_used=[],
        evidence_supported=False,
        premise_supported=False,
    )

    return LoreEvaluationSource(
        research_id="current_contacts",
        research_question=(
            "Does Garak currently retain "
            "contacts inside Cardassian "
            "Central Command?"
        ),
        lore_response=response,
    )


def build_clear_continuity(
) -> ContinuitySceneReview:
    return ContinuitySceneReview(
        audit=ContinuityScriptSceneAudit(
            segment_audits=[],
            has_contradiction=False,
            contradiction_count=0,
        ),
        assessment=(
            ContinuitySceneAssessment(
                status="clear",
                issues=[],
                contradiction_count=0,
                insufficient_context_count=0,
            )
        ),
    )


def build_bad_continuity(
) -> ContinuitySceneReview:
    issue = ContinuitySceneIssue(
        segment_id="script_text[1]",
        statement=(
            "Garak inserts the same data rod "
            "he still possesses into the console."
        ),
        state_id="rod_kira",
        established_state=(
            "Kira possesses the data rod."
        ),
        issue_type="contradiction",
        reason=(
            "The later statement claims Garak "
            "still possesses the same data rod "
            "without an established transfer "
            "from Kira."
        ),
    )

    return ContinuitySceneReview(
        audit=ContinuityScriptSceneAudit(
            segment_audits=[],
            has_contradiction=True,
            contradiction_count=1,
        ),
        assessment=(
            ContinuitySceneAssessment(
                status="contradiction",
                issues=[issue],
                contradiction_count=1,
                insufficient_context_count=0,
            )
        ),
    )


def build_grounded_scriptwriter_audit(
) -> ScriptSceneAudit:
    return ScriptSceneAudit(
        grounded=True,
        issues=[],
        reason=(
            "All screenplay segments are "
            "grounded or permitted."
        ),
    )


def build_bad_scriptwriter_audit(
) -> ScriptSceneAudit:
    issue = ScriptSceneAuditIssue(
        segment_id="script_text[0]",
        field="script_text",
        text=(
            "Garak says he still has a source "
            "inside Central Command."
        ),
        verdict="unsupported",
        reason=(
            "Current access to Central Command "
            "contacts was not established."
        ),
    )

    return ScriptSceneAudit(
        grounded=False,
        issues=[issue],
        reason=(
            "1 ScriptScene segment remains "
            "unsupported."
        ),
    )


def main() -> None:
    print(
        "Running Evaluation Agent v0.1N "
        "final adversarial production regression."
    )

    script_input = build_script_input()

    character_sources = [
        build_character_source(),
    ]

    lore_sources = [
        build_unresolved_lore_source(),
    ]

    # =====================================================
    # CASE 1
    #
    # Simultaneous failures:
    #
    # - Scriptwriter grounding failure
    # - Continuity contradiction
    # - Era/style violation
    # - Character-consistency violation
    #
    # Plus:
    #
    # - unresolved Lore evidence
    #
    # Final status MUST be rejected.
    # Failures take priority over insufficient evidence.
    # =====================================================

    bad_subject = (
        'GARAK\n'
        '"LOL, no worries. I still have a '
        'source inside Central Command. '
        'This whole thing is giving '
        'Dominion vibes."\n'
        "\n"
        "Garak inserts the same data rod "
        "he still possesses into the console."
    )

    rejected = review_production_subject(
        subject_text=bad_subject,
        script_input=script_input,
        scriptwriter_audit=(
            build_bad_scriptwriter_audit()
        ),
        continuity_review=(
            build_bad_continuity()
        ),
        character_sources=(
            character_sources
        ),
        lore_sources=lore_sources,
    )

    print(
        "\n===================================="
    )
    print("ADVERSARIAL REJECTED CASE")
    print(
        "===================================="
    )

    for result in (
        rejected.review.suite.results
    ):
        print(
            f"\n{result.criterion_id} "
            f"[{result.category}]"
        )
        print(
            f"Verdict: {result.verdict}"
        )
        print(
            f"Reason: {result.reason}"
        )

    rejected_assessment = (
        rejected.review.assessment
    )

    print("\n--- FINAL GATE ---")
    print(
        "Status:",
        rejected_assessment.status,
    )
    print(
        "Pass:",
        rejected_assessment.pass_count,
    )
    print(
        "Fail:",
        rejected_assessment.fail_count,
    )
    print(
        "Needs evidence:",
        rejected_assessment
        .insufficient_evidence_count,
    )
    print(
        "Blocking:",
        rejected_assessment
        .blocking_criterion_ids,
    )

    assert (
        rejected_assessment.status
        == "rejected"
    )

    assert (
        rejected_assessment.fail_count
        == 4
    )

    assert (
        rejected_assessment
        .insufficient_evidence_count
        == 1
    )

    assert (
        rejected_assessment.pass_count
        == 0
    )

    result_map = {
        result.criterion_id: result.verdict
        for result
        in rejected.review.suite.results
    }

    assert (
        result_map[
            "scriptwriter_grounding_review"
        ]
        == "fail"
    )

    assert (
        result_map[
            "continuity_agent_review"
        ]
        == "fail"
    )

    assert (
        result_map[
            "era_style_review"
        ]
        == "fail"
    )

    assert (
        result_map[
            "character_agent_guidance_elim_garak"
        ]
        == "fail"
    )

    assert (
        result_map[
            "lore_agent_evidence_current_contacts"
        ]
        == "insufficient_evidence"
    )

    # Insufficient evidence does not displace
    # explicit failures from the blocking list.
    assert set(
        rejected_assessment
        .blocking_criterion_ids
    ) == {
        "scriptwriter_grounding_review",
        "continuity_agent_review",
        "era_style_review",
        (
            "character_agent_guidance_"
            "elim_garak"
        ),
    }

    assert (
        "lore_agent_evidence_current_contacts"
        not in (
            rejected_assessment
            .blocking_criterion_ids
        )
    )

    # =====================================================
    # CASE 2
    #
    # Everything that CAN be judged passes.
    #
    # Lore still cannot establish the requested
    # current-contact premise.
    #
    # Final status MUST be needs_evidence,
    # not approved and not rejected.
    # =====================================================

    clean_subject = (
        'GARAK\n'
        '"A remarkable claim. I would prefer '
        'something more substantial before '
        'drawing conclusions."'
    )

    unresolved = review_production_subject(
        subject_text=clean_subject,
        script_input=script_input,
        scriptwriter_audit=(
            build_grounded_scriptwriter_audit()
        ),
        continuity_review=(
            build_clear_continuity()
        ),
        character_sources=(
            character_sources
        ),
        lore_sources=lore_sources,
    )

    print(
        "\n===================================="
    )
    print("NEEDS-EVIDENCE CASE")
    print(
        "===================================="
    )

    for result in (
        unresolved.review.suite.results
    ):
        print(
            f"\n{result.criterion_id} "
            f"[{result.category}]"
        )
        print(
            f"Verdict: {result.verdict}"
        )
        print(
            f"Reason: {result.reason}"
        )

    unresolved_assessment = (
        unresolved.review.assessment
    )

    print("\n--- FINAL GATE ---")
    print(
        "Status:",
        unresolved_assessment.status,
    )
    print(
        "Pass:",
        unresolved_assessment.pass_count,
    )
    print(
        "Fail:",
        unresolved_assessment.fail_count,
    )
    print(
        "Needs evidence:",
        unresolved_assessment
        .insufficient_evidence_count,
    )
    print(
        "Blocking:",
        unresolved_assessment
        .blocking_criterion_ids,
    )

    assert (
        unresolved_assessment.status
        == "needs_evidence"
    )

    assert (
        unresolved_assessment.pass_count
        == 4
    )

    assert (
        unresolved_assessment.fail_count
        == 0
    )

    assert (
        unresolved_assessment
        .insufficient_evidence_count
        == 1
    )

    assert (
        unresolved_assessment
        .blocking_criterion_ids
        == [
            (
                "lore_agent_evidence_"
                "current_contacts"
            )
        ]
    )

    # =====================================================
    # FINAL RESULT
    # =====================================================

    print(
        "\nPASS: Evaluation survived the "
        "final adversarial production "
        "regression. Multiple independent "
        "quality failures were preserved, "
        "explicit failures correctly outranked "
        "missing evidence, unresolved Lore "
        "evidence blocked approval without "
        "becoming a false failure, frozen "
        "upstream gates were not re-judged, "
        "and Python retained deterministic "
        "control of the final quality decision."
    )


if __name__ == "__main__":
    main()