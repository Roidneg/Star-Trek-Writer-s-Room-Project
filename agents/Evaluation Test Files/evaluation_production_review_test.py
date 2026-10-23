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
    ContinuityScriptSceneAudit,
    ContinuitySceneReview,
)

from agents.lore_agent import (
    LoreResponseModel,
)

from agents.scriptwriter_agent import (
    ScriptCharacterContext,
    ScriptSceneAudit,
    ScriptwriterInput,
)


def main() -> None:
    print(
        "Running Evaluation Agent v0.1M "
        "end-to-end production review test."
    )

    # =====================================================
    # Scriptwriter input
    # =====================================================

    script_input = ScriptwriterInput(
        story_context=(
            "A Cardassian defector meets Garak "
            "privately with sensitive intelligence."
        ),
        scene_objective=(
            "Garak evaluates the defector's claim "
            "without committing too quickly."
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
        prohibited_assumptions=[],
    )

    # =====================================================
    # Frozen Scriptwriter gate
    # =====================================================

    scriptwriter_audit = ScriptSceneAudit(
        grounded=True,
        issues=[],
        reason=(
            "All screenplay segments are "
            "grounded or permitted."
        ),
    )

    # =====================================================
    # Frozen Continuity gate
    # =====================================================

    continuity_review = ContinuitySceneReview(
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

    # =====================================================
    # Frozen Character output
    # =====================================================

    character_analysis = CharacterAnalysis(
        character_name="Elim Garak",
        story_role=(
            "Evaluate the Cardassian defector's "
            "claim before committing to action."
        ),
        motivations=[],
        likely_behavior=[
            (
                "Evaluate the claim carefully "
                "before committing to action."
            ),
        ],
        unlikely_behavior=[],
        relationships=[],
        dramatic_tension="",
        remaining_canon_questions=[],
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
        behavioral_constraints=[],
        era_language_notes=[],
        out_of_character_risks=[],
    )

    character_audit = (
        CharacterAnalysisAudit(
            grounded=True,
            issues=[],
            reason=(
                "All CharacterAnalysis claims "
                "are grounded."
            ),
        )
    )

    # =====================================================
    # Frozen Lore output
    # =====================================================

    lore_response = LoreResponseModel(
        topic="Elim Garak",
        summary=(
            "Garak participated in a joint "
            "Obsidian Order and Tal Shiar "
            "operation against the Founders."
        ),
        sources=[],
        tools_used=[],
        evidence_supported=True,
        premise_supported=True,
    )

    # =====================================================
    # Subject under final Evaluation
    # =====================================================

    subject_text = (
        "GARAK\n"
        '"A remarkable claim. I assume you '
        "brought something more substantial "
        'than your assurances?"'
    )

    result = review_production_subject(
        subject_text=subject_text,
        script_input=script_input,
        scriptwriter_audit=(
            scriptwriter_audit
        ),
        continuity_review=(
            continuity_review
        ),
        character_sources=[
            CharacterEvaluationSource(
                character_analysis=(
                    character_analysis
                ),
                audit=character_audit,
            ),
        ],
        lore_sources=[
            LoreEvaluationSource(
                research_id="garak_founders",
                research_question=(
                    "What does canon establish "
                    "about Garak's prior "
                    "involvement with the "
                    "Founders?"
                ),
                lore_response=lore_response,
            ),
        ],
    )

    print("\n--- MANIFEST ---")
    print(
        "LLM criteria:",
        len(result.manifest.criteria),
    )
    print(
        "Precomputed:",
        len(
            result.manifest
            .precomputed_results
        ),
    )

    print("\n--- FINAL RESULTS ---")

    for item in result.review.suite.results:
        print(
            f"\n{item.criterion_id} "
            f"[{item.category}]"
        )
        print(
            f"Verdict: {item.verdict}"
        )
        print(
            f"Reason: {item.reason}"
        )

    assessment = result.review.assessment

    print("\n--- FINAL QUALITY GATE ---")
    print(
        f"Status: {assessment.status}"
    )
    print(
        f"Pass: {assessment.pass_count}"
    )
    print(
        f"Fail: {assessment.fail_count}"
    )
    print(
        "Needs evidence: "
        f"{assessment.insufficient_evidence_count}"
    )
    print(
        "Blocking:",
        assessment.blocking_criterion_ids,
    )

    # =====================================================
    # Full production expectations
    # =====================================================

    assert len(
        result.manifest.criteria
    ) == 3

    assert len(
        result.manifest.precomputed_results
    ) == 2

    assert len(
        result.review.suite.results
    ) == 5

    categories = {
        item.category
        for item
        in result.review.suite.results
    }

    assert categories == {
        "canon_grounding",
        "character_consistency",
        "era_style",
        "upstream_constraint",
        "continuity",
    }

    assert assessment.status == "approved"

    assert assessment.pass_count == 5
    assert assessment.fail_count == 0

    assert (
        assessment
        .insufficient_evidence_count
        == 0
    )

    assert (
        assessment.blocking_criterion_ids
        == []
    )

    print(
        "\nPASS: Evaluation accepted real "
        "frozen upstream object types through "
        "one production-facing API, evaluated "
        "only unresolved semantic dimensions, "
        "preserved authoritative upstream "
        "quality gates, and returned one "
        "deterministic final approval decision."
    )


if __name__ == "__main__":
    main()