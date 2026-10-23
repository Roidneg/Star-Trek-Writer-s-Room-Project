from unittest.mock import patch

from agents.character_agent import (
    CharacterAnalysis,
    CharacterAnalysisAudit,
    CharacterClaimAudit,
    CharacterClaimAuditBatch,
    CharacterResearchPlan,
)

from workflows.character_research import (
    VOICE_RESEARCH_PLACEHOLDER,
    CharacterResearchProductionResult,
    research_character_for_production,
)


def main():
    # -----------------------------------------------------
    # Deterministic test artifacts
    # -----------------------------------------------------

    research_plan = CharacterResearchPlan(
        character_name="Elim Garak",
        story_context=(
            "A Cardassian defector approaches Garak with "
            "an allegation concerning Founder infiltration."
        ),
        canon_questions=[
            (
                "What does canon establish about Garak's "
                "knowledge of the Founders before the "
                "Dominion War?"
            ),
            (
                "What canon evidence establishes Garak's "
                "participation in events involving the Founders?"
            ),
        ],
        voice_questions=[
            (
                "What recurring sentence structures appear "
                "in Garak's dialogue in Deep Space Nine?"
            ),
            (
                "What forms of address recur in Garak's "
                "dialogue in Deep Space Nine?"
            ),
        ],
        story_era=(
            "Star Trek: Deep Space Nine, shortly before "
            "the Dominion War"
        ),
    )

    combined_canon_research = """
VERIFIED CANON EVIDENCE

Garak participated in a mission involving the Founders.


UNRESOLVED CANON QUESTIONS

No unresolved canon questions.


UNSUPPORTED CANON PREMISES

No unsupported canon premises found.
    """.strip()

    character_analysis = CharacterAnalysis(
        character_name="Elim Garak",
        story_role=(
            "Garak must evaluate the defector's allegation."
        ),
        motivations=[],
        likely_behavior=[],
        unlikely_behavior=[],
        relationships=[],
        dramatic_tension=(
            "The allegation creates uncertainty about "
            "whether the threat is credible."
        ),
        remaining_canon_questions=[],
        remaining_voice_questions=[
            (
                "What recurring sentence structures appear "
                "in Garak's dialogue in Deep Space Nine?"
            ),
        ],
        speech_patterns=[],
        preferred_vocabulary=[],
        communication_style=[],
        behavioral_constraints=[],
        era_language_notes=[],
        out_of_character_risks=[],
    )

    claim_audit = CharacterClaimAuditBatch(
        claims=[
            CharacterClaimAudit(
                field="story_role",
                claim=(
                    "Garak must evaluate the defector's "
                    "allegation."
                ),
                supported=True,
                evidence_excerpt=(
                    "A Cardassian defector approaches Garak "
                    "with an allegation concerning Founder "
                    "infiltration."
                ),
                reason=(
                    "The supplied story context establishes "
                    "the immediate story task."
                ),
            )
        ]
    )

    analysis_audit = CharacterAnalysisAudit(
        grounded=True,
        issues=[],
        reason=(
            "All submitted CharacterAnalysis claims passed "
            "the deterministic grounding gate."
        ),
    )

    canon_answers = [
        {
            "question": research_plan.canon_questions[0],
            "answer": "mock_lore_result_one",
        },
        {
            "question": research_plan.canon_questions[1],
            "answer": "mock_lore_result_two",
        },
    ]

    # -----------------------------------------------------
    # Mock the already-tested Character research workflow
    # and frozen audit components.
    #
    # This test is ONLY about the new production adapter.
    # -----------------------------------------------------

    with (
        patch(
            "workflows.character_research.research_character_canon",
            return_value=(
                research_plan,
                canon_answers,
                combined_canon_research,
                character_analysis,
            ),
        ) as mock_research,
        patch(
            "workflows.character_research.audit_character_claims",
            return_value=claim_audit,
        ) as mock_claim_audit,
        patch(
            "workflows.character_research."
            "validate_character_claim_audit_coverage",
        ) as mock_coverage,
        patch(
            "workflows.character_research."
            "build_character_analysis_audit",
            return_value=analysis_audit,
        ) as mock_build_audit,
    ):

        result = research_character_for_production(
            character_name="Elim Garak",
            story_context=research_plan.story_context,
            story_era=research_plan.story_era,
            debug=False,
        )

    # -----------------------------------------------------
    # Production result type
    # -----------------------------------------------------

    assert isinstance(
        result,
        CharacterResearchProductionResult,
    )

    # -----------------------------------------------------
    # Character research workflow invocation
    # -----------------------------------------------------

    mock_research.assert_called_once_with(
        character_name="Elim Garak",
        story_context=research_plan.story_context,
        story_era=research_plan.story_era,
        debug=False,
    )

    # -----------------------------------------------------
    # Frozen claim audit receives the exact artifacts
    # produced by Character research.
    # -----------------------------------------------------

    mock_claim_audit.assert_called_once_with(
        research_plan=research_plan,
        canon_research=combined_canon_research,
        voice_research=VOICE_RESEARCH_PLACEHOLDER,
        character_analysis=character_analysis,
    )

    # -----------------------------------------------------
    # Python-owned audit coverage gate must run.
    # -----------------------------------------------------

    mock_coverage.assert_called_once_with(
        character_analysis=character_analysis,
        audit_batch=claim_audit,
    )

    # -----------------------------------------------------
    # Deterministic aggregate audit must consume the
    # exact claim audit.
    # -----------------------------------------------------

    mock_build_audit.assert_called_once_with(
        claim_audit
    )

    # -----------------------------------------------------
    # Artifact preservation
    # -----------------------------------------------------

    assert result.research_plan == research_plan

    assert result.canon_answers == canon_answers

    assert (
        result.combined_canon_research
        == combined_canon_research
    )

    assert (
        result.voice_research
        == VOICE_RESEARCH_PLACEHOLDER
    )

    assert (
        result.character_analysis
        == character_analysis
    )

    assert result.claim_audit == claim_audit

    assert result.analysis_audit == analysis_audit

    # -----------------------------------------------------
    # Grounding status must be preserved rather than
    # reinterpreted by the production wrapper.
    # -----------------------------------------------------

    assert result.analysis_audit.grounded is True

    # -----------------------------------------------------
    # Output
    # -----------------------------------------------------

    print(
        "\n--- Character Research Production Test ---"
    )

    print(
        "Existing Character research workflow reused: PASS"
    )

    print(
        "Exact research artifacts routed to claim audit: PASS"
    )

    print(
        "Voice-research state preserved: PASS"
    )

    print(
        "Claim audit artifact preserved: PASS"
    )

    print(
        "Python audit-coverage gate invoked: PASS"
    )

    print(
        "Deterministic aggregate audit invoked: PASS"
    )

    print(
        "Production package preserved exactly: PASS"
    )

    print(
        "Character grounding verdict preserved: PASS"
    )

    print(
        "\nCharacter research → production adapter "
        "contract: PASS"
    )


if __name__ == "__main__":
    main()