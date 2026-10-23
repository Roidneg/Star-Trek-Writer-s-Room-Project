from unittest.mock import patch

from agents.story_architect import StoryPlan

from agents.character_agent import (
    CharacterAnalysis,
    CharacterAnalysisAudit,
    CharacterClaimAudit,
    CharacterClaimAuditBatch,
    CharacterResearchPlan,
)

from workflows.character_research import (
    CharacterResearchProductionResult,
    VOICE_RESEARCH_PLACEHOLDER,
)

from workflows.writers_room import (
    WritersRoomCharacterRequest,
    WritersRoomState,
    run_character_stage,
)


def main():
    # -----------------------------------------------------
    # Deterministic StoryPlan
    # -----------------------------------------------------

    story_plan = StoryPlan(
        working_title="The Defector's Warning",
        premise=(
            "Garak is approached by a Cardassian defector "
            "with an allegation concerning Founder infiltration."
        ),
        setting="Deep Space Nine",
        timeframe="Shortly before the Dominion War",
        main_characters=[
            "Elim Garak",
            "Cardassian defector",
        ],
        central_conflict=(
            "Garak must determine how seriously to treat "
            "the defector's allegation."
        ),
        stakes=(
            "Acting on false information may cause serious "
            "consequences, while ignoring a genuine threat "
            "may also carry significant risk."
        ),
        lore_questions=[
            (
                "What does canon establish about Garak's "
                "knowledge of the Founders before the Dominion War?"
            ),
            (
                "What does canon establish about Cardassian "
                "political conditions shortly before the Dominion War?"
            ),
        ],
    )

    # -----------------------------------------------------
    # Explicit Character request
    # -----------------------------------------------------

    request = WritersRoomCharacterRequest(
        character_name="Elim Garak",
        story_context=(
            "A Cardassian defector approaches Garak with "
            "an allegation concerning Founder infiltration. "
            "Garak must decide how seriously to treat the "
            "claim and whether to involve DS9 command staff."
        ),
        story_era=(
            "Star Trek: Deep Space Nine, shortly before "
            "the Dominion War"
        ),
    )

    # -----------------------------------------------------
    # Frozen Character artifacts
    # -----------------------------------------------------

    research_plan = CharacterResearchPlan(
        character_name="Elim Garak",
        story_context=request.story_context,
        canon_questions=[
            (
                "What does canon establish about Garak's "
                "knowledge of the Founders before the Dominion War?"
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
        story_era=request.story_era,
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
            "The allegation creates uncertainty over "
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
            "All CharacterAnalysis claims passed the "
            "frozen grounding gate."
        ),
    )

    production_result = (
        CharacterResearchProductionResult(
            research_plan=research_plan,
            canon_answers=[],
            combined_canon_research=(
                combined_canon_research
            ),
            voice_research=(
                VOICE_RESEARCH_PLACEHOLDER
            ),
            character_analysis=(
                character_analysis
            ),
            claim_audit=claim_audit,
            analysis_audit=analysis_audit,
        )
    )

    # -----------------------------------------------------
    # Prerequisite protection
    # -----------------------------------------------------

    empty_state = WritersRoomState(
        story_idea="Test story idea.",
    )

    try:
        run_character_stage(
            state=empty_state,
            requests=[request],
        )

    except ValueError as error:
        assert (
            "Character stage requires a completed StoryPlan."
            in str(error)
        )

    else:
        raise AssertionError(
            "Writer's Room allowed Character to run "
            "without a StoryPlan."
        )

    # -----------------------------------------------------
    # Empty request protection
    # -----------------------------------------------------

    state_with_plan = WritersRoomState(
        story_idea="Test story idea.",
        story_plan=story_plan,
    )

    try:
        run_character_stage(
            state=state_with_plan,
            requests=[],
        )

    except ValueError as error:
        assert (
            "Character stage requires at least one "
            "Character request."
            in str(error)
        )

    else:
        raise AssertionError(
            "Writer's Room allowed Character stage "
            "to run without an explicit request."
        )

    # -----------------------------------------------------
    # Duplicate request protection
    # -----------------------------------------------------

    duplicate_request = (
        WritersRoomCharacterRequest(
            character_name="elim garak",
            story_context=request.story_context,
            story_era=request.story_era,
        )
    )

    try:
        run_character_stage(
            state=state_with_plan,
            requests=[
                request,
                duplicate_request,
            ],
        )

    except ValueError as error:
        assert (
            "Character stage received a duplicate "
            "character request"
            in str(error)
        )

    else:
        raise AssertionError(
            "Writer's Room allowed duplicate "
            "Character requests."
        )

    # -----------------------------------------------------
    # Valid Character orchestration
    # -----------------------------------------------------

    with patch(
        "workflows.writers_room."
        "research_character_for_production",
        return_value=production_result,
    ) as mock_character:

        updated_state = run_character_stage(
            state=state_with_plan,
            requests=[request],
        )

        mock_character.assert_called_once_with(
            character_name="Elim Garak",
            story_context=request.story_context,
            story_era=request.story_era,
            debug=False,
        )

    # -----------------------------------------------------
    # Original state preservation
    # -----------------------------------------------------

    assert (
        state_with_plan.character_results
        == []
    )

    # -----------------------------------------------------
    # Result count
    # -----------------------------------------------------

    assert (
        len(updated_state.character_results)
        == 1
    )

    result = (
        updated_state.character_results[0]
    )

    # -----------------------------------------------------
    # Stable Writer's Room identity
    # -----------------------------------------------------

    assert (
        result.character_id
        == "character_000"
    )

    assert (
        result.character_name
        == "Elim Garak"
    )

    # -----------------------------------------------------
    # Complete Character artifact preservation
    # -----------------------------------------------------

    assert (
        result.research_plan
        == research_plan
    )

    assert (
        result.canon_research
        == combined_canon_research
    )

    assert (
        result.voice_research
        == VOICE_RESEARCH_PLACEHOLDER
    )

    assert (
        result.analysis
        == character_analysis
    )

    assert (
        result.claim_audit
        == claim_audit
    )

    assert (
        result.audit
        == analysis_audit
    )

    # -----------------------------------------------------
    # Character grounding status preserved
    # -----------------------------------------------------

    assert result.audit.grounded is True

    # -----------------------------------------------------
    # Stage overwrite protection
    # -----------------------------------------------------

    try:
        run_character_stage(
            state=updated_state,
            requests=[request],
        )

    except ValueError as error:
        assert (
            "Character stage has already completed."
            in str(error)
        )

    else:
        raise AssertionError(
            "Writer's Room allowed Character stage "
            "to overwrite existing Character artifacts."
        )

    # -----------------------------------------------------
    # Identity protection
    # -----------------------------------------------------

    mismatched_plan = research_plan.model_copy(
        update={
            "character_name": "Different Character",
        }
    )

    mismatched_result = (
        production_result.model_copy(
            update={
                "research_plan": mismatched_plan,
            }
        )
    )

    fresh_state = WritersRoomState(
        story_idea="Test story idea.",
        story_plan=story_plan,
    )

    with patch(
        "workflows.writers_room."
        "research_character_for_production",
        return_value=mismatched_result,
    ):

        try:
            run_character_stage(
                state=fresh_state,
                requests=[request],
            )

        except ValueError as error:
            assert (
                "Character production result changed "
                "the requested character identity."
                in str(error)
            )

        else:
            raise AssertionError(
                "Writer's Room accepted a Character "
                "artifact with changed identity."
            )

    # -----------------------------------------------------
    # Output
    # -----------------------------------------------------

    print(
        "\n--- Writer's Room Character Test ---"
    )

    print(
        "StoryPlan prerequisite enforced: PASS"
    )

    print(
        "Explicit Character request required: PASS"
    )

    print(
        "Duplicate Character requests rejected: PASS"
    )

    print(
        "Character production workflow routed correctly: PASS"
    )

    print(
        "Stable Character ID assigned: PASS"
    )

    print(
        "Complete Character package preserved: PASS"
    )

    print(
        "Character grounding verdict preserved: PASS"
    )

    print(
        "Original state not mutated: PASS"
    )

    print(
        "Character identity protected: PASS"
    )

    print(
        "Stage overwrite protection: PASS"
    )

    print(
        "\nWriter's Room → Character Agent "
        "orchestration contract: PASS"
    )


if __name__ == "__main__":
    main()