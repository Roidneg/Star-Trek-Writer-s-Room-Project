from unittest.mock import patch

from agents.story_architect import StoryPlan

from agents.lore_agent import LoreResponseModel

from agents.character_agent import (
    CharacterAnalysis,
    CharacterAnalysisAudit,
    CharacterClaimAudit,
    CharacterClaimAuditBatch,
    CharacterResearchPlan,
)

from agents.scriptwriter_agent import (
    ScriptScene,
)

from workflows.scriptwriter_production import (
    ScriptwriterProductionResult,
)

from workflows.writers_room import (
    WritersRoomCharacterResult,
    WritersRoomLoreResult,
    WritersRoomScriptCharacterRequest,
    WritersRoomScriptRequest,
    WritersRoomState,
    run_scriptwriter_stage,
)


def main():
    # -----------------------------------------------------
    # StoryPlan
    # -----------------------------------------------------

    story_plan = StoryPlan(
        working_title="The Defector's Warning",
        premise=(
            "A Cardassian defector approaches Garak with "
            "an allegation concerning Founder infiltration."
        ),
        setting="Deep Space Nine",
        timeframe="Shortly before the Dominion War",
        main_characters=[
            "Elim Garak",
            "Cardassian Defector",
        ],
        central_conflict=(
            "Garak must decide how seriously to treat "
            "the defector's allegation."
        ),
        stakes=(
            "Ignoring a genuine threat may be dangerous, "
            "while acting on false information may also "
            "carry consequences."
        ),
        lore_questions=[
            (
                "What does canon establish about Garak's "
                "knowledge of the Founders before the "
                "Dominion War?"
            ),
            (
                "What does canon establish about Cardassian "
                "political conditions shortly before the "
                "Dominion War?"
            ),
        ],
    )

    # -----------------------------------------------------
    # Lore results
    #
    # One verified result and one unresolved result.
    # Only the verified result may become Scriptwriter
    # canon evidence.
    # -----------------------------------------------------

    verified_lore = LoreResponseModel(
        topic="Garak and the Founders",
        summary=(
            "Garak participated in a mission involving "
            "the Founders before the Dominion War."
        ),
        sources=[
            "https://memory-alpha.fandom.com/wiki/Test"
        ],
        tools_used=[
            "memory_alpha_search_tool",
        ],
        evidence_supported=True,
        premise_supported=True,
    )

    unresolved_lore = LoreResponseModel(
        topic="Cardassian politics",
        summary=(
            "The retrieved evidence does not fully establish "
            "the requested political timeframe."
        ),
        sources=[
            "https://memory-alpha.fandom.com/wiki/Test_Two"
        ],
        tools_used=[
            "memory_alpha_search_tool",
        ],
        evidence_supported=False,
        premise_supported=True,
    )

    lore_results = [
        WritersRoomLoreResult(
            research_id="lore_000",
            research_question=(
                story_plan.lore_questions[0]
            ),
            response=verified_lore,
        ),
        WritersRoomLoreResult(
            research_id="lore_001",
            research_question=(
                story_plan.lore_questions[1]
            ),
            response=unresolved_lore,
        ),
    ]

    # -----------------------------------------------------
    # Grounded Character package
    # -----------------------------------------------------

    character_research_plan = (
        CharacterResearchPlan(
            character_name="Elim Garak",
            story_context=(
                "A Cardassian defector approaches Garak "
                "with an allegation concerning Founder "
                "infiltration."
            ),
            canon_questions=[
                (
                    "What does canon establish about "
                    "Garak's knowledge of the Founders?"
                ),
                (
                    "What canon evidence establishes "
                    "Garak's participation in events "
                    "involving the Founders?"
                ),
            ],
            voice_questions=[
                (
                    "What recurring sentence structures "
                    "appear in Garak's dialogue?"
                ),
                (
                    "What forms of address recur in "
                    "Garak's dialogue?"
                ),
            ],
            story_era=(
                "Star Trek: Deep Space Nine, shortly "
                "before the Dominion War"
            ),
        )
    )

    character_analysis = CharacterAnalysis(
        character_name="Elim Garak",
        story_role=(
            "Garak must evaluate the defector's allegation."
        ),
        motivations=[],
        likely_behavior=[
            (
                "Evaluate the defector's claim before "
                "deciding whether to involve DS9 command."
            ),
        ],
        unlikely_behavior=[],
        relationships=[],
        dramatic_tension=(
            "The allegation forces Garak to determine "
            "whether the information warrants action."
        ),
        remaining_canon_questions=[],
        remaining_voice_questions=[],
        speech_patterns=[],
        preferred_vocabulary=[],
        communication_style=[],
        behavioral_constraints=[
            (
                "Do not give Garak current Cardassian "
                "contacts unless supplied evidence "
                "establishes them."
            ),
        ],
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
                    "The current story context establishes "
                    "the immediate task."
                ),
            )
        ]
    )

    character_audit = CharacterAnalysisAudit(
        grounded=True,
        issues=[],
        reason=(
            "Character analysis passed its frozen "
            "grounding gate."
        ),
    )

    character_result = WritersRoomCharacterResult(
        character_id="character_000",
        character_name="Elim Garak",
        research_plan=character_research_plan,
        canon_research=(
            "VERIFIED CANON EVIDENCE\n\n"
            "Garak participated in a mission involving "
            "the Founders."
        ),
        voice_research=(
            "No dialogue or voice research has been "
            "performed yet. Treat all voice-related "
            "questions as unresolved."
        ),
        analysis=character_analysis,
        claim_audit=claim_audit,
        audit=character_audit,
    )

    # -----------------------------------------------------
    # Base Writer's Room state
    # -----------------------------------------------------

    state = WritersRoomState(
        story_idea="Test story idea.",
        story_plan=story_plan,
        lore_results=lore_results,
        character_results=[
            character_result,
        ],
    )

    # -----------------------------------------------------
    # Script request
    #
    # Garak is linked to grounded Character guidance.
    # The fictional defector is intentionally unlinked.
    # -----------------------------------------------------

    request = WritersRoomScriptRequest(
        story_context=(
            "A Cardassian defector privately approaches "
            "Garak with an allegation that the Founders "
            "have infiltrated the Cardassian government."
        ),
        scene_objective=(
            "Establish the allegation and force Garak "
            "to decide whether it deserves further attention."
        ),
        story_era=(
            "Star Trek: Deep Space Nine, shortly before "
            "the Dominion War"
        ),
        characters=[
            WritersRoomScriptCharacterRequest(
                character_name="Elim Garak",
                character_result_id="character_000",
            ),
            WritersRoomScriptCharacterRequest(
                character_name="Cardassian Defector",
                character_result_id=None,
            ),
        ],
        era_style_constraints=[
            (
                "Use screenplay conventions appropriate "
                "to Star Trek: Deep Space Nine."
            ),
        ],
        scene_requirements=[
            (
                "The Cardassian defector must present "
                "the infiltration allegation."
            ),
        ],
        prohibited_assumptions=[
            (
                "Garak has current contacts inside "
                "Cardassian Central Command."
            ),
        ],
    )

    # -----------------------------------------------------
    # Deterministic returned ScriptScene
    # -----------------------------------------------------

    scene = ScriptScene(
        scene_heading=(
            "INT. PRIVATE MEETING AREA - NIGHT"
        ),
        setting_description=(
            "A quiet alcove provides enough privacy "
            "for a guarded conversation."
        ),
        characters_present=[
            "Elim Garak",
            "Cardassian Defector",
        ],
        script_text=(
            "CARDASSIAN DEFECTOR\n"
            "I believe the Founders have infiltrated "
            "the government.\n\n"
            "ELIM GARAK\n"
            "That is a considerable accusation."
        ),
        canon_facts_used=[
            (
                "Garak participated in a mission involving "
                "the Founders before the Dominion War."
            ),
        ],
        invented_story_elements=[
            (
                "The Cardassian defector privately presents "
                "the infiltration allegation."
            ),
        ],
    )

    # -----------------------------------------------------
    # Prerequisite: StoryPlan
    # -----------------------------------------------------

    no_plan_state = WritersRoomState(
        story_idea="Test.",
    )

    try:
        run_scriptwriter_stage(
            state=no_plan_state,
            request=request,
        )

    except ValueError as error:
        assert (
            "Scriptwriter stage requires a completed StoryPlan."
            in str(error)
        )

    else:
        raise AssertionError(
            "Scriptwriter ran without a StoryPlan."
        )

    # -----------------------------------------------------
    # Prerequisite: Lore stage
    # -----------------------------------------------------

    no_lore_state = WritersRoomState(
        story_idea="Test.",
        story_plan=story_plan,
    )

    try:
        run_scriptwriter_stage(
            state=no_lore_state,
            request=request,
        )

    except ValueError as error:
        assert (
            "Scriptwriter stage requires a completed Lore stage."
            in str(error)
        )

    else:
        raise AssertionError(
            "Scriptwriter ran without Lore results."
        )

    # -----------------------------------------------------
    # Duplicate character protection
    # -----------------------------------------------------

    duplicate_request = request.model_copy(
        update={
            "characters": [
                WritersRoomScriptCharacterRequest(
                    character_name="Elim Garak",
                    character_result_id="character_000",
                ),
                WritersRoomScriptCharacterRequest(
                    character_name="elim garak",
                    character_result_id="character_000",
                ),
            ]
        }
    )

    try:
        run_scriptwriter_stage(
            state=state,
            request=duplicate_request,
        )

    except ValueError as error:
        assert (
            "Scriptwriter stage received a duplicate "
            "character request"
            in str(error)
        )

    else:
        raise AssertionError(
            "Writer's Room allowed duplicate "
            "Scriptwriter characters."
        )

    # -----------------------------------------------------
    # Unknown Character result protection
    # -----------------------------------------------------

    unknown_character_request = request.model_copy(
        update={
            "characters": [
                WritersRoomScriptCharacterRequest(
                    character_name="Elim Garak",
                    character_result_id="character_999",
                ),
            ]
        }
    )

    try:
        run_scriptwriter_stage(
            state=state,
            request=unknown_character_request,
        )

    except ValueError as error:
        assert (
            "Unknown Writer's Room Character result ID"
            in str(error)
        )

    else:
        raise AssertionError(
            "Writer's Room accepted an unknown "
            "Character result ID."
        )

    # -----------------------------------------------------
    # Identity mismatch protection
    # -----------------------------------------------------

    mismatched_request = request.model_copy(
        update={
            "characters": [
                WritersRoomScriptCharacterRequest(
                    character_name="Julian Bashir",
                    character_result_id="character_000",
                ),
            ]
        }
    )

    try:
        run_scriptwriter_stage(
            state=state,
            request=mismatched_request,
        )

    except ValueError as error:
        assert (
            "does not match the linked Character result identity"
            in str(error)
        )

    else:
        raise AssertionError(
            "Writer's Room accepted mismatched "
            "Character identity."
        )

    # -----------------------------------------------------
    # Valid orchestration
    # -----------------------------------------------------

    captured_script_input = None

    def fake_write_scene_for_production(
        script_input,
    ):
        nonlocal captured_script_input

        captured_script_input = script_input

        return ScriptwriterProductionResult(
            script_input=script_input,
            scene=scene,
            audit=(
                __import__(
                    "agents.scriptwriter_agent",
                    fromlist=["ScriptSceneAudit"],
                )
                .ScriptSceneAudit(
                    grounded=True,
                    issues=[],
                    reason=(
                        "Frozen Scriptwriter production "
                        "contract passed."
                    ),
                )
            ),
        )

    with patch(
        "workflows.writers_room."
        "write_scene_for_production",
        side_effect=fake_write_scene_for_production,
    ) as mock_scriptwriter:

        updated_state = run_scriptwriter_stage(
            state=state,
            request=request,
        )

        assert mock_scriptwriter.call_count == 1

    # -----------------------------------------------------
    # Verify constructed ScriptwriterInput
    # -----------------------------------------------------

    assert captured_script_input is not None

    assert (
        captured_script_input.story_context
        == request.story_context
    )

    assert (
        captured_script_input.scene_objective
        == request.scene_objective
    )

    assert (
        captured_script_input.story_era
        == request.story_era
    )

    # Only evidence_supported=True Lore should survive.
    assert (
        captured_script_input.verified_canon_facts
        == [
            verified_lore.summary
        ]
    )

    # Unresolved Lore must NOT be promoted.
    assert (
        unresolved_lore.summary
        not in captured_script_input.verified_canon_facts
    )

    assert (
        len(captured_script_input.characters)
        == 2
    )

    garak_context = (
        captured_script_input.characters[0]
    )

    defector_context = (
        captured_script_input.characters[1]
    )

    assert (
        garak_context.character_name
        == "Elim Garak"
    )

    assert (
        "Garak must evaluate the defector's allegation."
        in garak_context.grounded_character_guidance
    )

    assert (
        "Evaluate the defector's claim before deciding "
        "whether to involve DS9 command."
        in garak_context.grounded_character_guidance
    )

    # No voice evidence exists in the grounded analysis.
    assert (
        garak_context.grounded_voice_guidance
        == []
    )

    # Fictional character receives no fabricated
    # Character Agent guidance.
    assert (
        defector_context.grounded_character_guidance
        == []
    )

    assert (
        defector_context.grounded_voice_guidance
        == []
    )

    assert (
        captured_script_input.era_style_constraints
        == request.era_style_constraints
    )

    assert (
        captured_script_input.scene_requirements
        == request.scene_requirements
    )

    assert (
        captured_script_input.prohibited_assumptions
        == request.prohibited_assumptions
    )

    # -----------------------------------------------------
    # Result preservation
    # -----------------------------------------------------

    assert state.script_result is None

    assert updated_state.script_result is not None

    assert (
        updated_state.script_result.scene
        == scene
    )

    assert (
        updated_state.script_result.script_input
        == captured_script_input
    )

    assert (
        updated_state.script_result.audit.grounded
        is True
    )

    # -----------------------------------------------------
    # Stage overwrite protection
    # -----------------------------------------------------

    try:
        run_scriptwriter_stage(
            state=updated_state,
            request=request,
        )

    except ValueError as error:
        assert (
            "Scriptwriter stage has already completed."
            in str(error)
        )

    else:
        raise AssertionError(
            "Writer's Room allowed Scriptwriter stage "
            "to overwrite its existing artifact."
        )

    # -----------------------------------------------------
    # Output
    # -----------------------------------------------------

    print(
        "\n--- Writer's Room Scriptwriter Test ---"
    )

    print("StoryPlan prerequisite enforced: PASS")
    print("Lore prerequisite enforced: PASS")
    print("Duplicate Scriptwriter characters rejected: PASS")
    print("Unknown Character artifact rejected: PASS")
    print("Character identity mismatch rejected: PASS")
    print("Verified Lore promoted correctly: PASS")
    print("Unresolved Lore not promoted: PASS")
    print("Grounded Character guidance routed: PASS")
    print("Fictional character left ungrounded by default: PASS")
    print("ScriptwriterInput assembled correctly: PASS")
    print("Scriptwriter production workflow routed once: PASS")
    print("Complete Scriptwriter result preserved: PASS")
    print("Original state not mutated: PASS")
    print("Stage overwrite protection: PASS")

    print(
        "\nWriter's Room → Scriptwriter Agent "
        "orchestration contract: PASS"
    )


if __name__ == "__main__":
    main()