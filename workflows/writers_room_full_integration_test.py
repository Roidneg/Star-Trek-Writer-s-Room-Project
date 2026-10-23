from unittest.mock import patch

from agents.story_architect import (
    StoryPlan,
)

from agents.lore_agent import (
    LoreResponseModel,
)

from agents.character_agent import (
    CharacterAnalysis,
    CharacterAnalysisAudit,
    CharacterClaimAudit,
    CharacterClaimAuditBatch,
    CharacterResearchPlan,
)

from agents.scriptwriter_agent import (
    ScriptScene,
    ScriptSceneAudit,
)

from agents.continuity_agent import (
    ContinuitySceneAssessment,
    ContinuitySceneLearningResult,
    ContinuitySceneReview,
    ContinuityScriptSceneAudit,
    StoryStateEntry,
    StoryStateLedger,
)

from agents.evaluation_agent import (
    EvaluationAssessment,
    EvaluationCriterionResult,
    EvaluationManifest,
    EvaluationProductionReview,
    EvaluationReview,
    EvaluationSuiteResult,
)

from workflows.character_research import (
    CharacterResearchProductionResult,
    VOICE_RESEARCH_PLACEHOLDER,
)

from workflows.scriptwriter_production import (
    ScriptwriterProductionResult,
)

from workflows.continuity_production import (
    ContinuityProductionResult,
)

from workflows.writers_room import (
    WritersRoomCharacterRequest,
    WritersRoomScriptCharacterRequest,
    WritersRoomScriptRequest,
    WritersRoomState,
    run_character_stage,
    run_continuity_stage,
    run_evaluation_stage,
    run_lore_stage,
    run_scriptwriter_stage,
    run_story_architect_stage,
)


def main():
    # =====================================================
    # DETERMINISTIC SPECIALIST ARTIFACTS
    # =====================================================

    # -----------------------------------------------------
    # Story Architect
    # -----------------------------------------------------

    story_plan = StoryPlan(
        working_title="The Defector's Warning",
        premise=(
            "A Cardassian defector secretly approaches "
            "Garak with an allegation that the Founders "
            "have infiltrated the Cardassian government."
        ),
        setting="Deep Space Nine",
        timeframe="Shortly before the Dominion War",
        main_characters=[
            "Elim Garak",
            "Cardassian Defector",
        ],
        central_conflict=(
            "Garak must determine whether the allegation "
            "deserves further investigation."
        ),
        stakes=(
            "Ignoring a genuine threat could be dangerous, "
            "while acting on false information could create "
            "serious consequences."
        ),
        lore_questions=[
            (
                "What does canon establish about Garak's "
                "experience with the Founders before the "
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
    # Lore
    #
    # One verified result and one unresolved result.
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
        topic="Cardassian political conditions",
        summary=(
            "The retrieved evidence does not fully establish "
            "the requested political context."
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

    # -----------------------------------------------------
    # Character
    # -----------------------------------------------------

    character_request = WritersRoomCharacterRequest(
        character_name="Elim Garak",
        story_context=(
            "A Cardassian defector secretly approaches "
            "Garak with an allegation that the Founders "
            "have infiltrated the Cardassian government."
        ),
        story_era=(
            "Star Trek: Deep Space Nine, shortly before "
            "the Dominion War"
        ),
    )

    character_research_plan = CharacterResearchPlan(
        character_name="Elim Garak",
        story_context=character_request.story_context,
        story_era=character_request.story_era,
        canon_questions=[
            (
                "What does canon establish about Garak's "
                "experience with the Founders?"
            ),
            (
                "What canon evidence establishes Garak's "
                "participation in events involving "
                "the Founders?"
            ),
        ],
        voice_questions=[
            (
                "What recurring sentence structures appear "
                "in Garak's dialogue?"
            ),
            (
                "What forms of address recur in "
                "Garak's dialogue?"
            ),
        ],
    )

    character_analysis = CharacterAnalysis(
        character_name="Elim Garak",
        story_role=(
            "Garak must evaluate the defector's allegation."
        ),
        motivations=[],
        likely_behavior=[
            (
                "Evaluate the allegation before deciding "
                "whether further action is warranted."
            ),
        ],
        unlikely_behavior=[],
        relationships=[],
        dramatic_tension=(
            "Garak must determine whether the allegation "
            "is credible enough to pursue."
        ),
        remaining_canon_questions=[],
        remaining_voice_questions=[
            (
                "What recurring sentence structures appear "
                "in Garak's dialogue?"
            ),
        ],
        speech_patterns=[],
        preferred_vocabulary=[],
        communication_style=[],
        behavioral_constraints=[
            (
                "Do not give Garak current contacts inside "
                "Cardassian Central Command without "
                "supporting evidence."
            ),
        ],
        era_language_notes=[],
        out_of_character_risks=[],
    )

    character_claim_audit = CharacterClaimAuditBatch(
        claims=[
            CharacterClaimAudit(
                field="story_role",
                claim=(
                    "Garak must evaluate the defector's "
                    "allegation."
                ),
                supported=True,
                evidence_excerpt=(
                    "A Cardassian defector secretly "
                    "approaches Garak with an allegation."
                ),
                reason=(
                    "The supplied story context establishes "
                    "the immediate dramatic task."
                ),
            )
        ]
    )

    character_analysis_audit = CharacterAnalysisAudit(
        grounded=True,
        issues=[],
        reason=(
            "Character analysis passed the frozen "
            "grounding gate."
        ),
    )

    character_production_result = (
        CharacterResearchProductionResult(
            research_plan=character_research_plan,
            canon_answers=[],
            combined_canon_research=(
                "VERIFIED CANON EVIDENCE\n\n"
                "Garak participated in a mission involving "
                "the Founders before the Dominion War.\n\n"
                "UNRESOLVED CANON QUESTIONS\n\n"
                "No unresolved questions.\n\n"
                "UNSUPPORTED CANON PREMISES\n\n"
                "No unsupported premises."
            ),
            voice_research=(
                VOICE_RESEARCH_PLACEHOLDER
            ),
            character_analysis=(
                character_analysis
            ),
            claim_audit=(
                character_claim_audit
            ),
            analysis_audit=(
                character_analysis_audit
            ),
        )
    )

    # -----------------------------------------------------
    # Scriptwriter request
    # -----------------------------------------------------

    script_request = WritersRoomScriptRequest(
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
            (
                "The defector must give Garak a data rod."
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
    # Scriptwriter scene
    # -----------------------------------------------------

    script_scene = ScriptScene(
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
            "The defector removes a data rod and places "
            "it in Garak's hand.\n\n"
            "CARDASSIAN DEFECTOR\n"
            "Everything I know is on that rod.\n\n"
            "ELIM GARAK\n"
            "That is a considerable accusation."
        ),
        canon_facts_used=[
            verified_lore.summary,
        ],
        invented_story_elements=[
            (
                "A fictional Cardassian defector "
                "approaches Garak."
            ),
            (
                "The defector gives Garak a data rod."
            ),
        ],
    )

    script_audit = ScriptSceneAudit(
        grounded=True,
        issues=[],
        reason=(
            "Frozen Scriptwriter production "
            "grounding gate passed."
        ),
    )

    # -----------------------------------------------------
    # Continuity
    #
    # Scene causes one durable state transition.
    # -----------------------------------------------------

    starting_ledger = StoryStateLedger(
        entries=[]
    )

    continuity_review = ContinuitySceneReview(
        audit=ContinuityScriptSceneAudit(
            segment_audits=[],
            has_contradiction=False,
            contradiction_count=0,
        ),
        assessment=ContinuitySceneAssessment(
            status="clear",
            issues=[],
            contradiction_count=0,
            insufficient_context_count=0,
        ),
    )

    updated_ledger = StoryStateLedger(
        entries=[
            StoryStateEntry(
                state_id="scene_0000_script_text_000",
                scene_index=0,
                statement=(
                    "The defector removes a data rod "
                    "and places it in Garak's hand."
                ),
                normalized_state=(
                    "Garak possesses the data rod."
                ),
            )
        ]
    )

    continuity_learning_result = (
        ContinuitySceneLearningResult(
            committed_state_ids=[
                "scene_0000_script_text_000",
            ],
            transient_segment_ids=[],
            ledger=updated_ledger,
        )
    )

    continuity_production_result = (
        ContinuityProductionResult(
            scene_index=0,
            review=continuity_review,
            previous_ledger=starting_ledger,
            updated_ledger=updated_ledger,
            learning_result=(
                continuity_learning_result
            ),
        )
    )

    # -----------------------------------------------------
    # Evaluation
    #
    # Deliberately use NEEDS_EVIDENCE instead of APPROVED.
    #
    # This proves Writer's Room preserves the evaluator's
    # verdict rather than converting successful orchestration
    # into automatic approval.
    # -----------------------------------------------------

    unresolved_criterion = EvaluationCriterionResult(
        criterion_id="lore_001",
        category="canon_grounding",
        criterion=(
            "The requested Cardassian political context "
            "must have sufficient canon evidence."
        ),
        verdict="insufficient_evidence",
        reason=(
            "The Lore Agent returned unresolved evidence."
        ),
    )

    evaluation_manifest = EvaluationManifest(
        criteria=[],
        precomputed_results=[
            unresolved_criterion,
        ],
    )

    evaluation_suite = EvaluationSuiteResult(
        results=[
            unresolved_criterion,
        ],
        pass_count=0,
        fail_count=0,
        insufficient_evidence_count=1,
        has_failure=False,
    )

    evaluation_assessment = EvaluationAssessment(
        status="needs_evidence",
        pass_count=0,
        fail_count=0,
        insufficient_evidence_count=1,
        blocking_criterion_ids=[
            "lore_001",
        ],
    )

    evaluation_review = EvaluationReview(
        suite=evaluation_suite,
        assessment=evaluation_assessment,
    )

    evaluation_production_result = (
        EvaluationProductionReview(
            manifest=evaluation_manifest,
            review=evaluation_review,
        )
    )

    # =====================================================
    # ADVERSARIAL CHECK 1:
    # BAD STAGE ORDER
    # =====================================================

    initial_state = WritersRoomState(
        story_idea=(
            "A Cardassian defector approaches Garak "
            "with a warning about Founder infiltration."
        ),
        story_state_ledger=starting_ledger,
    )

    with patch(
        "workflows.writers_room."
        "write_scene_for_production"
    ) as mock_scriptwriter:

        try:
            run_scriptwriter_stage(
                state=initial_state,
                request=script_request,
            )

        except ValueError as error:
            assert (
                "Scriptwriter stage requires a completed "
                "StoryPlan."
                in str(error)
            )

        else:
            raise AssertionError(
                "Writer's Room allowed Scriptwriter "
                "to run out of stage order."
            )

        mock_scriptwriter.assert_not_called()

    # =====================================================
    # ADVERSARIAL CHECK 2:
    # MIDWORKFLOW SPECIALIST FAILURE
    #
    # Story Architect and Lore succeed.
    # Character hard-fails.
    #
    # Existing state must remain intact.
    # No downstream artifacts magically appear.
    # =====================================================

    with (
        patch(
            "workflows.writers_room.create_story_plan",
            return_value=story_plan,
        ),
        patch(
            "workflows.writers_room.run_lore_query",
            side_effect=[
                verified_lore,
                unresolved_lore,
            ],
        ),
    ):
        failure_state = run_story_architect_stage(
            initial_state
        )

        failure_state = run_lore_stage(
            failure_state
        )

    before_character_failure = failure_state

    with patch(
        "workflows.writers_room."
        "research_character_for_production",
        side_effect=ValueError(
            "Character production grounding failed."
        ),
    ):

        try:
            run_character_stage(
                state=before_character_failure,
                requests=[
                    character_request,
                ],
            )

        except ValueError as error:
            assert (
                "Character production grounding failed."
                in str(error)
            )

        else:
            raise AssertionError(
                "Writer's Room swallowed a Character "
                "production failure."
            )

    assert (
        before_character_failure.character_results
        == []
    )

    assert (
        before_character_failure.script_result
        is None
    )

    assert (
        before_character_failure.continuity_result
        is None
    )

    assert (
        before_character_failure.evaluation_result
        is None
    )

    # =====================================================
    # FULL DETERMINISTIC ROOM RUN
    # =====================================================

    # -----------------------------------------------------
    # Stage 1: Story Architect
    # -----------------------------------------------------

    with patch(
        "workflows.writers_room.create_story_plan",
        return_value=story_plan,
    ) as mock_story_architect:

        state_after_architect = (
            run_story_architect_stage(
                initial_state
            )
        )

        mock_story_architect.assert_called_once_with(
            initial_state.story_idea
        )

    # Initial state remains untouched.
    assert initial_state.story_plan is None

    assert (
        state_after_architect.story_plan
        == story_plan
    )

    # -----------------------------------------------------
    # Stage 2: Lore
    # -----------------------------------------------------

    with patch(
        "workflows.writers_room.run_lore_query",
        side_effect=[
            verified_lore,
            unresolved_lore,
        ],
    ) as mock_lore:

        state_after_lore = run_lore_stage(
            state_after_architect
        )

        assert mock_lore.call_count == 2

    assert (
        state_after_architect.lore_results
        == []
    )

    assert (
        len(state_after_lore.lore_results)
        == 2
    )

    assert (
        state_after_lore
        .lore_results[0]
        .research_id
        == "lore_000"
    )

    assert (
        state_after_lore
        .lore_results[1]
        .research_id
        == "lore_001"
    )

    assert (
        state_after_lore
        .lore_results[0]
        .response
        == verified_lore
    )

    assert (
        state_after_lore
        .lore_results[1]
        .response
        == unresolved_lore
    )

    # -----------------------------------------------------
    # Stage 3: Character
    # -----------------------------------------------------

    with patch(
        "workflows.writers_room."
        "research_character_for_production",
        return_value=(
            character_production_result
        ),
    ) as mock_character:

        state_after_character = (
            run_character_stage(
                state=state_after_lore,
                requests=[
                    character_request,
                ],
            )
        )

        mock_character.assert_called_once_with(
            character_name="Elim Garak",
            story_context=(
                character_request.story_context
            ),
            story_era=(
                character_request.story_era
            ),
            debug=False,
        )

    assert (
        state_after_lore.character_results
        == []
    )

    assert (
        len(
            state_after_character
            .character_results
        )
        == 1
    )

    assert (
        state_after_character
        .character_results[0]
        .character_id
        == "character_000"
    )

    assert (
        state_after_character
        .character_results[0]
        .analysis
        == character_analysis
    )

    assert (
        state_after_character
        .character_results[0]
        .audit
        == character_analysis_audit
    )

    # -----------------------------------------------------
    # Stage 4: Scriptwriter
    #
    # Capture the exact input assembled from upstream state.
    # -----------------------------------------------------

    captured_script_input = {}

    def fake_write_scene_for_production(
        script_input,
    ):
        captured_script_input["value"] = (
            script_input
        )

        return ScriptwriterProductionResult(
            script_input=script_input,
            scene=script_scene,
            audit=script_audit,
        )

    with patch(
        "workflows.writers_room."
        "write_scene_for_production",
        side_effect=(
            fake_write_scene_for_production
        ),
    ) as mock_scriptwriter:

        state_after_scriptwriter = (
            run_scriptwriter_stage(
                state=state_after_character,
                request=script_request,
            )
        )

        assert mock_scriptwriter.call_count == 1

    assembled_script_input = (
        captured_script_input["value"]
    )

    # Only supported Lore becomes verified canon.
    assert (
        assembled_script_input
        .verified_canon_facts
        == [
            verified_lore.summary,
        ]
    )

    assert (
        unresolved_lore.summary
        not in (
            assembled_script_input
            .verified_canon_facts
        )
    )

    # Garak receives grounded Character guidance.
    garak_context = (
        assembled_script_input.characters[0]
    )

    assert (
        garak_context.character_name
        == "Elim Garak"
    )

    assert (
        character_analysis.story_role
        in (
            garak_context
            .grounded_character_guidance
        )
    )

    # Fictional character receives no fake evidence.
    defector_context = (
        assembled_script_input.characters[1]
    )

    assert (
        defector_context
        .grounded_character_guidance
        == []
    )

    assert (
        defector_context
        .grounded_voice_guidance
        == []
    )

    assert (
        state_after_character.script_result
        is None
    )

    assert (
        state_after_scriptwriter
        .script_result
        .scene
        == script_scene
    )

    # -----------------------------------------------------
    # Stage 5: Continuity
    # -----------------------------------------------------

    with patch(
        "workflows.writers_room."
        "review_scene_for_production",
        return_value=(
            continuity_production_result
        ),
    ) as mock_continuity:

        state_after_continuity = (
            run_continuity_stage(
                state=state_after_scriptwriter,
                scene_index=0,
            )
        )

        mock_continuity.assert_called_once_with(
            ledger=starting_ledger,
            scene=script_scene,
            scene_index=0,
        )

    # Previous state remains unchanged.
    assert (
        state_after_scriptwriter
        .story_state_ledger
        == starting_ledger
    )

    # Continuity alone advances authoritative state.
    assert (
        state_after_continuity
        .story_state_ledger
        == updated_ledger
    )

    assert (
        len(
            state_after_continuity
            .story_state_ledger
            .entries
        )
        == 1
    )

    assert (
        state_after_continuity
        .story_state_ledger
        .entries[0]
        .normalized_state
        == "Garak possesses the data rod."
    )

    # -----------------------------------------------------
    # Stage 6: Evaluation
    #
    # Capture every upstream source sent to final QA.
    # -----------------------------------------------------

    captured_evaluation = {}

    def fake_review_production_subject(
        subject_text,
        script_input,
        scriptwriter_audit,
        continuity_review,
        character_sources,
        lore_sources,
    ):
        captured_evaluation[
            "subject_text"
        ] = subject_text

        captured_evaluation[
            "script_input"
        ] = script_input

        captured_evaluation[
            "scriptwriter_audit"
        ] = scriptwriter_audit

        captured_evaluation[
            "continuity_review"
        ] = continuity_review

        captured_evaluation[
            "character_sources"
        ] = character_sources

        captured_evaluation[
            "lore_sources"
        ] = lore_sources

        return evaluation_production_result

    with patch(
        "workflows.writers_room."
        "review_production_subject",
        side_effect=(
            fake_review_production_subject
        ),
    ) as mock_evaluation:

        final_state = run_evaluation_stage(
            state_after_continuity
        )

        assert mock_evaluation.call_count == 1

    # =====================================================
    # FINAL ROOM ASSERTIONS
    # =====================================================

    # -----------------------------------------------------
    # Every production receipt survives.
    # -----------------------------------------------------

    assert final_state.story_plan == story_plan

    assert len(final_state.lore_results) == 2

    assert (
        len(final_state.character_results)
        == 1
    )

    assert final_state.script_result is not None

    assert final_state.continuity_result is not None

    assert final_state.evaluation_result is not None

    # -----------------------------------------------------
    # Evaluation received the exact Scriptwriter artifacts.
    # -----------------------------------------------------

    assert (
        captured_evaluation["script_input"]
        == assembled_script_input
    )

    assert (
        captured_evaluation[
            "scriptwriter_audit"
        ]
        == script_audit
    )

    # -----------------------------------------------------
    # Evaluation received exact Continuity review.
    # -----------------------------------------------------

    assert (
        captured_evaluation[
            "continuity_review"
        ]
        == continuity_review
    )

    # -----------------------------------------------------
    # Evaluation received Character evidence.
    # -----------------------------------------------------

    assert (
        len(
            captured_evaluation[
                "character_sources"
            ]
        )
        == 1
    )

    assert (
        captured_evaluation[
            "character_sources"
        ][0].character_analysis
        == character_analysis
    )

    assert (
        captured_evaluation[
            "character_sources"
        ][0].audit
        == character_analysis_audit
    )

    # -----------------------------------------------------
    # Evaluation receives supported AND unresolved Lore.
    # -----------------------------------------------------

    assert (
        len(
            captured_evaluation[
                "lore_sources"
            ]
        )
        == 2
    )

    assert (
        captured_evaluation[
            "lore_sources"
        ][0].lore_response
        == verified_lore
    )

    assert (
        captured_evaluation[
            "lore_sources"
        ][1].lore_response
        == unresolved_lore
    )

    # -----------------------------------------------------
    # Final Evaluation verdict preserved EXACTLY.
    #
    # Orchestration succeeded, but Evaluation said:
    #
    # NEEDS_EVIDENCE
    #
    # Writer's Room must not turn that into APPROVED.
    # -----------------------------------------------------

    assert (
        final_state
        .evaluation_result
        .production_review
        == evaluation_production_result
    )

    assert (
        final_state
        .evaluation_result
        .production_review
        .review
        .assessment
        .status
        == "needs_evidence"
    )

    assert (
        final_state
        .evaluation_result
        .production_review
        .review
        .assessment
        .blocking_criterion_ids
        == [
            "lore_001",
        ]
    )

    # -----------------------------------------------------
    # Provenance survives final assembly.
    # -----------------------------------------------------

    assert (
        final_state
        .evaluation_result
        .character_result_ids
        == [
            "character_000",
        ]
    )

    assert (
        final_state
        .evaluation_result
        .lore_research_ids
        == [
            "lore_000",
            "lore_001",
        ]
    )

    # -----------------------------------------------------
    # Earlier Writer's Room states remain immutable.
    # -----------------------------------------------------

    assert initial_state.story_plan is None

    assert (
        state_after_architect.lore_results
        == []
    )

    assert (
        state_after_lore.character_results
        == []
    )

    assert (
        state_after_character.script_result
        is None
    )

    assert (
        state_after_scriptwriter
        .continuity_result
        is None
    )

    assert (
        state_after_continuity
        .evaluation_result
        is None
    )

    # =====================================================
    # OUTPUT
    # =====================================================

    print(
        "\n--- Writer's Room Full Integration Test ---"
    )

    print(
        "Bad stage order rejected before specialist work: PASS"
    )

    print(
        "Midworkflow specialist failure propagated: PASS"
    )

    print(
        "Midworkflow failure created no downstream artifacts: PASS"
    )

    print(
        "Story Architect artifact preserved through final state: PASS"
    )

    print(
        "Lore artifacts accumulated with stable provenance: PASS"
    )

    print(
        "Character package accumulated with grounding audit: PASS"
    )

    print(
        "Verified Lore promoted to Scriptwriter evidence: PASS"
    )

    print(
        "Unresolved Lore withheld from Scriptwriter canon facts: PASS"
    )

    print(
        "Grounded Character guidance routed to Scriptwriter: PASS"
    )

    print(
        "Fictional character received no fabricated guidance: PASS"
    )

    print(
        "Scriptwriter result accumulated in production state: PASS"
    )

    print(
        "Exact ScriptScene routed into Continuity: PASS"
    )

    print(
        "Story ledger advanced only through Continuity: PASS"
    )

    print(
        "Normalized durable state preserved in final ledger: PASS"
    )

    print(
        "Evaluation received exact Scriptwriter artifacts: PASS"
    )

    print(
        "Evaluation received exact Continuity artifact: PASS"
    )

    print(
        "Evaluation received exact Character evidence: PASS"
    )

    print(
        "Evaluation received supported and unresolved Lore: PASS"
    )

    print(
        "Final Evaluation provenance chain preserved: PASS"
    )

    print(
        "Final NEEDS_EVIDENCE verdict preserved exactly: PASS"
    )

    print(
        "Earlier Writer's Room states remained immutable: PASS"
    )

    print(
        "\n============================================"
    )

    print(
        "WRITER'S ROOM FULL INTEGRATION CONTRACT: PASS"
    )

    print(
        "============================================"
    )


if __name__ == "__main__":
    main()