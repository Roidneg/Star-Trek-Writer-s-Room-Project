from unittest.mock import patch

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
    ScriptCharacterContext,
    ScriptScene,
    ScriptSceneAudit,
    ScriptwriterInput,
)

from agents.continuity_agent import (
    ContinuitySceneAssessment,
    ContinuitySceneLearningResult,
    ContinuitySceneReview,
    ContinuityScriptSceneAudit,
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

from workflows.scriptwriter_production import (
    ScriptwriterProductionResult,
)

from workflows.continuity_production import (
    ContinuityProductionResult,
)

from workflows.writers_room import (
    WritersRoomCharacterResult,
    WritersRoomEvaluationResult,
    WritersRoomLoreResult,
    WritersRoomState,
    run_evaluation_stage,
)


def main():
    # -----------------------------------------------------
    # Deterministic Scriptwriter artifacts
    # -----------------------------------------------------

    script_input = ScriptwriterInput(
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
            ScriptCharacterContext(
                character_name="Elim Garak",
                grounded_character_guidance=[
                    (
                        "Garak must evaluate the "
                        "defector's allegation."
                    ),
                ],
                grounded_voice_guidance=[],
            ),
            ScriptCharacterContext(
                character_name="Cardassian Defector",
                grounded_character_guidance=[],
                grounded_voice_guidance=[],
            ),
        ],
        verified_canon_facts=[
            (
                "Garak participated in a mission "
                "involving the Founders before the "
                "Dominion War."
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

    scene = ScriptScene(
        scene_heading=(
            "INT. PRIVATE MEETING AREA - NIGHT"
        ),
        setting_description=(
            "A quiet private alcove provides enough "
            "seclusion for a guarded conversation."
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
                "Garak participated in a mission "
                "involving the Founders before the "
                "Dominion War."
            ),
        ],
        invented_story_elements=[
            (
                "A fictional Cardassian defector "
                "approaches Garak."
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

    script_result = ScriptwriterProductionResult(
        script_input=script_input,
        scene=scene,
        audit=script_audit,
    )

    # -----------------------------------------------------
    # Deterministic Continuity artifacts
    # -----------------------------------------------------

    ledger = StoryStateLedger(
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

    learning_result = ContinuitySceneLearningResult(
        committed_state_ids=[],
        transient_segment_ids=[],
        ledger=ledger,
    )

    continuity_result = ContinuityProductionResult(
        scene_index=1,
        review=continuity_review,
        previous_ledger=ledger,
        updated_ledger=ledger,
        learning_result=learning_result,
    )

    # -----------------------------------------------------
    # Deterministic Character artifact
    # -----------------------------------------------------

    character_research_plan = CharacterResearchPlan(
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

    character_analysis = CharacterAnalysis(
        character_name="Elim Garak",
        story_role=(
            "Garak must evaluate the defector's allegation."
        ),
        motivations=[],
        likely_behavior=[
            (
                "Evaluate the claim before deciding "
                "whether further action is warranted."
            ),
        ],
        unlikely_behavior=[],
        relationships=[],
        dramatic_tension=(
            "Garak must determine whether the "
            "allegation deserves attention."
        ),
        remaining_canon_questions=[],
        remaining_voice_questions=[],
        speech_patterns=[],
        preferred_vocabulary=[],
        communication_style=[],
        behavioral_constraints=[
            (
                "Do not give Garak current Cardassian "
                "contacts without supplied evidence."
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
                    "Garak must evaluate the "
                    "defector's allegation."
                ),
                supported=True,
                evidence_excerpt=(
                    "A Cardassian defector approaches "
                    "Garak with an allegation concerning "
                    "Founder infiltration."
                ),
                reason=(
                    "The supplied story context "
                    "establishes the immediate task."
                ),
            )
        ]
    )

    character_audit = CharacterAnalysisAudit(
        grounded=True,
        issues=[],
        reason=(
            "Frozen Character grounding gate passed."
        ),
    )

    character_result = WritersRoomCharacterResult(
        character_id="character_000",
        character_name="Elim Garak",
        research_plan=character_research_plan,
        canon_research=(
            "VERIFIED CANON EVIDENCE\n\n"
            "Garak participated in a mission "
            "involving the Founders."
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
    # Lore artifacts
    #
    # Evaluation must receive BOTH:
    # - supported Lore
    # - unresolved Lore
    #
    # Writer's Room does not reinterpret either.
    # -----------------------------------------------------

    supported_lore = LoreResponseModel(
        topic="Garak and the Founders",
        summary=(
            "Garak participated in a mission "
            "involving the Founders before the "
            "Dominion War."
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
            "The retrieved evidence does not fully "
            "establish the requested political context."
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
                "What does canon establish about "
                "Garak and the Founders?"
            ),
            response=supported_lore,
        ),
        WritersRoomLoreResult(
            research_id="lore_001",
            research_question=(
                "What does canon establish about "
                "Cardassian political conditions?"
            ),
            response=unresolved_lore,
        ),
    ]

    # -----------------------------------------------------
    # Deterministic frozen Evaluation result
    # -----------------------------------------------------

    criterion_result = EvaluationCriterionResult(
        criterion_id="scriptwriter_grounding_review",
        category="upstream_constraint",
        criterion=(
            "The scene must pass the frozen "
            "Scriptwriter grounding gate."
        ),
        verdict="pass",
        reason=(
            "Frozen Scriptwriter grounding result "
            "was accepted directly."
        ),
    )

    evaluation_manifest = EvaluationManifest(
        criteria=[],
        precomputed_results=[
            criterion_result,
        ],
    )

    evaluation_suite = EvaluationSuiteResult(
        results=[
            criterion_result,
        ],
        pass_count=1,
        fail_count=0,
        insufficient_evidence_count=0,
        has_failure=False,
    )

    evaluation_assessment = EvaluationAssessment(
        status="approved",
        pass_count=1,
        fail_count=0,
        insufficient_evidence_count=0,
        blocking_criterion_ids=[],
    )

    evaluation_review = EvaluationReview(
        suite=evaluation_suite,
        assessment=evaluation_assessment,
    )

    production_review = EvaluationProductionReview(
        manifest=evaluation_manifest,
        review=evaluation_review,
    )

    # -----------------------------------------------------
    # Complete valid Writer's Room state
    # -----------------------------------------------------

    state = WritersRoomState(
        story_idea="Test story idea.",
        lore_results=lore_results,
        character_results=[
            character_result,
        ],
        script_result=script_result,
        story_state_ledger=ledger,
        continuity_result=continuity_result,
    )

    # -----------------------------------------------------
    # Scriptwriter prerequisite
    # -----------------------------------------------------

    no_script_state = WritersRoomState(
        story_idea="Test.",
    )

    with patch(
        "workflows.writers_room."
        "review_production_subject"
    ) as mock_evaluation:

        try:
            run_evaluation_stage(
                state=no_script_state,
            )

        except ValueError as error:
            assert (
                "Evaluation stage requires a completed "
                "Scriptwriter stage."
                in str(error)
            )

        else:
            raise AssertionError(
                "Writer's Room allowed Evaluation "
                "without a Scriptwriter result."
            )

        mock_evaluation.assert_not_called()

    # -----------------------------------------------------
    # Continuity prerequisite
    # -----------------------------------------------------

    no_continuity_state = WritersRoomState(
        story_idea="Test.",
        script_result=script_result,
    )

    with patch(
        "workflows.writers_room."
        "review_production_subject"
    ) as mock_evaluation:

        try:
            run_evaluation_stage(
                state=no_continuity_state,
            )

        except ValueError as error:
            assert (
                "Evaluation stage requires a completed "
                "Continuity stage."
                in str(error)
            )

        else:
            raise AssertionError(
                "Writer's Room allowed Evaluation "
                "without a Continuity result."
            )

        mock_evaluation.assert_not_called()

    # -----------------------------------------------------
    # Lore prerequisite
    # -----------------------------------------------------

    no_lore_state = WritersRoomState(
        story_idea="Test.",
        script_result=script_result,
        continuity_result=continuity_result,
    )

    with patch(
        "workflows.writers_room."
        "review_production_subject"
    ) as mock_evaluation:

        try:
            run_evaluation_stage(
                state=no_lore_state,
            )

        except ValueError as error:
            assert (
                "Evaluation stage requires at least one "
                "Lore result."
                in str(error)
            )

        else:
            raise AssertionError(
                "Writer's Room allowed Evaluation "
                "without Lore evidence."
            )

        mock_evaluation.assert_not_called()

    # -----------------------------------------------------
    # Matching Character prerequisite
    #
    # The fictional defector does not require a Character
    # artifact, but at least one Scriptwriter character must
    # match a real Character Agent result.
    # -----------------------------------------------------

    no_character_state = WritersRoomState(
        story_idea="Test.",
        lore_results=lore_results,
        script_result=script_result,
        continuity_result=continuity_result,
    )

    with patch(
        "workflows.writers_room."
        "review_production_subject"
    ) as mock_evaluation:

        try:
            run_evaluation_stage(
                state=no_character_state,
            )

        except ValueError as error:
            assert (
                "Evaluation stage requires at least one "
                "Character result matching the "
                "Scriptwriter scene."
                in str(error)
            )

        else:
            raise AssertionError(
                "Writer's Room allowed Evaluation "
                "without matching Character evidence."
            )

        mock_evaluation.assert_not_called()

    # -----------------------------------------------------
    # Valid Evaluation orchestration
    # -----------------------------------------------------

    captured = {}

    def fake_review_production_subject(
        subject_text,
        script_input,
        scriptwriter_audit,
        continuity_review,
        character_sources,
        lore_sources,
    ):
        captured["subject_text"] = subject_text
        captured["script_input"] = script_input
        captured["scriptwriter_audit"] = (
            scriptwriter_audit
        )
        captured["continuity_review"] = (
            continuity_review
        )
        captured["character_sources"] = (
            character_sources
        )
        captured["lore_sources"] = lore_sources

        return production_review

    with patch(
        "workflows.writers_room."
        "review_production_subject",
        side_effect=fake_review_production_subject,
    ) as mock_evaluation:

        updated_state = run_evaluation_stage(
            state=state,
        )

        assert mock_evaluation.call_count == 1

    # -----------------------------------------------------
    # Complete screenplay subject assembled correctly
    # -----------------------------------------------------

    expected_subject = "\n\n".join(
        [
            scene.scene_heading,
            scene.setting_description,
            scene.script_text,
        ]
    )

    assert (
        captured["subject_text"]
        == expected_subject
    )

    # -----------------------------------------------------
    # Exact Scriptwriter artifacts routed
    # -----------------------------------------------------

    assert (
        captured["script_input"]
        == script_input
    )

    assert (
        captured["scriptwriter_audit"]
        == script_audit
    )

    # -----------------------------------------------------
    # Exact Continuity artifact routed
    # -----------------------------------------------------

    assert (
        captured["continuity_review"]
        == continuity_review
    )

    # -----------------------------------------------------
    # Only matching Character Agent artifacts routed
    #
    # The fictional Cardassian Defector does not cause the
    # Writer's Room to fabricate a CharacterEvaluationSource.
    # -----------------------------------------------------

    assert (
        len(captured["character_sources"])
        == 1
    )

    character_source = (
        captured["character_sources"][0]
    )

    assert (
        character_source.character_analysis
        == character_analysis
    )

    assert (
        character_source.audit
        == character_audit
    )

    # -----------------------------------------------------
    # ALL Lore artifacts routed
    #
    # Supported and unresolved Lore both reach Evaluation.
    # Evaluation owns what those statuses mean.
    # -----------------------------------------------------

    assert (
        len(captured["lore_sources"])
        == 2
    )

    assert (
        captured["lore_sources"][0].research_id
        == "lore_000"
    )

    assert (
        captured["lore_sources"][0].lore_response
        == supported_lore
    )

    assert (
        captured["lore_sources"][1].research_id
        == "lore_001"
    )

    assert (
        captured["lore_sources"][1].lore_response
        == unresolved_lore
    )

    assert (
        captured["lore_sources"][1]
        .lore_response
        .evidence_supported
        is False
    )

    # -----------------------------------------------------
    # Writer's Room result provenance
    # -----------------------------------------------------

    assert isinstance(
        updated_state.evaluation_result,
        WritersRoomEvaluationResult,
    )

    assert (
        updated_state
        .evaluation_result
        .subject_text
        == expected_subject
    )

    assert (
        updated_state
        .evaluation_result
        .character_result_ids
        == [
            "character_000",
        ]
    )

    assert (
        updated_state
        .evaluation_result
        .lore_research_ids
        == [
            "lore_000",
            "lore_001",
        ]
    )

    # -----------------------------------------------------
    # Frozen Evaluation result preserved exactly
    # -----------------------------------------------------

    assert (
        updated_state
        .evaluation_result
        .production_review
        == production_review
    )

    assert (
        updated_state
        .evaluation_result
        .production_review
        .review
        .assessment
        .status
        == "approved"
    )

    # -----------------------------------------------------
    # Original state not mutated
    # -----------------------------------------------------

    assert state.evaluation_result is None

    # -----------------------------------------------------
    # Frozen Evaluation failure propagation
    # -----------------------------------------------------

    failure_state = WritersRoomState(
        story_idea="Test.",
        lore_results=lore_results,
        character_results=[
            character_result,
        ],
        script_result=script_result,
        continuity_result=continuity_result,
    )

    with patch(
        "workflows.writers_room."
        "review_production_subject",
        side_effect=ValueError(
            "Evaluation manifest coverage validation failed."
        ),
    ):

        try:
            run_evaluation_stage(
                state=failure_state,
            )

        except ValueError as error:
            assert (
                "Evaluation manifest coverage validation failed."
                in str(error)
            )

        else:
            raise AssertionError(
                "Writer's Room swallowed a frozen "
                "Evaluation failure."
            )

    assert (
        failure_state.evaluation_result
        is None
    )

    # -----------------------------------------------------
    # Stage overwrite protection
    # -----------------------------------------------------

    with patch(
        "workflows.writers_room."
        "review_production_subject"
    ) as mock_evaluation:

        try:
            run_evaluation_stage(
                state=updated_state,
            )

        except ValueError as error:
            assert (
                "Evaluation stage has already completed."
                in str(error)
            )

        else:
            raise AssertionError(
                "Writer's Room allowed Evaluation "
                "to overwrite its final result."
            )

        mock_evaluation.assert_not_called()

    # -----------------------------------------------------
    # Output
    # -----------------------------------------------------

    print(
        "\n--- Writer's Room Evaluation Test ---"
    )

    print(
        "Scriptwriter prerequisite enforced: PASS"
    )

    print(
        "Continuity prerequisite enforced: PASS"
    )

    print(
        "Lore prerequisite enforced: PASS"
    )

    print(
        "Matching Character evidence required: PASS"
    )

    print(
        "Complete screenplay subject assembled correctly: PASS"
    )

    print(
        "Exact Scriptwriter input routed to Evaluation: PASS"
    )

    print(
        "Exact Scriptwriter audit routed to Evaluation: PASS"
    )

    print(
        "Exact Continuity review routed to Evaluation: PASS"
    )

    print(
        "Matching Character artifact routed exactly: PASS"
    )

    print(
        "Fictional character does not fabricate evidence: PASS"
    )

    print(
        "Supported Lore routed to Evaluation: PASS"
    )

    print(
        "Unresolved Lore preserved and routed to Evaluation: PASS"
    )

    print(
        "Evaluation provenance IDs preserved: PASS"
    )

    print(
        "Final Evaluation verdict preserved exactly: PASS"
    )

    print(
        "Original Writer's Room state not mutated: PASS"
    )

    print(
        "Frozen Evaluation failure propagated: PASS"
    )

    print(
        "Stage overwrite protection: PASS"
    )

    print(
        "\nWriter's Room → Evaluation Agent "
        "orchestration contract: PASS"
    )


if __name__ == "__main__":
    main()