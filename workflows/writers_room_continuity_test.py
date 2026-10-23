from unittest.mock import patch

from agents.scriptwriter_agent import (
    ScriptCharacterContext,
    ScriptScene,
    ScriptSceneAudit,
    ScriptwriterInput,
)

from agents.continuity_agent import (
    ContinuitySceneAssessment,
    ContinuitySceneIssue,
    ContinuitySceneLearningResult,
    ContinuitySceneReview,
    ContinuityScriptSceneAudit,
    StoryStateEntry,
    StoryStateLedger,
)

from workflows.scriptwriter_production import (
    ScriptwriterProductionResult,
)

from workflows.continuity_production import (
    ContinuityProductionResult,
)

from workflows.writers_room import (
    WritersRoomState,
    run_continuity_stage,
)


def main():
    # -----------------------------------------------------
    # Deterministic Scriptwriter artifacts
    # -----------------------------------------------------

    script_input = ScriptwriterInput(
        story_context=(
            "A Cardassian defector privately approaches "
            "Garak with an allegation concerning Founder "
            "infiltration."
        ),
        scene_objective=(
            "The defector gives Garak a data rod "
            "containing the alleged evidence."
        ),
        story_era=(
            "Star Trek: Deep Space Nine, shortly before "
            "the Dominion War"
        ),
        characters=[
            ScriptCharacterContext(
                character_name="Elim Garak",
                grounded_character_guidance=[],
                grounded_voice_guidance=[],
            ),
            ScriptCharacterContext(
                character_name="Cardassian Defector",
                grounded_character_guidance=[],
                grounded_voice_guidance=[],
            ),
        ],
        verified_canon_facts=[],
        era_style_constraints=[],
        scene_requirements=[
            (
                "The Cardassian defector gives "
                "Garak a data rod."
            ),
        ],
        prohibited_assumptions=[],
    )

    scene = ScriptScene(
        scene_heading=(
            "INT. PRIVATE MEETING AREA - NIGHT"
        ),
        setting_description=(
            "Garak and the defector speak in private."
        ),
        characters_present=[
            "Elim Garak",
            "Cardassian Defector",
        ],
        script_text=(
            "The defector removes a data rod and places "
            "it in Garak's hand.\n\n"
            "CARDASSIAN DEFECTOR\n"
            "Everything I know is on that rod."
        ),
        canon_facts_used=[],
        invented_story_elements=[
            (
                "The Cardassian defector gives "
                "Garak a data rod."
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
    # Existing authoritative story state
    # -----------------------------------------------------

    original_ledger = StoryStateLedger(
        entries=[
            StoryStateEntry(
                state_id="state_000",
                scene_index=0,
                statement=(
                    "The Cardassian defector possesses "
                    "the data rod."
                ),
                normalized_state=(
                    "The Cardassian defector possesses "
                    "the data rod."
                ),
            )
        ]
    )

    state = WritersRoomState(
        story_idea="Test story idea.",
        script_result=script_result,
        story_state_ledger=original_ledger,
    )

    # -----------------------------------------------------
    # Deterministic clear Continuity package
    # -----------------------------------------------------

    clear_review = ContinuitySceneReview(
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
                state_id="state_000",
                scene_index=0,
                statement=(
                    "The Cardassian defector possesses "
                    "the data rod."
                ),
                normalized_state=(
                    "The Cardassian defector possesses "
                    "the data rod."
                ),
            ),
            StoryStateEntry(
                state_id="scene_0001_script_text_000",
                scene_index=1,
                statement=(
                    "The defector removes a data rod "
                    "and places it in Garak's hand."
                ),
                normalized_state=(
                    "Garak possesses the data rod."
                ),
                supersedes_state_id="state_000",
            ),
        ]
    )

    learning_result = ContinuitySceneLearningResult(
        committed_state_ids=[
            "scene_0001_script_text_000",
        ],
        transient_segment_ids=[],
        ledger=updated_ledger,
    )

    clear_production_result = (
        ContinuityProductionResult(
            scene_index=1,
            review=clear_review,
            previous_ledger=original_ledger,
            updated_ledger=updated_ledger,
            learning_result=learning_result,
        )
    )

    # -----------------------------------------------------
    # Scriptwriter prerequisite
    # -----------------------------------------------------

    no_script_state = WritersRoomState(
        story_idea="Test story idea.",
    )

    with patch(
        "workflows.writers_room."
        "review_scene_for_production"
    ) as mock_continuity:

        try:
            run_continuity_stage(
                state=no_script_state,
                scene_index=1,
            )

        except ValueError as error:
            assert (
                "Continuity stage requires a completed "
                "Scriptwriter stage."
                in str(error)
            )

        else:
            raise AssertionError(
                "Writer's Room allowed Continuity "
                "without a Scriptwriter result."
            )

        mock_continuity.assert_not_called()

    # -----------------------------------------------------
    # Negative scene-index protection
    # -----------------------------------------------------

    with patch(
        "workflows.writers_room."
        "review_scene_for_production"
    ) as mock_continuity:

        try:
            run_continuity_stage(
                state=state,
                scene_index=-1,
            )

        except ValueError as error:
            assert (
                "Continuity stage scene_index cannot "
                "be negative."
                in str(error)
            )

        else:
            raise AssertionError(
                "Writer's Room allowed a negative "
                "Continuity scene index."
            )

        mock_continuity.assert_not_called()

    # -----------------------------------------------------
    # Clear Continuity orchestration
    # -----------------------------------------------------

    with patch(
        "workflows.writers_room."
        "review_scene_for_production",
        return_value=clear_production_result,
    ) as mock_continuity:

        updated_state = run_continuity_stage(
            state=state,
            scene_index=1,
        )

        mock_continuity.assert_called_once_with(
            ledger=original_ledger,
            scene=scene,
            scene_index=1,
        )

    # -----------------------------------------------------
    # Exact artifact preservation
    # -----------------------------------------------------

    assert (
        updated_state.continuity_result
        == clear_production_result
    )

    assert (
        updated_state.story_state_ledger
        == updated_ledger
    )

    # -----------------------------------------------------
    # Original Writer's Room state not mutated
    # -----------------------------------------------------

    assert state.continuity_result is None

    assert (
        state.story_state_ledger
        == original_ledger
    )

    assert (
        len(state.story_state_ledger.entries)
        == 1
    )

    # -----------------------------------------------------
    # Contradiction result
    #
    # Writer's Room must preserve the contradiction for
    # Evaluation and accept Continuity's unchanged ledger.
    # -----------------------------------------------------

    contradiction_issue = ContinuitySceneIssue(
        segment_id="script_text_000",
        statement=(
            "The defector still possesses the data rod."
        ),
        state_id="state_000",
        established_state=(
            "Garak possesses the data rod."
        ),
        issue_type="contradiction",
        reason=(
            "The statement conflicts with the "
            "established possession state."
        ),
    )

    contradiction_review = ContinuitySceneReview(
        audit=ContinuityScriptSceneAudit(
            segment_audits=[],
            has_contradiction=True,
            contradiction_count=1,
        ),
        assessment=ContinuitySceneAssessment(
            status="contradiction",
            issues=[
                contradiction_issue,
            ],
            contradiction_count=1,
            insufficient_context_count=0,
        ),
    )

    contradiction_result = (
        ContinuityProductionResult(
            scene_index=1,
            review=contradiction_review,
            previous_ledger=original_ledger,
            updated_ledger=original_ledger,
            learning_result=None,
        )
    )

    contradiction_state = WritersRoomState(
        story_idea="Test story idea.",
        script_result=script_result,
        story_state_ledger=original_ledger,
    )

    with patch(
        "workflows.writers_room."
        "review_scene_for_production",
        return_value=contradiction_result,
    ):

        blocked_state = run_continuity_stage(
            state=contradiction_state,
            scene_index=1,
        )

    assert (
        blocked_state.continuity_result
        == contradiction_result
    )

    assert (
        blocked_state
        .continuity_result
        .review
        .assessment
        .status
        == "contradiction"
    )

    # Writer's Room must not manufacture a ledger change.
    assert (
        blocked_state.story_state_ledger
        == original_ledger
    )

    assert (
        blocked_state
        .continuity_result
        .learning_result
        is None
    )

    # -----------------------------------------------------
    # Needs-context result
    #
    # Same rule: preserve QA evidence and unchanged ledger.
    # -----------------------------------------------------

    context_issue = ContinuitySceneIssue(
        segment_id="script_text_001",
        statement=(
            "Garak returns the device."
        ),
        state_id="state_000",
        established_state=(
            "The Cardassian defector possesses "
            "the data rod."
        ),
        issue_type="insufficient_context",
        reason=(
            "The supplied scene does not establish "
            "which device the statement refers to."
        ),
    )

    needs_context_review = ContinuitySceneReview(
        audit=ContinuityScriptSceneAudit(
            segment_audits=[],
            has_contradiction=False,
            contradiction_count=0,
        ),
        assessment=ContinuitySceneAssessment(
            status="needs_context",
            issues=[
                context_issue,
            ],
            contradiction_count=0,
            insufficient_context_count=1,
        ),
    )

    needs_context_result = (
        ContinuityProductionResult(
            scene_index=1,
            review=needs_context_review,
            previous_ledger=original_ledger,
            updated_ledger=original_ledger,
            learning_result=None,
        )
    )

    context_state = WritersRoomState(
        story_idea="Test story idea.",
        script_result=script_result,
        story_state_ledger=original_ledger,
    )

    with patch(
        "workflows.writers_room."
        "review_scene_for_production",
        return_value=needs_context_result,
    ):

        unresolved_state = run_continuity_stage(
            state=context_state,
            scene_index=1,
        )

    assert (
        unresolved_state
        .continuity_result
        .review
        .assessment
        .status
        == "needs_context"
    )

    assert (
        unresolved_state.story_state_ledger
        == original_ledger
    )

    assert (
        unresolved_state
        .continuity_result
        .learning_result
        is None
    )

    # -----------------------------------------------------
    # Scene-index identity protection
    # -----------------------------------------------------

    wrong_index_result = (
        clear_production_result.model_copy(
            update={
                "scene_index": 99,
            }
        )
    )

    fresh_state = WritersRoomState(
        story_idea="Test story idea.",
        script_result=script_result,
        story_state_ledger=original_ledger,
    )

    with patch(
        "workflows.writers_room."
        "review_scene_for_production",
        return_value=wrong_index_result,
    ):

        try:
            run_continuity_stage(
                state=fresh_state,
                scene_index=1,
            )

        except ValueError as error:
            assert (
                "Continuity production result changed "
                "the requested scene index."
                in str(error)
            )

        else:
            raise AssertionError(
                "Writer's Room accepted a Continuity "
                "result for the wrong scene index."
            )

    # -----------------------------------------------------
    # Frozen Continuity failure propagation
    # -----------------------------------------------------

    failure_state = WritersRoomState(
        story_idea="Test story idea.",
        script_result=script_result,
        story_state_ledger=original_ledger,
    )

    with patch(
        "workflows.writers_room."
        "review_scene_for_production",
        side_effect=ValueError(
            "Sequential scene learning requires "
            "additional continuity context."
        ),
    ):

        try:
            run_continuity_stage(
                state=failure_state,
                scene_index=1,
            )

        except ValueError as error:
            assert (
                "Sequential scene learning requires"
                in str(error)
            )

        else:
            raise AssertionError(
                "Writer's Room swallowed a frozen "
                "Continuity failure."
            )

    assert (
        failure_state.continuity_result
        is None
    )

    assert (
        failure_state.story_state_ledger
        == original_ledger
    )

    # -----------------------------------------------------
    # Stage overwrite protection
    # -----------------------------------------------------

    with patch(
        "workflows.writers_room."
        "review_scene_for_production"
    ) as mock_continuity:

        try:
            run_continuity_stage(
                state=updated_state,
                scene_index=1,
            )

        except ValueError as error:
            assert (
                "Continuity stage has already completed."
                in str(error)
            )

        else:
            raise AssertionError(
                "Writer's Room allowed Continuity "
                "to overwrite an existing result."
            )

        mock_continuity.assert_not_called()

    # -----------------------------------------------------
    # Output
    # -----------------------------------------------------

    print(
        "\n--- Writer's Room Continuity Test ---"
    )

    print(
        "Scriptwriter prerequisite enforced: PASS"
    )

    print(
        "Negative scene index rejected before specialist work: PASS"
    )

    print(
        "Exact ScriptScene routed to Continuity: PASS"
    )

    print(
        "Exact current ledger routed to Continuity: PASS"
    )

    print(
        "Clear Continuity result preserved exactly: PASS"
    )

    print(
        "Authoritative updated ledger adopted exactly: PASS"
    )

    print(
        "Original Writer's Room state not mutated: PASS"
    )

    print(
        "Contradiction preserved for downstream Evaluation: PASS"
    )

    print(
        "Contradiction does not advance story ledger: PASS"
    )

    print(
        "Needs-context result preserved for downstream Evaluation: PASS"
    )

    print(
        "Needs-context result does not advance story ledger: PASS"
    )

    print(
        "Scene-index identity protected: PASS"
    )

    print(
        "Frozen Continuity failure propagated: PASS"
    )

    print(
        "Stage overwrite protection: PASS"
    )

    print(
        "\nWriter's Room → Continuity Agent "
        "orchestration contract: PASS"
    )


if __name__ == "__main__":
    main()