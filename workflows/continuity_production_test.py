from unittest.mock import patch

from agents.scriptwriter_agent import (
    ScriptScene,
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

from workflows.continuity_production import (
    ContinuityProductionResult,
    review_scene_for_production,
)


def main():
    # -----------------------------------------------------
    # Deterministic existing story state
    # -----------------------------------------------------

    original_ledger = StoryStateLedger(
        entries=[
            StoryStateEntry(
                state_id="state_000",
                scene_index=0,
                statement=(
                    "The Cardassian defector possesses "
                    "a data rod."
                ),
                normalized_state=(
                    "The Cardassian defector possesses "
                    "the data rod."
                ),
            )
        ]
    )

    # -----------------------------------------------------
    # Deterministic ScriptScene
    # -----------------------------------------------------

    scene = ScriptScene(
        scene_heading=(
            "INT. PRIVATE MEETING AREA - NIGHT"
        ),
        setting_description=(
            "Garak and the Cardassian defector speak "
            "in a secluded alcove."
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
                "the data rod to Garak."
            ),
        ],
    )

    # -----------------------------------------------------
    # Clear Continuity review
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

    # -----------------------------------------------------
    # Updated ledger produced by frozen state learning
    # -----------------------------------------------------

    updated_ledger = StoryStateLedger(
        entries=[
            StoryStateEntry(
                state_id="state_000",
                scene_index=0,
                statement=(
                    "The Cardassian defector possesses "
                    "a data rod."
                ),
                normalized_state=(
                    "The Cardassian defector possesses "
                    "the data rod."
                ),
            ),
            StoryStateEntry(
                state_id="state_001",
                scene_index=1,
                statement=(
                    "The defector removes a data rod and "
                    "places it in Garak's hand."
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
            "state_001",
        ],
        transient_segment_ids=[],
        ledger=updated_ledger,
    )

    # -----------------------------------------------------
    # Negative scene index protection
    #
    # No Continuity work should occur.
    # -----------------------------------------------------

    with patch(
        "workflows.continuity_production."
        "review_script_scene_continuity"
    ) as mock_review:

        try:
            review_scene_for_production(
                ledger=original_ledger,
                scene=scene,
                scene_index=-1,
            )

        except ValueError as error:
            assert (
                "scene_index cannot be negative."
                in str(error)
            )

        else:
            raise AssertionError(
                "Continuity production adapter allowed "
                "a negative scene index."
            )

        mock_review.assert_not_called()

    # -----------------------------------------------------
    # CLEAR PATH
    #
    # Review passes -> frozen sequential state learning runs.
    # -----------------------------------------------------

    with (
        patch(
            "workflows.continuity_production."
            "review_script_scene_continuity",
            return_value=clear_review,
        ) as mock_review,
        patch(
            "workflows.continuity_production."
            "learn_scene_states_in_order",
            return_value=learning_result,
        ) as mock_learning,
    ):

        clear_result = review_scene_for_production(
            ledger=original_ledger,
            scene=scene,
            scene_index=1,
        )

        mock_review.assert_called_once_with(
            ledger=original_ledger,
            scene=scene,
        )

        mock_learning.assert_called_once_with(
            ledger=original_ledger,
            scene=scene,
            scene_index=1,
        )

    # -----------------------------------------------------
    # Clear-path result contract
    # -----------------------------------------------------

    assert isinstance(
        clear_result,
        ContinuityProductionResult,
    )

    assert clear_result.scene_index == 1

    assert (
        clear_result.review
        == clear_review
    )

    assert (
        clear_result.previous_ledger
        == original_ledger
    )

    assert (
        clear_result.updated_ledger
        == updated_ledger
    )

    assert (
        clear_result.learning_result
        == learning_result
    )

    # -----------------------------------------------------
    # Original ledger was not mutated
    # -----------------------------------------------------

    assert (
        len(original_ledger.entries)
        == 1
    )

    assert (
        original_ledger.entries[0].state_id
        == "state_000"
    )

    # -----------------------------------------------------
    # CONTRADICTION PATH
    #
    # Review blocks -> state learning must NOT run.
    # -----------------------------------------------------

    contradiction_issue = ContinuitySceneIssue(
        segment_id="script_000",
        statement=(
            "The defector still holds the data rod."
        ),
        state_id="state_001",
        established_state=(
            "Garak possesses the data rod."
        ),
        issue_type="contradiction",
        reason=(
            "The later statement conflicts with the "
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

    with (
        patch(
            "workflows.continuity_production."
            "review_script_scene_continuity",
            return_value=contradiction_review,
        ) as mock_review,
        patch(
            "workflows.continuity_production."
            "learn_scene_states_in_order",
        ) as mock_learning,
    ):

        blocked_result = review_scene_for_production(
            ledger=original_ledger,
            scene=scene,
            scene_index=1,
        )

        mock_review.assert_called_once_with(
            ledger=original_ledger,
            scene=scene,
        )

        mock_learning.assert_not_called()

    # -----------------------------------------------------
    # Blocked-path result contract
    # -----------------------------------------------------

    assert (
        blocked_result.review
        == contradiction_review
    )

    assert (
        blocked_result.previous_ledger
        == original_ledger
    )

    assert (
        blocked_result.updated_ledger
        == original_ledger
    )

    assert (
        blocked_result.learning_result
        is None
    )

    # -----------------------------------------------------
    # NEEDS CONTEXT PATH
    #
    # Same rule: preserve review, do not learn state.
    # -----------------------------------------------------

    context_issue = ContinuitySceneIssue(
        segment_id="script_001",
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
            "The supplied story state does not establish "
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

    with (
        patch(
            "workflows.continuity_production."
            "review_script_scene_continuity",
            return_value=needs_context_review,
        ) as mock_review,
        patch(
            "workflows.continuity_production."
            "learn_scene_states_in_order",
        ) as mock_learning,
    ):

        context_result = review_scene_for_production(
            ledger=original_ledger,
            scene=scene,
            scene_index=1,
        )

        mock_review.assert_called_once()

        mock_learning.assert_not_called()

    assert (
        context_result.review
        == needs_context_review
    )

    assert (
        context_result.updated_ledger
        == original_ledger
    )

    assert (
        context_result.learning_result
        is None
    )

    # -----------------------------------------------------
    # Frozen learning failure propagation
    #
    # A clear review does NOT give the adapter permission
    # to swallow a later sequential-learning failure.
    # -----------------------------------------------------

    with (
        patch(
            "workflows.continuity_production."
            "review_script_scene_continuity",
            return_value=clear_review,
        ),
        patch(
            "workflows.continuity_production."
            "learn_scene_states_in_order",
            side_effect=ValueError(
                "Sequential scene learning requires "
                "additional continuity context."
            ),
        ),
    ):

        try:
            review_scene_for_production(
                ledger=original_ledger,
                scene=scene,
                scene_index=1,
            )

        except ValueError as error:
            assert (
                "Sequential scene learning requires"
                in str(error)
            )

        else:
            raise AssertionError(
                "Continuity production adapter swallowed "
                "a frozen state-learning failure."
            )

    # -----------------------------------------------------
    # Output
    # -----------------------------------------------------

    print(
        "\n--- Continuity Production Test ---"
    )

    print(
        "Negative scene index rejected before specialist work: PASS"
    )

    print(
        "Frozen Continuity review reused directly: PASS"
    )

    print(
        "Clear review permits sequential state learning: PASS"
    )

    print(
        "Previous ledger preserved exactly: PASS"
    )

    print(
        "Updated ledger preserved exactly: PASS"
    )

    print(
        "Original ledger not mutated: PASS"
    )

    print(
        "Contradiction blocks ledger learning: PASS"
    )

    print(
        "Contradiction review preserved for downstream QA: PASS"
    )

    print(
        "Needs-context review blocks ledger learning: PASS"
    )

    print(
        "Needs-context review preserved for downstream QA: PASS"
    )

    print(
        "Frozen learning failure propagated: PASS"
    )

    print(
        "\nContinuity → production adapter "
        "contract: PASS"
    )


if __name__ == "__main__":
    main()