from unittest.mock import patch

from agents.story_architect import StoryPlan

from workflows.writers_room import (
    WritersRoomState,
    run_story_architect_stage,
)


def main():
    story_idea = (
        "Shortly before the Dominion War, Garak is approached "
        "by a Cardassian defector who claims the Founders have "
        "infiltrated the Cardassian government."
    )

    expected_story_plan = StoryPlan(
        working_title="The Defector's Warning",
        premise=(
            "Garak is approached by a Cardassian defector "
            "with an allegation concerning Founder infiltration."
        ),
        setting="Deep Space Nine",
        timeframe="Shortly before the Dominion War",
        main_characters=[
            "Garak",
            "Cardassian defector",
        ],
        central_conflict=(
            "Garak must determine how seriously to treat "
            "the defector's allegation."
        ),
        stakes=(
            "Acting on unreliable information could create "
            "serious consequences, while ignoring a genuine "
            "threat could also carry significant risk."
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

    initial_state = WritersRoomState(
        story_idea=story_idea,
    )

    assert initial_state.story_plan is None

    # Patch the function where Writer's Room imported it.
    with patch(
        "workflows.writers_room.create_story_plan",
        return_value=expected_story_plan,
    ) as mock_story_architect:

        updated_state = run_story_architect_stage(
            initial_state
        )

        # ---------------------------------------------
        # Invocation ownership
        # ---------------------------------------------

        mock_story_architect.assert_called_once_with(
            story_idea
        )

    # ---------------------------------------------
    # Artifact preservation
    # ---------------------------------------------

    assert initial_state.story_plan is None

    assert updated_state.story_idea == story_idea

    assert updated_state.story_plan is not None

    assert isinstance(
        updated_state.story_plan,
        StoryPlan,
    )

    assert (
        updated_state.story_plan
        == expected_story_plan
    )

    # ---------------------------------------------
    # Stage overwrite protection
    # ---------------------------------------------

    try:
        run_story_architect_stage(
            updated_state
        )

    except ValueError as error:
        assert (
            "Story Architect stage has already completed."
            in str(error)
        )

    else:
        raise AssertionError(
            "Writer's Room allowed the Story Architect "
            "stage to overwrite an existing StoryPlan."
        )

    print(
        "\n--- Writer's Room Story Architect Test ---"
    )
    print("Story idea routed correctly: PASS")
    print("StoryPlan artifact preserved: PASS")
    print("Original state not mutated: PASS")
    print("Stage overwrite protection: PASS")
    print(
        "\nWriter's Room → Story Architect "
        "orchestration contract: PASS"
    )


if __name__ == "__main__":
    main()