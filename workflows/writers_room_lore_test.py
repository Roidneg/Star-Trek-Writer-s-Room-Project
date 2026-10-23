from unittest.mock import (
    call,
    patch,
)

from agents.story_architect import StoryPlan
from agents.lore_agent import LoreResponseModel

from workflows.writers_room import (
    WritersRoomState,
    run_lore_stage,
)


def main():
    lore_question_one = (
        "What does canon establish about Garak's "
        "knowledge of the Founders before the Dominion War?"
    )

    lore_question_two = (
        "What does canon establish about Cardassian "
        "political conditions shortly before the Dominion War?"
    )

    story_plan = StoryPlan(
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
            lore_question_one,
            lore_question_two,
        ],
    )

    supported_response = LoreResponseModel(
        topic="Garak and the Founders",
        summary=(
            "Supplied test evidence establishes the "
            "requested factual premise and constraints."
        ),
        sources=[
            "https://memory-alpha.fandom.com/wiki/Test_One"
        ],
        tools_used=[
            "memory_alpha_search_tool",
        ],
        evidence_supported=True,
        premise_supported=True,
    )

    unresolved_response = LoreResponseModel(
        topic="Cardassian politics",
        summary=(
            "Supplied test evidence establishes part of "
            "the subject but not every requested constraint."
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
    # Prerequisite protection
    # -----------------------------------------------------

    empty_state = WritersRoomState(
        story_idea="Test story idea.",
    )

    try:
        run_lore_stage(
            empty_state
        )

    except ValueError as error:
        assert (
            "Lore stage requires a completed StoryPlan."
            in str(error)
        )

    else:
        raise AssertionError(
            "Writer's Room allowed Lore to run "
            "without a completed StoryPlan."
        )

    # -----------------------------------------------------
    # Valid StoryPlan state
    # -----------------------------------------------------

    initial_state = WritersRoomState(
        story_idea=(
            "Shortly before the Dominion War, Garak is "
            "approached by a Cardassian defector."
        ),
        story_plan=story_plan,
    )

    assert initial_state.lore_results == []

    # -----------------------------------------------------
    # Lore orchestration
    # -----------------------------------------------------

    with patch(
        "workflows.writers_room.run_lore_query",
        side_effect=[
            supported_response,
            unresolved_response,
        ],
    ) as mock_lore:

        updated_state = run_lore_stage(
            initial_state
        )

        assert mock_lore.call_count == 2

        assert mock_lore.call_args_list == [
            call(lore_question_one),
            call(lore_question_two),
        ]

    # -----------------------------------------------------
    # Original state preservation
    # -----------------------------------------------------

    assert initial_state.lore_results == []

    # -----------------------------------------------------
    # Result count and ordering
    # -----------------------------------------------------

    assert len(updated_state.lore_results) == 2

    first_result = updated_state.lore_results[0]
    second_result = updated_state.lore_results[1]

    # -----------------------------------------------------
    # Stable Writer's Room IDs
    # -----------------------------------------------------

    assert first_result.research_id == "lore_000"
    assert second_result.research_id == "lore_001"

    # -----------------------------------------------------
    # Exact provenance preservation
    # -----------------------------------------------------

    assert (
        first_result.research_question
        == lore_question_one
    )

    assert (
        second_result.research_question
        == lore_question_two
    )

    # -----------------------------------------------------
    # Frozen Lore artifacts preserved exactly
    # -----------------------------------------------------

    assert (
        first_result.response
        == supported_response
    )

    assert (
        second_result.response
        == unresolved_response
    )

    # Writer's Room must preserve Lore status rather
    # than filtering unresolved evidence.
    assert first_result.response.evidence_supported is True
    assert second_result.response.evidence_supported is False

    assert first_result.response.premise_supported is True
    assert second_result.response.premise_supported is True

    # -----------------------------------------------------
    # Stage overwrite protection
    # -----------------------------------------------------

    try:
        run_lore_stage(
            updated_state
        )

    except ValueError as error:
        assert (
            "Lore stage has already completed."
            in str(error)
        )

    else:
        raise AssertionError(
            "Writer's Room allowed the Lore stage "
            "to overwrite existing Lore artifacts."
        )

    print("\n--- Writer's Room Lore Test ---")
    print("StoryPlan prerequisite enforced: PASS")
    print("Lore questions routed in order: PASS")
    print("Stable research IDs assigned: PASS")
    print("Question provenance preserved: PASS")
    print("Lore artifacts preserved exactly: PASS")
    print("Unresolved Lore status preserved: PASS")
    print("Original state not mutated: PASS")
    print("Stage overwrite protection: PASS")

    print(
        "\nWriter's Room → Lore Agent "
        "orchestration contract: PASS"
    )


if __name__ == "__main__":
    main()