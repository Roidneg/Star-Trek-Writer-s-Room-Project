from agents.continuity_agent import (
    StoryStateEntry,
    StoryStateLedger,
    add_story_state,
    get_active_story_states,
    get_story_state_semantic_statement,
    learn_scene_states_in_order,
    review_script_scene_continuity,
)
from agents.scriptwriter_agent import ScriptScene


def main() -> None:
    print(
        "Running Continuity Agent v0.1S "
        "two-scene integration test."
    )

    # -----------------------------------------------------
    # Initial authoritative story state
    # -----------------------------------------------------

    ledger = StoryStateLedger()

    ledger = add_story_state(
        ledger,
        StoryStateEntry(
            state_id="rod_defector",
            scene_index=0,
            statement=(
                "The Cardassian defector "
                "possesses the data rod."
            ),
        ),
    )

    # -----------------------------------------------------
    # Scene 1:
    # valid possession transfer
    # -----------------------------------------------------

    scene_one = ScriptScene(
        scene_heading=(
            "INT. PRIVATE MEETING AREA - NIGHT"
        ),
        setting_description=(
            "Garak and the Cardassian defector "
            "stand beside a table."
        ),
        characters_present=[
            "Elim Garak",
            "Cardassian Defector",
        ],
        script_text=(
            "The Cardassian defector hands "
            "the data rod to Garak."
        ),
        canon_facts_used=[],
        invented_story_elements=[],
    )

    learning_result = (
        learn_scene_states_in_order(
            ledger=ledger,
            scene=scene_one,
            scene_index=1,
        )
    )

    ledger_after_scene_one = (
        learning_result.ledger
    )

    print("\n--- AFTER SCENE 1 ---")

    active_states = get_active_story_states(
        ledger_after_scene_one
    )

    for state in active_states:
        print(
            f"{state.state_id} | "
            f"{state.statement}"
        )
        print(
            "Normalized: "
            f"{get_story_state_semantic_statement(state)}"
        )

    assert len(active_states) == 1

    assert active_states[0].statement == (
        "The Cardassian defector hands "
        "the data rod to Garak."
    )

    assert (
        get_story_state_semantic_statement(
            active_states[0]
        )
        == "Garak possesses the data rod."
    )

    # -----------------------------------------------------
    # Scene 2:
    # genuine contradiction
    #
    # No transfer away from Garak is established.
    # The later action assumes the defector still
    # possesses the same rod.
    # -----------------------------------------------------

    scene_two = ScriptScene(
        scene_heading=(
            "INT. CORRIDOR - LATER"
        ),
        setting_description=(
            "The Cardassian defector stands "
            "alone in the corridor."
        ),
        characters_present=[
            "Cardassian Defector",
        ],
        script_text=(
            "The Cardassian defector inserts "
            "the same data rod he still possesses "
            "into a wall console."
        ),
        canon_facts_used=[],
        invented_story_elements=[],
    )

    review = review_script_scene_continuity(
        ledger=ledger_after_scene_one,
        scene=scene_two,
    )

    print("\n--- SCENE 2 REVIEW ---")
    print(
        f"Status: "
        f"{review.assessment.status}"
    )
    print(
        "Contradictions: "
        f"{review.assessment.contradiction_count}"
    )
    print(
        "Needs context: "
        f"{review.assessment.insufficient_context_count}"
    )

    for issue in review.assessment.issues:
        print(
            f"\n{issue.segment_id} | "
            f"{issue.issue_type}"
        )
        print(
            f"Statement: "
            f"{issue.statement}"
        )
        print(
            f"Against: "
            f"{issue.established_state}"
        )
        print(
            f"Reason: "
            f"{issue.reason}"
        )

    assert (
        review.assessment.status
        == "contradiction"
    )

    assert (
        review.assessment.contradiction_count
        >= 1
    )

    assert any(
        issue.issue_type == "contradiction"
        and (
            issue.established_state
            == "Garak possesses the data rod."
        )
        for issue in review.assessment.issues
    )

    print(
        "\nPASS: Continuity learned normalized "
        "state from Scene 1, carried that "
        "authoritative state into Scene 2, "
        "and detected a cross-scene "
        "contradiction."
    )


if __name__ == "__main__":
    main()