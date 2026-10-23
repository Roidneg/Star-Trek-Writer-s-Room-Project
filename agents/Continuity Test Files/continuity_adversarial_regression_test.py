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


def semantic_states(
    ledger: StoryStateLedger,
) -> list[str]:
    return [
        get_story_state_semantic_statement(
            entry
        )
        for entry in get_active_story_states(
            ledger
        )
    ]


def main() -> None:
    print(
        "Running Continuity Agent v0.1T "
        "final adversarial regression."
    )

    # =====================================================
    # INITIAL AUTHORITATIVE STORY STATE
    # =====================================================

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

    ledger = add_story_state(
        ledger,
        StoryStateEntry(
            state_id="console_intact",
            scene_index=0,
            statement=(
                "The wall console is intact."
            ),
        ),
    )

    ledger = add_story_state(
        ledger,
        StoryStateEntry(
            state_id="defector_alive",
            scene_index=0,
            statement=(
                "The Cardassian defector "
                "is alive."
            ),
        ),
    )

    print("\n--- INITIAL ACTIVE STATES ---")

    for state in semantic_states(ledger):
        print(state)

    # =====================================================
    # SCENE 1
    #
    # Tests:
    # - possession progression
    # - transient event rejection
    # - second possession progression
    # - physical-condition progression
    # - sequential state learning
    # =====================================================

    scene_one = ScriptScene(
        scene_heading=(
            "INT. PRIVATE MEETING AREA - NIGHT"
        ),
        setting_description=(
            "Garak, Kira, and the Cardassian "
            "defector stand near a wall console."
        ),
        characters_present=[
            "Elim Garak",
            "Kira Nerys",
            "Cardassian Defector",
        ],
        script_text=(
            "The Cardassian defector hands "
            "the data rod to Garak.\n"
            "\n"
            "Garak smiles.\n"
            "\n"
            "Garak hands the data rod "
            "to Kira.\n"
            "\n"
            "Kira destroys the wall console."
        ),
        canon_facts_used=[],
        invented_story_elements=[],
    )

    scene_one_result = (
        learn_scene_states_in_order(
            ledger=ledger,
            scene=scene_one,
            scene_index=1,
        )
    )

    ledger_after_scene_one = (
        scene_one_result.ledger
    )

    print("\n--- SCENE 1 LEARNING ---")

    print(
        "Committed states:",
        scene_one_result.committed_state_ids,
    )

    print(
        "Transient segments:",
        scene_one_result.transient_segment_ids,
    )

    # Three durable changes:
    #
    # defector -> Garak
    # Garak -> Kira
    # console intact -> destroyed
    assert len(
        scene_one_result.committed_state_ids
    ) == 3

    # Garak smiles must not become story state.
    assert len(
        scene_one_result.transient_segment_ids
    ) == 1

    active_after_scene_one = (
        semantic_states(
            ledger_after_scene_one
        )
    )

    print("\n--- ACTIVE AFTER SCENE 1 ---")

    for state in active_after_scene_one:
        print(state)

    # We should have exactly three active dimensions:
    #
    # rod possession
    # console condition
    # defector alive/dead condition
    assert len(active_after_scene_one) == 3

    lowered_states = [
        state.lower()
        for state in active_after_scene_one
    ]

    assert any(
        "kira" in state
        and "data rod" in state
        for state in lowered_states
    )

    assert any(
        "console" in state
        and "destroyed" in state
        for state in lowered_states
    )

    assert any(
        "cardassian defector" in state
        and "alive" in state
        for state in lowered_states
    )

    assert not any(
        "smiles" in state
        for state in lowered_states
    )

    # =====================================================
    # SCENE 2
    #
    # Unrelated action involving same character.
    #
    # This MUST NOT become:
    # - contradiction
    # - state progression against alive/dead
    # - needs_context
    #
    # It should simply be continuity-clear.
    # =====================================================

    scene_two = ScriptScene(
        scene_heading=(
            "INT. CORRIDOR - LATER"
        ),
        setting_description=(
            "The Cardassian defector stands "
            "near the doorway."
        ),
        characters_present=[
            "Cardassian Defector",
        ],
        script_text=(
            "The Cardassian defector "
            "leaves the room."
        ),
        canon_facts_used=[],
        invented_story_elements=[],
    )

    scene_two_review = (
        review_script_scene_continuity(
            ledger=ledger_after_scene_one,
            scene=scene_two,
        )
    )

    print("\n--- SCENE 2 REVIEW ---")
    print(
        "Status:",
        scene_two_review.assessment.status,
    )
    print(
        "Contradictions:",
        scene_two_review.assessment
        .contradiction_count,
    )
    print(
        "Needs context:",
        scene_two_review.assessment
        .insufficient_context_count,
    )

    assert (
        scene_two_review.assessment.status
        == "clear"
    )

    assert (
        scene_two_review.assessment
        .contradiction_count
        == 0
    )

    assert (
        scene_two_review.assessment
        .insufficient_context_count
        == 0
    )

    # =====================================================
    # SCENE 3
    #
    # Two explicit cross-scene contradictions:
    #
    # 1. Garak still possesses rod
    #    vs Kira possesses rod.
    #
    # 2. Console intact
    #    vs console destroyed.
    # =====================================================

    scene_three = ScriptScene(
        scene_heading=(
            "INT. OPERATIONS AREA - LATER"
        ),
        setting_description=(
            "Garak stands beside the damaged "
            "operations equipment."
        ),
        characters_present=[
            "Elim Garak",
        ],
        script_text=(
            "Garak inserts the same data rod "
            "he still possesses into a console.\n"
            "\n"
            "The wall console is intact "
            "and operational."
        ),
        canon_facts_used=[],
        invented_story_elements=[],
    )

    scene_three_review = (
        review_script_scene_continuity(
            ledger=ledger_after_scene_one,
            scene=scene_three,
        )
    )

    print("\n--- SCENE 3 ADVERSARIAL REVIEW ---")
    print(
        "Status:",
        scene_three_review.assessment.status,
    )
    print(
        "Contradictions:",
        scene_three_review.assessment
        .contradiction_count,
    )
    print(
        "Needs context:",
        scene_three_review.assessment
        .insufficient_context_count,
    )

    for issue in (
        scene_three_review.assessment.issues
    ):
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
        scene_three_review.assessment.status
        == "contradiction"
    )

    assert (
        scene_three_review.assessment
        .contradiction_count
        >= 2
    )

    # Confirm possession contradiction.
    assert any(
        issue.issue_type == "contradiction"
        and "garak" in issue.statement.lower()
        and "kira" in (
            issue.established_state.lower()
        )
        and "data rod" in (
            issue.established_state.lower()
        )
        for issue
        in scene_three_review.assessment.issues
    )

    # Confirm console-condition contradiction.
    assert any(
        issue.issue_type == "contradiction"
        and "console" in issue.statement.lower()
        and "destroyed" in (
            issue.established_state.lower()
        )
        for issue
        in scene_three_review.assessment.issues
    )

    # =====================================================
    # FAILED SCENE MUST NOT ALTER LEDGER
    # =====================================================

    entry_count_before_failure = len(
        ledger_after_scene_one.entries
    )

    active_before_failure = {
        (
            entry.state_id,
            get_story_state_semantic_statement(
                entry
            ),
        )
        for entry in get_active_story_states(
            ledger_after_scene_one
        )
    }

    try:
        learn_scene_states_in_order(
            ledger=ledger_after_scene_one,
            scene=scene_three,
            scene_index=3,
        )

    except ValueError as exc:
        print(
            "\n--- EXPECTED LEARNING FAILURE ---"
        )
        print(str(exc))

    else:
        raise AssertionError(
            "Contradictory adversarial scene "
            "was incorrectly committed."
        )

    # Original authoritative ledger must remain
    # completely unchanged.
    assert (
        len(ledger_after_scene_one.entries)
        == entry_count_before_failure
    )

    active_after_failure = {
        (
            entry.state_id,
            get_story_state_semantic_statement(
                entry
            ),
        )
        for entry in get_active_story_states(
            ledger_after_scene_one
        )
    }

    assert (
        active_after_failure
        == active_before_failure
    )

    assert not any(
        entry.scene_index == 3
        for entry
        in ledger_after_scene_one.entries
    )

    # =====================================================
    # FINAL RESULT
    # =====================================================

    print(
        "\nPASS: Continuity survived the "
        "final adversarial regression: "
        "sequential state progression, "
        "normalized cross-scene memory, "
        "transient-event exclusion, "
        "state-dimension separation, "
        "multiple contradiction detection, "
        "and atomic failure behavior all "
        "remained intact."
    )


if __name__ == "__main__":
    main()