from agents.continuity_agent import (
    StoryStateEntry,
    StoryStateLedger,
    add_story_state,
    get_active_story_states,
    learn_scene_states_in_order,
)
from agents.scriptwriter_agent import ScriptScene


def main() -> None:
    print(
        "Running Continuity Agent v0.1R "
        "transactional scene-learning test."
    )

    ledger = StoryStateLedger()

    ledger = add_story_state(
        ledger,
        StoryStateEntry(
            state_id="rod_intact",
            scene_index=0,
            statement=(
                "The data rod is intact."
            ),
        ),
    )

    original_entry_count = len(
        ledger.entries
    )

    original_active_ids = {
        entry.state_id
        for entry in get_active_story_states(
            ledger
        )
    }

    scene = ScriptScene(
        scene_heading=(
            "INT. PRIVATE MEETING AREA - NIGHT"
        ),
        setting_description=(
            "Garak and Kira stand beside "
            "a console."
        ),
        characters_present=[
            "Elim Garak",
            "Kira Nerys",
        ],
        script_text=(
            "Garak destroys the data rod.\n"
            "\n"
            "Kira inserts the same intact "
            "data rod into the console."
        ),
        canon_facts_used=[],
        invented_story_elements=[],
    )

    try:
        learn_scene_states_in_order(
            ledger=ledger,
            scene=scene,
            scene_index=1,
        )
    except ValueError as exc:
        print("\n--- EXPECTED FAILURE ---")
        print(str(exc))
    else:
        raise AssertionError(
            "Contradictory later scene event "
            "did not stop sequential learning."
        )

    # -----------------------------------------------------
    # Critical transaction test
    # -----------------------------------------------------

    assert (
        len(ledger.entries)
        == original_entry_count
    )

    active_ids = {
        entry.state_id
        for entry in get_active_story_states(
            ledger
        )
    }

    assert (
        active_ids
        == original_active_ids
    )

    assert "rod_intact" in active_ids

    assert not any(
        entry.scene_index == 1
        for entry in ledger.entries
    )

    print("\n--- ORIGINAL LEDGER AFTER FAILURE ---")

    for state in ledger.entries:
        print(
            f"{state.state_id} | "
            f"{state.statement}"
        )

    print(
        "\nPASS: a later continuity failure "
        "aborted sequential scene learning "
        "without leaking earlier working-copy "
        "commits into the original ledger."
    )


if __name__ == "__main__":
    main()