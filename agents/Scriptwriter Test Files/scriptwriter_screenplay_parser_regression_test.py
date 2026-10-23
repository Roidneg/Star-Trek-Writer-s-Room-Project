import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from agents.scriptwriter_agent import (
    ScriptScene,
    enumerate_script_scene_segments,
    is_standalone_speaker_cue,
    resolve_script_speaker_name,
)

from agents.scriptwriter_agent import (
    ScriptScene,
    enumerate_script_scene_segments,
    is_standalone_speaker_cue,
    resolve_script_speaker_name,
)


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    print("\n--- Scriptwriter Screenplay Parser Regression Test ---")

    # -----------------------------------------------------
    # TEST 1
    # Standard speaker cue must still work.
    # -----------------------------------------------------

    assert_true(
        is_standalone_speaker_cue("GARAK"),
        "Standard GARAK speaker cue was not recognized.",
    )

    print("Standard speaker cue: PASS")

    # -----------------------------------------------------
    # TEST 2
    # Screenplay continuation cues must resolve to the
    # underlying supplied character.
    #
    # This exact form appeared in the live failure:
    #
    #     GARAK (cont.)
    # -----------------------------------------------------

    assert_true(
        is_standalone_speaker_cue("GARAK (cont.)"),
        "GARAK (cont.) was not recognized as a speaker cue.",
    )

    resolved_name = resolve_script_speaker_name(
        "GARAK (cont.)",
        [
            "Elim Garak",
            "Cardassian Courier",
        ],
    )

    assert_true(
        resolved_name == "Elim Garak",
        (
            "GARAK (cont.) did not resolve to Elim Garak. "
            f"Received: {resolved_name!r}"
        ),
    )

    print("Continuation speaker cue: PASS")

    # -----------------------------------------------------
    # TEST 3
    # Wrapped dialogue must remain one semantic dialogue
    # statement instead of being broken into physical-line
    # fragments.
    #
    # This mirrors the live Scriptwriter output.
    # -----------------------------------------------------

    scene = ScriptScene(
        scene_heading=(
            "INT. PRIVATE ROOM - DEEP SPACE NINE - NIGHT"
        ),
        setting_description=(
            "A private room aboard Deep Space Nine."
        ),
        characters_present=[
            "Elim Garak",
            "Cardassian Courier",
        ],
        script_text=(
            "GARAK\n"
            "Information is a weapon, my friend, but only if its edge is\n"
            "sharp. A damaged rod may hold truth, or it may be a trap.\n"
            "I must consider the risk of bringing unverified data to\n"
            "Commander Sisko.\n"
            "\n"
            "GARAK (cont.)\n"
            "I will examine the fragment. If the cipher yields anything\n"
            "coherent, I will decide whether Sisko needs to hear it.\n"
            "Until then, keep this between us.\n"
            "\n"
            "CARDASSIAN COURIER\n"
            "As you wish, Garak.\n"
        ),
        canon_facts_used=[],
        invented_story_elements=[],
    )

    segments = enumerate_script_scene_segments(
        scene
    )

    for segment in segments:
        print(
            segment.segment_id,
            "|",
            segment.segment_type,
            "|",
            segment.speaker,
            "|",
            segment.text,
        )

    garak_dialogue = [
        segment
        for segment in segments
        if (
            segment.segment_type == "dialogue"
            and segment.speaker is not None
            and segment.speaker.upper().startswith(
                "GARAK"
            )
        )
    ]

    garak_dialogue_text = [
        segment.text
        for segment in garak_dialogue
    ]

    expected_dialogue = [
        (
            "Information is a weapon, my friend, "
            "but only if its edge is sharp."
        ),
        (
            "A damaged rod may hold truth, "
            "or it may be a trap."
        ),
        (
            "I must consider the risk of bringing "
            "unverified data to Commander Sisko."
        ),
        "I will examine the fragment.",
        (
            "If the cipher yields anything coherent, "
            "I will decide whether Sisko needs to hear it."
        ),
        "Until then, keep this between us.",
    ]

    assert_true(
        garak_dialogue_text == expected_dialogue,
        (
            "\nWrapped Garak dialogue was parsed incorrectly.\n\n"
            f"Expected:\n{expected_dialogue}\n\n"
            f"Received:\n{garak_dialogue_text}"
        ),
    )

    print("Wrapped dialogue preservation: PASS")

    # -----------------------------------------------------
    # TEST 4
    # Continuation dialogue itself must belong to Garak,
    # never become objective screenplay action.
    # -----------------------------------------------------

    continuation_texts = {
        "I will examine the fragment.",
        (
            "If the cipher yields anything coherent, "
            "I will decide whether Sisko needs to hear it."
        ),
        "Until then, keep this between us.",
    }

    misclassified = [
        segment
        for segment in segments
        if (
            segment.text in continuation_texts
            and segment.segment_type != "dialogue"
        )
    ]

    assert_true(
        not misclassified,
        (
            "Dialogue following GARAK (cont.) was "
            "misclassified as screenplay action:\n"
            + "\n".join(
                repr(segment)
                for segment in misclassified
            )
        ),
    )

    # -----------------------------------------------------
    # TEST 5
    # Wrapped screenplay action must be reconstructed
    # before semantic sentence segmentation.
    #
    # This reproduces the wrapping seen in the saved
    # production Scriptwriter scene.
    # -----------------------------------------------------

    action_scene = ScriptScene(
        scene_heading=(
            "INT. PRIVATE ROOM - DEEP SPACE NINE - NIGHT"
        ),
        setting_description=(
            "A private room aboard Deep Space Nine."
        ),
        characters_present=[
            "Elim Garak",
            "Cardassian Courier",
        ],
        script_text=(
            "He places the rod on the table, "
            "the cloth unraveling to reveal\n"
            "a scarred metal cylinder.\n"
            "\n"
            "Garak\n"
            "lifts the rod, his fingers brushing "
            "the worn surface, and\n"
            "turns the holo-screen toward a "
            "flickering data analysis program.\n"
            "\n"
            "The scene ends on his thoughtful "
            "expression as the screen\n"
            "begins to decode the corrupted files.\n"
        ),
        canon_facts_used=[],
        invented_story_elements=[],
    )

    action_segments = enumerate_script_scene_segments(
        action_scene
    )

    parsed_actions = [
        segment.text
        for segment in action_segments
        if segment.segment_type == "action"
    ]

    expected_actions = [
        (
            "He places the rod on the table, "
            "the cloth unraveling to reveal "
            "a scarred metal cylinder."
        ),
        (
            "Garak lifts the rod, his fingers "
            "brushing the worn surface, and "
            "turns the holo-screen toward a "
            "flickering data analysis program."
        ),
        (
            "The scene ends on his thoughtful "
            "expression as the screen begins "
            "to decode the corrupted files."
        ),
    ]

    assert_true(
        parsed_actions == expected_actions,
        (
            "\nWrapped screenplay action was "
            "parsed incorrectly.\n\n"
            f"Expected:\n{expected_actions}\n\n"
            f"Received:\n{parsed_actions}"
        ),
    )

    print("Wrapped action preservation: PASS")

    print("Continuation dialogue attribution: PASS")

    print(
        "\nScriptwriter screenplay parser regression: PASS"
    )



if __name__ == "__main__":
    main()