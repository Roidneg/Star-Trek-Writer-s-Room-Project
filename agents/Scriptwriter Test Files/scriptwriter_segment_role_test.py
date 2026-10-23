from agents.scriptwriter_agent import (
    ScriptScene,
    enumerate_script_scene_segments,
)


scene = ScriptScene(
    scene_heading="INT. PRIVATE MEETING AREA - NIGHT",
    setting_description=(
        "Garak and the Cardassian Defector speak privately."
    ),
    characters_present=[
        "Elim Garak",
        "Cardassian Defector",
    ],
    script_text=(
        "CARDASSIAN DEFECTOR\n"
        "The Founders have placed agents within "
        "the Cardassian government.\n\n"

        "CARDASSIAN DEFECTOR\n"
        "I think someone has been following me.\n\n"

        "The Cardassian government had been infiltrated "
        "by the Founders.\n\n"

        "CARDASSIAN DEFECTOR\n"
        "My longtime source inside Central Command confirmed it."
    ),
    canon_facts_used=[],
    invented_story_elements=[],
)


segments = enumerate_script_scene_segments(scene)


actual = [
    (
        segment.segment_id,
        segment.segment_type,
        segment.speaker,
        segment.text,
    )
    for segment in segments
]


expected = [
    (
        "scene_heading",
        "scene_heading",
        None,
        "INT. PRIVATE MEETING AREA - NIGHT",
    ),
    (
        "setting_description",
        "setting_description",
        None,
        "Garak and the Cardassian Defector speak privately.",
    ),
    (
        "script_text[0]",
        "speaker_cue",
        "CARDASSIAN DEFECTOR",
        "CARDASSIAN DEFECTOR",
    ),
    (
        "script_text[1]",
        "dialogue",
        "CARDASSIAN DEFECTOR",
        (
            "The Founders have placed agents within "
            "the Cardassian government."
        ),
    ),
    (
        "script_text[2]",
        "speaker_cue",
        "CARDASSIAN DEFECTOR",
        "CARDASSIAN DEFECTOR",
    ),
    (
        "script_text[3]",
        "dialogue",
        "CARDASSIAN DEFECTOR",
        "I think someone has been following me.",
    ),
    (
        "script_text[4]",
        "action",
        None,
        (
            "The Cardassian government had been infiltrated "
            "by the Founders."
        ),
    ),
    (
        "script_text[5]",
        "speaker_cue",
        "CARDASSIAN DEFECTOR",
        "CARDASSIAN DEFECTOR",
    ),
    (
        "script_text[6]",
        "dialogue",
        "CARDASSIAN DEFECTOR",
        (
            "My longtime source inside Central Command "
            "confirmed it."
        ),
    ),
]


assert actual == expected, (
    "Deterministic screenplay role parsing failed.\n\n"
    f"EXPECTED:\n{expected}\n\n"
    f"ACTUAL:\n{actual}"
)


print(
    "PASS: ScriptScene screenplay roles and speakers "
    "are deterministic."
)

for segment in segments:
    print(
        f"{segment.segment_id} | "
        f"{segment.segment_type} | "
        f"speaker={segment.speaker!r} | "
        f"{segment.text}"
    )