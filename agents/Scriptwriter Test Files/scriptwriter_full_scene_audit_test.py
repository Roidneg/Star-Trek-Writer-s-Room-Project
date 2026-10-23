from agents.scriptwriter_agent import (
    ScriptCharacterContext,
    ScriptwriterInput,
    ScriptScene,
    audit_script_scene,
    build_script_scene_audit,
    enumerate_script_scene_segments,
)


script_input = ScriptwriterInput(
    story_context=(
        "A Cardassian defector secretly approaches Garak with a claim "
        "that the Founders have infiltrated the Cardassian government. "
        "Garak must decide how seriously to take the claim and whether "
        "to involve DS9."
    ),
    scene_objective=(
        "Establish the defector's claim and show Garak evaluating it. "
        "Do not resolve whether the claim is true. End before Garak "
        "decides whether to involve DS9 command."
    ),
    story_era=(
        "Star Trek: Deep Space Nine, shortly before the Dominion War"
    ),
    characters=[
        ScriptCharacterContext(
            character_name="Elim Garak",
            grounded_character_guidance=[
                (
                    "Consult his knowledge of the Founders and prior "
                    "mission experience to evaluate the claim."
                )
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
            "Garak had knowledge of the Founders and the Dominion "
            "before the Dominion War."
        ),
        (
            "Garak joined Enabran Tain on a joint Obsidian Order/"
            "Tal Shiar mission to destroy the Founders' homeworld."
        ),
        (
            "Garak later received information directly from a "
            "Female Changeling."
        ),
    ],
    era_style_constraints=[
        "Write in a style appropriate to Star Trek: Deep Space Nine.",
        "Use screenplay-style action and dialogue.",
        "Avoid modern slang.",
        (
            "Do not introduce technological capabilities beyond "
            "what the scene requires."
        ),
    ],
    scene_requirements=[
        "The Cardassian defector approaches Garak privately.",
        (
            "The defector claims that the Founders have infiltrated "
            "the Cardassian government."
        ),
        "Garak evaluates the credibility of the claim.",
        (
            "The scene ends before Garak decides whether to involve "
            "DS9 command."
        ),
    ],
    prohibited_assumptions=[
        (
            "Garak has current contacts inside Cardassian "
            "Central Command."
        ),
        "Garak has a current Cardassian intelligence network.",
        (
            "Garak can contact former Cardassian "
            "intelligence associates."
        ),
    ],
)


scene = ScriptScene(
    scene_heading=(
        "INT. QUIET CORNER OF QUARK'S BAR - NIGHT"
    ),
    setting_description=(
        "A dimly lit alcove behind the bar, shielded from the main "
        "bustle. The hum of the station's life support is faint. "
        "A single table with a holo-lamp casts soft light over "
        "two figures."
    ),
    characters_present=[
        "Elim Garak",
        "Cardassian Defector",
    ],
    script_text=(
        "The CARDASSIAN DEFECTOR, a gaunt officer in a civilian coat, "
        "slips into the alcove and slides into the seat opposite GARAK, "
        "who studies the holo-lamp with a measured gaze.\n\n"
        "CARDASSIAN DEFECTOR\n"
        "(voice low, urgent)\n"
        "Garak. I come with news that may change the balance of power.\n\n"
        "GARAK\n"
        "(tilting his head, amused)\n"
        "News is a commodity, my friend. What makes yours worth the risk?\n\n"
        "CARDASSIAN DEFECTOR\n"
        "The Founders—those shape-shifters—have placed agents within "
        "the Cardassian government. High officials, even those close "
        "to Central Command, are being guided by changeling whispers.\n\n"
        "GARAK\n"
        "(leaning back, fingers steepled)\n"
        "You speak of infiltration at the highest echelons. "
        "How did you come by this intelligence?\n\n"
        "CARDASSIAN DEFECTOR\n"
        "A contact within the Tal Shiar, disillusioned, fed me "
        "intercepted transmissions. The patterns match the Founders' "
        "signatures—subtle, yet unmistakable.\n\n"
        "GARAK\n"
        "(eyes narrowing, recalling)\n"
        "I have seen the hand of the Founders before. "
        "My own past missions with Enabran Tain, the joint Obsidian "
        "Order–Tal Shiar strike against their homeworld, taught me "
        "the shape of their shadow. "
        "And I have heard a Female Changeling speak directly to me.\n\n"
        "CARDASSIAN DEFECTOR\n"
        "Then you understand the danger. If the Dominion moves through "
        "Cardassian channels, the war will begin before we are ready.\n\n"
        "GARAK\n"
        "(softly, almost to himself)\n"
        "Information is a weapon, but also a poison. "
        "I must weigh its truth against the cost of believing it.\n\n"
        "CARDASSIAN DEFECTOR\n"
        "I ask only that you consider it seriously. "
        "The lives of many may hinge on your judgment.\n\n"
        "GARAK\n"
        "(standing, smoothing his coat)\n"
        "I will contemplate your claim. "
        "For now, we remain in the shadows.\n\n"
        "The DEFECTOR nods, rising to leave. "
        "GARAK watches the holo-lamp flicker, his expression "
        "unreadable as the scene fades."
    ),
    canon_facts_used=[
        (
            "Garak had knowledge of the Founders and the Dominion "
            "before the Dominion War."
        ),
        (
            "Garak joined Enabran Tain on a joint Obsidian Order/"
            "Tal Shiar mission to destroy the Founders' homeworld."
        ),
        (
            "Garak later received information directly from a "
            "Female Changeling."
        ),
    ],
    invented_story_elements=[
        (
            "Cardassian defector approaches Garak in a private "
            "alcove of Quark's bar."
        ),
        (
            "Defector claims the Founders have infiltrated the "
            "Cardassian government."
        ),
        (
            "Garak evaluates the credibility of the claim based "
            "on his past experience."
        ),
        (
            "Scene ends before Garak decides whether to involve "
            "DS9 command."
        ),
    ],
)


segments = enumerate_script_scene_segments(scene)

print(
    f"\nAuditing {len(segments)} deterministic scene segments."
)


semantic_audit = audit_script_scene(
    script_input,
    scene,
)

scene_audit = build_script_scene_audit(
    script_input,
    scene,
    semantic_audit,
)


verdict_counts = {
    "grounded": 0,
    "permitted_story_invention": 0,
    "unsupported": 0,
    "prohibited": 0,
}

for judgment in semantic_audit.judgments:
    verdict_counts[judgment.verdict] += 1


print("\n--- VERDICT COUNTS ---")

for verdict, count in verdict_counts.items():
    print(f"{verdict}: {count}")


print("\n--- AUDIT ISSUES ---")

if not scene_audit.issues:
    print("No issues.")

for issue in scene_audit.issues:
    print(
        f"\n{issue.segment_id} | {issue.verdict}"
    )
    print(issue.text)
    print(f"REASON: {issue.reason}")


print("\n--- FINAL SCENE VERDICT ---")
print(f"grounded: {scene_audit.grounded}")
print(scene_audit.reason)