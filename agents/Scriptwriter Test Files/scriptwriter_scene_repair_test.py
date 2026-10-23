from agents.scriptwriter_agent import (
    ScriptCharacterContext,
    ScriptwriterInput,
    ScriptScene,
    ScriptSceneAudit,
    ScriptSceneAuditIssue,
    repair_script_scene,
    audit_script_scene,
    build_script_scene_audit,
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
        "Do not resolve whether the claim is true."
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
    ],
    scene_requirements=[
        "The Cardassian defector approaches Garak privately.",
        (
            "The defector claims that the Founders have infiltrated "
            "the Cardassian government."
        ),
        "Garak evaluates the credibility of the claim.",
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
        "A private alcove where the two Cardassians speak quietly."
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
        "High officials, even those close to Central Command, "
        "are being guided by changeling whispers.\n\n"

        "CARDASSIAN DEFECTOR\n"
        "A contact within the Tal Shiar, disillusioned, "
        "fed me intercepted transmissions.\n\n"

        "CARDASSIAN DEFECTOR\n"
        "The patterns match the Founders' signatures—"
        "subtle, yet unmistakable.\n\n"

        "GARAK\n"
        "I will consider your claim."
    ),
    canon_facts_used=[],
    invented_story_elements=[
        (
            "A Cardassian defector privately presents Garak "
            "with a claim of Founder infiltration."
        )
    ],
)


scene_audit = ScriptSceneAudit(
    grounded=False,
    issues=[
        ScriptSceneAuditIssue(
            segment_id="scene_heading",
            field="scene_heading",
            text=(
                "INT. QUIET CORNER OF QUARK'S BAR - NIGHT"
            ),
            verdict="unsupported",
            reason=(
                "The scene heading names the canon location "
                "Quark's Bar, which is not established by "
                "supplied evidence."
            ),
        ),
        ScriptSceneAuditIssue(
            segment_id="script_text[3]",
            field="script_text",
            text=(
                "High officials, even those close to Central Command, "
                "are being guided by changeling whispers."
            ),
            verdict="unsupported",
            reason=(
                "The statement expands the supplied infiltration "
                "allegation into unsupported factual detail."
            ),
        ),
        ScriptSceneAuditIssue(
            segment_id="script_text[7]",
            field="script_text",
            text=(
                "The patterns match the Founders' signatures—"
                "subtle, yet unmistakable."
            ),
            verdict="unsupported",
            reason=(
                "The statement invents an unsupported capability "
                "to identify Founder signatures."
            ),
        ),
    ],
    reason=(
        "3 ScriptScene segment(s) remain unsupported or prohibited."
    ),
)


print("\n--- ORIGINAL SCENE ---")
print(scene.model_dump_json(indent=2))


repaired_scene = repair_script_scene(
    script_input,
    scene,
    scene_audit,
)


print("\n--- REPAIRED SCENE ---")
print(repaired_scene.model_dump_json(indent=2))


semantic_audit = audit_script_scene(
    script_input,
    repaired_scene,
)


repaired_audit = build_script_scene_audit(
    script_input,
    repaired_scene,
    semantic_audit,
)


print("\n--- REPAIRED SCENE AUDIT ---")
print(repaired_audit.model_dump_json(indent=2))


if not repaired_audit.grounded:
    raise AssertionError(
        "Scriptwriter repair did not produce a grounded scene."
    )


print(
    "\nPASS: Scriptwriter repaired the audited scene "
    "and the repaired scene passed semantic audit."
)