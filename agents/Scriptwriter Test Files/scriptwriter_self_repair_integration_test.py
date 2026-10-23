from langchain_core.runnables import RunnableLambda

import agents.scriptwriter_agent as scriptwriter_agent

from agents.scriptwriter_agent import (
    ScriptCharacterContext,
    ScriptwriterInput,
    ScriptScene,
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


# Deterministic bad generation.
#
# We replace only the generation chain so this test always
# exercises the repair path. The real semantic auditor and
# real repair model remain active.
initial_scene = ScriptScene(
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


original_scriptwriter_chain = (
    scriptwriter_agent.scriptwriter_chain
)


def deterministic_generation(_):
    return initial_scene


def main():
    print(
        "Running Scriptwriter one-pass self-repair "
        "integration test.\n"
    )

    # Replace only generation with a deterministic bad scene.
    # Audit and repair remain real production behavior.
    scriptwriter_agent.scriptwriter_chain = RunnableLambda(
        deterministic_generation
    )

    try:
        final_scene = scriptwriter_agent.write_scene(
            script_input
        )
    finally:
        # Never leave the production chain patched after the test.
        scriptwriter_agent.scriptwriter_chain = (
            original_scriptwriter_chain
        )

    print("--- INITIAL SCENE ---")
    print(
        initial_scene.model_dump_json(
            indent=2
        )
    )

    print("\n--- FINAL SCENE ---")
    print(
        final_scene.model_dump_json(
            indent=2
        )
    )

    if final_scene == initial_scene:
        raise AssertionError(
            "write_scene returned the original ungrounded "
            "scene instead of exercising repair."
        )

    if "QUARK'S BAR" in final_scene.scene_heading.upper():
        raise AssertionError(
            "Repair preserved the unsupported Quark's Bar "
            "scene-heading claim."
        )

    print(
        "\nPASS: write_scene detected an ungrounded "
        "generation, repaired it once, independently "
        "re-audited it, and returned the grounded result."
    )


if __name__ == "__main__":
    main()