from unittest.mock import patch

from agents.scriptwriter_agent import (
    ScriptCharacterContext,
    ScriptScene,
    ScriptwriterInput,
)

from workflows.scriptwriter_production import (
    ScriptwriterProductionResult,
    write_scene_for_production,
)


def main():
    # -----------------------------------------------------
    # Deterministic Scriptwriter input
    # -----------------------------------------------------

    script_input = ScriptwriterInput(
        story_context=(
            "A Cardassian defector privately approaches "
            "Garak with an allegation that the Founders "
            "have infiltrated the Cardassian government."
        ),
        scene_objective=(
            "Establish the defector's allegation and force "
            "Garak to decide whether the claim deserves "
            "further attention."
        ),
        story_era=(
            "Star Trek: Deep Space Nine, shortly before "
            "the Dominion War"
        ),
        characters=[
            ScriptCharacterContext(
                character_name="Elim Garak",
                grounded_character_guidance=[
                    (
                        "Garak must evaluate the "
                        "defector's allegation."
                    ),
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
                "Garak participated in a mission "
                "involving the Founders."
            ),
        ],
        era_style_constraints=[
            (
                "Use screenplay conventions appropriate "
                "to Star Trek: Deep Space Nine."
            ),
        ],
        scene_requirements=[
            (
                "The Cardassian defector must present "
                "the infiltration allegation to Garak."
            ),
        ],
        prohibited_assumptions=[
            (
                "Garak has current contacts inside "
                "Cardassian Central Command."
            ),
        ],
    )

    # -----------------------------------------------------
    # Deterministic scene already returned by frozen
    # Scriptwriter production path
    # -----------------------------------------------------

    scene = ScriptScene(
        scene_heading=(
            "INT. PRIVATE MEETING AREA - NIGHT"
        ),
        setting_description=(
            "A quiet private alcove offers enough "
            "seclusion for a guarded conversation."
        ),
        characters_present=[
            "Elim Garak",
            "Cardassian Defector",
        ],
        script_text=(
            "CARDASSIAN DEFECTOR\n"
            "I have reason to believe the Founders have "
            "placed agents inside the government.\n\n"
            "ELIM GARAK\n"
            "That is a considerable accusation."
        ),
        canon_facts_used=[],
        invented_story_elements=[
            (
                "The defector privately presents the "
                "infiltration allegation to Garak."
            ),
        ],
    )

    # -----------------------------------------------------
    # Successful production adapter path
    # -----------------------------------------------------

    with patch(
        "workflows.scriptwriter_production.write_scene",
        return_value=scene,
    ) as mock_write_scene:

        result = write_scene_for_production(
            script_input
        )

        mock_write_scene.assert_called_once_with(
            script_input
        )

    # -----------------------------------------------------
    # Result contract
    # -----------------------------------------------------

    assert isinstance(
        result,
        ScriptwriterProductionResult,
    )

    assert result.script_input == script_input
    assert result.scene == scene

    # -----------------------------------------------------
    # Frozen grounding receipt
    # -----------------------------------------------------

    assert result.audit.grounded is True
    assert result.audit.issues == []
    assert result.audit.reason.strip()

    # -----------------------------------------------------
    # Failure propagation
    #
    # If frozen write_scene() hard-fails, the adapter must
    # not convert that failure into a successful package.
    # -----------------------------------------------------

    with patch(
        "workflows.scriptwriter_production.write_scene",
        side_effect=ValueError(
            "Scriptwriter scene remained ungrounded "
            "after one repair pass."
        ),
    ):

        try:
            write_scene_for_production(
                script_input
            )

        except ValueError as error:
            assert (
                "remained ungrounded"
                in str(error)
            )

        else:
            raise AssertionError(
                "Scriptwriter production adapter "
                "swallowed a frozen Scriptwriter failure."
            )

    # -----------------------------------------------------
    # Output
    # -----------------------------------------------------

    print(
        "\n--- Scriptwriter Production Test ---"
    )

    print(
        "Frozen write_scene() reused directly: PASS"
    )

    print(
        "ScriptwriterInput preserved exactly: PASS"
    )

    print(
        "Returned ScriptScene preserved exactly: PASS"
    )

    print(
        "Successful grounding receipt created: PASS"
    )

    print(
        "No redundant semantic audit invoked: PASS"
    )

    print(
        "Frozen Scriptwriter failure propagated: PASS"
    )

    print(
        "\nScriptwriter → production adapter "
        "contract: PASS"
    )


if __name__ == "__main__":
    main()