from agents.scriptwriter_agent import (
    ScriptCharacterContext,
    ScriptwriterInput,
    ScriptScene,
    audit_script_scene,
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


test_cases = [
    {
        "text": (
            "The Founders have placed agents within "
            "the Cardassian government."
        ),
        "expected": "grounded",
        "description": (
            "Dialogue restates the supplied fictional allegation."
        ),
    },
    {
        "text": (
            "I think someone has been following me."
        ),
        "expected": "permitted_story_invention",
        "description": (
            "Dialogue introduces a new story-specific suspicion "
            "without establishing it as objective truth."
        ),
    },
    {
        "text": (
            "The Cardassian government had been infiltrated "
            "by the Founders."
        ),
        "expected": "unsupported",
        "description": (
            "Objective action or narration asserts the unresolved "
            "allegation as world fact."
        ),
    },
    {
        "text": (
            "My longtime source inside Central Command confirmed it."
        ),
        "expected": "unsupported",
        "description": (
            "The fictional Cardassian Defector introduces an "
            "unsupported source and current relationship of his own."
        ),
    },
    {
        "text": (
            "A contact within the Tal Shiar fed me "
            "intercepted transmissions."
        ),
        "expected": "permitted_story_invention",
        "description": (
            "A fictional character invents a story-specific source "
            "involving an institution already established by the "
            "supplied canon evidence."
        ),
    },
    {
        "text": (
            "My contact within the Tal Shiar fed me "
            "intercepted transmissions."
        ),
        "expected": "unsupported",
        "description": (
            "An established canon character invents unsupported "
            "current contact and access."
        ),
    },
{
    "text": (
        "A contact within the Tal Shiar, disillusioned, "
        "fed me intercepted transmissions."
    ),
    "expected": "permitted_story_invention",
    "description": (
        "A fictional character invents a story-specific Tal Shiar "
        "contact and a fictional descriptive trait for that contact."
    ),
},

]


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
        "My longtime source inside Central Command confirmed it.\n\n"

        "CARDASSIAN DEFECTOR\n"
        "A contact within the Tal Shiar fed me "
        "intercepted transmissions.\n\n"
        
        "CARDASSIAN DEFECTOR\n"
        "A contact within the Tal Shiar, disillusioned, "
        "fed me intercepted transmissions.\n\n"

        "GARAK\n"
        "My contact within the Tal Shiar fed me "
        "intercepted transmissions."
    ),
    canon_facts_used=[],
    invented_story_elements=[],
)


def main():
    print(
        "Running Scriptwriter dialogue epistemic regression.\n"
    )

    segments = enumerate_script_scene_segments(scene)

    print(
        f"Auditing {len(segments)} deterministic segments.\n"
    )

    semantic_audit = audit_script_scene(
        script_input,
        scene,
    )

    if len(segments) != len(semantic_audit.judgments):
        raise AssertionError(
            "Semantic audit returned the wrong number of judgments. "
            f"Expected {len(segments)}, "
            f"received {len(semantic_audit.judgments)}."
        )

    expected_by_text = {
        case["text"]: case
        for case in test_cases
    }

    observed_cases = 0
    failures = []

    for segment, judgment in zip(
        segments,
        semantic_audit.judgments,
    ):
        case = expected_by_text.get(segment.text)

        if case is None:
            continue

        observed_cases += 1

        actual = judgment.verdict
        expected = case["expected"]

        print("----------------------------------------")
        print(f"TEXT: {segment.text}")
        print(f"EXPECTED: {expected}")
        print(f"ACTUAL: {actual}")
        print(f"REASON: {judgment.reason}")

        if actual == expected:
            print("RESULT: PASS")
        else:
            print("RESULT: FAIL")

            failures.append(
                (
                    segment.text,
                    expected,
                    actual,
                )
            )

        print()

    if observed_cases != len(test_cases):
        raise AssertionError(
            "Not every epistemic test case was found in the "
            "deterministic scene segments. "
            f"Expected {len(test_cases)}, "
            f"found {observed_cases}."
        )

    if failures:
        print("--- REGRESSION FAILED ---")

        for text, expected, actual in failures:
            print(f"\n{text}")
            print(f"Expected: {expected}")
            print(f"Actual:   {actual}")

        raise AssertionError(
            f"{len(failures)} dialogue epistemic "
            "regression case(s) failed."
        )

    print(
        "PASS: All dialogue epistemic regression cases "
        "received the expected verdict."
    )


if __name__ == "__main__":
    main()