from agents.evaluation_agent import (
    prepare_era_style_evaluation,
    review_subject_against_criteria,
)

from agents.scriptwriter_agent import (
    ScriptCharacterContext,
    ScriptwriterInput,
)


def build_script_input(
    era_style_constraints: list[str],
) -> ScriptwriterInput:
    return ScriptwriterInput(
        story_context=(
            "A Cardassian defector meets Garak "
            "privately with sensitive intelligence."
        ),
        scene_objective=(
            "Garak evaluates the defector's claim."
        ),
        story_era=(
            "Deep Space Nine, shortly before "
            "the Dominion War."
        ),
        characters=[
            ScriptCharacterContext(
                character_name="Elim Garak",
                grounded_character_guidance=[],
                grounded_voice_guidance=[],
            ),
        ],
        verified_canon_facts=[],
        era_style_constraints=(
            era_style_constraints
        ),
        scene_requirements=[],
        prohibited_assumptions=[],
    )


def main() -> None:
    print(
        "Running Evaluation Agent v0.1K "
        "era/style preparation tests."
    )

    # -----------------------------------------------------
    # Explicit constraints -> Evaluation criterion
    # -----------------------------------------------------

    script_input = build_script_input(
        [
            (
                "Use restrained, professional "
                "DS9-era dialogue rather than "
                "modern internet slang."
            ),
            (
                "Use Star Trek-era technological "
                "terminology rather than modern "
                "consumer-tech vocabulary."
            ),
            (
                "Favor concise dramatic exchanges "
                "appropriate to a 1990s-era "
                "Star Trek television scene."
            ),
        ]
    )

    prepared = prepare_era_style_evaluation(
        script_input
    )

    print("\n--- PREPARED STYLE CRITERION ---")

    assert prepared.criterion is not None

    assert (
        prepared.precomputed_result
        is None
    )

    print(
        "Criterion ID:",
        prepared.criterion.criterion_id,
    )

    print(
        "Category:",
        prepared.criterion.category,
    )

    print("\nContext:")

    for item in (
        prepared.criterion.evaluation_context
    ):
        print(item)

    assert (
        prepared.criterion.criterion_id
        == "era_style_review"
    )

    assert (
        prepared.criterion.category
        == "era_style"
    )

    assert any(
        "Deep Space Nine"
        in item
        for item
        in prepared.criterion.evaluation_context
    )

    assert any(
        "modern internet slang"
        in item
        for item
        in prepared.criterion.evaluation_context
    )

    # -----------------------------------------------------
    # Good scene -> pass
    # -----------------------------------------------------

    good_review = review_subject_against_criteria(
        subject_text=(
            'GARAK\n'
            '"A remarkable accusation. '
            'I assume you brought something '
            'more substantial than your word?"'
        ),
        criteria=[
            prepared.criterion,
        ],
    )

    print("\n--- ERA-APPROPRIATE SCENE ---")
    print(
        "Verdict:",
        good_review.suite.results[0].verdict,
    )
    print(
        "Reason:",
        good_review.suite.results[0].reason,
    )

    assert (
        good_review.suite.results[0].verdict
        == "pass"
    )

    # -----------------------------------------------------
    # Explicit modern slang violation -> fail
    # -----------------------------------------------------

    bad_review = review_subject_against_criteria(
        subject_text=(
            'GARAK\n'
            '"LOL, that intelligence is totally '
            'giving Dominion vibes."'
        ),
        criteria=[
            prepared.criterion,
        ],
    )

    print("\n--- STYLE VIOLATION ---")
    print(
        "Verdict:",
        bad_review.suite.results[0].verdict,
    )
    print(
        "Reason:",
        bad_review.suite.results[0].reason,
    )

    assert (
        bad_review.suite.results[0].verdict
        == "fail"
    )

    # -----------------------------------------------------
    # No explicit constraints -> insufficient evidence
    # -----------------------------------------------------

    no_constraints = (
        prepare_era_style_evaluation(
            build_script_input([])
        )
    )

    print("\n--- NO STYLE CONSTRAINTS ---")

    assert no_constraints.criterion is None

    assert (
        no_constraints.precomputed_result
        is not None
    )

    print(
        "Verdict:",
        no_constraints
        .precomputed_result
        .verdict,
    )

    print(
        "Reason:",
        no_constraints
        .precomputed_result
        .reason,
    )

    assert (
        no_constraints
        .precomputed_result
        .verdict
        == "insufficient_evidence"
    )

    print(
        "\nPASS: Evaluation used the exact "
        "Scriptwriter story era and explicit "
        "era/style constraints as its bounded "
        "style authority, detected an explicit "
        "style violation, and refused to invent "
        "style requirements when none were "
        "supplied."
    )


if __name__ == "__main__":
    main()