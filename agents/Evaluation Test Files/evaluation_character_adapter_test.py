from agents.evaluation_agent import (
    prepare_character_evaluation,
)

from agents.character_agent import (
    CharacterAnalysis,
    CharacterAnalysisAudit,
    CharacterAnalysisIssue,
)


def build_analysis() -> CharacterAnalysis:
    return CharacterAnalysis(
        character_name="Elim Garak",
        story_role=(
            "Evaluate the Cardassian defector's "
            "claim before deciding whether to "
            "involve DS9 command."
        ),
        motivations=[],
        likely_behavior=[
            (
                "Evaluate the claim before "
                "deciding whether to involve "
                "DS9 command."
            ),
        ],
        unlikely_behavior=[],
        relationships=[],
        dramatic_tension="",
        remaining_canon_questions=[
            (
                "Does Garak currently retain "
                "contacts inside Cardassian "
                "Central Command?"
            ),
        ],
        remaining_voice_questions=[],
        speech_patterns=[],
        preferred_vocabulary=[],
        communication_style=[],
        behavioral_constraints=[
            (
                "Do not assume current access "
                "to historical contacts."
            ),
        ],
        era_language_notes=[],
        out_of_character_risks=[],
    )


def main() -> None:
    print(
        "Running Evaluation Agent v0.1I "
        "grounded Character guidance "
        "adapter tests."
    )

    analysis = build_analysis()

    # -----------------------------------------------------
    # Grounded analysis -> Evaluation criterion
    # -----------------------------------------------------

    grounded_audit = CharacterAnalysisAudit(
        grounded=True,
        issues=[],
        reason=(
            "All CharacterAnalysis claims "
            "are grounded."
        ),
    )

    grounded = prepare_character_evaluation(
        character_analysis=analysis,
        audit=grounded_audit,
    )

    print("\n--- GROUNDED GUIDANCE ---")

    assert grounded.criterion is not None

    assert (
        grounded.precomputed_result
        is None
    )

    print(
        "Criterion ID:",
        grounded.criterion.criterion_id,
    )

    print(
        "Category:",
        grounded.criterion.category,
    )

    print("\nContext:")

    for item in (
        grounded.criterion.evaluation_context
    ):
        print(item)

    assert (
        grounded.criterion.category
        == "character_consistency"
    )

    # Unresolved research questions must NOT
    # leak into authoritative Evaluation context.
    assert not any(
        "Does Garak currently retain"
        in item
        for item
        in grounded.criterion.evaluation_context
    )

    assert any(
        "Evaluate the claim"
        in item
        for item
        in grounded.criterion.evaluation_context
    )

    assert any(
        "Do not assume current access"
        in item
        for item
        in grounded.criterion.evaluation_context
    )

    # -----------------------------------------------------
    # Ungrounded analysis -> insufficient evidence
    # -----------------------------------------------------

    issue = CharacterAnalysisIssue(
        field="likely_behavior[0]",
        claim=(
            "Consult his current "
            "Cardassian contacts."
        ),
        reason=(
            "Historical contact evidence "
            "does not establish current access."
        ),
    )

    ungrounded_audit = (
        CharacterAnalysisAudit(
            grounded=False,
            issues=[issue],
            reason=(
                "1 CharacterAnalysis claim "
                "remains unsupported."
            ),
        )
    )

    ungrounded = (
        prepare_character_evaluation(
            character_analysis=analysis,
            audit=ungrounded_audit,
        )
    )

    print("\n--- UNGROUNDED GUIDANCE ---")

    assert ungrounded.criterion is None

    assert (
        ungrounded.precomputed_result
        is not None
    )

    print(
        "Verdict:",
        ungrounded.precomputed_result.verdict,
    )

    print(
        "Reason:",
        ungrounded.precomputed_result.reason,
    )

    assert (
        ungrounded.precomputed_result.verdict
        == "insufficient_evidence"
    )

    assert (
        ungrounded.precomputed_result.category
        == "character_consistency"
    )

    # -----------------------------------------------------
    # Malformed audit must fail closed
    # -----------------------------------------------------

    malformed = CharacterAnalysisAudit(
        grounded=True,
        issues=[issue],
        reason="Malformed fixture.",
    )

    try:
        prepare_character_evaluation(
            character_analysis=analysis,
            audit=malformed,
        )

    except ValueError as exc:
        print(
            "\n--- EXPECTED INTEGRITY "
            "FAILURE ---"
        )
        print(str(exc))

    else:
        raise AssertionError(
            "Malformed CharacterAnalysis "
            "audit was incorrectly accepted."
        )

    print(
        "\nPASS: Evaluation exposed only "
        "grounded Character Agent guidance, "
        "excluded unresolved research "
        "questions from authoritative context, "
        "and converted ungrounded guidance "
        "into insufficient evidence instead "
        "of allowing unsupported character "
        "claims downstream."
    )


if __name__ == "__main__":
    main()