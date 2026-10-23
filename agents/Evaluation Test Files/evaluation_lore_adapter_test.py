from agents.evaluation_agent import (
    prepare_lore_evaluation,
)

from agents.lore_agent import (
    LoreResponseModel,
)


def make_response(
    *,
    summary: str,
    evidence_supported: bool,
    premise_supported: bool,
) -> LoreResponseModel:
    return LoreResponseModel(
        topic="Elim Garak",
        summary=summary,
        sources=[],
        tools_used=[],
        evidence_supported=(
            evidence_supported
        ),
        premise_supported=(
            premise_supported
        ),
    )


def main() -> None:
    print(
        "Running Evaluation Agent v0.1J "
        "frozen Lore evidence adapter tests."
    )

    # -----------------------------------------------------
    # Supported evidence -> canon criterion
    # -----------------------------------------------------

    supported_response = make_response(
        summary=(
            "Garak participated in a joint "
            "Obsidian Order and Tal Shiar "
            "operation against the Founders."
        ),
        evidence_supported=True,
        premise_supported=True,
    )

    supported = prepare_lore_evaluation(
        research_id="garak_founders",
        research_question=(
            "What does canon establish about "
            "Garak's prior involvement with "
            "the Founders?"
        ),
        lore_response=supported_response,
    )

    print("\n--- SUPPORTED LORE ---")

    assert supported.criterion is not None
    assert supported.precomputed_result is None

    print(
        "Criterion ID:",
        supported.criterion.criterion_id,
    )

    print(
        "Category:",
        supported.criterion.category,
    )

    print("\nContext:")

    for item in (
        supported.criterion.evaluation_context
    ):
        print(item)

    assert (
        supported.criterion.criterion_id
        == "lore_agent_evidence_garak_founders"
    )

    assert (
        supported.criterion.category
        == "canon_grounding"
    )

    assert any(
        "Obsidian Order" in item
        for item in (
            supported
            .criterion
            .evaluation_context
        )
    )

    # -----------------------------------------------------
    # Premise unsupported -> insufficient evidence
    # -----------------------------------------------------

    unsupported_premise = make_response(
        summary=(
            "The retrieved evidence does not "
            "establish the requested premise."
        ),
        evidence_supported=False,
        premise_supported=False,
    )

    premise_result = prepare_lore_evaluation(
        research_id="current_contacts",
        research_question=(
            "Does Garak currently retain "
            "contacts inside Cardassian "
            "Central Command?"
        ),
        lore_response=unsupported_premise,
    )

    print("\n--- UNSUPPORTED PREMISE ---")

    assert premise_result.criterion is None

    assert (
        premise_result.precomputed_result
        is not None
    )

    print(
        "Verdict:",
        premise_result
        .precomputed_result
        .verdict,
    )

    print(
        "Reason:",
        premise_result
        .precomputed_result
        .reason,
    )

    assert (
        premise_result
        .precomputed_result
        .verdict
        == "insufficient_evidence"
    )

    # -----------------------------------------------------
    # Premise supported, but full requested
    # constraints/evidence unresolved
    # -----------------------------------------------------

    constrained_response = make_response(
        summary=(
            "Garak historically maintained "
            "contacts inside Cardassian "
            "institutions."
        ),
        evidence_supported=False,
        premise_supported=True,
    )

    constrained = prepare_lore_evaluation(
        research_id="current_access",
        research_question=(
            "Does Garak retain those contacts "
            "during the current story period?"
        ),
        lore_response=constrained_response,
    )

    print("\n--- MISSING CONSTRAINT SUPPORT ---")

    assert constrained.criterion is None

    assert (
        constrained.precomputed_result
        is not None
    )

    print(
        "Verdict:",
        constrained
        .precomputed_result
        .verdict,
    )

    print(
        "Reason:",
        constrained
        .precomputed_result
        .reason,
    )

    assert (
        constrained
        .precomputed_result
        .verdict
        == "insufficient_evidence"
    )

    # -----------------------------------------------------
    # Impossible Lore flag combination must fail closed
    # -----------------------------------------------------

    malformed_response = make_response(
        summary=(
            "Malformed evidence-state fixture."
        ),
        evidence_supported=True,
        premise_supported=False,
    )

    try:
        prepare_lore_evaluation(
            research_id="malformed_flags",
            research_question=(
                "Malformed Lore evidence test."
            ),
            lore_response=malformed_response,
        )

    except ValueError as exc:
        print(
            "\n--- EXPECTED FLAG "
            "INTEGRITY FAILURE ---"
        )
        print(str(exc))

    else:
        raise AssertionError(
            "Impossible Lore evidence flags "
            "were incorrectly accepted."
        )

    # -----------------------------------------------------
    # Supported evidence requires usable summary
    # -----------------------------------------------------

    empty_supported_response = make_response(
        summary="",
        evidence_supported=True,
        premise_supported=True,
    )

    try:
        prepare_lore_evaluation(
            research_id="empty_summary",
            research_question=(
                "Supported Lore summary test."
            ),
            lore_response=(
                empty_supported_response
            ),
        )

    except ValueError as exc:
        print(
            "\n--- EXPECTED EMPTY "
            "SUMMARY FAILURE ---"
        )
        print(str(exc))

    else:
        raise AssertionError(
            "Supported Lore response with "
            "empty summary was incorrectly "
            "accepted."
        )

    print(
        "\nPASS: Evaluation consumed the "
        "actual frozen LoreResponseModel "
        "contract, exposed only fully "
        "supported Lore evidence as canon "
        "context, converted unresolved "
        "premises or constraints into "
        "insufficient evidence, and failed "
        "closed on malformed Lore states."
    )


if __name__ == "__main__":
    main()