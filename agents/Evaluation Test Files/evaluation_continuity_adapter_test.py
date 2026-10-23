from agents.evaluation_agent import (
    adapt_continuity_review,
)

from agents.continuity_agent import (
    ContinuitySceneIssue,
    ContinuitySceneAssessment,
    ContinuityScriptSceneAudit,
    ContinuitySceneReview,
)


def make_review(
    status: str,
    contradiction_count: int,
    insufficient_context_count: int,
    issues: list[ContinuitySceneIssue],
) -> ContinuitySceneReview:
    return ContinuitySceneReview(
        audit=ContinuityScriptSceneAudit(
            segment_audits=[],
            has_contradiction=(
                contradiction_count > 0
            ),
            contradiction_count=(
                contradiction_count
            ),
        ),
        assessment=ContinuitySceneAssessment(
            status=status,
            issues=issues,
            contradiction_count=(
                contradiction_count
            ),
            insufficient_context_count=(
                insufficient_context_count
            ),
        ),
    )


def main() -> None:
    print(
        "Running Evaluation Agent v0.1F "
        "frozen Continuity adapter tests."
    )

    # -----------------------------------------------------
    # Clear Continuity review -> pass
    # -----------------------------------------------------

    clear_review = make_review(
        status="clear",
        contradiction_count=0,
        insufficient_context_count=0,
        issues=[],
    )

    clear_result = adapt_continuity_review(
        clear_review
    )

    print("\n--- CLEAR ---")
    print(
        f"Verdict: {clear_result.verdict}"
    )
    print(
        f"Reason: {clear_result.reason}"
    )

    assert clear_result.verdict == "pass"

    assert (
        clear_result.category
        == "continuity"
    )

    # -----------------------------------------------------
    # Needs context -> insufficient evidence
    # -----------------------------------------------------

    context_issue = ContinuitySceneIssue(
        segment_id="script_text[2]",
        statement=(
            "Garak accesses the restricted file."
        ),
        state_id="access_state",
        established_state=(
            "Garak previously lacked access "
            "to the restricted file."
        ),
        issue_type="insufficient_context",
        reason=(
            "No supplied transition establishes "
            "that Garak gained access."
        ),
    )

    needs_context_review = make_review(
        status="needs_context",
        contradiction_count=0,
        insufficient_context_count=1,
        issues=[
            context_issue,
        ],
    )

    context_result = adapt_continuity_review(
        needs_context_review
    )

    print("\n--- NEEDS CONTEXT ---")
    print(
        f"Verdict: {context_result.verdict}"
    )
    print(
        f"Reason: {context_result.reason}"
    )

    assert (
        context_result.verdict
        == "insufficient_evidence"
    )

    # -----------------------------------------------------
    # Contradiction -> fail
    # -----------------------------------------------------

    contradiction_issue = (
        ContinuitySceneIssue(
            segment_id="script_text[4]",
            statement=(
                "The wall console is intact."
            ),
            state_id="console_destroyed",
            established_state=(
                "The wall console is destroyed."
            ),
            issue_type="contradiction",
            reason=(
                "The later intact condition "
                "conflicts with the established "
                "destroyed condition."
            ),
        )
    )

    contradiction_review = make_review(
        status="contradiction",
        contradiction_count=1,
        insufficient_context_count=0,
        issues=[
            contradiction_issue,
        ],
    )

    contradiction_result = (
        adapt_continuity_review(
            contradiction_review
        )
    )

    print("\n--- CONTRADICTION ---")
    print(
        f"Verdict: "
        f"{contradiction_result.verdict}"
    )
    print(
        f"Reason: "
        f"{contradiction_result.reason}"
    )

    assert (
        contradiction_result.verdict
        == "fail"
    )

    assert (
        "script_text[4]"
        in contradiction_result.reason
    )

    # -----------------------------------------------------
    # Malformed upstream result must be rejected
    # -----------------------------------------------------

    malformed_review = (
        ContinuitySceneReview(
            audit=ContinuityScriptSceneAudit(
                segment_audits=[],
                has_contradiction=False,
                contradiction_count=0,
            ),
            assessment=(
                ContinuitySceneAssessment(
                    status="clear",
                    issues=[
                        contradiction_issue,
                    ],
                    contradiction_count=0,
                    insufficient_context_count=0,
                )
            ),
        )
    )

    try:
        adapt_continuity_review(
            malformed_review
        )
    except ValueError as exc:
        print(
            "\n--- EXPECTED INTEGRITY "
            "FAILURE ---"
        )
        print(str(exc))
    else:
        raise AssertionError(
            "Malformed Continuity review "
            "was incorrectly accepted."
        )

    print(
        "\nPASS: Evaluation consumed the "
        "frozen Continuity quality gate "
        "without re-judging continuity, "
        "mapped its statuses deterministically, "
        "and rejected malformed upstream data."
    )


if __name__ == "__main__":
    main()