from agents.evaluation_agent import (
    adapt_scriptwriter_audit,
)

from agents.scriptwriter_agent import (
    ScriptSceneAudit,
    ScriptSceneAuditIssue,
)


def main() -> None:
    print(
        "Running Evaluation Agent v0.1G "
        "frozen Scriptwriter adapter tests."
    )

    # -----------------------------------------------------
    # Grounded Scriptwriter audit -> pass
    # -----------------------------------------------------

    grounded_audit = ScriptSceneAudit(
        grounded=True,
        issues=[],
        reason=(
            "All ScriptScene segments are "
            "grounded or permitted story "
            "inventions."
        ),
    )

    grounded_result = (
        adapt_scriptwriter_audit(
            grounded_audit
        )
    )

    print("\n--- GROUNDED ---")
    print(
        f"Verdict: "
        f"{grounded_result.verdict}"
    )
    print(
        f"Category: "
        f"{grounded_result.category}"
    )
    print(
        f"Reason: "
        f"{grounded_result.reason}"
    )

    assert grounded_result.verdict == "pass"

    assert (
        grounded_result.category
        == "upstream_constraint"
    )

    # -----------------------------------------------------
    # Unsupported segment -> fail
    # -----------------------------------------------------

    unsupported_audit = ScriptSceneAudit(
        grounded=False,
        issues=[
            ScriptSceneAuditIssue(
                segment_id="script_text[2]",
                field="script_text",
                text=(
                    "Garak contacts his current "
                    "source inside Central Command."
                ),
                verdict="unsupported",
                reason=(
                    "Current access to such a "
                    "source was not established."
                ),
            ),
        ],
        reason=(
            "1 ScriptScene segment remains "
            "unsupported or prohibited."
        ),
    )

    unsupported_result = (
        adapt_scriptwriter_audit(
            unsupported_audit
        )
    )

    print("\n--- UNSUPPORTED ---")
    print(
        f"Verdict: "
        f"{unsupported_result.verdict}"
    )
    print(
        f"Reason: "
        f"{unsupported_result.reason}"
    )

    assert unsupported_result.verdict == "fail"

    assert (
        "1 unsupported"
        in unsupported_result.reason
    )

    assert (
        "script_text[2]"
        in unsupported_result.reason
    )

    # -----------------------------------------------------
    # Mixed unsupported + prohibited -> fail
    # -----------------------------------------------------

    mixed_audit = ScriptSceneAudit(
        grounded=False,
        issues=[
            ScriptSceneAuditIssue(
                segment_id="scene_heading",
                field="scene_heading",
                text=(
                    "INT. QUIET CORNER OF "
                    "QUARK'S BAR - NIGHT"
                ),
                verdict="unsupported",
                reason=(
                    "The canon-specific location "
                    "was not supplied."
                ),
            ),
            ScriptSceneAuditIssue(
                segment_id="script_text[5]",
                field="script_text",
                text=(
                    "Garak contacts Cardassian "
                    "Central Command."
                ),
                verdict="prohibited",
                reason=(
                    "The supplied prohibition "
                    "forbids current Central "
                    "Command contact."
                ),
            ),
        ],
        reason=(
            "2 ScriptScene segments remain "
            "unsupported or prohibited."
        ),
    )

    mixed_result = adapt_scriptwriter_audit(
        mixed_audit
    )

    print("\n--- MIXED BLOCKERS ---")
    print(
        f"Verdict: "
        f"{mixed_result.verdict}"
    )
    print(
        f"Reason: "
        f"{mixed_result.reason}"
    )

    assert mixed_result.verdict == "fail"

    assert (
        "1 unsupported"
        in mixed_result.reason
    )

    assert (
        "1 prohibited"
        in mixed_result.reason
    )

    # -----------------------------------------------------
    # Malformed grounded audit must fail closed
    # -----------------------------------------------------

    malformed_grounded = ScriptSceneAudit(
        grounded=True,
        issues=[
            ScriptSceneAuditIssue(
                segment_id="script_text[0]",
                field="script_text",
                text="Invalid fixture.",
                verdict="unsupported",
                reason=(
                    "Grounded flag conflicts "
                    "with this issue."
                ),
            ),
        ],
        reason="Malformed fixture.",
    )

    try:
        adapt_scriptwriter_audit(
            malformed_grounded
        )
    except ValueError as exc:
        print(
            "\n--- EXPECTED GROUNDED "
            "INTEGRITY FAILURE ---"
        )
        print(str(exc))
    else:
        raise AssertionError(
            "Malformed grounded Scriptwriter "
            "audit was incorrectly accepted."
        )

    # -----------------------------------------------------
    # Ungrounded with zero issues must also fail closed
    # -----------------------------------------------------

    malformed_ungrounded = ScriptSceneAudit(
        grounded=False,
        issues=[],
        reason="Malformed fixture.",
    )

    try:
        adapt_scriptwriter_audit(
            malformed_ungrounded
        )
    except ValueError as exc:
        print(
            "\n--- EXPECTED UNGROUNDED "
            "INTEGRITY FAILURE ---"
        )
        print(str(exc))
    else:
        raise AssertionError(
            "Malformed ungrounded Scriptwriter "
            "audit was incorrectly accepted."
        )

    print(
        "\nPASS: Evaluation consumed the "
        "frozen Scriptwriter grounding gate "
        "without re-auditing screenplay "
        "semantics, mapped blocking issues "
        "deterministically, and rejected "
        "malformed upstream results."
    )


if __name__ == "__main__":
    main()