from agents.continuity_agent import (
    ContinuityStateCandidate,
    classify_state_candidate,
)


def run_case(
    name: str,
    statement: str,
    expected_verdict: str,
) -> None:
    candidate = ContinuityStateCandidate(
        segment_id=name,
        statement=statement,
    )

    judgment = classify_state_candidate(
        candidate
    )

    print(f"\n--- {name} ---")
    print(f"Statement: {statement}")
    print(f"Expected: {expected_verdict}")
    print(f"Actual: {judgment.verdict}")
    print(f"Reason: {judgment.reason}")

    assert (
        judgment.verdict
        == expected_verdict
    ), (
        f"{name} failed: expected "
        f"{expected_verdict!r}, got "
        f"{judgment.verdict!r}"
    )


def main() -> None:
    print(
        "Running Continuity Agent v0.1M "
        "state-candidate classification tests."
    )

    run_case(
        name="Possession Becomes Durable",
        statement=(
            "Garak takes the data rod."
        ),
        expected_verdict="durable_state",
    )

    run_case(
        name="Destroyed Condition Is Durable",
        statement=(
            "The console is destroyed."
        ),
        expected_verdict="durable_state",
    )

    run_case(
        name="Gesture Is Transient",
        statement=(
            "Garak smiles."
        ),
        expected_verdict="transient_event",
    )

    run_case(
        name="Vague Action Needs Context",
        statement=(
            "Garak does something "
            "to the data rod."
        ),
        expected_verdict=(
            "insufficient_context"
        ),
    )

    print(
        "\nPASS: Continuity distinguished "
        "durable story state from transient "
        "events and unresolved candidates."
    )


if __name__ == "__main__":
    main()