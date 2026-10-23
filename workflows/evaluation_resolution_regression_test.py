"""
Pure-Python regression coverage for workflows.evaluation_resolution.

Run from the project root:

    python -m workflows.evaluation_resolution_regression_test

No provider calls are made.
"""

from __future__ import annotations

from types import SimpleNamespace

from workflows.evaluation_resolution import (
    CharacterResolutionSource,
    LoreResolutionSource,
    build_evaluation_resolution_plan,
)


def assert_true(
    condition: bool,
    message: str,
) -> None:
    if not condition:
        raise AssertionError(message)


def assert_raises(
    expected_exception: type[Exception],
    callable_object,
    expected_message_fragment: str,
) -> None:
    try:
        callable_object()
    except expected_exception as error:
        if (
            expected_message_fragment
            not in str(error)
        ):
            raise AssertionError(
                "Exception message did not contain "
                f"{expected_message_fragment!r}.\n\n"
                f"Received:\n{error}"
            ) from error
        return
    except Exception as error:
        raise AssertionError(
            "Expected "
            f"{expected_exception.__name__}, "
            f"received {type(error).__name__}: "
            f"{error}"
        ) from error

    raise AssertionError(
        "Expected "
        f"{expected_exception.__name__} "
        "but no exception was raised."
    )


def criterion_result(
    criterion_id: str,
    category: str,
    verdict: str,
) -> SimpleNamespace:
    return SimpleNamespace(
        criterion_id=criterion_id,
        category=category,
        criterion=(
            f"Regression criterion for {criterion_id}."
        ),
        verdict=verdict,
        reason=(
            f"Regression result for {criterion_id}."
        ),
    )


def production_review(
    *,
    status: str,
    blocking_criterion_ids: list[str],
    results: list[SimpleNamespace],
) -> SimpleNamespace:
    return SimpleNamespace(
        review=SimpleNamespace(
            suite=SimpleNamespace(
                results=results,
            ),
            assessment=SimpleNamespace(
                status=status,
                blocking_criterion_ids=(
                    blocking_criterion_ids
                ),
            ),
        )
    )


def test_live_flight_resolution() -> None:
    """
    Reproduce the exact blocker families returned by the
    first complete live Writer's Room flight.
    """

    review = production_review(
        status="needs_evidence",
        blocking_criterion_ids=[
            "lore_agent_evidence_lore_000",
            (
                "character_agent_guidance_"
                "elim_garak"
            ),
        ],
        results=[
            criterion_result(
                "lore_agent_evidence_lore_000",
                "canon_grounding",
                "insufficient_evidence",
            ),
            criterion_result(
                "lore_agent_evidence_lore_001",
                "canon_grounding",
                "pass",
            ),
            criterion_result(
                (
                    "character_agent_guidance_"
                    "elim_garak"
                ),
                "character_consistency",
                "insufficient_evidence",
            ),
            criterion_result(
                "era_style_review",
                "era_style",
                "pass",
            ),
            criterion_result(
                "scriptwriter_grounding_review",
                "upstream_constraint",
                "pass",
            ),
            criterion_result(
                "continuity_agent_review",
                "continuity",
                "pass",
            ),
        ],
    )

    plan = build_evaluation_resolution_plan(
        production_review=review,
        lore_sources=[
            LoreResolutionSource(
                research_id="lore_000"
            ),
            LoreResolutionSource(
                research_id="lore_001"
            ),
        ],
        character_sources=[
            CharacterResolutionSource(
                character_id="character_000",
                character_name="Elim Garak",
            )
        ],
    )

    assert_true(
        plan.evaluation_status
        == "needs_evidence",
        "Live-flight plan lost Evaluation status.",
    )

    assert_true(
        plan.resolution_status
        == "needs_resolution",
        "Live-flight blockers were not routed "
        "to evidence resolution.",
    )

    assert_true(
        len(plan.targets) == 2,
        "Expected exactly two live-flight "
        "resolution targets.",
    )

    lore_target = plan.targets[0]
    character_target = plan.targets[1]

    assert_true(
        lore_target.criterion_id
        == "lore_agent_evidence_lore_000",
        "Lore blocker criterion ID changed.",
    )

    assert_true(
        lore_target.owner == "lore",
        "Lore blocker did not route to Lore.",
    )

    assert_true(
        lore_target.action
        == "refresh_lore_evidence",
        "Lore blocker received wrong action.",
    )

    assert_true(
        lore_target.artifact_id == "lore_000",
        "Lore blocker did not map to lore_000.",
    )

    assert_true(
        character_target.criterion_id
        == (
            "character_agent_guidance_"
            "elim_garak"
        ),
        "Character blocker criterion ID changed.",
    )

    assert_true(
        character_target.owner == "character",
        "Character blocker did not route "
        "to Character.",
    )

    assert_true(
        character_target.action
        == "refresh_character_evidence",
        "Character blocker received wrong action.",
    )

    assert_true(
        character_target.artifact_id
        == "character_000",
        "Character blocker did not map "
        "to character_000.",
    )

    assert_true(
        character_target.subject_name
        == "Elim Garak",
        "Character blocker lost subject identity.",
    )

    print(
        "Live-flight blocker routing: PASS"
    )


def test_approved_no_action() -> None:
    review = production_review(
        status="approved",
        blocking_criterion_ids=[],
        results=[
            criterion_result(
                "era_style_review",
                "era_style",
                "pass",
            )
        ],
    )

    plan = build_evaluation_resolution_plan(
        production_review=review,
    )

    assert_true(
        plan.evaluation_status == "approved",
        "Approved Evaluation status changed.",
    )

    assert_true(
        plan.resolution_status == "no_action",
        "Approved Evaluation incorrectly "
        "requested resolution.",
    )

    assert_true(
        plan.targets == [],
        "Approved Evaluation produced "
        "resolution targets.",
    )

    print(
        "Approved no-action routing: PASS"
    )


def test_rejected_not_evidence_loop() -> None:
    review = production_review(
        status="rejected",
        blocking_criterion_ids=[
            "scriptwriter_grounding_review",
        ],
        results=[
            criterion_result(
                "scriptwriter_grounding_review",
                "upstream_constraint",
                "fail",
            )
        ],
    )

    plan = build_evaluation_resolution_plan(
        production_review=review,
    )

    assert_true(
        plan.evaluation_status == "rejected",
        "Rejected Evaluation status changed.",
    )

    assert_true(
        plan.resolution_status
        == "not_applicable",
        "Rejected Evaluation incorrectly "
        "entered evidence resolution.",
    )

    assert_true(
        plan.targets == [],
        "Rejected Evaluation produced "
        "evidence-resolution targets.",
    )

    print(
        "Rejected evidence-loop exclusion: PASS"
    )


def test_missing_lore_source_fails_closed() -> None:
    review = production_review(
        status="needs_evidence",
        blocking_criterion_ids=[
            "lore_agent_evidence_lore_999",
        ],
        results=[
            criterion_result(
                "lore_agent_evidence_lore_999",
                "canon_grounding",
                "insufficient_evidence",
            )
        ],
    )

    assert_raises(
        ValueError,
        lambda: build_evaluation_resolution_plan(
            production_review=review,
            lore_sources=[
                LoreResolutionSource(
                    research_id="lore_000"
                )
            ],
        ),
        (
            "does not map to a supplied "
            "Writer's Room Lore artifact"
        ),
    )

    print(
        "Unknown Lore blocker fail-closed: PASS"
    )


def test_inconsistent_blocker_verdict_fails_closed() -> None:
    review = production_review(
        status="needs_evidence",
        blocking_criterion_ids=[
            "lore_agent_evidence_lore_000",
        ],
        results=[
            criterion_result(
                "lore_agent_evidence_lore_000",
                "canon_grounding",
                "pass",
            )
        ],
    )

    assert_raises(
        ValueError,
        lambda: build_evaluation_resolution_plan(
            production_review=review,
            lore_sources=[
                LoreResolutionSource(
                    research_id="lore_000"
                )
            ],
        ),
        (
            "must have "
            "verdict='insufficient_evidence'"
        ),
    )

    print(
        "Inconsistent blocker verdict "
        "fail-closed: PASS"
    )


def test_non_lore_resolution_families() -> None:
    """
    Prove the two non-research evidence families are
    routed without inventing an artifact ID.
    """

    review = production_review(
        status="needs_evidence",
        blocking_criterion_ids=[
            "era_style_review",
            "continuity_agent_review",
        ],
        results=[
            criterion_result(
                "era_style_review",
                "era_style",
                "insufficient_evidence",
            ),
            criterion_result(
                "continuity_agent_review",
                "continuity",
                "insufficient_evidence",
            ),
        ],
    )

    plan = build_evaluation_resolution_plan(
        production_review=review,
    )

    assert_true(
        len(plan.targets) == 2,
        "Expected two non-research "
        "resolution targets.",
    )

    era_target = plan.targets[0]
    continuity_target = plan.targets[1]

    assert_true(
        (
            era_target.owner,
            era_target.action,
            era_target.artifact_id,
        )
        == (
            "scriptwriter_input",
            "supply_era_style_constraints",
            None,
        ),
        "Era/style blocker routed incorrectly.",
    )

    assert_true(
        (
            continuity_target.owner,
            continuity_target.action,
            continuity_target.artifact_id,
        )
        == (
            "continuity",
            "resolve_continuity_context",
            None,
        ),
        "Continuity blocker routed incorrectly.",
    )

    print(
        "Era/Continuity blocker routing: PASS"
    )


def main() -> None:
    print(
        "\n--- Evaluation Resolution Regression Test ---"
    )

    test_live_flight_resolution()
    test_approved_no_action()
    test_rejected_not_evidence_loop()
    test_missing_lore_source_fails_closed()
    test_inconsistent_blocker_verdict_fails_closed()
    test_non_lore_resolution_families()

    print(
        "\n============================================"
    )
    print(
        "EVALUATION RESOLUTION REGRESSION: PASS"
    )
    print(
        "============================================"
    )


if __name__ == "__main__":
    main()
