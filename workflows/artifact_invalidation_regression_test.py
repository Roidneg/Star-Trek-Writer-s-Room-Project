"""
Pure-Python regression coverage for workflows.artifact_invalidation.

Run from the project root:

    python -m workflows.artifact_invalidation_regression_test

No provider calls are made.
"""

from __future__ import annotations

from workflows.artifact_invalidation import (
    build_artifact_invalidation_plan,
)
from workflows.evaluation_resolution import (
    EvaluationResolutionPlan,
    EvaluationResolutionTarget,
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


def approved_plan() -> EvaluationResolutionPlan:
    return EvaluationResolutionPlan(
        evaluation_status="approved",
        resolution_status="no_action",
        targets=[],
        reason=(
            "Evaluation approved the scene."
        ),
    )


def rejected_plan() -> EvaluationResolutionPlan:
    return EvaluationResolutionPlan(
        evaluation_status="rejected",
        resolution_status="not_applicable",
        targets=[],
        reason=(
            "Evaluation rejected the scene."
        ),
    )


def lore_target() -> EvaluationResolutionTarget:
    return EvaluationResolutionTarget(
        criterion_id="lore_agent_evidence_lore_000",
        category="canon_grounding",
        owner="lore",
        action="refresh_lore_evidence",
        artifact_id="lore_000",
        subject_name=None,
        reason=(
            "Lore evidence is insufficient."
        ),
    )


def character_target() -> EvaluationResolutionTarget:
    return EvaluationResolutionTarget(
        criterion_id=(
            "character_agent_guidance_"
            "elim_garak"
        ),
        category="character_consistency",
        owner="character",
        action="refresh_character_evidence",
        artifact_id="character_000",
        subject_name="Elim Garak",
        reason=(
            "Character evidence is insufficient."
        ),
    )


def era_style_target() -> EvaluationResolutionTarget:
    return EvaluationResolutionTarget(
        criterion_id="era_style_review",
        category="era_style",
        owner="scriptwriter_input",
        action="supply_era_style_constraints",
        artifact_id=None,
        subject_name=None,
        reason=(
            "Era/style evidence is insufficient."
        ),
    )


def continuity_target() -> EvaluationResolutionTarget:
    return EvaluationResolutionTarget(
        criterion_id="continuity_agent_review",
        category="continuity",
        owner="continuity",
        action="resolve_continuity_context",
        artifact_id=None,
        subject_name=None,
        reason=(
            "Continuity context is insufficient."
        ),
    )


def needs_evidence_plan(
    targets: list[
        EvaluationResolutionTarget
    ],
) -> EvaluationResolutionPlan:
    return EvaluationResolutionPlan(
        evaluation_status="needs_evidence",
        resolution_status="needs_resolution",
        targets=targets,
        reason=(
            "Evaluation needs more evidence."
        ),
    )


def test_lore_invalidation() -> None:
    plan = build_artifact_invalidation_plan(
        needs_evidence_plan(
            [lore_target()]
        )
    )

    assert_true(
        plan.trigger == "on_material_change",
        "Lore invalidation should be conditional "
        "on material change.",
    )

    assert_true(
        plan.invalidated_artifacts == [
            "script_result",
            "continuity_result",
            "story_state_ledger",
            "evaluation_result",
        ],
        "Lore invalidation set is incorrect.",
    )

    assert_true(
        plan.preserved_artifact_families == [
            "story_plan",
            "character_results",
        ],
        "Lore refresh should preserve StoryPlan "
        "and Character results.",
    )

    assert_true(
        plan.requires_ledger_rebuild is True,
        "Lore-driven Scriptwriter invalidation must "
        "require Continuity ledger rebuild.",
    )

    print(
        "Lore invalidation routing: PASS"
    )


def test_character_invalidation() -> None:
    plan = build_artifact_invalidation_plan(
        needs_evidence_plan(
            [character_target()]
        )
    )

    assert_true(
        plan.invalidated_artifacts == [
            "script_result",
            "continuity_result",
            "story_state_ledger",
            "evaluation_result",
        ],
        "Character invalidation set is incorrect.",
    )

    assert_true(
        plan.preserved_artifact_families == [
            "story_plan",
            "lore_results",
        ],
        "Character refresh should preserve StoryPlan "
        "and Lore results.",
    )

    assert_true(
        plan.requires_ledger_rebuild is True,
        "Character-driven Scriptwriter invalidation "
        "must require ledger rebuild.",
    )

    print(
        "Character invalidation routing: PASS"
    )


def test_lore_and_character_combined() -> None:
    plan = build_artifact_invalidation_plan(
        needs_evidence_plan(
            [
                lore_target(),
                character_target(),
            ]
        )
    )

    assert_true(
        plan.invalidated_artifacts == [
            "script_result",
            "continuity_result",
            "story_state_ledger",
            "evaluation_result",
        ],
        "Combined invalidation should de-duplicate "
        "shared downstream artifacts.",
    )

    assert_true(
        plan.preserved_artifact_families == [
            "story_plan",
        ],
        "Combined Lore/Character refresh should "
        "preserve only StoryPlan among tracked "
        "upstream families.",
    )

    assert_true(
        len(plan.causes) == 2,
        "Combined plan should preserve both "
        "invalidation causes.",
    )

    print(
        "Combined invalidation de-duplication: PASS"
    )


def test_era_style_invalidation() -> None:
    plan = build_artifact_invalidation_plan(
        needs_evidence_plan(
            [era_style_target()]
        )
    )

    assert_true(
        plan.invalidated_artifacts == [
            "script_result",
            "continuity_result",
            "story_state_ledger",
            "evaluation_result",
        ],
        "Era/style invalidation set is incorrect.",
    )

    assert_true(
        plan.preserved_artifact_families == [
            "story_plan",
            "lore_results",
            "character_results",
        ],
        "Era/style refresh should preserve upstream "
        "StoryPlan, Lore, and Character artifacts.",
    )

    print(
        "Era/style invalidation routing: PASS"
    )


def test_continuity_only_invalidation() -> None:
    plan = build_artifact_invalidation_plan(
        needs_evidence_plan(
            [continuity_target()]
        )
    )

    assert_true(
        plan.invalidated_artifacts == [
            "continuity_result",
            "story_state_ledger",
            "evaluation_result",
        ],
        "Continuity-only invalidation set is incorrect.",
    )

    assert_true(
        "script_result"
        in plan.preserved_artifact_families,
        "Continuity-only resolution must preserve "
        "Scriptwriter result.",
    )

    assert_true(
        plan.preserved_artifact_families == [
            "story_plan",
            "lore_results",
            "character_results",
            "script_result",
        ],
        "Continuity-only resolution preserved the "
        "wrong upstream families.",
    )

    assert_true(
        plan.requires_ledger_rebuild is True,
        "Continuity resolution must rebuild the ledger.",
    )

    print(
        "Continuity-only invalidation routing: PASS"
    )


def test_approved_no_invalidation() -> None:
    plan = build_artifact_invalidation_plan(
        approved_plan()
    )

    assert_true(
        plan.trigger == "none",
        "Approved plan should not have an "
        "invalidation trigger.",
    )

    assert_true(
        plan.invalidated_artifacts == [],
        "Approved Evaluation should invalidate nothing.",
    )

    assert_true(
        plan.requires_ledger_rebuild is False,
        "Approved Evaluation should not require "
        "ledger rebuild.",
    )

    print(
        "Approved no-invalidation routing: PASS"
    )


def test_rejected_not_invalidation_path() -> None:
    plan = build_artifact_invalidation_plan(
        rejected_plan()
    )

    assert_true(
        plan.trigger == "not_applicable",
        "Rejected Evaluation should not enter "
        "the evidence invalidation path.",
    )

    assert_true(
        plan.invalidated_artifacts == [],
        "Rejected Evaluation should not invalidate "
        "through evidence-resolution rules.",
    )

    print(
        "Rejected invalidation exclusion: PASS"
    )


def test_duplicate_resolution_target_fails_closed() -> None:
    target = lore_target()

    duplicate_plan = (
        needs_evidence_plan(
            [
                target,
                target.model_copy(),
            ]
        )
    )

    assert_raises(
        ValueError,
        lambda: build_artifact_invalidation_plan(
            duplicate_plan
        ),
        "Duplicate resolution target criterion ID",
    )

    print(
        "Duplicate target fail-closed: PASS"
    )


def test_inconsistent_approved_plan_fails_closed() -> None:
    bad_plan = EvaluationResolutionPlan(
        evaluation_status="approved",
        resolution_status="no_action",
        targets=[
            lore_target()
        ],
        reason=(
            "Deliberately inconsistent test plan."
        ),
    )

    assert_raises(
        ValueError,
        lambda: build_artifact_invalidation_plan(
            bad_plan
        ),
        "internally inconsistent",
    )

    print(
        "Inconsistent approved plan fail-closed: PASS"
    )


def test_inconsistent_needs_evidence_fails_closed() -> None:
    bad_plan = EvaluationResolutionPlan(
        evaluation_status="needs_evidence",
        resolution_status="needs_resolution",
        targets=[],
        reason=(
            "Deliberately incomplete test plan."
        ),
    )

    assert_raises(
        ValueError,
        lambda: build_artifact_invalidation_plan(
            bad_plan
        ),
        "at least one resolution target",
    )

    print(
        "Empty NEEDS_EVIDENCE plan fail-closed: PASS"
    )


def main() -> None:
    print(
        "\n--- Artifact Invalidation Regression Test ---"
    )

    test_lore_invalidation()
    test_character_invalidation()
    test_lore_and_character_combined()
    test_era_style_invalidation()
    test_continuity_only_invalidation()
    test_approved_no_invalidation()
    test_rejected_not_invalidation_path()
    test_duplicate_resolution_target_fails_closed()
    test_inconsistent_approved_plan_fails_closed()
    test_inconsistent_needs_evidence_fails_closed()

    print(
        "\n============================================"
    )
    print(
        "ARTIFACT INVALIDATION REGRESSION: PASS"
    )
    print(
        "============================================"
    )


if __name__ == "__main__":
    main()
