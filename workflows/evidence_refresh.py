"""
Targeted evidence refresh execution for the Writer's Room.

Version 0.1 supports Lore evidence refresh only.

This module sits between deterministic Evaluation resolution and
checkpoint mutation.

Flow:

    EvaluationResolutionTarget
            ↓
    refresh exactly one Lore artifact
            ↓
    build candidate replacement
            ↓
    compare material authority
            ↓
      unchanged? ---------------- yes → preserve downstream artifacts
            |
            no
            ↓
    attach deterministic invalidation plan

This module does NOT:
- mutate WritersRoomState;
- write or edit checkpoints;
- invalidate completed artifacts;
- rerun Story Architect;
- rerun Character;
- rerun Scriptwriter;
- rerun Continuity;
- rerun Evaluation;
- invent a new Lore research ID;
- rewrite the existing Lore research question.

The current Lore refresh deliberately re-runs the same authoritative
research question through the frozen Lore Agent. Future versions may
add a separately proven targeted follow-up research strategy.

Core rules:

    Refresh one source.
    Preserve its identity.
    Measure material change.
    Mutate nothing here.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from workflows.artifact_invalidation import (
    ArtifactInvalidationPlan,
    build_artifact_invalidation_plan,
)

from workflows.artifact_material_change import (
    MaterialChangeReport,
    build_lore_material_snapshot,
    compare_lore_material_change,
)

from workflows.evaluation_resolution import (
    EvaluationResolutionPlan,
    EvaluationResolutionTarget,
)


# ---------------------------------------------------------
# Refresh Output
# ---------------------------------------------------------


class LoreEvidenceRefreshResult(BaseModel):
    """
    Complete non-mutating result of one targeted Lore refresh.

    candidate_artifact is the proposed replacement for the existing
    Writer's Room Lore artifact. The caller must not commit it merely
    because refresh execution completed.

    Commit logic belongs to a later state-mutation layer.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    artifact_type: Literal["lore"] = "lore"

    research_id: str = Field(
        min_length=1,
    )

    research_question: str = Field(
        min_length=1,
    )

    refresh_status: Literal[
        "unchanged",
        "material_change",
    ]

    material_change: MaterialChangeReport

    candidate_artifact: Any

    invalidation_plan: (
        ArtifactInvalidationPlan
        | None
    ) = None

    reason: str = Field(
        min_length=1,
    )


# ---------------------------------------------------------
# Target Validation
# ---------------------------------------------------------


def validate_lore_refresh_target(
    target: EvaluationResolutionTarget,
) -> None:
    """
    Ensure a resolution target belongs to the Lore refresh path.

    Fail closed rather than treating semantically similar target data
    as equivalent.
    """

    if target.category != "canon_grounding":
        raise ValueError(
            "Lore refresh requires "
            "category='canon_grounding'."
        )

    if target.owner != "lore":
        raise ValueError(
            "Lore refresh requires owner='lore'."
        )

    if (
        target.action
        != "refresh_lore_evidence"
    ):
        raise ValueError(
            "Lore refresh requires "
            "action='refresh_lore_evidence'."
        )

    if (
        target.artifact_id is None
        or not target.artifact_id.strip()
    ):
        raise ValueError(
            "Lore refresh target requires a "
            "Writer's Room Lore artifact_id."
        )

    if target.subject_name is not None:
        raise ValueError(
            "Lore refresh target must not carry a "
            "Character subject_name."
        )


# ---------------------------------------------------------
# Artifact Validation
# ---------------------------------------------------------


def _validate_refresh_artifact_identity(
    target: EvaluationResolutionTarget,
    current_lore_result: Any,
) -> tuple[str, str]:
    """
    Validate and return the stable Lore ID and exact research question.
    """

    # Reuse the material-snapshot contract because it already enforces
    # the downstream-authoritative Lore structure.
    snapshot = (
        build_lore_material_snapshot(
            current_lore_result
        )
    )

    research_id = snapshot.research_id

    if research_id != target.artifact_id:
        raise ValueError(
            "Lore refresh target does not match the "
            "supplied Writer's Room Lore artifact.\n"
            f"Target artifact_id: "
            f"{target.artifact_id!r}\n"
            f"Supplied research_id: "
            f"{research_id!r}"
        )

    research_question = (
        snapshot.research_question
    )

    return (
        research_id,
        research_question,
    )


def _build_candidate_lore_artifact(
    current_lore_result: Any,
    refreshed_response: Any,
) -> Any:
    """
    Build a validated candidate artifact while preserving all Python-owned
    Writer's Room identity fields.

    Real production input is WritersRoomLoreResult, a Pydantic model.
    Regression tests may use another Pydantic model with the same contract.

    We intentionally reconstruct through model_validate rather than relying
    only on model_copy(update=...), because Pydantic model_copy does not
    revalidate updated values.
    """

    if not isinstance(
        current_lore_result,
        BaseModel,
    ):
        raise TypeError(
            "Lore refresh requires the current Lore "
            "artifact to be a Pydantic BaseModel."
        )

    artifact_type = type(
        current_lore_result
    )

    if not hasattr(
        artifact_type,
        "model_validate",
    ):
        raise TypeError(
            "Lore artifact type does not support "
            "Pydantic model validation."
        )

    current_payload = (
        current_lore_result.model_dump(
            mode="python"
        )
    )

    if "response" not in current_payload:
        raise ValueError(
            "Lore artifact does not contain a "
            "response field."
        )

    current_payload[
        "response"
    ] = refreshed_response

    try:
        candidate = (
            artifact_type.model_validate(
                current_payload
            )
        )
    except Exception as error:
        raise ValueError(
            "Frozen Lore workflow returned a response "
            "that cannot be stored in the current "
            "Writer's Room Lore artifact contract."
        ) from error

    # Re-run the material snapshot contract immediately so malformed
    # authority states are rejected before comparison.
    build_lore_material_snapshot(
        candidate
    )

    return candidate


def _verify_candidate_identity_preserved(
    current_lore_result: Any,
    candidate_lore_result: Any,
) -> None:
    """
    Ensure refresh execution changed only the Lore response payload.

    research_id and research_question are Python-owned provenance and
    cannot be rewritten by a refresh.
    """

    old_snapshot = (
        build_lore_material_snapshot(
            current_lore_result
        )
    )

    new_snapshot = (
        build_lore_material_snapshot(
            candidate_lore_result
        )
    )

    if (
        old_snapshot.research_id
        != new_snapshot.research_id
    ):
        raise ValueError(
            "Lore refresh changed the Python-owned "
            "research_id."
        )

    if (
        old_snapshot.research_question
        != new_snapshot.research_question
    ):
        raise ValueError(
            "Lore refresh changed the authoritative "
            "research question."
        )


# ---------------------------------------------------------
# Runner Resolution
# ---------------------------------------------------------


def _get_default_lore_runner() -> Callable[..., Any]:
    """
    Import the frozen provider-backed Lore workflow lazily.

    Keeping this import out of module initialization lets deterministic
    regression tests import evidence_refresh without initializing the
    Lore model or external research tools.
    """

    from agents.lore_agent import (
        run_lore_query,
    )

    return run_lore_query


def _invoke_lore_runner(
    lore_runner: Callable[..., Any],
    research_question: str,
    debug: bool,
) -> Any:
    """
    Invoke the refresh dependency using the frozen Lore signature.

    The runner receives the exact existing research question.
    """

    try:
        refreshed_response = lore_runner(
            query=research_question,
            debug=debug,
        )
    except TypeError as error:
        raise TypeError(
            "Lore refresh runner must support the "
            "frozen signature "
            "runner(query=<str>, debug=<bool>)."
        ) from error

    if refreshed_response is None:
        raise ValueError(
            "Lore refresh runner returned None."
        )

    return refreshed_response


# ---------------------------------------------------------
# Invalidation Planning
# ---------------------------------------------------------


def _build_single_target_resolution_plan(
    target: EvaluationResolutionTarget,
) -> EvaluationResolutionPlan:
    """
    Reconstruct the smallest valid NEEDS_EVIDENCE plan needed by the
    deterministic invalidation planner.

    The target itself was already produced by evaluation_resolution.
    """

    return EvaluationResolutionPlan(
        evaluation_status="needs_evidence",
        resolution_status="needs_resolution",
        targets=[
            target
        ],
        reason=(
            "Single-target Lore refresh requires "
            "conditional downstream invalidation "
            "only if material authority changes."
        ),
    )


def _build_material_change_invalidation_plan(
    target: EvaluationResolutionTarget,
) -> ArtifactInvalidationPlan:
    return build_artifact_invalidation_plan(
        _build_single_target_resolution_plan(
            target
        )
    )


# ---------------------------------------------------------
# Public API
# ---------------------------------------------------------


def refresh_lore_evidence(
    *,
    target: EvaluationResolutionTarget,
    current_lore_result: Any,
    lore_runner: (
        Callable[..., Any]
        | None
    ) = None,
    debug: bool = False,
) -> LoreEvidenceRefreshResult:
    """
    Re-run exactly one existing Writer's Room Lore research request.

    Parameters
    ----------
    target:
        One deterministic EvaluationResolutionTarget produced for a
        canon-grounding blocker.

    current_lore_result:
        The existing WritersRoomLoreResult associated with target.artifact_id.

    lore_runner:
        Optional dependency injection for deterministic regression tests.
        When omitted, the frozen agents.lore_agent.run_lore_query function
        is imported lazily and used.

    debug:
        Forwarded unchanged to the Lore workflow.

    Returns
    -------
    LoreEvidenceRefreshResult
        A proposed replacement artifact plus material-change information.
        Writer's Room state is never mutated by this function.

    Behavior
    --------
    1. Validate target ownership.
    2. Validate stable Lore artifact identity.
    3. Re-run the exact existing research question once.
    4. Build a validated candidate artifact with the same ID/question.
    5. Compare current vs refreshed authoritative Lore material.
    6. If unchanged, return no invalidation plan.
    7. If materially changed, attach the deterministic downstream
       invalidation plan.
    """

    validate_lore_refresh_target(
        target
    )

    (
        research_id,
        research_question,
    ) = _validate_refresh_artifact_identity(
        target=target,
        current_lore_result=(
            current_lore_result
        ),
    )

    runner = (
        lore_runner
        if lore_runner is not None
        else _get_default_lore_runner()
    )

    if not callable(
        runner
    ):
        raise TypeError(
            "Lore refresh runner must be callable."
        )

    refreshed_response = (
        _invoke_lore_runner(
            lore_runner=runner,
            research_question=(
                research_question
            ),
            debug=debug,
        )
    )

    candidate_artifact = (
        _build_candidate_lore_artifact(
            current_lore_result=(
                current_lore_result
            ),
            refreshed_response=(
                refreshed_response
            ),
        )
    )

    _verify_candidate_identity_preserved(
        current_lore_result=(
            current_lore_result
        ),
        candidate_lore_result=(
            candidate_artifact
        ),
    )

    material_change = (
        compare_lore_material_change(
            old_result=(
                current_lore_result
            ),
            new_result=(
                candidate_artifact
            ),
        )
    )

    if not material_change.material_change:
        return LoreEvidenceRefreshResult(
            research_id=research_id,
            research_question=(
                research_question
            ),
            refresh_status="unchanged",
            material_change=(
                material_change
            ),
            candidate_artifact=(
                candidate_artifact
            ),
            invalidation_plan=None,
            reason=(
                "Lore refresh completed, but the "
                "downstream-authoritative Lore content "
                "did not materially change. Existing "
                "Scriptwriter, Continuity, ledger, and "
                "Evaluation artifacts should remain "
                "preserved."
            ),
        )

    invalidation_plan = (
        _build_material_change_invalidation_plan(
            target
        )
    )

    if (
        invalidation_plan.trigger
        != "on_material_change"
    ):
        raise AssertionError(
            "Material Lore change did not produce "
            "a conditional invalidation plan."
        )

    if not (
        {
            "script_result",
            "continuity_result",
            "story_state_ledger",
            "evaluation_result",
        }
        .issubset(
            set(
                invalidation_plan
                .invalidated_artifacts
            )
        )
    ):
        raise AssertionError(
            "Material Lore change produced an "
            "incomplete downstream invalidation plan."
        )

    return LoreEvidenceRefreshResult(
        research_id=research_id,
        research_question=research_question,
        refresh_status="material_change",
        material_change=material_change,
        candidate_artifact=(
            candidate_artifact
        ),
        invalidation_plan=(
            invalidation_plan
        ),
        reason=(
            "Lore refresh materially changed "
            "downstream-authoritative canon evidence. "
            "The candidate replacement is ready for a "
            "future commit layer, and dependent "
            "Scriptwriter/Continuity/Evaluation artifacts "
            "must be invalidated if that candidate is "
            "committed."
        ),
    )
