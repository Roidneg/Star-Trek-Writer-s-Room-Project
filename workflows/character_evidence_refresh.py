"""
Targeted Character evidence refresh execution for the Writer's Room.

Version 0.1 supports Character evidence refresh only.

This module mirrors the proven Lore refresh boundary:

    EvaluationResolutionTarget
            ↓
    refresh exactly one Character production package
            ↓
    preserve Python-owned identity and story inputs
            ↓
    build candidate Writer's Room Character artifact
            ↓
    compare downstream-authoritative Character guidance
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
- rerun Scriptwriter;
- rerun Continuity;
- rerun Evaluation;
- invent a new Character ID;
- change character identity;
- change story_context;
- change story_era.

The Character production workflow is allowed to generate a fresh
research plan and fresh research questions. Those are part of the
refresh process. Python preserves the stable Writer's Room identity
and the exact story inputs that define the Character task.

Core rules:

    Refresh one Character source.
    Preserve task identity.
    Measure material downstream change.
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
    build_character_material_snapshot,
    compare_character_material_change,
)

from workflows.evaluation_resolution import (
    EvaluationResolutionPlan,
    EvaluationResolutionTarget,
)


# ---------------------------------------------------------
# Refresh Output
# ---------------------------------------------------------


class CharacterEvidenceRefreshResult(BaseModel):
    """
    Complete non-mutating result of one targeted Character refresh.

    candidate_artifact is the proposed replacement for the existing
    Writer's Room Character artifact.

    State commit belongs to a later layer.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    artifact_type: Literal["character"] = "character"

    character_id: str = Field(
        min_length=1,
    )

    character_name: str = Field(
        min_length=1,
    )

    story_context: str = Field(
        min_length=1,
    )

    story_era: str = Field(
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
# Structural Helpers
# ---------------------------------------------------------


def _read_attribute(
    value: Any,
    attribute_name: str,
    context: str,
) -> Any:
    if not hasattr(
        value,
        attribute_name,
    ):
        raise TypeError(
            f"{context} is missing required attribute "
            f"{attribute_name!r}."
        )

    return getattr(
        value,
        attribute_name,
    )


def _required_text(
    value: Any,
    label: str,
) -> str:
    if not isinstance(
        value,
        str,
    ):
        raise TypeError(
            f"{label} must be a string."
        )

    if not value.strip():
        raise ValueError(
            f"{label} cannot be empty."
        )

    # Preserve exact content. Stripping is used only for validation.
    return value


# ---------------------------------------------------------
# Target Validation
# ---------------------------------------------------------


def validate_character_refresh_target(
    target: EvaluationResolutionTarget,
) -> None:
    """
    Ensure a resolution target belongs to the Character refresh path.
    """

    if (
        target.category
        != "character_consistency"
    ):
        raise ValueError(
            "Character refresh requires "
            "category='character_consistency'."
        )

    if target.owner != "character":
        raise ValueError(
            "Character refresh requires "
            "owner='character'."
        )

    if (
        target.action
        != "refresh_character_evidence"
    ):
        raise ValueError(
            "Character refresh requires "
            "action='refresh_character_evidence'."
        )

    if (
        target.artifact_id is None
        or not target.artifact_id.strip()
    ):
        raise ValueError(
            "Character refresh target requires a "
            "Writer's Room Character artifact_id."
        )

    if (
        target.subject_name is None
        or not target.subject_name.strip()
    ):
        raise ValueError(
            "Character refresh target requires "
            "subject_name."
        )


# ---------------------------------------------------------
# Existing Artifact Identity
# ---------------------------------------------------------


def _validate_refresh_artifact_identity(
    *,
    target: EvaluationResolutionTarget,
    current_character_result: Any,
) -> tuple[
    str,
    str,
    str,
    str,
]:
    """
    Validate the existing Character artifact and return:

        character_id
        character_name
        story_context
        story_era

    The material snapshot validates downstream Character structure.
    The research plan provides the immutable story inputs for refresh.
    """

    snapshot = (
        build_character_material_snapshot(
            current_character_result
        )
    )

    character_id = snapshot.character_id
    character_name = snapshot.character_name

    if (
        character_id
        != target.artifact_id
    ):
        raise ValueError(
            "Character refresh target does not match "
            "the supplied Writer's Room Character artifact.\n"
            f"Target artifact_id: "
            f"{target.artifact_id!r}\n"
            f"Supplied character_id: "
            f"{character_id!r}"
        )

    if (
        character_name
        .strip()
        .casefold()
        != target.subject_name
        .strip()
        .casefold()
    ):
        raise ValueError(
            "Character refresh target subject_name "
            "does not match the supplied Character "
            "artifact identity."
        )

    research_plan = _read_attribute(
        current_character_result,
        "research_plan",
        "Character result",
    )

    plan_character_name = _required_text(
        _read_attribute(
            research_plan,
            "character_name",
            "Character research plan",
        ),
        "Character research_plan.character_name",
    )

    if (
        plan_character_name
        .strip()
        .casefold()
        != character_name
        .strip()
        .casefold()
    ):
        raise ValueError(
            "Character research plan identity does "
            "not match Writer's Room Character identity."
        )

    story_context = _required_text(
        _read_attribute(
            research_plan,
            "story_context",
            "Character research plan",
        ),
        "Character research_plan.story_context",
    )

    story_era = _required_text(
        _read_attribute(
            research_plan,
            "story_era",
            "Character research plan",
        ),
        "Character research_plan.story_era",
    )

    return (
        character_id,
        character_name,
        story_context,
        story_era,
    )


# ---------------------------------------------------------
# Production Runner
# ---------------------------------------------------------


def _get_default_character_runner() -> Callable[..., Any]:
    """
    Import provider-backed Character production lazily so deterministic
    regression tests can import this module without initializing Groq.
    """

    from workflows.character_research import (
        research_character_for_production,
    )

    return research_character_for_production


def _invoke_character_runner(
    *,
    character_runner: Callable[..., Any],
    character_name: str,
    story_context: str,
    story_era: str,
    debug: bool,
) -> Any:
    try:
        production_result = (
            character_runner(
                character_name=character_name,
                story_context=story_context,
                story_era=story_era,
                debug=debug,
            )
        )
    except TypeError as error:
        raise TypeError(
            "Character refresh runner must support "
            "the frozen signature "
            "runner(character_name=<str>, "
            "story_context=<str>, "
            "story_era=<str>, "
            "debug=<bool>)."
        ) from error

    if production_result is None:
        raise ValueError(
            "Character refresh runner returned None."
        )

    return production_result


# ---------------------------------------------------------
# Production Result Validation
# ---------------------------------------------------------


def _validate_production_identity(
    *,
    production_result: Any,
    character_name: str,
    story_context: str,
    story_era: str,
) -> None:
    """
    Ensure the frozen Character workflow preserved the Python-owned
    task identity supplied to the refresh.
    """

    research_plan = _read_attribute(
        production_result,
        "research_plan",
        "Character production result",
    )

    returned_name = _required_text(
        _read_attribute(
            research_plan,
            "character_name",
            "Character production research plan",
        ),
        "Character production character_name",
    )

    returned_context = _required_text(
        _read_attribute(
            research_plan,
            "story_context",
            "Character production research plan",
        ),
        "Character production story_context",
    )

    returned_era = _required_text(
        _read_attribute(
            research_plan,
            "story_era",
            "Character production research plan",
        ),
        "Character production story_era",
    )

    if returned_name != character_name:
        raise ValueError(
            "Character production refresh changed "
            "the exact character_name."
        )

    if returned_context != story_context:
        raise ValueError(
            "Character production refresh changed "
            "the exact story_context."
        )

    if returned_era != story_era:
        raise ValueError(
            "Character production refresh changed "
            "the exact story_era."
        )


def _build_candidate_character_artifact(
    *,
    current_character_result: Any,
    production_result: Any,
) -> Any:
    """
    Build a validated candidate Writer's Room Character artifact from
    the frozen Character production package while preserving the
    Python-owned Character ID and display identity.
    """

    if not isinstance(
        current_character_result,
        BaseModel,
    ):
        raise TypeError(
            "Character refresh requires the current "
            "Character artifact to be a Pydantic BaseModel."
        )

    artifact_type = type(
        current_character_result
    )

    current_payload = (
        current_character_result.model_dump(
            mode="python"
        )
    )

    required_artifact_fields = {
        "character_id",
        "character_name",
        "research_plan",
        "canon_research",
        "voice_research",
        "analysis",
        "claim_audit",
        "audit",
    }

    missing_fields = (
        required_artifact_fields
        - set(current_payload)
    )

    if missing_fields:
        raise ValueError(
            "Character artifact is missing required "
            "Writer's Room fields: "
            + ", ".join(
                sorted(
                    missing_fields
                )
            )
        )

    # Python-owned stable identity remains from the existing artifact.
    # The complete research/analysis package is replaced by the new
    # frozen Character production result.
    current_payload.update(
        {
            "research_plan": (
                _read_attribute(
                    production_result,
                    "research_plan",
                    "Character production result",
                )
            ),
            "canon_research": (
                _read_attribute(
                    production_result,
                    "combined_canon_research",
                    "Character production result",
                )
            ),
            "voice_research": (
                _read_attribute(
                    production_result,
                    "voice_research",
                    "Character production result",
                )
            ),
            "analysis": (
                _read_attribute(
                    production_result,
                    "character_analysis",
                    "Character production result",
                )
            ),
            "claim_audit": (
                _read_attribute(
                    production_result,
                    "claim_audit",
                    "Character production result",
                )
            ),
            "audit": (
                _read_attribute(
                    production_result,
                    "analysis_audit",
                    "Character production result",
                )
            ),
        }
    )

    try:
        candidate = (
            artifact_type.model_validate(
                current_payload
            )
        )
    except Exception as error:
        raise ValueError(
            "Frozen Character workflow returned a "
            "production package that cannot be stored "
            "in the current Writer's Room Character "
            "artifact contract."
        ) from error

    # Re-run downstream material contract immediately.
    build_character_material_snapshot(
        candidate
    )

    return candidate


def _verify_candidate_identity_preserved(
    *,
    current_character_result: Any,
    candidate_character_result: Any,
    story_context: str,
    story_era: str,
) -> None:
    old_snapshot = (
        build_character_material_snapshot(
            current_character_result
        )
    )

    new_snapshot = (
        build_character_material_snapshot(
            candidate_character_result
        )
    )

    if (
        old_snapshot.character_id
        != new_snapshot.character_id
    ):
        raise ValueError(
            "Character refresh changed the "
            "Python-owned character_id."
        )

    if (
        old_snapshot.character_name
        != new_snapshot.character_name
    ):
        raise ValueError(
            "Character refresh changed the "
            "Python-owned character_name."
        )

    candidate_plan = _read_attribute(
        candidate_character_result,
        "research_plan",
        "Candidate Character result",
    )

    if (
        _read_attribute(
            candidate_plan,
            "story_context",
            "Candidate Character research plan",
        )
        != story_context
    ):
        raise ValueError(
            "Character refresh candidate changed "
            "story_context."
        )

    if (
        _read_attribute(
            candidate_plan,
            "story_era",
            "Candidate Character research plan",
        )
        != story_era
    ):
        raise ValueError(
            "Character refresh candidate changed "
            "story_era."
        )


# ---------------------------------------------------------
# Invalidation Planning
# ---------------------------------------------------------


def _build_single_target_resolution_plan(
    target: EvaluationResolutionTarget,
) -> EvaluationResolutionPlan:
    return EvaluationResolutionPlan(
        evaluation_status="needs_evidence",
        resolution_status="needs_resolution",
        targets=[
            target
        ],
        reason=(
            "Single-target Character refresh requires "
            "conditional downstream invalidation only "
            "if grounded Character authority changes."
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


def refresh_character_evidence(
    *,
    target: EvaluationResolutionTarget,
    current_character_result: Any,
    character_runner: (
        Callable[..., Any]
        | None
    ) = None,
    debug: bool = False,
) -> CharacterEvidenceRefreshResult:
    """
    Re-run exactly one existing Writer's Room Character task.

    Parameters
    ----------
    target:
        One deterministic EvaluationResolutionTarget for a Character
        consistency evidence blocker.

    current_character_result:
        Existing WritersRoomCharacterResult associated with
        target.artifact_id.

    character_runner:
        Optional dependency injection for deterministic regression
        tests. When omitted, the frozen
        workflows.character_research.research_character_for_production
        function is imported lazily.

    debug:
        Forwarded unchanged to the Character production workflow.

    Returns
    -------
    CharacterEvidenceRefreshResult
        Proposed replacement Character artifact plus material-change
        information. Writer's Room state is never mutated here.
    """

    validate_character_refresh_target(
        target
    )

    (
        character_id,
        character_name,
        story_context,
        story_era,
    ) = _validate_refresh_artifact_identity(
        target=target,
        current_character_result=(
            current_character_result
        ),
    )

    runner = (
        character_runner
        if character_runner is not None
        else _get_default_character_runner()
    )

    if not callable(
        runner
    ):
        raise TypeError(
            "Character refresh runner must be callable."
        )

    production_result = (
        _invoke_character_runner(
            character_runner=runner,
            character_name=character_name,
            story_context=story_context,
            story_era=story_era,
            debug=debug,
        )
    )

    _validate_production_identity(
        production_result=production_result,
        character_name=character_name,
        story_context=story_context,
        story_era=story_era,
    )

    candidate_artifact = (
        _build_candidate_character_artifact(
            current_character_result=(
                current_character_result
            ),
            production_result=(
                production_result
            ),
        )
    )

    _verify_candidate_identity_preserved(
        current_character_result=(
            current_character_result
        ),
        candidate_character_result=(
            candidate_artifact
        ),
        story_context=story_context,
        story_era=story_era,
    )

    # -----------------------------------------------------
    # Grounding Degradation Guard
    #
    # A targeted evidence refresh is allowed to improve an
    # ungrounded Character artifact into grounded authority.
    # It must NOT replace an already-grounded authoritative
    # Character artifact with an ungrounded candidate.
    #
    # Scriptwriter is forbidden from consuming ungrounded
    # Character guidance. Promoting a degraded candidate
    # would therefore invalidate downstream work and then
    # leave the Writer's Room unable to resume.
    #
    # Fail before material-change planning/commit so a
    # multi-target evidence transaction can remain atomic
    # and preserve the last known-good Character authority.
    # -----------------------------------------------------

    old_material_snapshot = (
        build_character_material_snapshot(
            current_character_result
        )
    )

    new_material_snapshot = (
        build_character_material_snapshot(
            candidate_artifact
        )
    )

    if (
        old_material_snapshot.grounded
        and not new_material_snapshot.grounded
    ):
        candidate_audit = _read_attribute(
            candidate_artifact,
            "audit",
            "Candidate Character artifact",
        )

        candidate_issues = getattr(
            candidate_audit,
            "issues",
            None,
        )

        issue_summary = ""

        if candidate_issues:
            issue_summary = "\n".join(
                (
                    f"- {issue.field}: "
                    f"{issue.claim}\n"
                    f"  Reason: {issue.reason}"
                )
                for issue in candidate_issues
            )

        if not issue_summary:
            issue_summary = (
                "- Candidate was ungrounded, but detailed "
                "audit issues were not available."
            )

        raise ValueError(
            "Character evidence refresh degraded an "
            "already-grounded authoritative artifact "
            "to ungrounded. Refusing to promote the "
            "candidate; preserve the last known-good "
            "Character authority.\n\n"
            "Candidate grounding issues:\n"
            + issue_summary
        )


    material_change = (
        compare_character_material_change(
            old_result=(
                current_character_result
            ),
            new_result=(
                candidate_artifact
            ),
        )
    )

    if not material_change.material_change:
        return CharacterEvidenceRefreshResult(
            character_id=character_id,
            character_name=character_name,
            story_context=story_context,
            story_era=story_era,
            refresh_status="unchanged",
            material_change=material_change,
            candidate_artifact=(
                candidate_artifact
            ),
            invalidation_plan=None,
            reason=(
                "Character refresh completed, but "
                "downstream-authoritative grounded "
                "Character guidance did not materially "
                "change. Existing Scriptwriter, "
                "Continuity, ledger, and Evaluation "
                "artifacts should remain preserved."
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
            "Material Character change did not "
            "produce a conditional invalidation plan."
        )

    expected_invalidations = {
        "script_result",
        "continuity_result",
        "story_state_ledger",
        "evaluation_result",
    }

    if not expected_invalidations.issubset(
        set(
            invalidation_plan
            .invalidated_artifacts
        )
    ):
        raise AssertionError(
            "Material Character change produced an "
            "incomplete downstream invalidation plan."
        )

    return CharacterEvidenceRefreshResult(
        character_id=character_id,
        character_name=character_name,
        story_context=story_context,
        story_era=story_era,
        refresh_status="material_change",
        material_change=material_change,
        candidate_artifact=(
            candidate_artifact
        ),
        invalidation_plan=(
            invalidation_plan
        ),
        reason=(
            "Character refresh materially changed "
            "downstream-authoritative grounded guidance. "
            "The candidate replacement is ready for a "
            "future commit layer, and dependent "
            "Scriptwriter/Continuity/Evaluation artifacts "
            "must be invalidated if that candidate is "
            "committed."
        ),
    )
