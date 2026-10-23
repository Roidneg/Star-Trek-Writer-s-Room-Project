"""
Deterministic downstream artifact invalidation planning.

This module consumes the output of workflows.evaluation_resolution and
answers one question:

    If a requested evidence resolution materially changes its
    authoritative upstream source, which completed Writer's Room
    artifacts become stale?

It does NOT:
- call an LLM;
- refresh Lore or Character evidence;
- mutate WritersRoomState;
- delete checkpoint data;
- decide whether a refresh actually changed an artifact;
- rebuild the Continuity ledger.

Core rule:

    Refresh narrowly.
    Invalidate only proven dependents.
    Do not invalidate anything until the upstream artifact
    materially changes.

Current Writer's Room dependencies:

    Lore ---------------------> Scriptwriter
      +-----------------------> Evaluation

    Character ----------------> Scriptwriter
      +-----------------------> Evaluation

    ScriptwriterInput --------> Scriptwriter

    Scriptwriter -------------> Continuity
      +-----------------------> Evaluation

    Continuity ---------------> Story-state ledger
      +-----------------------> Evaluation

Story Architect is not invalidated by the current evidence-resolution
loop.

Lore and Character are sibling evidence sources in the current
architecture. Refreshing Lore does not automatically invalidate
Character, and refreshing Character does not automatically invalidate
Lore.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from workflows.evaluation_resolution import (
    EvaluationResolutionPlan,
    EvaluationResolutionTarget,
)


ArtifactName = Literal[
    "script_result",
    "continuity_result",
    "story_state_ledger",
    "evaluation_result",
]

PreservedArtifactFamily = Literal[
    "story_plan",
    "lore_results",
    "character_results",
    "script_result",
]


CANONICAL_INVALIDATION_ORDER: tuple[
    ArtifactName,
    ...,
] = (
    "script_result",
    "continuity_result",
    "story_state_ledger",
    "evaluation_result",
)


ALL_PRESERVABLE_FAMILIES: tuple[
    PreservedArtifactFamily,
    ...,
] = (
    "story_plan",
    "lore_results",
    "character_results",
    "script_result",
)


class ArtifactInvalidationCause(BaseModel):
    criterion_id: str = Field(
        min_length=1,
    )

    resolution_owner: Literal[
        "lore",
        "character",
        "scriptwriter_input",
        "continuity",
    ]

    resolution_action: Literal[
        "refresh_lore_evidence",
        "refresh_character_evidence",
        "supply_era_style_constraints",
        "resolve_continuity_context",
    ]

    invalidates: list[
        ArtifactName
    ] = Field(
        default_factory=list
    )

    reason: str = Field(
        min_length=1,
    )


class ArtifactInvalidationPlan(BaseModel):
    evaluation_status: Literal[
        "approved",
        "needs_evidence",
        "rejected",
    ]

    resolution_status: Literal[
        "no_action",
        "needs_resolution",
        "not_applicable",
    ]

    trigger: Literal[
        "none",
        "on_material_change",
        "not_applicable",
    ]

    invalidated_artifacts: list[
        ArtifactName
    ] = Field(
        default_factory=list
    )

    preserved_artifact_families: list[
        PreservedArtifactFamily
    ] = Field(
        default_factory=list
    )

    causes: list[
        ArtifactInvalidationCause
    ] = Field(
        default_factory=list
    )

    requires_ledger_rebuild: bool = False

    reason: str = Field(
        min_length=1,
    )


def _ordered_artifacts(
    artifacts: set[ArtifactName],
) -> list[ArtifactName]:
    """
    Return invalidations in pipeline order.

    Stable ordering matters for tests, diagnostics, and later
    checkpoint mutation logic.
    """

    return [
        artifact
        for artifact
        in CANONICAL_INVALIDATION_ORDER
        if artifact in artifacts
    ]


def _invalidation_for_target(
    target: EvaluationResolutionTarget,
) -> ArtifactInvalidationCause:
    """
    Map one evidence-resolution target to its proven downstream
    dependencies.

    This function describes what becomes stale only if the target's
    authoritative source materially changes.
    """

    if (
        target.owner == "lore"
        and target.action
        == "refresh_lore_evidence"
    ):
        invalidates: set[
            ArtifactName
        ] = {
            "script_result",
            "continuity_result",
            "story_state_ledger",
            "evaluation_result",
        }

        return ArtifactInvalidationCause(
            criterion_id=target.criterion_id,
            resolution_owner=target.owner,
            resolution_action=target.action,
            invalidates=_ordered_artifacts(
                invalidates
            ),
            reason=(
                "ScriptwriterInput derives verified canon facts "
                "from Writer's Room Lore results, and final "
                "Evaluation also consumes the complete Lore "
                "artifact set. A materially changed Lore result "
                "therefore makes the existing Scriptwriter scene "
                "and its downstream Continuity/Evaluation "
                "artifacts stale."
            ),
        )

    if (
        target.owner == "character"
        and target.action
        == "refresh_character_evidence"
    ):
        invalidates = {
            "script_result",
            "continuity_result",
            "story_state_ledger",
            "evaluation_result",
        }

        return ArtifactInvalidationCause(
            criterion_id=target.criterion_id,
            resolution_owner=target.owner,
            resolution_action=target.action,
            invalidates=_ordered_artifacts(
                invalidates
            ),
            reason=(
                "ScriptwriterInput derives linked character "
                "guidance from the authoritative Character "
                "artifact, and final Evaluation also consumes "
                "matching Character results. A materially "
                "changed Character artifact therefore makes "
                "the existing Scriptwriter scene and its "
                "downstream Continuity/Evaluation artifacts "
                "stale."
            ),
        )

    if (
        target.owner == "scriptwriter_input"
        and target.action
        == "supply_era_style_constraints"
    ):
        invalidates = {
            "script_result",
            "continuity_result",
            "story_state_ledger",
            "evaluation_result",
        }

        return ArtifactInvalidationCause(
            criterion_id=target.criterion_id,
            resolution_owner=target.owner,
            resolution_action=target.action,
            invalidates=_ordered_artifacts(
                invalidates
            ),
            reason=(
                "Era/style constraints are part of the "
                "authoritative ScriptwriterInput. A materially "
                "changed constraint set requires a new grounded "
                "Scriptwriter result, which makes downstream "
                "Continuity state and Evaluation stale."
            ),
        )

    if (
        target.owner == "continuity"
        and target.action
        == "resolve_continuity_context"
    ):
        invalidates = {
            "continuity_result",
            "story_state_ledger",
            "evaluation_result",
        }

        return ArtifactInvalidationCause(
            criterion_id=target.criterion_id,
            resolution_owner=target.owner,
            resolution_action=target.action,
            invalidates=_ordered_artifacts(
                invalidates
            ),
            reason=(
                "Continuity context affects the authoritative "
                "Continuity review and the story-state ledger. "
                "Final Evaluation consumes the Continuity "
                "review. Scriptwriter itself is upstream of "
                "Continuity and remains valid."
            ),
        )

    raise ValueError(
        "Unsupported evidence-resolution target for "
        "artifact invalidation:\n"
        f"- criterion_id={target.criterion_id!r}\n"
        f"- owner={target.owner!r}\n"
        f"- action={target.action!r}"
    )


def _preserved_families(
    resolution_plan: EvaluationResolutionPlan,
    invalidated_artifacts: set[
        ArtifactName
    ],
) -> list[
    PreservedArtifactFamily
]:
    """
    Identify upstream artifact families that are explicitly preserved.

    A family being refreshed is not listed as preserved even when other
    members of that family may remain unchanged.
    """

    directly_refreshed_families: set[str] = set()

    for target in resolution_plan.targets:
        if target.owner == "lore":
            directly_refreshed_families.add(
                "lore_results"
            )

        elif target.owner == "character":
            directly_refreshed_families.add(
                "character_results"
            )

    preserved: list[
        PreservedArtifactFamily
    ] = []

    for family in ALL_PRESERVABLE_FAMILIES:
        if family in directly_refreshed_families:
            continue

        if family in invalidated_artifacts:
            continue

        preserved.append(
            family
        )

    return preserved


def build_artifact_invalidation_plan(
    resolution_plan: EvaluationResolutionPlan,
) -> ArtifactInvalidationPlan:
    """
    Build the deterministic downstream invalidation plan for one
    EvaluationResolutionPlan.

    Important:
        The returned invalidations are conditional on a MATERIAL change
        to the refreshed upstream source. A future comparison/apply layer
        must prove that change before mutating Writer's Room state.
    """

    # -----------------------------------------------------
    # Approved
    # -----------------------------------------------------

    if (
        resolution_plan.evaluation_status
        == "approved"
    ):
        if (
            resolution_plan.resolution_status
            != "no_action"
            or resolution_plan.targets
        ):
            raise ValueError(
                "Approved Evaluation resolution plan is "
                "internally inconsistent."
            )

        return ArtifactInvalidationPlan(
            evaluation_status="approved",
            resolution_status="no_action",
            trigger="none",
            invalidated_artifacts=[],
            preserved_artifact_families=list(
                ALL_PRESERVABLE_FAMILIES
            ),
            causes=[],
            requires_ledger_rebuild=False,
            reason=(
                "Evaluation is approved. No evidence "
                "resolution or artifact invalidation is "
                "required."
            ),
        )

    # -----------------------------------------------------
    # Rejected
    # -----------------------------------------------------

    if (
        resolution_plan.evaluation_status
        == "rejected"
    ):
        if (
            resolution_plan.resolution_status
            != "not_applicable"
            or resolution_plan.targets
        ):
            raise ValueError(
                "Rejected Evaluation resolution plan is "
                "internally inconsistent."
            )

        return ArtifactInvalidationPlan(
            evaluation_status="rejected",
            resolution_status="not_applicable",
            trigger="not_applicable",
            invalidated_artifacts=[],
            preserved_artifact_families=list(
                ALL_PRESERVABLE_FAMILIES
            ),
            causes=[],
            requires_ledger_rebuild=False,
            reason=(
                "Evaluation rejected the scene. The "
                "evidence-resolution invalidation path "
                "does not apply to failed criteria."
            ),
        )

    # -----------------------------------------------------
    # Needs Evidence
    # -----------------------------------------------------

    if (
        resolution_plan.evaluation_status
        != "needs_evidence"
    ):
        raise ValueError(
            "Unsupported Evaluation status in "
            "artifact invalidation plan."
        )

    if (
        resolution_plan.resolution_status
        != "needs_resolution"
    ):
        raise ValueError(
            "NEEDS_EVIDENCE must use "
            "resolution_status='needs_resolution'."
        )

    if not resolution_plan.targets:
        raise ValueError(
            "NEEDS_EVIDENCE invalidation requires at "
            "least one resolution target."
        )

    seen_criterion_ids: set[str] = set()
    causes: list[
        ArtifactInvalidationCause
    ] = []

    combined_invalidations: set[
        ArtifactName
    ] = set()

    for target in resolution_plan.targets:
        if (
            target.criterion_id
            in seen_criterion_ids
        ):
            raise ValueError(
                "Duplicate resolution target criterion ID: "
                f"{target.criterion_id!r}"
            )

        seen_criterion_ids.add(
            target.criterion_id
        )

        cause = _invalidation_for_target(
            target
        )

        causes.append(
            cause
        )

        combined_invalidations.update(
            cause.invalidates
        )

    ordered_invalidations = (
        _ordered_artifacts(
            combined_invalidations
        )
    )

    requires_ledger_rebuild = (
        "story_state_ledger"
        in combined_invalidations
    )

    preserved = _preserved_families(
        resolution_plan=resolution_plan,
        invalidated_artifacts=(
            combined_invalidations
        ),
    )

    return ArtifactInvalidationPlan(
        evaluation_status="needs_evidence",
        resolution_status="needs_resolution",
        trigger="on_material_change",
        invalidated_artifacts=(
            ordered_invalidations
        ),
        preserved_artifact_families=preserved,
        causes=causes,
        requires_ledger_rebuild=(
            requires_ledger_rebuild
        ),
        reason=(
            f"{len(resolution_plan.targets)} evidence "
            "resolution target(s) map to "
            f"{len(ordered_invalidations)} unique stale "
            "downstream artifact(s) if the refreshed "
            "upstream authority materially changes. "
            "No Writer's Room state should be mutated "
            "until material change is proven."
        ),
    )
