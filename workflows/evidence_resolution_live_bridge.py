"""
Live checkpoint bridge for Writer's Room evidence resolution.

This module connects a completed Writer's Room Evaluation artifact to
the proven evidence-resolution subsystem without replacing the live
smoke test's per-stage downstream checkpointing.

Production-safe flow:

    completed Evaluation checkpoint
            ↓
    build deterministic EvaluationResolutionPlan
            ↓
    if approved/rejected:
        no evidence refresh
        no checkpoint rewrite
            ↓
    if NEEDS_EVIDENCE:
        resolve all supported Lore / Character blockers in memory
            ↓
        persist combined upstream resolution exactly once
            ↓
        return the saved state + earliest stale stage

The caller should then resume the normal Writer's Room stage loop from
that stale stage. Scriptwriter / Continuity / Evaluation keep their
existing per-stage checkpoint boundaries.

Why not resume the whole suffix here?
-------------------------------------
The live smoke test deliberately checkpoints after each expensive stage.
That behavior protected successful work during provider rate limits.
This bridge preserves that operational resilience.

This module does NOT:
- rerun Scriptwriter;
- rerun Continuity;
- rerun Evaluation;
- replace the normal live stage checkpoint cadence;
- loop evidence resolution indefinitely.

Version 0.1 supports only the Lore and Character target families already
proven by evidence_resolution_orchestrator.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from workflows.evaluation_resolution import (
    CharacterResolutionSource,
    EvaluationResolutionPlan,
    LoreResolutionSource,
    build_evaluation_resolution_plan,
)
from workflows.evidence_resolution_checkpoint import (
    EvidenceResolutionCheckpointResult,
    persist_evidence_resolution,
)
from workflows.evidence_resolution_orchestrator import (
    EvidenceResolutionOrchestrationResult,
    orchestrate_evidence_resolution,
)


LiveResolutionOutcome = Literal[
    "no_action",
    "not_applicable",
    "resolved_no_material_change",
    "resolved_material_change",
]


class EvidenceResolutionLiveBridgeResult(BaseModel):
    """
    Result of one bounded live evidence-resolution transaction.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    outcome: LiveResolutionOutcome

    resolution_plan: EvaluationResolutionPlan

    orchestration_result: (
        EvidenceResolutionOrchestrationResult
        | None
    ) = None

    persistence_result: (
        EvidenceResolutionCheckpointResult
        | None
    ) = None

    earliest_stale_stage: (
        Literal[
            "scriptwriter",
            "continuity",
            "evaluation",
        ]
        | None
    ) = None

    final_state: Any

    checkpoint_changed: bool

    backup_path: str | None = None

    reason: str = Field(
        min_length=1,
    )


# ---------------------------------------------------------
# State / Evaluation Validation
# ---------------------------------------------------------


def _require_state_model(
    state: Any,
) -> BaseModel:
    if not isinstance(
        state,
        BaseModel,
    ):
        raise TypeError(
            "Live evidence-resolution bridge requires "
            "Writer's Room state to be a Pydantic BaseModel."
        )

    required_fields = {
        "lore_results",
        "character_results",
        "evaluation_result",
    }

    missing_fields = (
        required_fields
        - set(
            state.__class__.model_fields
        )
    )

    if missing_fields:
        raise TypeError(
            "Writer's Room state is missing required "
            "live-resolution fields: "
            + ", ".join(
                sorted(
                    missing_fields
                )
            )
        )

    if (
        getattr(
            state,
            "evaluation_result",
        )
        is None
    ):
        raise ValueError(
            "Live evidence resolution requires a "
            "completed Evaluation result."
        )

    return state


def _semantic_dump(
    value: Any,
) -> Any:
    if isinstance(
        value,
        BaseModel,
    ):
        return value.model_dump(
            mode="json"
        )

    if isinstance(
        value,
        list,
    ):
        return [
            _semantic_dump(
                item
            )
            for item in value
        ]

    return value


def _read_evaluation_routing(
    state: BaseModel,
) -> tuple[
    Any,
    list[str],
    list[str],
]:
    evaluation_result = getattr(
        state,
        "evaluation_result",
    )

    required_fields = (
        "production_review",
        "lore_research_ids",
        "character_result_ids",
    )

    for field_name in required_fields:
        if not hasattr(
            evaluation_result,
            field_name,
        ):
            raise TypeError(
                "Writer's Room Evaluation result is "
                f"missing required field {field_name!r}."
            )

    production_review = getattr(
        evaluation_result,
        "production_review",
    )

    lore_ids = list(
        getattr(
            evaluation_result,
            "lore_research_ids",
        )
    )

    character_ids = list(
        getattr(
            evaluation_result,
            "character_result_ids",
        )
    )

    if len(lore_ids) != len(
        set(lore_ids)
    ):
        raise ValueError(
            "Evaluation routing contains duplicate "
            "Lore research IDs."
        )

    if len(character_ids) != len(
        set(character_ids)
    ):
        raise ValueError(
            "Evaluation routing contains duplicate "
            "Character result IDs."
        )

    return (
        production_review,
        lore_ids,
        character_ids,
    )


# ---------------------------------------------------------
# Resolution Source Construction
# ---------------------------------------------------------


def _index_unique_artifacts(
    *,
    artifacts: list[Any],
    id_field: str,
    label: str,
) -> dict[str, Any]:
    index: dict[str, Any] = {}

    for artifact in artifacts:
        artifact_id = getattr(
            artifact,
            id_field,
            None,
        )

        if (
            not isinstance(
                artifact_id,
                str,
            )
            or not artifact_id.strip()
        ):
            raise ValueError(
                f"{label} artifact is missing a valid "
                f"{id_field}."
            )

        if artifact_id in index:
            raise ValueError(
                f"Writer's Room contains duplicate "
                f"{label} artifact ID "
                f"{artifact_id!r}."
            )

        index[
            artifact_id
        ] = artifact

    return index


def _build_resolution_sources(
    *,
    state: BaseModel,
    lore_ids: list[str],
    character_ids: list[str],
) -> tuple[
    list[LoreResolutionSource],
    list[CharacterResolutionSource],
]:
    lore_index = (
        _index_unique_artifacts(
            artifacts=list(
                getattr(
                    state,
                    "lore_results",
                )
            ),
            id_field="research_id",
            label="Lore",
        )
    )

    character_index = (
        _index_unique_artifacts(
            artifacts=list(
                getattr(
                    state,
                    "character_results",
                )
            ),
            id_field="character_id",
            label="Character",
        )
    )

    lore_sources: list[
        LoreResolutionSource
    ] = []

    for research_id in lore_ids:
        if research_id not in lore_index:
            raise ValueError(
                "Evaluation references unknown Writer's "
                "Room Lore artifact ID: "
                f"{research_id!r}"
            )

        lore_sources.append(
            LoreResolutionSource(
                research_id=research_id
            )
        )

    character_sources: list[
        CharacterResolutionSource
    ] = []

    for character_id in character_ids:
        if character_id not in character_index:
            raise ValueError(
                "Evaluation references unknown Writer's "
                "Room Character artifact ID: "
                f"{character_id!r}"
            )

        character_artifact = (
            character_index[
                character_id
            ]
        )

        character_name = getattr(
            character_artifact,
            "character_name",
            None,
        )

        if (
            not isinstance(
                character_name,
                str,
            )
            or not character_name.strip()
        ):
            raise ValueError(
                "Character artifact referenced by "
                "Evaluation has no valid character_name."
            )

        character_sources.append(
            CharacterResolutionSource(
                character_id=character_id,
                character_name=(
                    character_name
                ),
            )
        )

    return (
        lore_sources,
        character_sources,
    )


# ---------------------------------------------------------
# Checkpoint Concurrency Base
# ---------------------------------------------------------


def _capture_matching_checkpoint(
    *,
    state: BaseModel,
    checkpoint_path: Path,
) -> bytes:
    if not isinstance(
        checkpoint_path,
        Path,
    ):
        raise TypeError(
            "checkpoint_path must be pathlib.Path."
        )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "Live Writer's Room checkpoint does not exist:\n"
            f"{checkpoint_path}"
        )

    checkpoint_bytes = (
        checkpoint_path.read_bytes()
    )

    if not checkpoint_bytes:
        raise ValueError(
            "Live Writer's Room checkpoint is empty."
        )

    try:
        checkpoint_state = (
            type(state)
            .model_validate_json(
                checkpoint_bytes.decode(
                    "utf-8"
                )
            )
        )
    except Exception as error:
        raise ValueError(
            "Live Writer's Room checkpoint could not "
            "be validated before evidence resolution."
        ) from error

    if (
        _semantic_dump(
            checkpoint_state
        )
        != _semantic_dump(
            state
        )
    ):
        raise ValueError(
            "Supplied Writer's Room state does not match "
            "the current checkpoint. Refusing to begin "
            "evidence resolution from divergent state."
        )

    return checkpoint_bytes


# ---------------------------------------------------------
# Public API
# ---------------------------------------------------------


def resolve_live_evaluation_checkpoint(
    *,
    state: Any,
    checkpoint_path: Path,
    lore_runner: Any = None,
    character_runner: Any = None,
    debug: bool = False,
) -> EvidenceResolutionLiveBridgeResult:
    """
    Resolve one completed live Evaluation checkpoint.

    If Evaluation is approved or rejected, no checkpoint rewrite occurs.

    If Evaluation needs evidence:
    1. Build deterministic resolution targets from the exact Evaluation
       source routing stored in WritersRoomEvaluationResult.
    2. Resolve every supported Lore / Character target in memory.
    3. Persist the final combined upstream state once.
    4. Return earliest_stale_stage to the caller.

    The caller remains responsible for resuming downstream Writer's Room
    stages using its normal per-stage checkpoint cadence.
    """

    current_state = _require_state_model(
        state
    )

    (
        production_review,
        lore_ids,
        character_ids,
    ) = _read_evaluation_routing(
        current_state
    )

    (
        lore_sources,
        character_sources,
    ) = _build_resolution_sources(
        state=current_state,
        lore_ids=lore_ids,
        character_ids=character_ids,
    )

    resolution_plan = (
        build_evaluation_resolution_plan(
            production_review=(
                production_review
            ),
            lore_sources=(
                lore_sources
            ),
            character_sources=(
                character_sources
            ),
        )
    )

    # -----------------------------------------------------
    # Evaluation already final / outside evidence loop
    # -----------------------------------------------------

    if (
        resolution_plan.resolution_status
        == "no_action"
    ):
        return EvidenceResolutionLiveBridgeResult(
            outcome="no_action",
            resolution_plan=(
                resolution_plan
            ),
            orchestration_result=None,
            persistence_result=None,
            earliest_stale_stage=None,
            final_state=current_state,
            checkpoint_changed=False,
            backup_path=None,
            reason=(
                "Evaluation is approved. No evidence "
                "resolution or checkpoint rewrite is needed."
            ),
        )

    if (
        resolution_plan.resolution_status
        == "not_applicable"
    ):
        return EvidenceResolutionLiveBridgeResult(
            outcome="not_applicable",
            resolution_plan=(
                resolution_plan
            ),
            orchestration_result=None,
            persistence_result=None,
            earliest_stale_stage=None,
            final_state=current_state,
            checkpoint_changed=False,
            backup_path=None,
            reason=(
                "Evaluation is rejected. Evidence refresh "
                "is not the applicable recovery path."
            ),
        )

    if (
        resolution_plan.resolution_status
        != "needs_resolution"
    ):
        raise AssertionError(
            "Unexpected Evaluation resolution status."
        )

    # Capture the exact checkpoint only after the deterministic
    # plan has proven that evidence work is actually required.
    original_checkpoint_bytes = (
        _capture_matching_checkpoint(
            state=current_state,
            checkpoint_path=(
                checkpoint_path
            ),
        )
    )

    orchestration_result = (
        orchestrate_evidence_resolution(
            resolution_plan=(
                resolution_plan
            ),
            state=current_state,
            lore_runner=lore_runner,
            character_runner=(
                character_runner
            ),
            debug=debug,
        )
    )

    persistence_result = (
        persist_evidence_resolution(
            checkpoint_path=(
                checkpoint_path
            ),
            original_checkpoint_bytes=(
                original_checkpoint_bytes
            ),
            orchestration_result=(
                orchestration_result
            ),
        )
    )

    final_state = (
        persistence_result.saved_state
    )

    if (
        orchestration_result
        .material_change_detected
    ):
        if (
            orchestration_result
            .earliest_stale_stage
            is None
        ):
            raise AssertionError(
                "Material live evidence resolution "
                "did not identify a stale resume stage."
            )

        return EvidenceResolutionLiveBridgeResult(
            outcome="resolved_material_change",
            resolution_plan=(
                resolution_plan
            ),
            orchestration_result=(
                orchestration_result
            ),
            persistence_result=(
                persistence_result
            ),
            earliest_stale_stage=(
                orchestration_result
                .earliest_stale_stage
            ),
            final_state=final_state,
            checkpoint_changed=True,
            backup_path=(
                persistence_result
                .backup_path
            ),
            reason=(
                "Live evidence blockers were resolved and "
                "their combined upstream state was persisted "
                "once. At least one authoritative source "
                "materially changed; resume the normal "
                "Writer's Room stage loop from "
                f"{orchestration_result.earliest_stale_stage}."
            ),
        )

    return EvidenceResolutionLiveBridgeResult(
        outcome="resolved_no_material_change",
        resolution_plan=(
            resolution_plan
        ),
        orchestration_result=(
            orchestration_result
        ),
        persistence_result=(
            persistence_result
        ),
        earliest_stale_stage=None,
        final_state=final_state,
        checkpoint_changed=True,
        backup_path=(
            persistence_result
            .backup_path
        ),
        reason=(
            "Live evidence refresh completed and refreshed "
            "research provenance was persisted, but no "
            "downstream-authoritative material changed. "
            "Do not automatically repeat the same evidence "
            "resolution cycle or rerun downstream stages."
        ),
    )
