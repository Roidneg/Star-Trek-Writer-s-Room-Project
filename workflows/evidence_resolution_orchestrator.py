"""
Evidence-resolution orchestration for the Writer's Room.

Version 0.1 orchestrates the two upstream evidence families proven by
the first live Writer's Room flight:

- Lore
- Character

It deliberately does NOT yet support:
- era/style resolution;
- Continuity-context resolution;
- checkpoint persistence;
- Scriptwriter resume;
- Continuity resume;
- Evaluation rerun.

The orchestrator coordinates already-proven components:

    EvaluationResolutionPlan
            ↓
    deterministic target ordering
            ↓
    refresh one upstream artifact
            ↓
    commit candidate to evolving in-memory state
            ↓
    move to next upstream target
            ↓
    union material invalidations
            ↓
    report earliest stale stage

Important sequencing rule:

    Resolve ALL targeted upstream evidence first.
    Resume downstream work only once afterward.

This avoids repeatedly regenerating Scriptwriter / Continuity /
Evaluation while multiple upstream blockers are still unresolved.

Core rules:

    One target at a time.
    Carry committed state forward.
    Never mutate the caller's state in place.
    Never persist from this layer.
    Fail closed on unsupported target families.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from workflows.character_evidence_refresh import (
    CharacterEvidenceRefreshResult,
    refresh_character_evidence,
)
from workflows.character_evidence_refresh_commit import (
    CharacterRefreshCommitResult,
    commit_character_refresh_to_state,
)
from workflows.evaluation_resolution import (
    EvaluationResolutionPlan,
    EvaluationResolutionTarget,
)
from workflows.evidence_refresh import (
    LoreEvidenceRefreshResult,
    refresh_lore_evidence,
)
from workflows.evidence_refresh_commit import (
    LoreRefreshCommitResult,
    commit_lore_refresh_to_state,
)


SupportedResolutionOwner = Literal[
    "lore",
    "character",
]

RefreshStatus = Literal[
    "unchanged",
    "material_change",
]

CommitStatus = Literal[
    "committed_without_invalidation",
    "committed_with_invalidation",
]

ResolutionRunStatus = Literal[
    "no_action",
    "resolved",
    "not_applicable",
]

EarliestStaleStage = Literal[
    "scriptwriter",
    "continuity",
    "evaluation",
]


CANONICAL_INVALIDATION_ORDER: tuple[
    str,
    ...,
] = (
    "script_result",
    "continuity_result",
    "story_state_ledger",
    "evaluation_result",
)


TARGET_OWNER_ORDER: dict[
    str,
    int,
] = {
    "lore": 0,
    "character": 1,
}


class EvidenceResolutionExecutionRecord(BaseModel):
    """
    One completed upstream evidence-resolution target.
    """

    criterion_id: str = Field(
        min_length=1,
    )

    owner: SupportedResolutionOwner

    action: Literal[
        "refresh_lore_evidence",
        "refresh_character_evidence",
    ]

    artifact_id: str = Field(
        min_length=1,
    )

    subject_name: str | None = None

    refresh_status: RefreshStatus

    commit_status: CommitStatus

    material_change: bool

    invalidated_artifacts: list[str] = Field(
        default_factory=list
    )

    reason: str = Field(
        min_length=1,
    )


class EvidenceResolutionOrchestrationResult(BaseModel):
    """
    Non-persistent result of processing an EvaluationResolutionPlan.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    run_status: ResolutionRunStatus

    evaluation_status: Literal[
        "approved",
        "needs_evidence",
        "rejected",
    ]

    processed_targets: list[
        EvidenceResolutionExecutionRecord
    ] = Field(
        default_factory=list
    )

    material_change_detected: bool = False

    cumulative_invalidated_artifacts: list[
        str
    ] = Field(
        default_factory=list
    )

    earliest_stale_stage: (
        EarliestStaleStage
        | None
    ) = None

    final_state: Any

    reason: str = Field(
        min_length=1,
    )


# ---------------------------------------------------------
# Structural Validation
# ---------------------------------------------------------


def _require_state_model(
    state: Any,
) -> BaseModel:
    if not isinstance(
        state,
        BaseModel,
    ):
        raise TypeError(
            "Evidence-resolution orchestration requires "
            "Writer's Room state to be a Pydantic BaseModel."
        )

    required_fields = {
        "lore_results",
        "character_results",
        "script_result",
        "story_state_ledger",
        "continuity_result",
        "evaluation_result",
    }

    state_fields = set(
        state.__class__.model_fields
    )

    missing_fields = (
        required_fields
        - state_fields
    )

    if missing_fields:
        raise TypeError(
            "Writer's Room state is missing required "
            "orchestration fields: "
            + ", ".join(
                sorted(
                    missing_fields
                )
            )
        )

    return state


def _clone_validated_state(
    state: BaseModel,
) -> BaseModel:
    """
    Start orchestration from a validated semantic copy.

    The original caller-owned state must never be mutated.
    """

    state_type = type(
        state
    )

    try:
        return state_type.model_validate(
            state.model_dump(
                mode="python"
            )
        )
    except Exception as error:
        raise ValueError(
            "Writer's Room state failed full Pydantic "
            "validation before evidence resolution."
        ) from error


def _validate_plan_contract(
    plan: EvaluationResolutionPlan,
) -> None:
    if not isinstance(
        plan,
        EvaluationResolutionPlan,
    ):
        raise TypeError(
            "Evidence-resolution orchestrator requires "
            "an EvaluationResolutionPlan."
        )

    if (
        plan.evaluation_status
        == "approved"
    ):
        if (
            plan.resolution_status
            != "no_action"
            or plan.targets
        ):
            raise ValueError(
                "Approved Evaluation resolution plan is "
                "internally inconsistent."
            )

        return

    if (
        plan.evaluation_status
        == "rejected"
    ):
        if (
            plan.resolution_status
            != "not_applicable"
            or plan.targets
        ):
            raise ValueError(
                "Rejected Evaluation resolution plan is "
                "internally inconsistent."
            )

        return

    if (
        plan.evaluation_status
        != "needs_evidence"
    ):
        raise ValueError(
            "Unsupported Evaluation status in "
            "evidence-resolution orchestration."
        )

    if (
        plan.resolution_status
        != "needs_resolution"
    ):
        raise ValueError(
            "NEEDS_EVIDENCE orchestration requires "
            "resolution_status='needs_resolution'."
        )

    if not plan.targets:
        raise ValueError(
            "NEEDS_EVIDENCE orchestration requires "
            "at least one resolution target."
        )

    seen_criterion_ids: set[str] = set()
    seen_artifact_targets: set[
        tuple[
            str,
            str,
        ]
    ] = set()

    for target in plan.targets:
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

        if target.owner not in {
            "lore",
            "character",
        }:
            raise ValueError(
                "Evidence-resolution orchestrator v0.1 "
                "supports only Lore and Character targets. "
                "Unsupported target:\n"
                f"- criterion_id={target.criterion_id!r}\n"
                f"- owner={target.owner!r}\n"
                f"- action={target.action!r}"
            )

        if (
            target.artifact_id is None
            or not target.artifact_id.strip()
        ):
            raise ValueError(
                "Supported evidence-resolution target "
                "requires artifact_id."
            )

        artifact_key = (
            target.owner,
            target.artifact_id,
        )

        if (
            artifact_key
            in seen_artifact_targets
        ):
            raise ValueError(
                "Multiple resolution targets refer to the "
                "same upstream artifact in one orchestration "
                f"run: {artifact_key!r}"
            )

        seen_artifact_targets.add(
            artifact_key
        )


# ---------------------------------------------------------
# Deterministic Target Ordering
# ---------------------------------------------------------


def _target_sort_key(
    target: EvaluationResolutionTarget,
) -> tuple[
    int,
    str,
    str,
]:
    """
    Stable v0.1 ordering:

        Lore before Character,
        then artifact_id,
        then criterion_id.

    This ordering is independent of LLM/output list order.
    """

    owner_order = TARGET_OWNER_ORDER.get(
        target.owner
    )

    if owner_order is None:
        raise ValueError(
            "Unsupported target owner in deterministic "
            f"ordering: {target.owner!r}"
        )

    artifact_id = (
        target.artifact_id
        if target.artifact_id is not None
        else ""
    )

    return (
        owner_order,
        artifact_id,
        target.criterion_id,
    )


def _ordered_targets(
    plan: EvaluationResolutionPlan,
) -> list[
    EvaluationResolutionTarget
]:
    return sorted(
        plan.targets,
        key=_target_sort_key,
    )


# ---------------------------------------------------------
# Artifact Lookup
# ---------------------------------------------------------


def _find_unique_artifact(
    *,
    state: BaseModel,
    collection_name: str,
    id_field: str,
    artifact_id: str,
) -> Any:
    collection = getattr(
        state,
        collection_name,
    )

    if not isinstance(
        collection,
        list,
    ):
        raise TypeError(
            f"Writer's Room {collection_name} must be a list."
        )

    matches = [
        artifact
        for artifact in collection
        if (
            getattr(
                artifact,
                id_field,
                None,
            )
            == artifact_id
        )
    ]

    if not matches:
        raise ValueError(
            f"Writer's Room {collection_name} does not "
            f"contain artifact {artifact_id!r}."
        )

    if len(matches) > 1:
        raise ValueError(
            f"Writer's Room {collection_name} contains "
            f"duplicate artifact ID {artifact_id!r}."
        )

    return matches[0]


# ---------------------------------------------------------
# Execution Helpers
# ---------------------------------------------------------


def _execute_lore_target(
    *,
    state: BaseModel,
    target: EvaluationResolutionTarget,
    lore_runner: (
        Callable[..., Any]
        | None
    ),
    debug: bool,
) -> tuple[
    BaseModel,
    EvidenceResolutionExecutionRecord,
]:
    current_lore_result = (
        _find_unique_artifact(
            state=state,
            collection_name="lore_results",
            id_field="research_id",
            artifact_id=target.artifact_id,
        )
    )

    refresh_result: (
        LoreEvidenceRefreshResult
    ) = refresh_lore_evidence(
        target=target,
        current_lore_result=(
            current_lore_result
        ),
        lore_runner=lore_runner,
        debug=debug,
    )

    commit_result: (
        LoreRefreshCommitResult
    ) = commit_lore_refresh_to_state(
        state=state,
        refresh_result=refresh_result,
    )

    record = (
        EvidenceResolutionExecutionRecord(
            criterion_id=(
                target.criterion_id
            ),
            owner="lore",
            action="refresh_lore_evidence",
            artifact_id=(
                refresh_result.research_id
            ),
            subject_name=None,
            refresh_status=(
                refresh_result.refresh_status
            ),
            commit_status=(
                commit_result.commit_status
            ),
            material_change=(
                refresh_result
                .material_change
                .material_change
            ),
            invalidated_artifacts=list(
                commit_result
                .invalidated_artifacts
            ),
            reason=(
                "Lore evidence target refreshed and "
                "committed to the evolving in-memory "
                "Writer's Room state."
            ),
        )
    )

    return (
        commit_result.committed_state,
        record,
    )


def _execute_character_target(
    *,
    state: BaseModel,
    target: EvaluationResolutionTarget,
    character_runner: (
        Callable[..., Any]
        | None
    ),
    debug: bool,
) -> tuple[
    BaseModel,
    EvidenceResolutionExecutionRecord,
]:
    current_character_result = (
        _find_unique_artifact(
            state=state,
            collection_name="character_results",
            id_field="character_id",
            artifact_id=target.artifact_id,
        )
    )

    refresh_result: (
        CharacterEvidenceRefreshResult
    ) = refresh_character_evidence(
        target=target,
        current_character_result=(
            current_character_result
        ),
        character_runner=(
            character_runner
        ),
        debug=debug,
    )

    commit_result: (
        CharacterRefreshCommitResult
    ) = commit_character_refresh_to_state(
        state=state,
        refresh_result=refresh_result,
    )

    record = (
        EvidenceResolutionExecutionRecord(
            criterion_id=(
                target.criterion_id
            ),
            owner="character",
            action=(
                "refresh_character_evidence"
            ),
            artifact_id=(
                refresh_result.character_id
            ),
            subject_name=(
                refresh_result.character_name
            ),
            refresh_status=(
                refresh_result.refresh_status
            ),
            commit_status=(
                commit_result.commit_status
            ),
            material_change=(
                refresh_result
                .material_change
                .material_change
            ),
            invalidated_artifacts=list(
                commit_result
                .invalidated_artifacts
            ),
            reason=(
                "Character evidence target refreshed "
                "and committed to the evolving in-memory "
                "Writer's Room state."
            ),
        )
    )

    return (
        commit_result.committed_state,
        record,
    )


# ---------------------------------------------------------
# Aggregate Invalidation / Resume
# ---------------------------------------------------------


def _ordered_invalidations(
    records: list[
        EvidenceResolutionExecutionRecord
    ],
) -> list[str]:
    combined: set[str] = set()

    for record in records:
        combined.update(
            record.invalidated_artifacts
        )

    return [
        artifact
        for artifact
        in CANONICAL_INVALIDATION_ORDER
        if artifact in combined
    ]


def _earliest_stale_stage(
    invalidated_artifacts: list[str],
) -> EarliestStaleStage | None:
    if "script_result" in invalidated_artifacts:
        return "scriptwriter"

    if (
        "continuity_result"
        in invalidated_artifacts
        or "story_state_ledger"
        in invalidated_artifacts
    ):
        return "continuity"

    if (
        "evaluation_result"
        in invalidated_artifacts
    ):
        return "evaluation"

    return None


# ---------------------------------------------------------
# Public API
# ---------------------------------------------------------


def orchestrate_evidence_resolution(
    *,
    resolution_plan: EvaluationResolutionPlan,
    state: Any,
    lore_runner: (
        Callable[..., Any]
        | None
    ) = None,
    character_runner: (
        Callable[..., Any]
        | None
    ) = None,
    debug: bool = False,
) -> EvidenceResolutionOrchestrationResult:
    """
    Resolve all currently supported upstream evidence targets against
    one evolving in-memory Writer's Room state.

    v0.1 supports Lore and Character targets only.

    No checkpoint is written by this function.

    If multiple targets exist, all are resolved before the caller
    resumes any stale downstream stage.
    """

    _validate_plan_contract(
        resolution_plan
    )

    current_state = (
        _clone_validated_state(
            _require_state_model(
                state
            )
        )
    )

    # -----------------------------------------------------
    # Approved
    # -----------------------------------------------------

    if (
        resolution_plan.evaluation_status
        == "approved"
    ):
        return (
            EvidenceResolutionOrchestrationResult(
                run_status="no_action",
                evaluation_status="approved",
                processed_targets=[],
                material_change_detected=False,
                cumulative_invalidated_artifacts=[],
                earliest_stale_stage=None,
                final_state=current_state,
                reason=(
                    "Evaluation is approved. No "
                    "evidence-resolution work is required."
                ),
            )
        )

    # -----------------------------------------------------
    # Rejected
    # -----------------------------------------------------

    if (
        resolution_plan.evaluation_status
        == "rejected"
    ):
        return (
            EvidenceResolutionOrchestrationResult(
                run_status="not_applicable",
                evaluation_status="rejected",
                processed_targets=[],
                material_change_detected=False,
                cumulative_invalidated_artifacts=[],
                earliest_stale_stage=None,
                final_state=current_state,
                reason=(
                    "Evaluation rejected the scene. "
                    "Evidence-resolution refresh is not "
                    "the applicable recovery path."
                ),
            )
        )

    # -----------------------------------------------------
    # NEEDS_EVIDENCE
    # -----------------------------------------------------

    execution_records: list[
        EvidenceResolutionExecutionRecord
    ] = []

    for target in _ordered_targets(
        resolution_plan
    ):
        if target.owner == "lore":
            (
                current_state,
                record,
            ) = _execute_lore_target(
                state=current_state,
                target=target,
                lore_runner=lore_runner,
                debug=debug,
            )

        elif target.owner == "character":
            (
                current_state,
                record,
            ) = _execute_character_target(
                state=current_state,
                target=target,
                character_runner=(
                    character_runner
                ),
                debug=debug,
            )

        else:
            # _validate_plan_contract() should make this unreachable.
            raise AssertionError(
                "Unsupported evidence-resolution target "
                "escaped orchestration validation."
            )

        execution_records.append(
            record
        )

    cumulative_invalidations = (
        _ordered_invalidations(
            execution_records
        )
    )

    material_change_detected = any(
        record.material_change
        for record in execution_records
    )

    earliest_stale_stage = (
        _earliest_stale_stage(
            cumulative_invalidations
        )
    )

    if (
        material_change_detected
        and earliest_stale_stage is None
    ):
        raise AssertionError(
            "Evidence resolution reported material "
            "change without any stale downstream stage."
        )

    if (
        not material_change_detected
        and cumulative_invalidations
    ):
        raise AssertionError(
            "Evidence resolution accumulated invalidations "
            "without a material upstream change."
        )

    return EvidenceResolutionOrchestrationResult(
        run_status="resolved",
        evaluation_status="needs_evidence",
        processed_targets=(
            execution_records
        ),
        material_change_detected=(
            material_change_detected
        ),
        cumulative_invalidated_artifacts=(
            cumulative_invalidations
        ),
        earliest_stale_stage=(
            earliest_stale_stage
        ),
        final_state=current_state,
        reason=(
            f"Resolved {len(execution_records)} supported "
            "upstream evidence target(s) before downstream "
            "resume. "
            + (
                "At least one authoritative upstream "
                "artifact materially changed; resume from "
                f"{earliest_stale_stage}."
                if material_change_detected
                else (
                    "No downstream-authoritative material "
                    "change occurred; existing downstream "
                    "artifacts remain valid."
                )
            )
        ),
    )
