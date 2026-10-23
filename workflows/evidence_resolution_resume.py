"""
Downstream resume execution after evidence resolution.

Version 0.1 resumes the stale Writer's Room suffix identified by
EvidenceResolutionOrchestrationResult.

Supported resume points:
- scriptwriter
- continuity
- evaluation

The current Lore/Character evidence-resolution architecture normally
produces either:
- earliest_stale_stage=None
- earliest_stale_stage="scriptwriter"

The broader stage support is included now so the resume contract can
later accept Continuity-only or Evaluation-only invalidation without
being redesigned.

Important Scriptwriter dependency
---------------------------------
WritersRoomState does not currently store the WritersRoomScriptRequest.
The existing Writer's Room stage requires:

    run_scriptwriter_stage(
        state=state,
        request=script_request,
    )

Therefore a Scriptwriter resume MUST receive the original/current
script_request explicitly.

This module does NOT:
- resolve evidence blockers;
- refresh Lore or Character;
- persist checkpoints;
- alter upstream Lore / Character artifacts;
- infer or reconstruct a missing Scriptwriter request.

Core rule:

    Resume exactly the stale suffix.
    Never regenerate a valid upstream stage.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from workflows.evidence_resolution_orchestrator import (
    EvidenceResolutionOrchestrationResult,
)


ResumeStage = Literal[
    "scriptwriter",
    "continuity",
    "evaluation",
]

ResumeRunStatus = Literal[
    "no_resume",
    "resumed",
]


class EvidenceResolutionResumeResult(BaseModel):
    """
    Result of executing the stale downstream Writer's Room suffix.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    run_status: ResumeRunStatus

    started_at: (
        ResumeStage
        | None
    ) = None

    completed_stages: list[
        ResumeStage
    ] = Field(
        default_factory=list
    )

    final_state: Any

    reason: str = Field(
        min_length=1,
    )


# ---------------------------------------------------------
# State Helpers
# ---------------------------------------------------------


def _require_state_model(
    state: Any,
) -> BaseModel:
    if not isinstance(
        state,
        BaseModel,
    ):
        raise TypeError(
            "Evidence-resolution resume requires "
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
            "resume fields: "
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
            "validation before downstream resume."
        ) from error


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


def _capture_upstream_authority(
    state: BaseModel,
) -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "lore_results": _semantic_dump(
            getattr(
                state,
                "lore_results",
            )
        ),
        "character_results": _semantic_dump(
            getattr(
                state,
                "character_results",
            )
        ),
    }

    if hasattr(
        state,
        "story_plan",
    ):
        snapshot[
            "story_plan"
        ] = _semantic_dump(
            getattr(
                state,
                "story_plan",
            )
        )

    return snapshot


def _assert_upstream_authority_preserved(
    *,
    state: BaseModel,
    expected: dict[str, Any],
) -> None:
    current = _capture_upstream_authority(
        state
    )

    if current != expected:
        raise AssertionError(
            "Downstream resume unexpectedly changed "
            "upstream StoryPlan/Lore/Character authority."
        )


# ---------------------------------------------------------
# Orchestration Contract
# ---------------------------------------------------------


def _validate_orchestration_result(
    result: EvidenceResolutionOrchestrationResult,
) -> None:
    if not isinstance(
        result,
        EvidenceResolutionOrchestrationResult,
    ):
        raise TypeError(
            "Downstream resume requires an "
            "EvidenceResolutionOrchestrationResult."
        )

    if (
        result.run_status
        in {
            "no_action",
            "not_applicable",
        }
    ):
        if (
            result.earliest_stale_stage
            is not None
        ):
            raise ValueError(
                "Non-resolution orchestration result "
                "cannot contain a stale resume stage."
            )

        return

    if result.run_status != "resolved":
        raise ValueError(
            "Unsupported evidence-resolution "
            f"run_status: {result.run_status!r}"
        )

    if result.material_change_detected:
        if (
            result.earliest_stale_stage
            is None
        ):
            raise ValueError(
                "Material evidence resolution must "
                "identify an earliest stale stage."
            )

        if (
            not result
            .cumulative_invalidated_artifacts
        ):
            raise ValueError(
                "Material evidence resolution must "
                "carry invalidated artifacts."
            )

    else:
        if (
            result.earliest_stale_stage
            is not None
        ):
            raise ValueError(
                "Non-material evidence resolution "
                "cannot identify a stale stage."
            )

        if (
            result
            .cumulative_invalidated_artifacts
        ):
            raise ValueError(
                "Non-material evidence resolution "
                "cannot carry invalidated artifacts."
            )


# ---------------------------------------------------------
# Resume-State Preconditions
# ---------------------------------------------------------


def _require_empty_ledger(
    state: BaseModel,
) -> None:
    ledger = getattr(
        state,
        "story_state_ledger",
    )

    if hasattr(
        ledger,
        "entries",
    ):
        entries = getattr(
            ledger,
            "entries",
        )

        if entries:
            raise ValueError(
                "Scriptwriter/Continuity resume requires "
                "the invalidated story-state ledger to "
                "be empty."
            )


def _validate_resume_preconditions(
    *,
    state: BaseModel,
    resume_stage: ResumeStage,
    script_request: Any,
) -> None:
    script_result = getattr(
        state,
        "script_result",
    )

    continuity_result = getattr(
        state,
        "continuity_result",
    )

    evaluation_result = getattr(
        state,
        "evaluation_result",
    )

    if resume_stage == "scriptwriter":
        if script_request is None:
            raise ValueError(
                "Scriptwriter resume requires the "
                "WritersRoomScriptRequest explicitly. "
                "The request is not stored in "
                "WritersRoomState."
            )

        if script_result is not None:
            raise ValueError(
                "Scriptwriter resume requires "
                "script_result=None."
            )

        if continuity_result is not None:
            raise ValueError(
                "Scriptwriter resume requires "
                "continuity_result=None."
            )

        if evaluation_result is not None:
            raise ValueError(
                "Scriptwriter resume requires "
                "evaluation_result=None."
            )

        _require_empty_ledger(
            state
        )

        return

    if resume_stage == "continuity":
        if script_result is None:
            raise ValueError(
                "Continuity resume requires an existing "
                "script_result."
            )

        if continuity_result is not None:
            raise ValueError(
                "Continuity resume requires "
                "continuity_result=None."
            )

        if evaluation_result is not None:
            raise ValueError(
                "Continuity resume requires "
                "evaluation_result=None."
            )

        _require_empty_ledger(
            state
        )

        return

    if resume_stage == "evaluation":
        if script_result is None:
            raise ValueError(
                "Evaluation resume requires an existing "
                "script_result."
            )

        if continuity_result is None:
            raise ValueError(
                "Evaluation resume requires an existing "
                "continuity_result."
            )

        if evaluation_result is not None:
            raise ValueError(
                "Evaluation resume requires "
                "evaluation_result=None."
            )

        return

    raise ValueError(
        "Unsupported downstream resume stage: "
        f"{resume_stage!r}"
    )


# ---------------------------------------------------------
# Lazy Default Stage Runners
# ---------------------------------------------------------


def _default_scriptwriter_runner() -> Callable[..., Any]:
    from workflows.writers_room import (
        run_scriptwriter_stage,
    )

    return run_scriptwriter_stage


def _default_continuity_runner() -> Callable[..., Any]:
    from workflows.writers_room import (
        run_continuity_stage,
    )

    return run_continuity_stage


def _default_evaluation_runner() -> Callable[..., Any]:
    from workflows.writers_room import (
        run_evaluation_stage,
    )

    return run_evaluation_stage


# ---------------------------------------------------------
# Stage Invocation
# ---------------------------------------------------------


def _run_scriptwriter(
    *,
    state: BaseModel,
    script_request: Any,
    runner: (
        Callable[..., Any]
        | None
    ),
) -> BaseModel:
    stage_runner = (
        runner
        if runner is not None
        else _default_scriptwriter_runner()
    )

    if not callable(
        stage_runner
    ):
        raise TypeError(
            "Scriptwriter stage runner must be callable."
        )

    try:
        next_state = stage_runner(
            state=state,
            request=script_request,
        )
    except TypeError as error:
        raise TypeError(
            "Scriptwriter runner must support "
            "runner(state=<state>, request=<request>)."
        ) from error

    next_state = _require_state_model(
        next_state
    )

    if getattr(
        next_state,
        "script_result",
    ) is None:
        raise ValueError(
            "Scriptwriter runner completed without "
            "producing script_result."
        )

    if getattr(
        next_state,
        "continuity_result",
    ) is not None:
        raise ValueError(
            "Scriptwriter runner unexpectedly produced "
            "continuity_result."
        )

    if getattr(
        next_state,
        "evaluation_result",
    ) is not None:
        raise ValueError(
            "Scriptwriter runner unexpectedly produced "
            "evaluation_result."
        )

    return next_state


def _run_continuity(
    *,
    state: BaseModel,
    scene_index: int,
    runner: (
        Callable[..., Any]
        | None
    ),
) -> BaseModel:
    stage_runner = (
        runner
        if runner is not None
        else _default_continuity_runner()
    )

    if not callable(
        stage_runner
    ):
        raise TypeError(
            "Continuity stage runner must be callable."
        )

    try:
        next_state = stage_runner(
            state=state,
            scene_index=scene_index,
        )
    except TypeError as error:
        raise TypeError(
            "Continuity runner must support "
            "runner(state=<state>, scene_index=<int>)."
        ) from error

    next_state = _require_state_model(
        next_state
    )

    if getattr(
        next_state,
        "script_result",
    ) is None:
        raise ValueError(
            "Continuity runner lost script_result."
        )

    if getattr(
        next_state,
        "continuity_result",
    ) is None:
        raise ValueError(
            "Continuity runner completed without "
            "producing continuity_result."
        )

    if getattr(
        next_state,
        "evaluation_result",
    ) is not None:
        raise ValueError(
            "Continuity runner unexpectedly produced "
            "evaluation_result."
        )

    return next_state


def _run_evaluation(
    *,
    state: BaseModel,
    runner: (
        Callable[..., Any]
        | None
    ),
) -> BaseModel:
    stage_runner = (
        runner
        if runner is not None
        else _default_evaluation_runner()
    )

    if not callable(
        stage_runner
    ):
        raise TypeError(
            "Evaluation stage runner must be callable."
        )

    try:
        next_state = stage_runner(
            state
        )
    except TypeError as error:
        raise TypeError(
            "Evaluation runner must support "
            "runner(state)."
        ) from error

    next_state = _require_state_model(
        next_state
    )

    if getattr(
        next_state,
        "script_result",
    ) is None:
        raise ValueError(
            "Evaluation runner lost script_result."
        )

    if getattr(
        next_state,
        "continuity_result",
    ) is None:
        raise ValueError(
            "Evaluation runner lost continuity_result."
        )

    if getattr(
        next_state,
        "evaluation_result",
    ) is None:
        raise ValueError(
            "Evaluation runner completed without "
            "producing evaluation_result."
        )

    return next_state


# ---------------------------------------------------------
# Public API
# ---------------------------------------------------------


def resume_after_evidence_resolution(
    *,
    orchestration_result: EvidenceResolutionOrchestrationResult,
    script_request: Any = None,
    scriptwriter_runner: (
        Callable[..., Any]
        | None
    ) = None,
    continuity_runner: (
        Callable[..., Any]
        | None
    ) = None,
    evaluation_runner: (
        Callable[..., Any]
        | None
    ) = None,
    scene_index: int = 0,
) -> EvidenceResolutionResumeResult:
    """
    Resume exactly the stale Writer's Room suffix after evidence
    resolution.

    No checkpoint persistence is performed here.

    For Scriptwriter resume, script_request is mandatory because
    WritersRoomState does not currently own that request.
    """

    _validate_orchestration_result(
        orchestration_result
    )

    if not isinstance(
        scene_index,
        int,
    ):
        raise TypeError(
            "scene_index must be an int."
        )

    if scene_index < 0:
        raise ValueError(
            "scene_index cannot be negative."
        )

    current_state = _clone_validated_state(
        _require_state_model(
            orchestration_result.final_state
        )
    )

    upstream_authority = (
        _capture_upstream_authority(
            current_state
        )
    )

    resume_stage = (
        orchestration_result
        .earliest_stale_stage
    )

    # -----------------------------------------------------
    # Nothing stale
    # -----------------------------------------------------

    if resume_stage is None:
        return EvidenceResolutionResumeResult(
            run_status="no_resume",
            started_at=None,
            completed_stages=[],
            final_state=current_state,
            reason=(
                "Evidence resolution did not materially "
                "change downstream authority. No "
                "Writer's Room stage needs to rerun."
            ),
        )

    _validate_resume_preconditions(
        state=current_state,
        resume_stage=resume_stage,
        script_request=script_request,
    )

    completed_stages: list[
        ResumeStage
    ] = []

    # -----------------------------------------------------
    # Scriptwriter
    # -----------------------------------------------------

    if resume_stage == "scriptwriter":
        current_state = _run_scriptwriter(
            state=current_state,
            script_request=script_request,
            runner=scriptwriter_runner,
        )

        _assert_upstream_authority_preserved(
            state=current_state,
            expected=upstream_authority,
        )

        completed_stages.append(
            "scriptwriter"
        )

    # -----------------------------------------------------
    # Continuity
    #
    # Runs when starting at Scriptwriter OR Continuity.
    # -----------------------------------------------------

    if resume_stage in {
        "scriptwriter",
        "continuity",
    }:
        current_state = _run_continuity(
            state=current_state,
            scene_index=scene_index,
            runner=continuity_runner,
        )

        _assert_upstream_authority_preserved(
            state=current_state,
            expected=upstream_authority,
        )

        completed_stages.append(
            "continuity"
        )

    # -----------------------------------------------------
    # Evaluation
    #
    # Runs for every supported stale suffix.
    # -----------------------------------------------------

    current_state = _run_evaluation(
        state=current_state,
        runner=evaluation_runner,
    )

    _assert_upstream_authority_preserved(
        state=current_state,
        expected=upstream_authority,
    )

    completed_stages.append(
        "evaluation"
    )

    return EvidenceResolutionResumeResult(
        run_status="resumed",
        started_at=resume_stage,
        completed_stages=(
            completed_stages
        ),
        final_state=current_state,
        reason=(
            "Writer's Room resumed exactly the stale "
            f"downstream suffix beginning at {resume_stage}. "
            "Upstream StoryPlan/Lore/Character authority "
            "was preserved."
        ),
    )
