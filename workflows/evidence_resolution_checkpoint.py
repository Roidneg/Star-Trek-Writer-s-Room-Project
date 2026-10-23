"""
Checkpoint persistence for a completed multi-target evidence-resolution run.

Version 0.1 persists EvidenceResolutionOrchestrationResult exactly once
after all supported upstream evidence targets have been processed.

This is the preferred persistence path for multi-blocker resolution.

Flow:

    original checkpoint bytes captured before orchestration
            ↓
    orchestrate all Lore / Character evidence targets
            ↓
    one final in-memory Writer's Room state
            ↓
    verify checkpoint unchanged
            ↓
    preserve exact backup
            ↓
    verify checkpoint unchanged again
            ↓
    atomic replace
            ↓
    reload + validate + hash

This module does NOT:
- call any agent;
- execute evidence refresh;
- decide material change;
- compute target routing;
- resume Scriptwriter / Continuity / Evaluation;
- mutate the caller's state in place.

Core rule:

    Resolve many upstream blockers.
    Persist once.
"""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from workflows.evidence_resolution_orchestrator import (
    EvidenceResolutionOrchestrationResult,
)


class EvidenceResolutionCheckpointResult(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    persistence_status: Literal["saved"] = "saved"

    checkpoint_path: str = Field(
        min_length=1,
    )

    backup_path: str = Field(
        min_length=1,
    )

    original_sha256: str = Field(
        min_length=64,
        max_length=64,
    )

    saved_sha256: str = Field(
        min_length=64,
        max_length=64,
    )

    orchestration_status: Literal[
        "no_action",
        "resolved",
        "not_applicable",
    ]

    evaluation_status: Literal[
        "approved",
        "needs_evidence",
        "rejected",
    ]

    processed_target_count: int = Field(
        ge=0,
    )

    material_change_detected: bool

    cumulative_invalidated_artifacts: list[str] = Field(
        default_factory=list
    )

    earliest_stale_stage: (
        Literal[
            "scriptwriter",
            "continuity",
            "evaluation",
        ]
        | None
    ) = None

    saved_state: Any

    reason: str = Field(
        min_length=1,
    )


def sha256_bytes(
    data: bytes,
) -> str:
    return hashlib.sha256(
        data
    ).hexdigest()


def _require_checkpoint_bytes(
    original_checkpoint_bytes: bytes,
) -> None:
    if not isinstance(
        original_checkpoint_bytes,
        bytes,
    ):
        raise TypeError(
            "original_checkpoint_bytes must be bytes."
        )

    if not original_checkpoint_bytes:
        raise ValueError(
            "original_checkpoint_bytes cannot be empty."
        )

    try:
        original_checkpoint_bytes.decode(
            "utf-8"
        )
    except UnicodeDecodeError as error:
        raise ValueError(
            "Original checkpoint bytes are not valid UTF-8."
        ) from error


def _require_orchestration_result(
    result: EvidenceResolutionOrchestrationResult,
) -> EvidenceResolutionOrchestrationResult:
    if not isinstance(
        result,
        EvidenceResolutionOrchestrationResult,
    ):
        raise TypeError(
            "Evidence-resolution checkpoint persistence "
            "requires an EvidenceResolutionOrchestrationResult."
        )

    return result


def _validate_orchestration_contract(
    result: EvidenceResolutionOrchestrationResult,
) -> None:
    if result.run_status == "no_action":
        if result.evaluation_status != "approved":
            raise ValueError(
                "no_action orchestration must come from "
                "evaluation_status='approved'."
            )

        if result.processed_targets:
            raise ValueError(
                "no_action orchestration must not contain "
                "processed targets."
            )

        if result.material_change_detected:
            raise ValueError(
                "no_action orchestration cannot report "
                "material change."
            )

        if result.cumulative_invalidated_artifacts:
            raise ValueError(
                "no_action orchestration cannot contain "
                "invalidated artifacts."
            )

        if result.earliest_stale_stage is not None:
            raise ValueError(
                "no_action orchestration cannot contain "
                "a stale resume stage."
            )

        return

    if result.run_status == "not_applicable":
        if result.evaluation_status != "rejected":
            raise ValueError(
                "not_applicable orchestration must come from "
                "evaluation_status='rejected'."
            )

        if result.processed_targets:
            raise ValueError(
                "not_applicable orchestration must not contain "
                "processed targets."
            )

        if result.material_change_detected:
            raise ValueError(
                "not_applicable orchestration cannot report "
                "material change."
            )

        if result.cumulative_invalidated_artifacts:
            raise ValueError(
                "not_applicable orchestration cannot contain "
                "invalidated artifacts."
            )

        if result.earliest_stale_stage is not None:
            raise ValueError(
                "not_applicable orchestration cannot contain "
                "a stale resume stage."
            )

        return

    if result.run_status != "resolved":
        raise ValueError(
            "Unsupported evidence-resolution orchestration "
            f"status: {result.run_status!r}"
        )

    if result.evaluation_status != "needs_evidence":
        raise ValueError(
            "resolved orchestration must come from "
            "evaluation_status='needs_evidence'."
        )

    if not result.processed_targets:
        raise ValueError(
            "resolved orchestration requires at least one "
            "processed target."
        )

    if result.material_change_detected:
        if not result.cumulative_invalidated_artifacts:
            raise ValueError(
                "Material orchestration must contain "
                "invalidated artifacts."
            )

        if result.earliest_stale_stage is None:
            raise ValueError(
                "Material orchestration must report the "
                "earliest stale stage."
            )
    else:
        if result.cumulative_invalidated_artifacts:
            raise ValueError(
                "Non-material orchestration cannot contain "
                "invalidated artifacts."
            )

        if result.earliest_stale_stage is not None:
            raise ValueError(
                "Non-material orchestration cannot report "
                "a stale stage."
            )


def _require_pydantic_final_state(
    result: EvidenceResolutionOrchestrationResult,
) -> BaseModel:
    state = result.final_state

    if not isinstance(
        state,
        BaseModel,
    ):
        raise TypeError(
            "EvidenceResolutionOrchestrationResult.final_state "
            "must be a Pydantic BaseModel."
        )

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
            "Final evidence-resolution state failed full "
            "Pydantic validation before persistence."
        ) from error


def _serialize_state(
    state: BaseModel,
) -> bytes:
    return (
        state.model_dump_json(
            indent=2
        )
        + "\n"
    ).encode(
        "utf-8"
    )


def _verify_checkpoint_unchanged(
    *,
    checkpoint_path: Path,
    original_checkpoint_bytes: bytes,
) -> None:
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "Writer's Room checkpoint disappeared before "
            "evidence-resolution persistence:\n"
            f"{checkpoint_path}"
        )

    current_bytes = checkpoint_path.read_bytes()

    if current_bytes != original_checkpoint_bytes:
        raise ValueError(
            "Writer's Room checkpoint changed after the "
            "evidence-resolution transaction began. "
            "Refusing to overwrite newer state."
        )


def _write_exact_backup(
    *,
    checkpoint_path: Path,
    original_checkpoint_bytes: bytes,
) -> Path:
    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with tempfile.NamedTemporaryFile(
        mode="wb",
        prefix="writers_room_before_evidence_resolution_",
        suffix=".json",
        dir=checkpoint_path.parent,
        delete=False,
    ) as backup_file:
        backup_file.write(
            original_checkpoint_bytes
        )

        backup_path = Path(
            backup_file.name
        )

    if backup_path.read_bytes() != original_checkpoint_bytes:
        backup_path.unlink(
            missing_ok=True
        )

        raise IOError(
            "Evidence-resolution checkpoint backup did not "
            "preserve the exact original bytes."
        )

    return backup_path


def _atomic_replace(
    *,
    checkpoint_path: Path,
    candidate_bytes: bytes,
) -> None:
    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix="evidence_resolution_",
            suffix=".tmp",
            dir=checkpoint_path.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(
                temporary_file.name
            )

            temporary_file.write(
                candidate_bytes
            )

            temporary_file.flush()

        if temporary_path.read_bytes() != candidate_bytes:
            raise IOError(
                "Temporary evidence-resolution checkpoint "
                "does not match candidate bytes."
            )

        temporary_path.replace(
            checkpoint_path
        )

    finally:
        if temporary_path is not None:
            temporary_path.unlink(
                missing_ok=True
            )


def _reload_saved_state(
    *,
    checkpoint_path: Path,
    state_type: type[BaseModel],
) -> BaseModel:
    try:
        return state_type.model_validate_json(
            checkpoint_path.read_text(
                encoding="utf-8"
            )
        )
    except Exception as error:
        raise ValueError(
            "Saved evidence-resolution checkpoint failed "
            "post-write state validation."
        ) from error


def persist_evidence_resolution(
    *,
    checkpoint_path: Path,
    original_checkpoint_bytes: bytes,
    orchestration_result: EvidenceResolutionOrchestrationResult,
) -> EvidenceResolutionCheckpointResult:
    """
    Persist a completed multi-target evidence-resolution transaction.

    The caller should capture original_checkpoint_bytes before any
    refresh execution begins.

    The checkpoint is written exactly once after orchestration finishes.
    """

    if not isinstance(
        checkpoint_path,
        Path,
    ):
        raise TypeError(
            "checkpoint_path must be pathlib.Path."
        )

    _require_checkpoint_bytes(
        original_checkpoint_bytes
    )

    result = _require_orchestration_result(
        orchestration_result
    )

    _validate_orchestration_contract(
        result
    )

    validated_state = _require_pydantic_final_state(
        result
    )

    _verify_checkpoint_unchanged(
        checkpoint_path=checkpoint_path,
        original_checkpoint_bytes=(
            original_checkpoint_bytes
        ),
    )

    candidate_bytes = _serialize_state(
        validated_state
    )

    backup_path = _write_exact_backup(
        checkpoint_path=checkpoint_path,
        original_checkpoint_bytes=(
            original_checkpoint_bytes
        ),
    )

    try:
        # Close the race window between backup creation and replace.
        _verify_checkpoint_unchanged(
            checkpoint_path=checkpoint_path,
            original_checkpoint_bytes=(
                original_checkpoint_bytes
            ),
        )

        _atomic_replace(
            checkpoint_path=checkpoint_path,
            candidate_bytes=candidate_bytes,
        )

    except Exception:
        # Backup remains intentionally for provenance/diagnosis.
        raise

    saved_bytes = checkpoint_path.read_bytes()

    if saved_bytes != candidate_bytes:
        raise IOError(
            "Persisted evidence-resolution checkpoint "
            "does not match validated candidate bytes."
        )

    saved_state = _reload_saved_state(
        checkpoint_path=checkpoint_path,
        state_type=type(
            validated_state
        ),
    )

    if (
        saved_state.model_dump(
            mode="json"
        )
        != validated_state.model_dump(
            mode="json"
        )
    ):
        raise AssertionError(
            "Post-write checkpoint state does not match "
            "the final evidence-resolution state."
        )

    return EvidenceResolutionCheckpointResult(
        checkpoint_path=str(
            checkpoint_path
        ),
        backup_path=str(
            backup_path
        ),
        original_sha256=sha256_bytes(
            original_checkpoint_bytes
        ),
        saved_sha256=sha256_bytes(
            saved_bytes
        ),
        orchestration_status=(
            result.run_status
        ),
        evaluation_status=(
            result.evaluation_status
        ),
        processed_target_count=len(
            result.processed_targets
        ),
        material_change_detected=(
            result.material_change_detected
        ),
        cumulative_invalidated_artifacts=list(
            result.cumulative_invalidated_artifacts
        ),
        earliest_stale_stage=(
            result.earliest_stale_stage
        ),
        saved_state=saved_state,
        reason=(
            "Completed evidence-resolution orchestration "
            "was persisted exactly once after two exact-byte "
            "concurrency checks. The pre-resolution checkpoint "
            "was preserved as a unique backup."
        ),
    )
