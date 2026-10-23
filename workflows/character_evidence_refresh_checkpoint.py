"""
Checkpoint persistence gate for Character evidence-resolution commits.

Version 0.1 persists an already-validated
CharacterRefreshCommitResult.

This module is intentionally downstream of:
- character_evidence_refresh.py
- character_evidence_refresh_commit.py

It does NOT:
- call an LLM;
- execute Character research;
- decide material change;
- compute invalidation;
- mutate the caller's in-memory state;
- rerun any Writer's Room stage.

Flow:

    checkpoint bytes captured before refresh
            ↓
    CharacterRefreshCommitResult
            ↓
    verify checkpoint has not changed
            ↓
    verify committed state is valid
            ↓
    write unique backup of exact original bytes
            ↓
    verify checkpoint still has not changed
            ↓
    write validated candidate to unique temp file
            ↓
    atomic replace
            ↓
    reload and verify exact saved state

Core rules:

    Never overwrite a checkpoint that changed during refresh.
    Never persist an unvalidated state.
    Always preserve the exact pre-commit checkpoint.
"""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from workflows.character_evidence_refresh_commit import (
    CharacterRefreshCommitResult,
)


EXPECTED_MATERIAL_CHARACTER_INVALIDATIONS = [
    "script_result",
    "continuity_result",
    "story_state_ledger",
    "evaluation_result",
]


class CharacterEvidenceRefreshCheckpointResult(
    BaseModel
):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    persistence_status: Literal[
        "saved",
    ] = "saved"

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

    character_id: str = Field(
        min_length=1,
    )

    character_name: str = Field(
        min_length=1,
    )

    commit_status: Literal[
        "committed_without_invalidation",
        "committed_with_invalidation",
    ]

    invalidated_artifacts: list[str] = Field(
        default_factory=list
    )

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
            "Original checkpoint bytes are not "
            "valid UTF-8."
        ) from error


def _require_commit_result(
    commit_result: CharacterRefreshCommitResult,
) -> CharacterRefreshCommitResult:
    if not isinstance(
        commit_result,
        CharacterRefreshCommitResult,
    ):
        raise TypeError(
            "Character evidence-refresh checkpoint "
            "persistence requires a "
            "CharacterRefreshCommitResult."
        )

    return commit_result


def _require_pydantic_committed_state(
    commit_result: CharacterRefreshCommitResult,
) -> BaseModel:
    state = commit_result.committed_state

    if not isinstance(
        state,
        BaseModel,
    ):
        raise TypeError(
            "CharacterRefreshCommitResult."
            "committed_state must be a "
            "Pydantic BaseModel."
        )

    state_type = type(
        state
    )

    try:
        validated_state = (
            state_type.model_validate(
                state.model_dump(
                    mode="python"
                )
            )
        )
    except Exception as error:
        raise ValueError(
            "Committed Character evidence-refresh "
            "state failed full Pydantic validation "
            "before persistence."
        ) from error

    return validated_state


def _validate_commit_contract(
    commit_result: CharacterRefreshCommitResult,
) -> None:
    if (
        commit_result.commit_status
        == "committed_without_invalidation"
    ):
        if commit_result.invalidated_artifacts:
            raise ValueError(
                "Character commit status says no "
                "invalidation, but "
                "invalidated_artifacts is not empty."
            )

        return

    if (
        commit_result.commit_status
        == "committed_with_invalidation"
    ):
        if (
            commit_result.invalidated_artifacts
            != EXPECTED_MATERIAL_CHARACTER_INVALIDATIONS
        ):
            raise ValueError(
                "Material Character commit contains "
                "an unexpected invalidation set."
            )

        return

    raise ValueError(
        "Unsupported Character evidence-refresh "
        "commit status: "
        f"{commit_result.commit_status!r}"
    )


def _serialize_state(
    state: BaseModel,
) -> bytes:
    """
    Produce stable UTF-8 JSON checkpoint bytes.

    A terminal newline is included for readable
    runtime inspection and exact hash verification.
    """

    serialized = (
        state.model_dump_json(
            indent=2
        )
        + "\n"
    )

    return serialized.encode(
        "utf-8"
    )


def _validate_current_checkpoint_unchanged(
    *,
    checkpoint_path: Path,
    original_checkpoint_bytes: bytes,
) -> None:
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "Writer's Room checkpoint disappeared "
            "before Character evidence-refresh "
            "persistence:\n"
            f"{checkpoint_path}"
        )

    current_bytes = (
        checkpoint_path.read_bytes()
    )

    if (
        current_bytes
        != original_checkpoint_bytes
    ):
        raise ValueError(
            "Writer's Room checkpoint changed after "
            "the Character evidence refresh began. "
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
        prefix=(
            "writers_room_before_"
            "character_evidence_refresh_"
        ),
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

    if (
        backup_path.read_bytes()
        != original_checkpoint_bytes
    ):
        backup_path.unlink(
            missing_ok=True
        )

        raise IOError(
            "Character evidence-refresh checkpoint "
            "backup did not preserve the exact "
            "original bytes."
        )

    return backup_path


def _atomic_replace_checkpoint(
    *,
    checkpoint_path: Path,
    candidate_bytes: bytes,
) -> None:
    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=(
                "character_evidence_refresh_"
            ),
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

        if (
            temporary_path.read_bytes()
            != candidate_bytes
        ):
            raise IOError(
                "Temporary Character evidence-refresh "
                "checkpoint does not match "
                "candidate bytes."
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
        saved_text = (
            checkpoint_path.read_text(
                encoding="utf-8"
            )
        )

        saved_state = (
            state_type.model_validate_json(
                saved_text
            )
        )

    except Exception as error:
        raise ValueError(
            "Saved Character evidence-refresh "
            "checkpoint failed post-write "
            "state validation."
        ) from error

    return saved_state


def persist_character_refresh_commit(
    *,
    checkpoint_path: Path,
    original_checkpoint_bytes: bytes,
    commit_result: CharacterRefreshCommitResult,
) -> CharacterEvidenceRefreshCheckpointResult:
    """
    Persist one already-validated Character refresh commit.

    Parameters
    ----------
    checkpoint_path:
        Existing Writer's Room checkpoint to replace.

    original_checkpoint_bytes:
        Exact checkpoint bytes captured BEFORE Character
        refresh execution began. They provide the concurrency
        guard and become the backup artifact.

    commit_result:
        In-memory Character refresh commit produced by
        commit_character_refresh_to_state().

    Returns
    -------
    CharacterEvidenceRefreshCheckpointResult
        Provenance describing the saved checkpoint and
        exact pre-refresh backup.

    Important
    ---------
    This function assumes refresh execution and in-memory
    commit have already completed successfully.
    It performs no provider work.
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

    commit_result = (
        _require_commit_result(
            commit_result
        )
    )

    _validate_commit_contract(
        commit_result
    )

    validated_state = (
        _require_pydantic_committed_state(
            commit_result
        )
    )

    _validate_current_checkpoint_unchanged(
        checkpoint_path=(
            checkpoint_path
        ),
        original_checkpoint_bytes=(
            original_checkpoint_bytes
        ),
    )

    candidate_bytes = (
        _serialize_state(
            validated_state
        )
    )

    backup_path = (
        _write_exact_backup(
            checkpoint_path=(
                checkpoint_path
            ),
            original_checkpoint_bytes=(
                original_checkpoint_bytes
            ),
        )
    )

    # Close the race window between backup creation and
    # replacement. If the checkpoint changed after the
    # first check, preserve the backup and refuse overwrite.
    try:
        _validate_current_checkpoint_unchanged(
            checkpoint_path=(
                checkpoint_path
            ),
            original_checkpoint_bytes=(
                original_checkpoint_bytes
            ),
        )

        _atomic_replace_checkpoint(
            checkpoint_path=(
                checkpoint_path
            ),
            candidate_bytes=(
                candidate_bytes
            ),
        )

    except Exception:
        # Keep the backup for transaction provenance
        # and diagnosis.
        raise

    saved_bytes = (
        checkpoint_path.read_bytes()
    )

    if saved_bytes != candidate_bytes:
        raise IOError(
            "Persisted Character evidence-refresh "
            "checkpoint does not match the validated "
            "candidate bytes."
        )

    saved_state = (
        _reload_saved_state(
            checkpoint_path=(
                checkpoint_path
            ),
            state_type=type(
                validated_state
            ),
        )
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
            "Post-write checkpoint state does not "
            "match the committed Character "
            "evidence-refresh state."
        )

    original_sha256 = (
        sha256_bytes(
            original_checkpoint_bytes
        )
    )

    saved_sha256 = (
        sha256_bytes(
            saved_bytes
        )
    )

    return (
        CharacterEvidenceRefreshCheckpointResult(
            checkpoint_path=str(
                checkpoint_path
            ),
            backup_path=str(
                backup_path
            ),
            original_sha256=(
                original_sha256
            ),
            saved_sha256=(
                saved_sha256
            ),
            character_id=(
                commit_result.character_id
            ),
            character_name=(
                commit_result.character_name
            ),
            commit_status=(
                commit_result.commit_status
            ),
            invalidated_artifacts=list(
                commit_result
                .invalidated_artifacts
            ),
            saved_state=(
                saved_state
            ),
            reason=(
                "Validated Character evidence-refresh "
                "state was persisted atomically after "
                "two exact-byte concurrency checks. "
                "The pre-refresh checkpoint was "
                "preserved as a unique backup."
            ),
        )
    )
