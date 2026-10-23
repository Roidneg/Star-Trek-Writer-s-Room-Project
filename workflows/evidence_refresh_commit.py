"""
In-memory commit gate for targeted evidence refreshes.

Version 0.1 supports committing LoreEvidenceRefreshResult into an
existing Writer's Room state.

This module is intentionally separate from:
- evidence refresh execution;
- material-change detection;
- invalidation planning;
- checkpoint persistence.

Flow:

    LoreEvidenceRefreshResult
            ↓
    verify current state still matches refresh base
            ↓
    replace exactly one Lore artifact
            ↓
    material change?
      │
      ├── no
      │    preserve all downstream artifacts
      │
      └── yes
           clear only planned dependents
           reset Continuity-owned ledger
            ↓
    return new in-memory state

It does NOT:
- call an LLM;
- write a checkpoint;
- rerun any agent;
- mutate the supplied state object in place;
- invalidate StoryPlan;
- invalidate Character artifacts for a Lore-only refresh.

Core rules:

    Refresh result is a proposal.
    State commit must verify provenance again.
    Material authority controls downstream invalidation.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from workflows.artifact_material_change import (
    build_lore_material_snapshot,
)

from workflows.evidence_refresh import (
    LoreEvidenceRefreshResult,
)


EXPECTED_MATERIAL_LORE_INVALIDATIONS = [
    "script_result",
    "continuity_result",
    "story_state_ledger",
    "evaluation_result",
]


class LoreRefreshCommitResult(BaseModel):
    """
    Result of applying one validated Lore refresh to an in-memory
    Writer's Room state.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    research_id: str = Field(
        min_length=1,
    )

    commit_status: Literal[
        "committed_without_invalidation",
        "committed_with_invalidation",
    ]

    invalidated_artifacts: list[str] = Field(
        default_factory=list
    )

    committed_state: Any

    reason: str = Field(
        min_length=1,
    )


def _require_pydantic_state(
    state: Any,
) -> BaseModel:
    if not isinstance(
        state,
        BaseModel,
    ):
        raise TypeError(
            "Lore refresh commit requires Writer's Room "
            "state to be a Pydantic BaseModel."
        )

    required_fields = {
        "lore_results",
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
            "commit fields: "
            + ", ".join(
                sorted(
                    missing_fields
                )
            )
        )

    return state


def _find_lore_result_index(
    state: BaseModel,
    research_id: str,
) -> int:
    lore_results = getattr(
        state,
        "lore_results",
    )

    if not isinstance(
        lore_results,
        list,
    ):
        raise TypeError(
            "Writer's Room lore_results must be a list."
        )

    matching_indexes = [
        index
        for index, result
        in enumerate(
            lore_results
        )
        if (
            getattr(
                result,
                "research_id",
                None,
            )
            == research_id
        )
    ]

    if not matching_indexes:
        raise ValueError(
            "Writer's Room does not contain Lore artifact "
            f"{research_id!r}."
        )

    if len(matching_indexes) > 1:
        raise ValueError(
            "Writer's Room contains duplicate Lore artifact "
            f"ID {research_id!r}."
        )

    return matching_indexes[0]


def _snapshot_json(
    lore_result: Any,
) -> dict[str, Any]:
    return (
        build_lore_material_snapshot(
            lore_result
        )
        .model_dump(
            mode="json"
        )
    )


def _verify_refresh_base_matches_state(
    *,
    current_lore_result: Any,
    refresh_result: LoreEvidenceRefreshResult,
) -> None:
    """
    Prevent applying a refresh result to a state whose authoritative
    source changed after the refresh began.
    """

    current_snapshot = (
        _snapshot_json(
            current_lore_result
        )
    )

    expected_old_snapshot = (
        refresh_result
        .material_change
        .old_snapshot
    )

    if (
        current_snapshot
        != expected_old_snapshot
    ):
        raise ValueError(
            "Writer's Room Lore artifact changed after "
            "the refresh was produced. Refusing stale "
            "refresh commit."
        )


def _verify_candidate_matches_refresh_report(
    refresh_result: LoreEvidenceRefreshResult,
) -> None:
    candidate_snapshot = (
        _snapshot_json(
            refresh_result
            .candidate_artifact
        )
    )

    expected_new_snapshot = (
        refresh_result
        .material_change
        .new_snapshot
    )

    if (
        candidate_snapshot
        != expected_new_snapshot
    ):
        raise ValueError(
            "Lore refresh candidate no longer matches "
            "its material-change report."
        )

    if (
        candidate_snapshot[
            "research_id"
        ]
        != refresh_result.research_id
    ):
        raise ValueError(
            "Lore refresh candidate research_id does not "
            "match refresh result identity."
        )

    if (
        candidate_snapshot[
            "research_question"
        ]
        != refresh_result.research_question
    ):
        raise ValueError(
            "Lore refresh candidate research question "
            "does not match refresh result provenance."
        )


def _validate_refresh_status_contract(
    refresh_result: LoreEvidenceRefreshResult,
) -> None:
    material_change = (
        refresh_result
        .material_change
        .material_change
    )

    if (
        refresh_result.refresh_status
        == "unchanged"
    ):
        if material_change:
            raise ValueError(
                "Lore refresh status says unchanged but "
                "material-change report says changed."
            )

        if (
            refresh_result
            .invalidation_plan
            is not None
        ):
            raise ValueError(
                "Unchanged Lore refresh must not carry "
                "an invalidation plan."
            )

        return

    if (
        refresh_result.refresh_status
        == "material_change"
    ):
        if not material_change:
            raise ValueError(
                "Lore refresh status says material_change "
                "but material-change report says unchanged."
            )

        invalidation_plan = (
            refresh_result
            .invalidation_plan
        )

        if invalidation_plan is None:
            raise ValueError(
                "Material Lore refresh requires an "
                "invalidation plan."
            )

        if (
            invalidation_plan.trigger
            != "on_material_change"
        ):
            raise ValueError(
                "Material Lore invalidation plan must use "
                "trigger='on_material_change'."
            )

        if (
            invalidation_plan
            .invalidated_artifacts
            != EXPECTED_MATERIAL_LORE_INVALIDATIONS
        ):
            raise ValueError(
                "Material Lore refresh carries an "
                "unexpected invalidation set."
            )

        if (
            invalidation_plan
            .requires_ledger_rebuild
            is not True
        ):
            raise ValueError(
                "Material Lore refresh must require a "
                "Continuity ledger rebuild."
            )

        return

    raise ValueError(
        "Unsupported Lore refresh status: "
        f"{refresh_result.refresh_status!r}"
    )


def _replace_one_lore_artifact(
    *,
    state: BaseModel,
    lore_index: int,
    candidate_artifact: Any,
) -> list[Any]:
    """
    Return a new Lore list. Never mutate the state-owned list in place.
    """

    current_lore_results = list(
        getattr(
            state,
            "lore_results",
        )
    )

    current_lore_results[
        lore_index
    ] = candidate_artifact

    return current_lore_results


def _build_empty_ledger(
    current_ledger: Any,
) -> Any:
    """
    Rebuild the Continuity-owned ledger using its current concrete
    type and default constructor.

    This keeps the commit layer independent of provider-backed
    Writer's Room imports while still producing the real ledger type
    in production.
    """

    ledger_type = type(
        current_ledger
    )

    try:
        empty_ledger = ledger_type()
    except Exception as error:
        raise TypeError(
            "Current story_state_ledger type cannot be "
            "reconstructed from its default constructor."
        ) from error

    # If the ledger exposes entries, verify the new default is empty.
    if hasattr(
        empty_ledger,
        "entries",
    ):
        entries = getattr(
            empty_ledger,
            "entries",
        )

        if entries:
            raise ValueError(
                "Default reconstructed story_state_ledger "
                "is not empty."
            )

    return empty_ledger


def _validate_committed_state(
    state: BaseModel,
) -> BaseModel:
    """
    Revalidate the complete state through its own Pydantic model class.

    model_copy(update=...) intentionally does not validate updates.
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
            "Committed Writer's Room state failed "
            "full Pydantic validation."
        ) from error


def commit_lore_refresh_to_state(
    *,
    state: Any,
    refresh_result: LoreEvidenceRefreshResult,
) -> LoreRefreshCommitResult:
    """
    Apply one already-produced Lore refresh result to an in-memory
    Writer's Room state.

    The supplied state is never mutated in place.

    Unchanged authority:
        Replace the Lore artifact with the refreshed candidate and
        preserve Scriptwriter, Continuity, ledger, and Evaluation.

    Materially changed authority:
        Replace the Lore artifact and invalidate exactly:
        - script_result
        - continuity_result
        - story_state_ledger
        - evaluation_result

        StoryPlan and Character artifacts remain preserved.

    This function does not persist the returned state.
    """

    current_state = _require_pydantic_state(
        state
    )

    if (
        not isinstance(
            refresh_result,
            LoreEvidenceRefreshResult,
        )
    ):
        raise TypeError(
            "Lore refresh commit requires a "
            "LoreEvidenceRefreshResult."
        )

    _validate_refresh_status_contract(
        refresh_result
    )

    _verify_candidate_matches_refresh_report(
        refresh_result
    )

    lore_index = _find_lore_result_index(
        current_state,
        refresh_result.research_id,
    )

    current_lore_result = (
        getattr(
            current_state,
            "lore_results",
        )[
            lore_index
        ]
    )

    _verify_refresh_base_matches_state(
        current_lore_result=(
            current_lore_result
        ),
        refresh_result=(
            refresh_result
        ),
    )

    replacement_lore_results = (
        _replace_one_lore_artifact(
            state=current_state,
            lore_index=lore_index,
            candidate_artifact=(
                refresh_result
                .candidate_artifact
            ),
        )
    )

    # -----------------------------------------------------
    # Unchanged Authority
    # -----------------------------------------------------

    if (
        refresh_result.refresh_status
        == "unchanged"
    ):
        candidate_state = (
            current_state.model_copy(
                update={
                    "lore_results": (
                        replacement_lore_results
                    ),
                }
            )
        )

        committed_state = (
            _validate_committed_state(
                candidate_state
            )
        )

        # Downstream objects must remain semantically identical.
        for field_name in (
            "script_result",
            "story_state_ledger",
            "continuity_result",
            "evaluation_result",
        ):
            old_value = getattr(
                current_state,
                field_name,
            )

            new_value = getattr(
                committed_state,
                field_name,
            )

            old_dump = (
                old_value.model_dump(
                    mode="json"
                )
                if isinstance(
                    old_value,
                    BaseModel,
                )
                else old_value
            )

            new_dump = (
                new_value.model_dump(
                    mode="json"
                )
                if isinstance(
                    new_value,
                    BaseModel,
                )
                else new_value
            )

            if old_dump != new_dump:
                raise AssertionError(
                    "Unchanged Lore refresh unexpectedly "
                    f"changed downstream field "
                    f"{field_name!r}."
                )

        return LoreRefreshCommitResult(
            research_id=(
                refresh_result.research_id
            ),
            commit_status=(
                "committed_without_invalidation"
            ),
            invalidated_artifacts=[],
            committed_state=(
                committed_state
            ),
            reason=(
                "Refreshed Lore provenance was committed, "
                "but authoritative downstream content was "
                "unchanged. Existing Scriptwriter, "
                "Continuity, ledger, and Evaluation "
                "artifacts were preserved."
            ),
        )

    # -----------------------------------------------------
    # Material Authority Change
    # -----------------------------------------------------

    current_ledger = getattr(
        current_state,
        "story_state_ledger",
    )

    empty_ledger = (
        _build_empty_ledger(
            current_ledger
        )
    )

    candidate_state = (
        current_state.model_copy(
            update={
                "lore_results": (
                    replacement_lore_results
                ),
                "script_result": None,
                "continuity_result": None,
                "story_state_ledger": (
                    empty_ledger
                ),
                "evaluation_result": None,
            }
        )
    )

    committed_state = (
        _validate_committed_state(
            candidate_state
        )
    )

    if (
        getattr(
            committed_state,
            "script_result",
        )
        is not None
    ):
        raise AssertionError(
            "Material Lore refresh failed to invalidate "
            "script_result."
        )

    if (
        getattr(
            committed_state,
            "continuity_result",
        )
        is not None
    ):
        raise AssertionError(
            "Material Lore refresh failed to invalidate "
            "continuity_result."
        )

    if (
        getattr(
            committed_state,
            "evaluation_result",
        )
        is not None
    ):
        raise AssertionError(
            "Material Lore refresh failed to invalidate "
            "evaluation_result."
        )

    committed_ledger = getattr(
        committed_state,
        "story_state_ledger",
    )

    if (
        hasattr(
            committed_ledger,
            "entries",
        )
        and getattr(
            committed_ledger,
            "entries",
        )
    ):
        raise AssertionError(
            "Material Lore refresh failed to reset "
            "story_state_ledger."
        )

    # The current architecture treats Character as a sibling
    # authority to Lore. Lore refresh must not erase it.
    old_character_results = getattr(
        current_state,
        "character_results",
        None,
    )

    new_character_results = getattr(
        committed_state,
        "character_results",
        None,
    )

    if (
        old_character_results
        is not None
        and new_character_results
        is not None
    ):
        old_character_dump = [
            item.model_dump(
                mode="json"
            )
            if isinstance(
                item,
                BaseModel,
            )
            else item
            for item in old_character_results
        ]

        new_character_dump = [
            item.model_dump(
                mode="json"
            )
            if isinstance(
                item,
                BaseModel,
            )
            else item
            for item in new_character_results
        ]

        if (
            old_character_dump
            != new_character_dump
        ):
            raise AssertionError(
                "Lore refresh unexpectedly changed "
                "Character artifacts."
            )

    return LoreRefreshCommitResult(
        research_id=(
            refresh_result.research_id
        ),
        commit_status=(
            "committed_with_invalidation"
        ),
        invalidated_artifacts=list(
            EXPECTED_MATERIAL_LORE_INVALIDATIONS
        ),
        committed_state=(
            committed_state
        ),
        reason=(
            "Refreshed Lore authority materially changed. "
            "The candidate Lore artifact was committed "
            "in memory and exactly the proven downstream "
            "Scriptwriter, Continuity, story-state ledger, "
            "and Evaluation artifacts were invalidated. "
            "StoryPlan and Character artifacts remain "
            "preserved."
        ),
    )
