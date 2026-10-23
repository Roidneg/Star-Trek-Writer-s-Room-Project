"""
In-memory commit gate for targeted Character evidence refreshes.

Version 0.1 commits CharacterEvidenceRefreshResult into an existing
Writer's Room state.

This module is intentionally separate from:
- Character refresh execution;
- material-change detection;
- invalidation planning;
- checkpoint persistence.

Flow:

    CharacterEvidenceRefreshResult
            ↓
    verify current Character task still matches refresh base
            ↓
    replace exactly one Character artifact
            ↓
    material change?
      │
      ├── no
      │    preserve all downstream artifacts
      │
      └── yes
           clear only proven dependents
           reset Continuity-owned ledger
            ↓
    return new in-memory state

It does NOT:
- call an LLM;
- write a checkpoint;
- rerun any agent;
- mutate the supplied state object in place;
- invalidate StoryPlan;
- invalidate Lore artifacts for a Character-only refresh.

Core rules:

    Refresh result is a proposal.
    State commit must verify provenance again.
    Grounded Character authority controls downstream invalidation.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from workflows.artifact_material_change import (
    build_character_material_snapshot,
)

from workflows.character_evidence_refresh import (
    CharacterEvidenceRefreshResult,
)


EXPECTED_MATERIAL_CHARACTER_INVALIDATIONS = [
    "script_result",
    "continuity_result",
    "story_state_ledger",
    "evaluation_result",
]


class CharacterRefreshCommitResult(BaseModel):
    """
    Result of applying one validated Character refresh to an in-memory
    Writer's Room state.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
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
            "Character refresh commit requires Writer's Room "
            "state to be a Pydantic BaseModel."
        )

    required_fields = {
        "character_results",
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
            "Character commit fields: "
            + ", ".join(
                sorted(
                    missing_fields
                )
            )
        )

    return state


def _find_character_result_index(
    state: BaseModel,
    character_id: str,
) -> int:
    character_results = getattr(
        state,
        "character_results",
    )

    if not isinstance(
        character_results,
        list,
    ):
        raise TypeError(
            "Writer's Room character_results must be a list."
        )

    matching_indexes = [
        index
        for index, result
        in enumerate(
            character_results
        )
        if (
            getattr(
                result,
                "character_id",
                None,
            )
            == character_id
        )
    ]

    if not matching_indexes:
        raise ValueError(
            "Writer's Room does not contain Character artifact "
            f"{character_id!r}."
        )

    if len(matching_indexes) > 1:
        raise ValueError(
            "Writer's Room contains duplicate Character artifact "
            f"ID {character_id!r}."
        )

    return matching_indexes[0]


def _snapshot_json(
    character_result: Any,
) -> dict[str, Any]:
    return (
        build_character_material_snapshot(
            character_result
        )
        .model_dump(
            mode="json"
        )
    )


def _read_task_inputs(
    character_result: Any,
) -> tuple[str, str]:
    if not hasattr(
        character_result,
        "research_plan",
    ):
        raise TypeError(
            "Character artifact is missing research_plan."
        )

    research_plan = getattr(
        character_result,
        "research_plan",
    )

    if not hasattr(
        research_plan,
        "story_context",
    ):
        raise TypeError(
            "Character research_plan is missing story_context."
        )

    if not hasattr(
        research_plan,
        "story_era",
    ):
        raise TypeError(
            "Character research_plan is missing story_era."
        )

    story_context = getattr(
        research_plan,
        "story_context",
    )

    story_era = getattr(
        research_plan,
        "story_era",
    )

    if (
        not isinstance(
            story_context,
            str,
        )
        or not story_context.strip()
    ):
        raise ValueError(
            "Character research_plan.story_context "
            "must be a non-empty string."
        )

    if (
        not isinstance(
            story_era,
            str,
        )
        or not story_era.strip()
    ):
        raise ValueError(
            "Character research_plan.story_era "
            "must be a non-empty string."
        )

    return (
        story_context,
        story_era,
    )


def _verify_refresh_base_matches_state(
    *,
    current_character_result: Any,
    refresh_result: CharacterEvidenceRefreshResult,
) -> None:
    """
    Prevent applying a refresh result to a Character task whose
    authoritative guidance or stable story inputs changed after the
    refresh was produced.
    """

    current_snapshot = (
        _snapshot_json(
            current_character_result
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
            "Writer's Room Character artifact changed after "
            "the refresh was produced. Refusing stale "
            "refresh commit."
        )

    (
        current_story_context,
        current_story_era,
    ) = _read_task_inputs(
        current_character_result
    )

    if (
        current_story_context
        != refresh_result.story_context
    ):
        raise ValueError(
            "Writer's Room Character story_context changed "
            "after the refresh was produced. Refusing stale "
            "refresh commit."
        )

    if (
        current_story_era
        != refresh_result.story_era
    ):
        raise ValueError(
            "Writer's Room Character story_era changed "
            "after the refresh was produced. Refusing stale "
            "refresh commit."
        )


def _verify_candidate_matches_refresh_report(
    refresh_result: CharacterEvidenceRefreshResult,
) -> None:
    candidate = (
        refresh_result
        .candidate_artifact
    )

    candidate_snapshot = (
        _snapshot_json(
            candidate
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
            "Character refresh candidate no longer matches "
            "its material-change report."
        )

    if (
        candidate_snapshot[
            "character_id"
        ]
        != refresh_result.character_id
    ):
        raise ValueError(
            "Character refresh candidate character_id does "
            "not match refresh result identity."
        )

    if (
        candidate_snapshot[
            "character_name"
        ]
        != refresh_result.character_name
    ):
        raise ValueError(
            "Character refresh candidate character_name does "
            "not match refresh result identity."
        )

    (
        candidate_story_context,
        candidate_story_era,
    ) = _read_task_inputs(
        candidate
    )

    if (
        candidate_story_context
        != refresh_result.story_context
    ):
        raise ValueError(
            "Character refresh candidate story_context does "
            "not match refresh result provenance."
        )

    if (
        candidate_story_era
        != refresh_result.story_era
    ):
        raise ValueError(
            "Character refresh candidate story_era does "
            "not match refresh result provenance."
        )


def _validate_refresh_status_contract(
    refresh_result: CharacterEvidenceRefreshResult,
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
                "Character refresh status says unchanged but "
                "material-change report says changed."
            )

        if (
            refresh_result
            .invalidation_plan
            is not None
        ):
            raise ValueError(
                "Unchanged Character refresh must not carry "
                "an invalidation plan."
            )

        return

    if (
        refresh_result.refresh_status
        == "material_change"
    ):
        if not material_change:
            raise ValueError(
                "Character refresh status says material_change "
                "but material-change report says unchanged."
            )

        invalidation_plan = (
            refresh_result
            .invalidation_plan
        )

        if invalidation_plan is None:
            raise ValueError(
                "Material Character refresh requires an "
                "invalidation plan."
            )

        if (
            invalidation_plan.trigger
            != "on_material_change"
        ):
            raise ValueError(
                "Material Character invalidation plan must use "
                "trigger='on_material_change'."
            )

        if (
            invalidation_plan
            .invalidated_artifacts
            != EXPECTED_MATERIAL_CHARACTER_INVALIDATIONS
        ):
            raise ValueError(
                "Material Character refresh carries an "
                "unexpected invalidation set."
            )

        if (
            invalidation_plan
            .requires_ledger_rebuild
            is not True
        ):
            raise ValueError(
                "Material Character refresh must require a "
                "Continuity ledger rebuild."
            )

        return

    raise ValueError(
        "Unsupported Character refresh status: "
        f"{refresh_result.refresh_status!r}"
    )


def _replace_one_character_artifact(
    *,
    state: BaseModel,
    character_index: int,
    candidate_artifact: Any,
) -> list[Any]:
    """
    Return a new Character list without mutating state-owned storage.
    """

    current_character_results = list(
        getattr(
            state,
            "character_results",
        )
    )

    current_character_results[
        character_index
    ] = candidate_artifact

    return current_character_results


def _build_empty_ledger(
    current_ledger: Any,
) -> Any:
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


def commit_character_refresh_to_state(
    *,
    state: Any,
    refresh_result: CharacterEvidenceRefreshResult,
) -> CharacterRefreshCommitResult:
    """
    Apply one already-produced Character refresh result to an in-memory
    Writer's Room state.

    The supplied state is never mutated in place.

    Unchanged grounded authority:
        Replace the Character artifact with the refreshed candidate and
        preserve Scriptwriter, Continuity, ledger, and Evaluation.

    Materially changed grounded authority:
        Replace the Character artifact and invalidate exactly:
        - script_result
        - continuity_result
        - story_state_ledger
        - evaluation_result

        StoryPlan, Lore artifacts, and sibling Character artifacts remain
        preserved.

    This function does not persist the returned state.
    """

    current_state = _require_pydantic_state(
        state
    )

    if (
        not isinstance(
            refresh_result,
            CharacterEvidenceRefreshResult,
        )
    ):
        raise TypeError(
            "Character refresh commit requires a "
            "CharacterEvidenceRefreshResult."
        )

    _validate_refresh_status_contract(
        refresh_result
    )

    _verify_candidate_matches_refresh_report(
        refresh_result
    )

    character_index = (
        _find_character_result_index(
            current_state,
            refresh_result.character_id,
        )
    )

    current_character_result = (
        getattr(
            current_state,
            "character_results",
        )[
            character_index
        ]
    )

    _verify_refresh_base_matches_state(
        current_character_result=(
            current_character_result
        ),
        refresh_result=(
            refresh_result
        ),
    )

    replacement_character_results = (
        _replace_one_character_artifact(
            state=current_state,
            character_index=(
                character_index
            ),
            candidate_artifact=(
                refresh_result
                .candidate_artifact
            ),
        )
    )

    # -----------------------------------------------------
    # Unchanged Grounded Authority
    # -----------------------------------------------------

    if (
        refresh_result.refresh_status
        == "unchanged"
    ):
        candidate_state = (
            current_state.model_copy(
                update={
                    "character_results": (
                        replacement_character_results
                    ),
                }
            )
        )

        committed_state = (
            _validate_committed_state(
                candidate_state
            )
        )

        for field_name in (
            "lore_results",
            "script_result",
            "story_state_ledger",
            "continuity_result",
            "evaluation_result",
        ):
            if (
                _semantic_dump(
                    getattr(
                        current_state,
                        field_name,
                    )
                )
                != _semantic_dump(
                    getattr(
                        committed_state,
                        field_name,
                    )
                )
            ):
                raise AssertionError(
                    "Unchanged Character refresh unexpectedly "
                    f"changed preserved field "
                    f"{field_name!r}."
                )

        return CharacterRefreshCommitResult(
            character_id=(
                refresh_result.character_id
            ),
            character_name=(
                refresh_result.character_name
            ),
            commit_status=(
                "committed_without_invalidation"
            ),
            invalidated_artifacts=[],
            committed_state=(
                committed_state
            ),
            reason=(
                "Refreshed Character research provenance "
                "was committed, but downstream-authoritative "
                "grounded guidance was unchanged. Lore, "
                "Scriptwriter, Continuity, ledger, and "
                "Evaluation artifacts were preserved."
            ),
        )

    # -----------------------------------------------------
    # Material Grounded Authority Change
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
                "character_results": (
                    replacement_character_results
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
            "Material Character refresh failed to invalidate "
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
            "Material Character refresh failed to invalidate "
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
            "Material Character refresh failed to invalidate "
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
            "Material Character refresh failed to reset "
            "story_state_ledger."
        )

    # Lore is a sibling authority to Character and must survive.
    if (
        _semantic_dump(
            getattr(
                current_state,
                "lore_results",
            )
        )
        != _semantic_dump(
            getattr(
                committed_state,
                "lore_results",
            )
        )
    ):
        raise AssertionError(
            "Character refresh unexpectedly changed "
            "Lore artifacts."
        )

    # Sibling Character artifacts must survive unchanged.
    old_character_results = getattr(
        current_state,
        "character_results",
    )

    new_character_results = getattr(
        committed_state,
        "character_results",
    )

    for index, (
        old_character,
        new_character,
    ) in enumerate(
        zip(
            old_character_results,
            new_character_results,
        )
    ):
        if index == character_index:
            continue

        if (
            _semantic_dump(
                old_character
            )
            != _semantic_dump(
                new_character
            )
        ):
            raise AssertionError(
                "Character refresh unexpectedly changed "
                "a sibling Character artifact."
            )

    # StoryPlan is upstream and must also survive.
    if hasattr(
        current_state,
        "story_plan",
    ):
        if (
            _semantic_dump(
                getattr(
                    current_state,
                    "story_plan",
                )
            )
            != _semantic_dump(
                getattr(
                    committed_state,
                    "story_plan",
                )
            )
        ):
            raise AssertionError(
                "Character refresh unexpectedly changed "
                "StoryPlan."
            )

    return CharacterRefreshCommitResult(
        character_id=(
            refresh_result.character_id
        ),
        character_name=(
            refresh_result.character_name
        ),
        commit_status=(
            "committed_with_invalidation"
        ),
        invalidated_artifacts=list(
            EXPECTED_MATERIAL_CHARACTER_INVALIDATIONS
        ),
        committed_state=(
            committed_state
        ),
        reason=(
            "Refreshed Character authority materially "
            "changed. The candidate Character artifact "
            "was committed in memory and exactly the "
            "proven downstream Scriptwriter, Continuity, "
            "story-state ledger, and Evaluation artifacts "
            "were invalidated. StoryPlan, Lore, and sibling "
            "Character artifacts remain preserved."
        ),
    )
