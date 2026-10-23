"""
Deterministic Evaluation evidence-resolution planning.

This module belongs to the Writer's Room orchestration layer.

It does NOT:
- call an LLM;
- rerun any specialist agent;
- mutate Writer's Room state;
- invalidate downstream artifacts;
- repair screenplay content;
- reinterpret Evaluation verdicts.

Its only responsibility is to translate a completed Evaluation
production result into explicit upstream evidence-resolution targets.

Core rule:

    Evaluation owns the final verdict.
    Python owns deterministic blocker routing.

The resolver is intentionally fail-closed. A NEEDS_EVIDENCE blocker
must map to a known criterion family and, where applicable, to an
authoritative upstream artifact supplied by Writer's Room.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------
# Stable Evaluation Criterion IDs
# ---------------------------------------------------------

LORE_CRITERION_PREFIX = "lore_agent_evidence_"
CHARACTER_CRITERION_PREFIX = (
    "character_agent_guidance_"
)

ERA_STYLE_CRITERION_ID = "era_style_review"
CONTINUITY_CRITERION_ID = (
    "continuity_agent_review"
)
SCRIPTWRITER_CRITERION_ID = (
    "scriptwriter_grounding_review"
)


# ---------------------------------------------------------
# Resolution Source Descriptors
#
# These are intentionally small Writer's Room-facing
# descriptors. The resolver does not need complete Lore or
# Character production packages merely to identify ownership.
# ---------------------------------------------------------


class LoreResolutionSource(BaseModel):
    research_id: str = Field(
        min_length=1,
        description=(
            "Stable Writer's Room Lore artifact ID."
        ),
    )


class CharacterResolutionSource(BaseModel):
    character_id: str = Field(
        min_length=1,
        description=(
            "Stable Writer's Room Character artifact ID."
        ),
    )

    character_name: str = Field(
        min_length=1,
        description=(
            "Character identity represented by the artifact."
        ),
    )


# ---------------------------------------------------------
# Resolution Output
# ---------------------------------------------------------


class EvaluationResolutionTarget(BaseModel):
    criterion_id: str = Field(
        min_length=1,
    )

    category: Literal[
        "canon_grounding",
        "character_consistency",
        "era_style",
        "continuity",
    ]

    owner: Literal[
        "lore",
        "character",
        "scriptwriter_input",
        "continuity",
    ]

    action: Literal[
        "refresh_lore_evidence",
        "refresh_character_evidence",
        "supply_era_style_constraints",
        "resolve_continuity_context",
    ]

    artifact_id: str | None = None

    subject_name: str | None = None

    reason: str = Field(
        min_length=1,
        description=(
            "Deterministic explanation of why this "
            "upstream owner must resolve the blocker."
        ),
    )


class EvaluationResolutionPlan(BaseModel):
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

    targets: list[
        EvaluationResolutionTarget
    ] = Field(
        default_factory=list
    )

    reason: str = Field(
        min_length=1,
    )


# ---------------------------------------------------------
# Identifier Helpers
# ---------------------------------------------------------


def normalize_evaluation_identifier(
    value: str,
) -> str:
    """
    Match the normalization currently used by the frozen
    Evaluation Agent's stable criterion-ID builders.

    This helper is kept deterministic so the resolver can
    map criterion IDs back to Writer's Room artifacts
    without asking an LLM.
    """

    if not value.strip():
        raise ValueError(
            "Identifier source cannot be empty."
        )

    normalized = "".join(
        character.lower()
        if character.isalnum()
        else "_"
        for character in value
    ).strip("_")

    if not normalized:
        raise ValueError(
            "Identifier source does not contain usable "
            "identifier characters."
        )

    return normalized


def build_expected_lore_criterion_id(
    research_id: str,
) -> str:
    return (
        LORE_CRITERION_PREFIX
        + normalize_evaluation_identifier(
            research_id
        )
    )


def build_expected_character_criterion_id(
    character_name: str,
) -> str:
    return (
        CHARACTER_CRITERION_PREFIX
        + normalize_evaluation_identifier(
            character_name
        )
    )


# ---------------------------------------------------------
# Structural Readers
#
# EvaluationProductionReview is deliberately consumed by
# attribute contract rather than imported here. This keeps
# the resolver free of provider/model initialization while
# still accepting the real Pydantic production object.
# ---------------------------------------------------------


def _read_required_attribute(
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


def _read_evaluation_status(
    production_review: Any,
) -> str:
    review = _read_required_attribute(
        production_review,
        "review",
        "Evaluation production review",
    )

    assessment = _read_required_attribute(
        review,
        "assessment",
        "Evaluation review",
    )

    status = _read_required_attribute(
        assessment,
        "status",
        "Evaluation assessment",
    )

    if status not in {
        "approved",
        "needs_evidence",
        "rejected",
    }:
        raise ValueError(
            "Unsupported Evaluation assessment status: "
            f"{status!r}"
        )

    return status


def _read_blocking_criterion_ids(
    production_review: Any,
) -> list[str]:
    review = _read_required_attribute(
        production_review,
        "review",
        "Evaluation production review",
    )

    assessment = _read_required_attribute(
        review,
        "assessment",
        "Evaluation review",
    )

    blocking_ids = _read_required_attribute(
        assessment,
        "blocking_criterion_ids",
        "Evaluation assessment",
    )

    if not isinstance(
        blocking_ids,
        list,
    ):
        raise TypeError(
            "Evaluation blocking_criterion_ids must "
            "be a list."
        )

    normalized_ids: list[str] = []

    for index, criterion_id in enumerate(
        blocking_ids
    ):
        if (
            not isinstance(
                criterion_id,
                str,
            )
            or not criterion_id.strip()
        ):
            raise ValueError(
                "Evaluation blocking criterion ID at "
                f"index {index} is invalid."
            )

        normalized_ids.append(
            criterion_id
        )

    if len(normalized_ids) != len(
        set(normalized_ids)
    ):
        raise ValueError(
            "Evaluation blocking criterion IDs "
            "must be unique."
        )

    return normalized_ids


def _read_suite_results(
    production_review: Any,
) -> list[Any]:
    review = _read_required_attribute(
        production_review,
        "review",
        "Evaluation production review",
    )

    suite = _read_required_attribute(
        review,
        "suite",
        "Evaluation review",
    )

    results = _read_required_attribute(
        suite,
        "results",
        "Evaluation suite",
    )

    if not isinstance(
        results,
        list,
    ):
        raise TypeError(
            "Evaluation suite results must be a list."
        )

    if not results:
        raise ValueError(
            "Evaluation suite must contain at least "
            "one criterion result."
        )

    return results


def _index_suite_results_by_id(
    production_review: Any,
) -> dict[str, Any]:
    result_by_id: dict[str, Any] = {}

    for result in _read_suite_results(
        production_review
    ):
        criterion_id = _read_required_attribute(
            result,
            "criterion_id",
            "Evaluation criterion result",
        )

        if (
            not isinstance(
                criterion_id,
                str,
            )
            or not criterion_id.strip()
        ):
            raise ValueError(
                "Evaluation criterion result contains "
                "an invalid criterion_id."
            )

        if criterion_id in result_by_id:
            raise ValueError(
                "Evaluation suite contains duplicate "
                f"criterion ID {criterion_id!r}."
            )

        result_by_id[
            criterion_id
        ] = result

    return result_by_id


# ---------------------------------------------------------
# Source Indexes
# ---------------------------------------------------------


def _index_lore_sources(
    lore_sources: list[
        LoreResolutionSource
    ],
) -> dict[str, LoreResolutionSource]:
    criterion_map: dict[
        str,
        LoreResolutionSource,
    ] = {}

    seen_research_ids: set[str] = set()

    for source in lore_sources:
        if source.research_id in seen_research_ids:
            raise ValueError(
                "Duplicate Lore resolution source ID: "
                f"{source.research_id!r}"
            )

        seen_research_ids.add(
            source.research_id
        )

        criterion_id = (
            build_expected_lore_criterion_id(
                source.research_id
            )
        )

        if criterion_id in criterion_map:
            raise ValueError(
                "Multiple Lore artifacts normalize to "
                "the same Evaluation criterion ID: "
                f"{criterion_id!r}"
            )

        criterion_map[
            criterion_id
        ] = source

    return criterion_map


def _index_character_sources(
    character_sources: list[
        CharacterResolutionSource
    ],
) -> dict[
    str,
    CharacterResolutionSource,
]:
    criterion_map: dict[
        str,
        CharacterResolutionSource,
    ] = {}

    seen_character_ids: set[str] = set()

    for source in character_sources:
        if (
            source.character_id
            in seen_character_ids
        ):
            raise ValueError(
                "Duplicate Character resolution "
                f"source ID: {source.character_id!r}"
            )

        seen_character_ids.add(
            source.character_id
        )

        criterion_id = (
            build_expected_character_criterion_id(
                source.character_name
            )
        )

        if criterion_id in criterion_map:
            raise ValueError(
                "Multiple Character artifacts normalize "
                "to the same Evaluation criterion ID: "
                f"{criterion_id!r}"
            )

        criterion_map[
            criterion_id
        ] = source

    return criterion_map


# ---------------------------------------------------------
# Blocker Resolution
# ---------------------------------------------------------


def _resolve_lore_blocker(
    criterion_id: str,
    lore_criterion_map: dict[
        str,
        LoreResolutionSource,
    ],
) -> EvaluationResolutionTarget:
    source = lore_criterion_map.get(
        criterion_id
    )

    if source is None:
        raise ValueError(
            "Lore Evaluation blocker does not map to "
            "a supplied Writer's Room Lore artifact: "
            f"{criterion_id!r}"
        )

    return EvaluationResolutionTarget(
        criterion_id=criterion_id,
        category="canon_grounding",
        owner="lore",
        action="refresh_lore_evidence",
        artifact_id=source.research_id,
        subject_name=None,
        reason=(
            "Evaluation requires additional canon "
            "authority for this Lore criterion. The "
            "criterion deterministically maps to "
            f"Writer's Room Lore artifact "
            f"{source.research_id!r}."
        ),
    )


def _resolve_character_blocker(
    criterion_id: str,
    character_criterion_map: dict[
        str,
        CharacterResolutionSource,
    ],
) -> EvaluationResolutionTarget:
    source = character_criterion_map.get(
        criterion_id
    )

    if source is None:
        raise ValueError(
            "Character Evaluation blocker does not map "
            "to a supplied Writer's Room Character "
            f"artifact: {criterion_id!r}"
        )

    return EvaluationResolutionTarget(
        criterion_id=criterion_id,
        category="character_consistency",
        owner="character",
        action="refresh_character_evidence",
        artifact_id=source.character_id,
        subject_name=source.character_name,
        reason=(
            "Evaluation requires additional grounded "
            "character authority for this criterion. "
            "The criterion deterministically maps to "
            f"Writer's Room Character artifact "
            f"{source.character_id!r} "
            f"({source.character_name})."
        ),
    )


def _resolve_single_blocker(
    criterion_id: str,
    criterion_result: Any,
    lore_criterion_map: dict[
        str,
        LoreResolutionSource,
    ],
    character_criterion_map: dict[
        str,
        CharacterResolutionSource,
    ],
) -> EvaluationResolutionTarget:
    verdict = _read_required_attribute(
        criterion_result,
        "verdict",
        "Evaluation criterion result",
    )

    category = _read_required_attribute(
        criterion_result,
        "category",
        "Evaluation criterion result",
    )

    if verdict != "insufficient_evidence":
        raise ValueError(
            "NEEDS_EVIDENCE blocking criterion must "
            "have verdict='insufficient_evidence': "
            f"{criterion_id!r} has {verdict!r}."
        )

    if category == "canon_grounding":
        if not criterion_id.startswith(
            LORE_CRITERION_PREFIX
        ):
            raise ValueError(
                "Canon-grounding insufficient-evidence "
                "criterion is not a recognized Lore "
                f"criterion: {criterion_id!r}"
            )

        return _resolve_lore_blocker(
            criterion_id=criterion_id,
            lore_criterion_map=(
                lore_criterion_map
            ),
        )

    if category == "character_consistency":
        if not criterion_id.startswith(
            CHARACTER_CRITERION_PREFIX
        ):
            raise ValueError(
                "Character-consistency insufficient-"
                "evidence criterion is not a recognized "
                f"Character criterion: {criterion_id!r}"
            )

        return _resolve_character_blocker(
            criterion_id=criterion_id,
            character_criterion_map=(
                character_criterion_map
            ),
        )

    if category == "era_style":
        if criterion_id != ERA_STYLE_CRITERION_ID:
            raise ValueError(
                "Unknown era/style blocking criterion: "
                f"{criterion_id!r}"
            )

        return EvaluationResolutionTarget(
            criterion_id=criterion_id,
            category="era_style",
            owner="scriptwriter_input",
            action=(
                "supply_era_style_constraints"
            ),
            artifact_id=None,
            subject_name=None,
            reason=(
                "Evaluation lacks explicit bounded "
                "era/style authority. Resolution belongs "
                "to ScriptwriterInput constraints rather "
                "than to canon or character research."
            ),
        )

    if category == "continuity":
        if (
            criterion_id
            != CONTINUITY_CRITERION_ID
        ):
            raise ValueError(
                "Unknown Continuity blocking criterion: "
                f"{criterion_id!r}"
            )

        return EvaluationResolutionTarget(
            criterion_id=criterion_id,
            category="continuity",
            owner="continuity",
            action="resolve_continuity_context",
            artifact_id=None,
            subject_name=None,
            reason=(
                "Frozen Continuity reported unresolved "
                "story context. Resolution belongs to "
                "Continuity context/state rather than "
                "Lore or Character research."
            ),
        )

    if category == "upstream_constraint":
        if (
            criterion_id
            == SCRIPTWRITER_CRITERION_ID
        ):
            raise ValueError(
                "Scriptwriter grounding cannot be routed "
                "through the evidence-resolution loop. "
                "The frozen Scriptwriter adapter produces "
                "pass or fail for this criterion, not "
                "insufficient_evidence."
            )

        raise ValueError(
            "Unknown upstream-constraint blocking "
            f"criterion: {criterion_id!r}"
        )

    raise ValueError(
        "Unsupported Evaluation blocker category "
        f"{category!r} for criterion "
        f"{criterion_id!r}."
    )


# ---------------------------------------------------------
# Public API
# ---------------------------------------------------------


def build_evaluation_resolution_plan(
    production_review: Any,
    lore_sources: list[
        LoreResolutionSource
    ] | None = None,
    character_sources: list[
        CharacterResolutionSource
    ] | None = None,
) -> EvaluationResolutionPlan:
    """
    Convert one completed EvaluationProductionReview into
    a deterministic evidence-resolution plan.

    Parameters
    ----------
    production_review:
        The completed EvaluationProductionReview produced
        by the frozen Evaluation Agent.

    lore_sources:
        Writer's Room Lore artifact descriptors available
        to this Evaluation result.

    character_sources:
        Writer's Room Character artifact descriptors
        available to this Evaluation result.

    Returns
    -------
    EvaluationResolutionPlan
        A deterministic plan describing whether evidence
        resolution is needed and which upstream owners must
        address each blocking criterion.

    Important
    ---------
    This function never executes the resolution targets.
    It only identifies them.
    """

    lore_sources = (
        list(lore_sources)
        if lore_sources is not None
        else []
    )

    character_sources = (
        list(character_sources)
        if character_sources is not None
        else []
    )

    evaluation_status = (
        _read_evaluation_status(
            production_review
        )
    )

    blocking_ids = (
        _read_blocking_criterion_ids(
            production_review
        )
    )

    result_by_id = (
        _index_suite_results_by_id(
            production_review
        )
    )

    # -----------------------------------------------------
    # Approved
    #
    # No evidence-resolution work is allowed because
    # Evaluation already approved the scene.
    # -----------------------------------------------------

    if evaluation_status == "approved":
        if blocking_ids:
            raise ValueError(
                "Approved Evaluation cannot contain "
                "blocking criterion IDs."
            )

        return EvaluationResolutionPlan(
            evaluation_status="approved",
            resolution_status="no_action",
            targets=[],
            reason=(
                "Evaluation approved the scene. "
                "No evidence resolution is required."
            ),
        )

    # -----------------------------------------------------
    # Rejected
    #
    # A rejection is not an evidence-resolution request.
    # The failing criterion belongs to a repair/revision
    # path that is intentionally outside this module.
    # -----------------------------------------------------

    if evaluation_status == "rejected":
        if not blocking_ids:
            raise ValueError(
                "Rejected Evaluation must contain at "
                "least one blocking criterion ID."
            )

        for criterion_id in blocking_ids:
            result = result_by_id.get(
                criterion_id
            )

            if result is None:
                raise ValueError(
                    "Rejected Evaluation references "
                    "unknown blocking criterion ID "
                    f"{criterion_id!r}."
                )

            verdict = _read_required_attribute(
                result,
                "verdict",
                "Evaluation criterion result",
            )

            if verdict != "fail":
                raise ValueError(
                    "Rejected Evaluation blocking "
                    "criteria must have verdict='fail': "
                    f"{criterion_id!r} has "
                    f"{verdict!r}."
                )

        return EvaluationResolutionPlan(
            evaluation_status="rejected",
            resolution_status="not_applicable",
            targets=[],
            reason=(
                "Evaluation rejected the scene because "
                "one or more criteria failed. Evidence "
                "resolution is not the appropriate path."
            ),
        )

    # -----------------------------------------------------
    # Needs Evidence
    # -----------------------------------------------------

    if evaluation_status != "needs_evidence":
        raise AssertionError(
            "Unreachable Evaluation status branch."
        )

    if not blocking_ids:
        raise ValueError(
            "NEEDS_EVIDENCE Evaluation must contain "
            "at least one blocking criterion ID."
        )

    lore_criterion_map = (
        _index_lore_sources(
            lore_sources
        )
    )

    character_criterion_map = (
        _index_character_sources(
            character_sources
        )
    )

    targets: list[
        EvaluationResolutionTarget
    ] = []

    for criterion_id in blocking_ids:
        criterion_result = result_by_id.get(
            criterion_id
        )

        if criterion_result is None:
            raise ValueError(
                "Evaluation assessment references "
                "blocking criterion ID that is absent "
                "from the Evaluation suite: "
                f"{criterion_id!r}"
            )

        target = _resolve_single_blocker(
            criterion_id=criterion_id,
            criterion_result=(
                criterion_result
            ),
            lore_criterion_map=(
                lore_criterion_map
            ),
            character_criterion_map=(
                character_criterion_map
            ),
        )

        targets.append(
            target
        )

    if len(targets) != len(
        blocking_ids
    ):
        raise AssertionError(
            "Evidence-resolution target coverage "
            "does not match Evaluation blockers."
        )

    return EvaluationResolutionPlan(
        evaluation_status="needs_evidence",
        resolution_status="needs_resolution",
        targets=targets,
        reason=(
            f"Evaluation reported {len(targets)} "
            "insufficient-evidence blocker(s). "
            "Each blocker was deterministically "
            "mapped to an upstream resolution owner."
        ),
    )
