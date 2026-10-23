"""
Deterministic material-change detection for evidence-resolution refreshes.

This module answers:

    Did a refreshed authoritative upstream artifact change in a way
    that can affect current Writer's Room downstream behavior?

It does NOT:
- call an LLM;
- refresh any specialist;
- mutate Writer's Room state;
- invalidate artifacts;
- compare incidental diagnostics or provider metadata.

The comparison surface is intentionally limited to fields that the
current Writer's Room actually routes downstream.

Current material surfaces
-------------------------

Lore:
- stable research identity;
- research question;
- premise_supported;
- evidence_supported;
- source text and its provenance fingerprint when authority is valid.

Character:
- stable Writer's Room character identity;
- audit.grounded;
- CharacterAnalysis fields actually routed to Scriptwriter and/or
  final Evaluation, but only when audit.grounded=True.

Why exclude other data?
-----------------------

Character's raw research text, audit explanations, and remaining
questions are preserved without becoming scene/evaluation authority.
Their changes alone do not force downstream regeneration. Lore's
validated source identity and context are included in its fingerprint.

Core rule:

    Preserve provenance broadly.
    Detect material authority narrowly.
"""

from __future__ import annotations

from tools.lore_evidence import authoritative_lore_text, lore_evidence_fingerprint

from typing import Any, Literal

from pydantic import BaseModel, Field


MaterialArtifactType = Literal[
    "lore",
    "character",
    "era_style",
]


# ---------------------------------------------------------
# Material Snapshots
# ---------------------------------------------------------


class LoreMaterialSnapshot(BaseModel):
    research_id: str = Field(
        min_length=1,
    )

    research_question: str = Field(
        min_length=1,
    )

    premise_supported: bool

    evidence_supported: bool

    authoritative_summary: str

    authoritative_evidence_fingerprint: str = ""


class CharacterMaterialSnapshot(BaseModel):
    character_id: str = Field(
        min_length=1,
    )

    character_name: str = Field(
        min_length=1,
    )

    grounded: bool

    story_role: str = ""

    motivations: tuple[str, ...] = ()

    likely_behavior: tuple[str, ...] = ()

    unlikely_behavior: tuple[str, ...] = ()

    relationships: tuple[str, ...] = ()

    dramatic_tension: str = ""

    speech_patterns: tuple[str, ...] = ()

    preferred_vocabulary: tuple[str, ...] = ()

    communication_style: tuple[str, ...] = ()

    behavioral_constraints: tuple[str, ...] = ()

    era_language_notes: tuple[str, ...] = ()

    out_of_character_risks: tuple[str, ...] = ()


class EraStyleMaterialSnapshot(BaseModel):
    story_era: str = Field(
        min_length=1,
    )

    era_style_constraints: tuple[
        str,
        ...,
    ] = ()


class MaterialChangeReport(BaseModel):
    artifact_type: MaterialArtifactType

    material_change: bool

    changed_fields: list[str] = Field(
        default_factory=list
    )

    old_snapshot: dict[str, Any]

    new_snapshot: dict[str, Any]

    reason: str = Field(
        min_length=1,
    )


# ---------------------------------------------------------
# Generic Structural Readers
#
# These accept either real Pydantic production objects or
# lightweight regression objects with the same attributes.
# Importing Writer's Room / agent modules here would also
# initialize provider-backed model objects, which this
# deterministic layer intentionally avoids.
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


def _normalize_required_text(
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

    normalized = value.strip()

    if not normalized:
        raise ValueError(
            f"{label} cannot be empty."
        )

    return normalized


def _normalize_optional_text(
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

    return value.strip()


def _normalize_text_sequence(
    value: Any,
    label: str,
) -> tuple[str, ...]:
    if not isinstance(
        value,
        (list, tuple),
    ):
        raise TypeError(
            f"{label} must be a list or tuple."
        )

    normalized: list[str] = []

    for index, item in enumerate(
        value
    ):
        if not isinstance(
            item,
            str,
        ):
            raise TypeError(
                f"{label}[{index}] must be a string."
            )

        stripped_item = item.strip()

        # Writer's Room / Evaluation already filter blank
        # guidance items before using them downstream.
        if stripped_item:
            normalized.append(
                stripped_item
            )

    return tuple(
        normalized
    )


def _read_bool(
    value: Any,
    label: str,
) -> bool:
    if not isinstance(
        value,
        bool,
    ):
        raise TypeError(
            f"{label} must be a bool."
        )

    return value


# ---------------------------------------------------------
# Snapshot Builders
# ---------------------------------------------------------


def build_lore_material_snapshot(
    lore_result: Any,
) -> LoreMaterialSnapshot:
    """
    Build the exact Lore authority surface currently relevant
    to Scriptwriter and Evaluation.

    When evidence_supported=False or a historical response has no
    source package, the Lore summary is not authoritative downstream:
    - Scriptwriter does not promote it into verified canon.
    - Evaluation returns insufficient_evidence from the flags
      rather than using the summary as canon context.

    Non-authoritative wording is excluded from comparison. Valid
    evidence includes a provenance fingerprint, so source identity or
    source-context changes can trigger conservative invalidation.
    """

    research_id = _normalize_required_text(
        _read_attribute(
            lore_result,
            "research_id",
            "Lore result",
        ),
        "Lore research_id",
    )

    research_question = (
        _normalize_required_text(
            _read_attribute(
                lore_result,
                "research_question",
                "Lore result",
            ),
            "Lore research_question",
        )
    )

    response = _read_attribute(
        lore_result,
        "response",
        "Lore result",
    )

    premise_supported = _read_bool(
        _read_attribute(
            response,
            "premise_supported",
            "Lore response",
        ),
        "Lore premise_supported",
    )

    evidence_supported = _read_bool(
        _read_attribute(
            response,
            "evidence_supported",
            "Lore response",
        ),
        "Lore evidence_supported",
    )

    if (
        evidence_supported
        and not premise_supported
    ):
        raise ValueError(
            "Lore material snapshot received an "
            "impossible authority state: "
            "evidence_supported=True while "
            "premise_supported=False."
        )

    summary = _normalize_optional_text(
        _read_attribute(
            response,
            "summary",
            "Lore response",
        ),
        "Lore summary",
    )

    fingerprint = ""
    if evidence_supported and getattr(response, "evidence_package", None) is None:
        # Historical success flags cannot authorize pre-contract summaries.
        # Keep identity/premise for refresh comparison without promoting text.
        evidence_supported = False
        authoritative_summary = ""
    elif evidence_supported:
        authoritative_summary = authoritative_lore_text(response)
        fingerprint = lore_evidence_fingerprint(response)
    else:
        authoritative_summary = ""

    return LoreMaterialSnapshot(
        research_id=research_id,
        research_question=(
            research_question
        ),
        premise_supported=(
            premise_supported
        ),
        evidence_supported=(
            evidence_supported
        ),
        authoritative_summary=(
            authoritative_summary
        ),
        authoritative_evidence_fingerprint=fingerprint,
    )


def build_character_material_snapshot(
    character_result: Any,
) -> CharacterMaterialSnapshot:
    """
    Build the Character authority surface currently consumed
    downstream.

    If audit.grounded=False, Character analysis content is not
    authoritative:
    - Scriptwriter rejects it as character guidance.
    - Evaluation converts the ungrounded audit directly to
      insufficient_evidence.

    Therefore analysis wording changes while the artifact remains
    ungrounded are intentionally non-material.
    """

    character_id = _normalize_required_text(
        _read_attribute(
            character_result,
            "character_id",
            "Character result",
        ),
        "Character character_id",
    )

    character_name = _normalize_required_text(
        _read_attribute(
            character_result,
            "character_name",
            "Character result",
        ),
        "Character character_name",
    )

    audit = _read_attribute(
        character_result,
        "audit",
        "Character result",
    )

    grounded = _read_bool(
        _read_attribute(
            audit,
            "grounded",
            "Character audit",
        ),
        "Character audit.grounded",
    )

    if not grounded:
        return CharacterMaterialSnapshot(
            character_id=character_id,
            character_name=character_name,
            grounded=False,
        )

    analysis = _read_attribute(
        character_result,
        "analysis",
        "Character result",
    )

    analysis_character_name = (
        _normalize_required_text(
            _read_attribute(
                analysis,
                "character_name",
                "Character analysis",
            ),
            "Character analysis.character_name",
        )
    )

    if (
        analysis_character_name.casefold()
        != character_name.casefold()
    ):
        raise ValueError(
            "Character analysis identity does not "
            "match Writer's Room character identity."
        )

    return CharacterMaterialSnapshot(
        character_id=character_id,
        character_name=character_name,
        grounded=True,
        story_role=_normalize_optional_text(
            _read_attribute(
                analysis,
                "story_role",
                "Character analysis",
            ),
            "Character story_role",
        ),
        motivations=_normalize_text_sequence(
            _read_attribute(
                analysis,
                "motivations",
                "Character analysis",
            ),
            "Character motivations",
        ),
        likely_behavior=_normalize_text_sequence(
            _read_attribute(
                analysis,
                "likely_behavior",
                "Character analysis",
            ),
            "Character likely_behavior",
        ),
        unlikely_behavior=_normalize_text_sequence(
            _read_attribute(
                analysis,
                "unlikely_behavior",
                "Character analysis",
            ),
            "Character unlikely_behavior",
        ),
        relationships=_normalize_text_sequence(
            _read_attribute(
                analysis,
                "relationships",
                "Character analysis",
            ),
            "Character relationships",
        ),
        dramatic_tension=_normalize_optional_text(
            _read_attribute(
                analysis,
                "dramatic_tension",
                "Character analysis",
            ),
            "Character dramatic_tension",
        ),
        speech_patterns=_normalize_text_sequence(
            _read_attribute(
                analysis,
                "speech_patterns",
                "Character analysis",
            ),
            "Character speech_patterns",
        ),
        preferred_vocabulary=(
            _normalize_text_sequence(
                _read_attribute(
                    analysis,
                    "preferred_vocabulary",
                    "Character analysis",
                ),
                "Character preferred_vocabulary",
            )
        ),
        communication_style=(
            _normalize_text_sequence(
                _read_attribute(
                    analysis,
                    "communication_style",
                    "Character analysis",
                ),
                "Character communication_style",
            )
        ),
        behavioral_constraints=(
            _normalize_text_sequence(
                _read_attribute(
                    analysis,
                    "behavioral_constraints",
                    "Character analysis",
                ),
                "Character behavioral_constraints",
            )
        ),
        era_language_notes=(
            _normalize_text_sequence(
                _read_attribute(
                    analysis,
                    "era_language_notes",
                    "Character analysis",
                ),
                "Character era_language_notes",
            )
        ),
        out_of_character_risks=(
            _normalize_text_sequence(
                _read_attribute(
                    analysis,
                    "out_of_character_risks",
                    "Character analysis",
                ),
                "Character out_of_character_risks",
            )
        ),
    )


def build_era_style_material_snapshot(
    story_era: str,
    era_style_constraints: list[str] | tuple[str, ...],
) -> EraStyleMaterialSnapshot:
    """
    Build the ScriptwriterInput era/style authority surface.
    """

    return EraStyleMaterialSnapshot(
        story_era=_normalize_required_text(
            story_era,
            "story_era",
        ),
        era_style_constraints=(
            _normalize_text_sequence(
                era_style_constraints,
                "era_style_constraints",
            )
        ),
    )


# ---------------------------------------------------------
# Comparison Helpers
# ---------------------------------------------------------


def _changed_snapshot_fields(
    old_snapshot: BaseModel,
    new_snapshot: BaseModel,
    excluded_fields: set[str] | None = None,
) -> list[str]:
    excluded_fields = (
        set(excluded_fields)
        if excluded_fields
        else set()
    )

    old_data = old_snapshot.model_dump(
        mode="python"
    )

    new_data = new_snapshot.model_dump(
        mode="python"
    )

    if set(old_data) != set(
        new_data
    ):
        raise AssertionError(
            "Material snapshot schemas do not match."
        )

    return [
        field_name
        for field_name in old_data
        if (
            field_name
            not in excluded_fields
            and old_data[field_name]
            != new_data[field_name]
        )
    ]


def compare_lore_material_change(
    old_result: Any,
    new_result: Any,
) -> MaterialChangeReport:
    old_snapshot = (
        build_lore_material_snapshot(
            old_result
        )
    )

    new_snapshot = (
        build_lore_material_snapshot(
            new_result
        )
    )

    if (
        old_snapshot.research_id
        != new_snapshot.research_id
    ):
        raise ValueError(
            "Cannot compare Lore material change "
            "across different research IDs."
        )

    changed_fields = (
        _changed_snapshot_fields(
            old_snapshot,
            new_snapshot,
            excluded_fields={
                "research_id",
            },
        )
    )

    material_change = bool(
        changed_fields
    )

    if material_change:
        reason = (
            "Lore refresh materially changed "
            "downstream authoritative content: "
            + ", ".join(
                changed_fields
            )
            + "."
        )
    else:
        reason = (
            "Lore refresh did not change any "
            "downstream authoritative content."
        )

    return MaterialChangeReport(
        artifact_type="lore",
        material_change=(
            material_change
        ),
        changed_fields=(
            changed_fields
        ),
        old_snapshot=(
            old_snapshot.model_dump(
                mode="json"
            )
        ),
        new_snapshot=(
            new_snapshot.model_dump(
                mode="json"
            )
        ),
        reason=reason,
    )


def compare_character_material_change(
    old_result: Any,
    new_result: Any,
) -> MaterialChangeReport:
    old_snapshot = (
        build_character_material_snapshot(
            old_result
        )
    )

    new_snapshot = (
        build_character_material_snapshot(
            new_result
        )
    )

    if (
        old_snapshot.character_id
        != new_snapshot.character_id
    ):
        raise ValueError(
            "Cannot compare Character material change "
            "across different character IDs."
        )

    if (
        old_snapshot.character_name
        .casefold()
        != new_snapshot.character_name
        .casefold()
    ):
        raise ValueError(
            "Cannot compare Character material change "
            "across different character identities."
        )

    changed_fields = (
        _changed_snapshot_fields(
            old_snapshot,
            new_snapshot,
            excluded_fields={
                "character_id",
                "character_name",
            },
        )
    )

    material_change = bool(
        changed_fields
    )

    if material_change:
        reason = (
            "Character refresh materially changed "
            "downstream authoritative guidance: "
            + ", ".join(
                changed_fields
            )
            + "."
        )
    else:
        reason = (
            "Character refresh did not change any "
            "downstream authoritative guidance."
        )

    return MaterialChangeReport(
        artifact_type="character",
        material_change=(
            material_change
        ),
        changed_fields=(
            changed_fields
        ),
        old_snapshot=(
            old_snapshot.model_dump(
                mode="json"
            )
        ),
        new_snapshot=(
            new_snapshot.model_dump(
                mode="json"
            )
        ),
        reason=reason,
    )


def compare_era_style_material_change(
    *,
    old_story_era: str,
    old_constraints: list[str] | tuple[str, ...],
    new_story_era: str,
    new_constraints: list[str] | tuple[str, ...],
) -> MaterialChangeReport:
    old_snapshot = (
        build_era_style_material_snapshot(
            story_era=old_story_era,
            era_style_constraints=(
                old_constraints
            ),
        )
    )

    new_snapshot = (
        build_era_style_material_snapshot(
            story_era=new_story_era,
            era_style_constraints=(
                new_constraints
            ),
        )
    )

    changed_fields = (
        _changed_snapshot_fields(
            old_snapshot,
            new_snapshot,
        )
    )

    material_change = bool(
        changed_fields
    )

    if material_change:
        reason = (
            "Era/style authority materially changed: "
            + ", ".join(
                changed_fields
            )
            + "."
        )
    else:
        reason = (
            "Era/style refresh did not change the "
            "authoritative ScriptwriterInput constraints."
        )

    return MaterialChangeReport(
        artifact_type="era_style",
        material_change=(
            material_change
        ),
        changed_fields=(
            changed_fields
        ),
        old_snapshot=(
            old_snapshot.model_dump(
                mode="json"
            )
        ),
        new_snapshot=(
            new_snapshot.model_dump(
                mode="json"
            )
        ),
        reason=reason,
    )
