"""
Pure-Python regression coverage for workflows.artifact_material_change.

Run from the project root:

    python -m workflows.artifact_material_change_regression_test

No provider calls are made.
"""

from __future__ import annotations

from types import SimpleNamespace

from workflows.artifact_material_change import (
    compare_character_material_change,
    compare_era_style_material_change,
    compare_lore_material_change,
)


def assert_true(
    condition: bool,
    message: str,
) -> None:
    if not condition:
        raise AssertionError(message)


def assert_raises(
    expected_exception: type[Exception],
    callable_object,
    expected_message_fragment: str,
) -> None:
    try:
        callable_object()
    except expected_exception as error:
        if (
            expected_message_fragment
            not in str(error)
        ):
            raise AssertionError(
                "Exception message did not contain "
                f"{expected_message_fragment!r}.\n\n"
                f"Received:\n{error}"
            ) from error
        return
    except Exception as error:
        raise AssertionError(
            "Expected "
            f"{expected_exception.__name__}, "
            f"received {type(error).__name__}: "
            f"{error}"
        ) from error

    raise AssertionError(
        "Expected "
        f"{expected_exception.__name__} "
        "but no exception was raised."
    )


def lore_result(
    *,
    research_id: str = "lore_000",
    research_question: str = (
        "When does Garak directly encounter the Founders?"
    ),
    premise_supported: bool = True,
    evidence_supported: bool = False,
    summary: str = (
        "The available evidence does not establish "
        "a direct encounter."
    ),
) -> SimpleNamespace:
    return SimpleNamespace(
        research_id=research_id,
        research_question=research_question,
        response=SimpleNamespace(
            premise_supported=premise_supported,
            evidence_supported=evidence_supported,
            summary=summary,
        ),
    )


def character_analysis(
    *,
    character_name: str = "Elim Garak",
    story_role: str = "Information broker",
    motivations: list[str] | None = None,
    likely_behavior: list[str] | None = None,
    unlikely_behavior: list[str] | None = None,
    relationships: list[str] | None = None,
    dramatic_tension: str = (
        "He must judge whether uncertain information "
        "deserves action."
    ),
    speech_patterns: list[str] | None = None,
    preferred_vocabulary: list[str] | None = None,
    communication_style: list[str] | None = None,
    behavioral_constraints: list[str] | None = None,
    era_language_notes: list[str] | None = None,
    out_of_character_risks: list[str] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        character_name=character_name,
        story_role=story_role,
        motivations=(
            motivations
            if motivations is not None
            else ["Protect his interests."]
        ),
        likely_behavior=(
            likely_behavior
            if likely_behavior is not None
            else ["Evaluate uncertain information."]
        ),
        unlikely_behavior=(
            unlikely_behavior
            if unlikely_behavior is not None
            else ["Accept an allegation as proven."]
        ),
        relationships=(
            relationships
            if relationships is not None
            else []
        ),
        dramatic_tension=dramatic_tension,
        speech_patterns=(
            speech_patterns
            if speech_patterns is not None
            else []
        ),
        preferred_vocabulary=(
            preferred_vocabulary
            if preferred_vocabulary is not None
            else []
        ),
        communication_style=(
            communication_style
            if communication_style is not None
            else []
        ),
        behavioral_constraints=(
            behavioral_constraints
            if behavioral_constraints is not None
            else []
        ),
        era_language_notes=(
            era_language_notes
            if era_language_notes is not None
            else []
        ),
        out_of_character_risks=(
            out_of_character_risks
            if out_of_character_risks is not None
            else []
        ),
    )


def character_result(
    *,
    character_id: str = "character_000",
    character_name: str = "Elim Garak",
    grounded: bool = True,
    analysis: SimpleNamespace | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        character_id=character_id,
        character_name=character_name,
        audit=SimpleNamespace(
            grounded=grounded,
        ),
        analysis=(
            analysis
            if analysis is not None
            else character_analysis(
                character_name=character_name,
            )
        ),
    )


def test_supported_lore_summary_change_is_material() -> None:
    old = lore_result(
        evidence_supported=True,
        summary="Call to Arms occurs in 2373.",
    )

    new = lore_result(
        evidence_supported=True,
        summary=(
            "Call to Arms marks the official start "
            "of the Dominion War in 2373."
        ),
    )

    report = compare_lore_material_change(
        old,
        new,
    )

    assert_true(
        report.material_change is True,
        "Supported Lore summary change should "
        "be material.",
    )

    assert_true(
        "authoritative_summary"
        in report.changed_fields,
        "Supported Lore summary change was not "
        "reported.",
    )

    print(
        "Supported Lore summary change: PASS"
    )


def test_unsupported_lore_summary_change_is_not_material() -> None:
    old = lore_result(
        evidence_supported=False,
        summary=(
            "No direct encounter was established."
        ),
    )

    new = lore_result(
        evidence_supported=False,
        summary=(
            "Retrieved material still does not "
            "establish a direct encounter."
        ),
    )

    report = compare_lore_material_change(
        old,
        new,
    )

    assert_true(
        report.material_change is False,
        "Unsupported Lore wording alone should "
        "not be material.",
    )

    assert_true(
        report.changed_fields == [],
        "Unsupported Lore wording leaked into "
        "material comparison.",
    )

    print(
        "Unsupported Lore wording exclusion: PASS"
    )


def test_lore_support_transition_is_material() -> None:
    old = lore_result(
        evidence_supported=False,
        summary=(
            "No direct encounter was established."
        ),
    )

    new = lore_result(
        evidence_supported=True,
        summary=(
            "New evidence establishes a direct "
            "encounter."
        ),
    )

    report = compare_lore_material_change(
        old,
        new,
    )

    assert_true(
        report.material_change is True,
        "Lore support transition must be material.",
    )

    assert_true(
        "evidence_supported"
        in report.changed_fields,
        "Lore support transition was not reported.",
    )

    assert_true(
        "authoritative_summary"
        in report.changed_fields,
        "New supported Lore summary should become "
        "material.",
    )

    print(
        "Lore support transition: PASS"
    )


def test_lore_question_change_is_material() -> None:
    old = lore_result()

    new = lore_result(
        research_question=(
            "What is the earliest canonical direct "
            "Founder encounter for Garak?"
        )
    )

    report = compare_lore_material_change(
        old,
        new,
    )

    assert_true(
        report.material_change is True,
        "Lore research-question change should "
        "be material.",
    )

    assert_true(
        "research_question"
        in report.changed_fields,
        "Lore research-question change was not "
        "reported.",
    )

    print(
        "Lore question change: PASS"
    )


def test_lore_identity_mismatch_fails_closed() -> None:
    old = lore_result(
        research_id="lore_000"
    )

    new = lore_result(
        research_id="lore_001"
    )

    assert_raises(
        ValueError,
        lambda: compare_lore_material_change(
            old,
            new,
        ),
        "different research IDs",
    )

    print(
        "Lore identity mismatch fail-closed: PASS"
    )


def test_impossible_lore_authority_state_fails_closed() -> None:
    invalid = lore_result(
        premise_supported=False,
        evidence_supported=True,
        summary="Invalid supported summary.",
    )

    valid = lore_result(
        premise_supported=True,
        evidence_supported=True,
        summary="Supported summary.",
    )

    assert_raises(
        ValueError,
        lambda: compare_lore_material_change(
            invalid,
            valid,
        ),
        "impossible authority state",
    )

    print(
        "Impossible Lore authority fail-closed: PASS"
    )


def test_ungrounded_character_analysis_change_not_material() -> None:
    old = character_result(
        grounded=False,
        analysis=character_analysis(
            motivations=[
                "Old speculative motivation."
            ]
        ),
    )

    new = character_result(
        grounded=False,
        analysis=character_analysis(
            motivations=[
                "Different speculative motivation."
            ],
            likely_behavior=[
                "Different speculative behavior."
            ],
        ),
    )

    report = compare_character_material_change(
        old,
        new,
    )

    assert_true(
        report.material_change is False,
        "Ungrounded Character analysis wording "
        "should not be material.",
    )

    assert_true(
        report.changed_fields == [],
        "Ungrounded Character analysis leaked into "
        "material comparison.",
    )

    print(
        "Ungrounded Character wording exclusion: PASS"
    )


def test_character_grounding_transition_is_material() -> None:
    old = character_result(
        grounded=False,
    )

    new = character_result(
        grounded=True,
    )

    report = compare_character_material_change(
        old,
        new,
    )

    assert_true(
        report.material_change is True,
        "Character grounding transition must "
        "be material.",
    )

    assert_true(
        "grounded"
        in report.changed_fields,
        "Character grounding transition was not "
        "reported.",
    )

    print(
        "Character grounding transition: PASS"
    )


def test_grounded_character_guidance_change_is_material() -> None:
    old = character_result(
        grounded=True,
        analysis=character_analysis(
            likely_behavior=[
                "Evaluate uncertain information."
            ]
        ),
    )

    new = character_result(
        grounded=True,
        analysis=character_analysis(
            likely_behavior=[
                "Evaluate uncertain information.",
                (
                    "Seek corroboration before acting "
                    "when evidence is incomplete."
                ),
            ]
        ),
    )

    report = compare_character_material_change(
        old,
        new,
    )

    assert_true(
        report.material_change is True,
        "Grounded Character guidance change must "
        "be material.",
    )

    assert_true(
        "likely_behavior"
        in report.changed_fields,
        "Changed grounded behavior guidance was "
        "not reported.",
    )

    print(
        "Grounded Character guidance change: PASS"
    )


def test_character_identity_mismatch_fails_closed() -> None:
    old = character_result(
        character_id="character_000",
        character_name="Elim Garak",
    )

    new = character_result(
        character_id="character_001",
        character_name="Elim Garak",
    )

    assert_raises(
        ValueError,
        lambda: compare_character_material_change(
            old,
            new,
        ),
        "different character IDs",
    )

    print(
        "Character ID mismatch fail-closed: PASS"
    )


def test_character_name_mismatch_fails_closed() -> None:
    old = character_result(
        character_id="character_000",
        character_name="Elim Garak",
    )

    new = character_result(
        character_id="character_000",
        character_name="Benjamin Sisko",
        analysis=character_analysis(
            character_name="Benjamin Sisko"
        ),
    )

    assert_raises(
        ValueError,
        lambda: compare_character_material_change(
            old,
            new,
        ),
        "different character identities",
    )

    print(
        "Character identity mismatch fail-closed: PASS"
    )


def test_analysis_identity_mismatch_fails_closed() -> None:
    old = character_result(
        grounded=True,
    )

    new = character_result(
        grounded=True,
        analysis=character_analysis(
            character_name="Benjamin Sisko"
        ),
    )

    assert_raises(
        ValueError,
        lambda: compare_character_material_change(
            old,
            new,
        ),
        "analysis identity does not match",
    )

    print(
        "Character analysis identity fail-closed: PASS"
    )


def test_whitespace_normalization_not_material() -> None:
    old = character_result(
        grounded=True,
        analysis=character_analysis(
            story_role="Information broker",
            likely_behavior=[
                "Evaluate uncertain information."
            ],
        ),
    )

    new = character_result(
        grounded=True,
        analysis=character_analysis(
            story_role="  Information broker  ",
            likely_behavior=[
                "  Evaluate uncertain information.  "
            ],
        ),
    )

    report = compare_character_material_change(
        old,
        new,
    )

    assert_true(
        report.material_change is False,
        "Whitespace-only Character differences "
        "should not be material.",
    )

    print(
        "Whitespace normalization exclusion: PASS"
    )


def test_era_style_no_change() -> None:
    report = compare_era_style_material_change(
        old_story_era=(
            "Deep Space Nine, pre-Dominion War"
        ),
        old_constraints=[
            "Use restrained dramatic pacing.",
            "Avoid generic modern sci-fi phrasing.",
        ],
        new_story_era=(
            "Deep Space Nine, pre-Dominion War"
        ),
        new_constraints=[
            "Use restrained dramatic pacing.",
            "Avoid generic modern sci-fi phrasing.",
        ],
    )

    assert_true(
        report.material_change is False,
        "Identical era/style authority should not "
        "be material.",
    )

    print(
        "Era/style no-change detection: PASS"
    )


def test_era_style_change_is_material() -> None:
    report = compare_era_style_material_change(
        old_story_era=(
            "Deep Space Nine, pre-Dominion War"
        ),
        old_constraints=[
            "Use restrained dramatic pacing."
        ],
        new_story_era=(
            "Deep Space Nine, pre-Dominion War"
        ),
        new_constraints=[
            "Use restrained dramatic pacing.",
            (
                "Favor DS9-era conversational tension "
                "over action spectacle."
            ),
        ],
    )

    assert_true(
        report.material_change is True,
        "Era/style constraint change must "
        "be material.",
    )

    assert_true(
        "era_style_constraints"
        in report.changed_fields,
        "Era/style constraint change was not "
        "reported.",
    )

    print(
        "Era/style material change: PASS"
    )


def main() -> None:
    print(
        "\n--- Artifact Material Change Regression Test ---"
    )

    test_supported_lore_summary_change_is_material()
    test_unsupported_lore_summary_change_is_not_material()
    test_lore_support_transition_is_material()
    test_lore_question_change_is_material()
    test_lore_identity_mismatch_fails_closed()
    test_impossible_lore_authority_state_fails_closed()

    test_ungrounded_character_analysis_change_not_material()
    test_character_grounding_transition_is_material()
    test_grounded_character_guidance_change_is_material()
    test_character_identity_mismatch_fails_closed()
    test_character_name_mismatch_fails_closed()
    test_analysis_identity_mismatch_fails_closed()
    test_whitespace_normalization_not_material()

    test_era_style_no_change()
    test_era_style_change_is_material()

    print(
        "\n============================================"
    )
    print(
        "ARTIFACT MATERIAL CHANGE REGRESSION: PASS"
    )
    print(
        "============================================"
    )


if __name__ == "__main__":
    main()
