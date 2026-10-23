"""
Pure-Python regression coverage for workflows.evidence_refresh.

Run from the project root:

    python -m workflows.evidence_refresh_regression_test

No provider calls are made.
No checkpoint files are changed.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from workflows.evaluation_resolution import (
    EvaluationResolutionTarget,
)
from workflows.evidence_refresh import (
    refresh_lore_evidence,
)


# ---------------------------------------------------------
# Lightweight Lore Models
#
# These mimic only the Writer's Room Lore contract needed
# by evidence_refresh. They avoid importing the live Lore
# Agent or provider-backed modules.
# ---------------------------------------------------------


class FakeLoreResponse(BaseModel):
    premise_supported: bool
    evidence_supported: bool
    summary: str


class FakeLoreResult(BaseModel):
    research_id: str = Field(
        min_length=1,
    )

    research_question: str = Field(
        min_length=1,
    )

    response: FakeLoreResponse


# ---------------------------------------------------------
# Test Helpers
# ---------------------------------------------------------


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


def lore_target(
    *,
    artifact_id: str = "lore_000",
) -> EvaluationResolutionTarget:
    return EvaluationResolutionTarget(
        criterion_id=(
            "lore_agent_evidence_"
            f"{artifact_id}"
        ),
        category="canon_grounding",
        owner="lore",
        action="refresh_lore_evidence",
        artifact_id=artifact_id,
        subject_name=None,
        reason=(
            "Regression Lore evidence blocker."
        ),
    )


def current_lore_result(
    *,
    research_id: str = "lore_000",
    evidence_supported: bool = False,
    premise_supported: bool = True,
    summary: str = (
        "The available evidence does not "
        "establish a direct encounter."
    ),
) -> FakeLoreResult:
    return FakeLoreResult(
        research_id=research_id,
        research_question=(
            "When does Garak directly encounter "
            "the Founders?"
        ),
        response=FakeLoreResponse(
            premise_supported=(
                premise_supported
            ),
            evidence_supported=(
                evidence_supported
            ),
            summary=summary,
        ),
    )


# ---------------------------------------------------------
# Regression Cases
# ---------------------------------------------------------


def test_unchanged_refresh_preserves_downstream() -> None:
    current = current_lore_result(
        evidence_supported=False,
        summary=(
            "No direct encounter was established."
        ),
    )

    calls: list[tuple[str, bool]] = []

    def fake_runner(
        *,
        query: str,
        debug: bool,
    ) -> FakeLoreResponse:
        calls.append(
            (
                query,
                debug,
            )
        )

        return FakeLoreResponse(
            premise_supported=True,
            evidence_supported=False,
            summary=(
                "Retrieved material still does not "
                "establish a direct encounter."
            ),
        )

    result = refresh_lore_evidence(
        target=lore_target(),
        current_lore_result=current,
        lore_runner=fake_runner,
        debug=False,
    )

    assert_true(
        len(calls) == 1,
        "Lore refresh runner should be called "
        "exactly once.",
    )

    assert_true(
        calls[0][0]
        == current.research_question,
        "Lore refresh did not preserve the exact "
        "research question.",
    )

    assert_true(
        calls[0][1] is False,
        "Lore refresh changed debug routing.",
    )

    assert_true(
        result.refresh_status
        == "unchanged",
        "Unsupported wording-only Lore refresh "
        "should remain unchanged.",
    )

    assert_true(
        result.material_change.material_change
        is False,
        "Unsupported wording-only Lore refresh "
        "was incorrectly material.",
    )

    assert_true(
        result.invalidation_plan is None,
        "Unchanged Lore refresh should not produce "
        "an invalidation plan.",
    )

    assert_true(
        result.candidate_artifact.research_id
        == current.research_id,
        "Lore refresh changed stable research ID.",
    )

    assert_true(
        result.candidate_artifact.research_question
        == current.research_question,
        "Lore refresh changed the authoritative "
        "research question.",
    )

    print(
        "Unchanged Lore refresh preservation: PASS"
    )


def test_material_refresh_builds_invalidation_plan() -> None:
    current = current_lore_result(
        evidence_supported=False,
        summary=(
            "No direct encounter was established."
        ),
    )

    def fake_runner(
        *,
        query: str,
        debug: bool,
    ) -> FakeLoreResponse:
        return FakeLoreResponse(
            premise_supported=True,
            evidence_supported=True,
            summary=(
                "New evidence establishes the "
                "requested canon fact."
            ),
        )

    result = refresh_lore_evidence(
        target=lore_target(),
        current_lore_result=current,
        lore_runner=fake_runner,
    )

    assert_true(
        result.refresh_status
        == "material_change",
        "Lore support transition should be "
        "material.",
    )

    assert_true(
        result.material_change.material_change
        is True,
        "Lore support transition did not register "
        "as material.",
    )

    assert_true(
        result.invalidation_plan is not None,
        "Material Lore refresh should attach an "
        "invalidation plan.",
    )

    assert_true(
        result.invalidation_plan
        .invalidated_artifacts
        == [
            "script_result",
            "continuity_result",
            "story_state_ledger",
            "evaluation_result",
        ],
        "Material Lore refresh produced the wrong "
        "downstream invalidation set.",
    )

    assert_true(
        result.invalidation_plan
        .requires_ledger_rebuild
        is True,
        "Material Lore refresh should require a "
        "Continuity ledger rebuild.",
    )

    print(
        "Material Lore refresh invalidation: PASS"
    )


def test_refresh_preserves_python_owned_identity() -> None:
    current = current_lore_result(
        research_id="lore_000",
    )

    def fake_runner(
        *,
        query: str,
        debug: bool,
    ) -> FakeLoreResponse:
        return FakeLoreResponse(
            premise_supported=True,
            evidence_supported=True,
            summary=(
                "Supported replacement summary."
            ),
        )

    result = refresh_lore_evidence(
        target=lore_target(
            artifact_id="lore_000"
        ),
        current_lore_result=current,
        lore_runner=fake_runner,
    )

    assert_true(
        result.research_id == "lore_000",
        "Refresh result lost stable research ID.",
    )

    assert_true(
        result.research_question
        == current.research_question,
        "Refresh result lost the authoritative "
        "research question.",
    )

    assert_true(
        result.candidate_artifact.research_id
        == "lore_000",
        "Candidate artifact changed research ID.",
    )

    assert_true(
        result.candidate_artifact.research_question
        == current.research_question,
        "Candidate artifact changed research "
        "question.",
    )

    print(
        "Lore refresh identity preservation: PASS"
    )


def test_target_artifact_mismatch_fails_closed() -> None:
    current = current_lore_result(
        research_id="lore_000"
    )

    assert_raises(
        ValueError,
        lambda: refresh_lore_evidence(
            target=lore_target(
                artifact_id="lore_999"
            ),
            current_lore_result=current,
            lore_runner=(
                lambda **kwargs: FakeLoreResponse(
                    premise_supported=True,
                    evidence_supported=False,
                    summary="Unused.",
                )
            ),
        ),
        "does not match the supplied "
        "Writer's Room Lore artifact",
    )

    print(
        "Lore target/artifact mismatch "
        "fail-closed: PASS"
    )


def test_non_lore_target_fails_closed() -> None:
    bad_target = EvaluationResolutionTarget(
        criterion_id=(
            "character_agent_guidance_"
            "elim_garak"
        ),
        category="character_consistency",
        owner="character",
        action="refresh_character_evidence",
        artifact_id="character_000",
        subject_name="Elim Garak",
        reason="Wrong refresh family.",
    )

    assert_raises(
        ValueError,
        lambda: refresh_lore_evidence(
            target=bad_target,
            current_lore_result=(
                current_lore_result()
            ),
            lore_runner=(
                lambda **kwargs: FakeLoreResponse(
                    premise_supported=True,
                    evidence_supported=False,
                    summary="Unused.",
                )
            ),
        ),
        "category='canon_grounding'",
    )

    print(
        "Non-Lore target fail-closed: PASS"
    )


def test_none_runner_result_fails_closed() -> None:
    current = current_lore_result()

    def fake_runner(
        *,
        query: str,
        debug: bool,
    ):
        return None

    assert_raises(
        ValueError,
        lambda: refresh_lore_evidence(
            target=lore_target(),
            current_lore_result=current,
            lore_runner=fake_runner,
        ),
        "returned None",
    )

    print(
        "None Lore response fail-closed: PASS"
    )


def test_wrong_runner_signature_fails_closed() -> None:
    current = current_lore_result()

    def bad_runner(
        question: str,
    ) -> FakeLoreResponse:
        return FakeLoreResponse(
            premise_supported=True,
            evidence_supported=False,
            summary="Unused.",
        )

    assert_raises(
        TypeError,
        lambda: refresh_lore_evidence(
            target=lore_target(),
            current_lore_result=current,
            lore_runner=bad_runner,
        ),
        "frozen signature",
    )

    print(
        "Lore runner signature fail-closed: PASS"
    )


def test_non_pydantic_artifact_fails_closed() -> None:
    class FakePlainLore:
        def __init__(self) -> None:
            self.research_id = "lore_000"
            self.research_question = (
                "When does Garak directly encounter "
                "the Founders?"
            )
            self.response = FakeLoreResponse(
                premise_supported=True,
                evidence_supported=False,
                summary="Unsupported.",
            )

    plain_artifact = FakePlainLore()

    def fake_runner(
        *,
        query: str,
        debug: bool,
    ) -> FakeLoreResponse:
        return FakeLoreResponse(
            premise_supported=True,
            evidence_supported=False,
            summary="Still unsupported.",
        )

    assert_raises(
        TypeError,
        lambda: refresh_lore_evidence(
            target=lore_target(),
            current_lore_result=(
                plain_artifact
            ),
            lore_runner=fake_runner,
        ),
        "Pydantic BaseModel",
    )

    print(
        "Non-Pydantic artifact fail-closed: PASS"
    )


def test_invalid_refreshed_response_fails_closed() -> None:
    current = current_lore_result()

    def fake_runner(
        *,
        query: str,
        debug: bool,
    ):
        return {
            "premise_supported": True,
            # Intentionally missing evidence_supported
            "summary": "Malformed response.",
        }

    assert_raises(
        ValueError,
        lambda: refresh_lore_evidence(
            target=lore_target(),
            current_lore_result=current,
            lore_runner=fake_runner,
        ),
        "cannot be stored",
    )

    print(
        "Malformed Lore response fail-closed: PASS"
    )


def test_impossible_refreshed_authority_fails_closed() -> None:
    current = current_lore_result()

    def fake_runner(
        *,
        query: str,
        debug: bool,
    ) -> FakeLoreResponse:
        return FakeLoreResponse(
            premise_supported=False,
            evidence_supported=True,
            summary=(
                "Impossible authority state."
            ),
        )

    assert_raises(
        ValueError,
        lambda: refresh_lore_evidence(
            target=lore_target(),
            current_lore_result=current,
            lore_runner=fake_runner,
        ),
        "impossible authority state",
    )

    print(
        "Impossible refreshed Lore authority "
        "fail-closed: PASS"
    )


def main() -> None:
    print(
        "\n--- Evidence Refresh Regression Test ---"
    )

    test_unchanged_refresh_preserves_downstream()
    test_material_refresh_builds_invalidation_plan()
    test_refresh_preserves_python_owned_identity()
    test_target_artifact_mismatch_fails_closed()
    test_non_lore_target_fails_closed()
    test_none_runner_result_fails_closed()
    test_wrong_runner_signature_fails_closed()
    test_non_pydantic_artifact_fails_closed()
    test_invalid_refreshed_response_fails_closed()
    test_impossible_refreshed_authority_fails_closed()

    print(
        "\n============================================"
    )
    print(
        "EVIDENCE REFRESH REGRESSION: PASS"
    )
    print(
        "============================================"
    )


if __name__ == "__main__":
    main()
