"""
Pure-Python regression coverage for
workflows.character_evidence_refresh_checkpoint.

Run from the project root:

    python -m workflows.character_evidence_refresh_checkpoint_regression_test

No provider calls are made.
The real Writer's Room checkpoint is never touched.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from pydantic import BaseModel, Field

from workflows.character_evidence_refresh import (
    refresh_character_evidence,
)
from workflows.character_evidence_refresh_checkpoint import (
    persist_character_refresh_commit,
    sha256_bytes,
)
from workflows.character_evidence_refresh_commit import (
    commit_character_refresh_to_state,
)
from workflows.evaluation_resolution import (
    EvaluationResolutionTarget,
)


# ---------------------------------------------------------
# Lightweight Models
# ---------------------------------------------------------


class FakeResearchPlan(BaseModel):
    character_name: str
    story_context: str
    story_era: str

    canon_questions: list[str] = Field(
        default_factory=list
    )

    voice_questions: list[str] = Field(
        default_factory=list
    )


class FakeCharacterAnalysis(BaseModel):
    character_name: str
    story_role: str = ""

    motivations: list[str] = Field(
        default_factory=list
    )

    likely_behavior: list[str] = Field(
        default_factory=list
    )

    unlikely_behavior: list[str] = Field(
        default_factory=list
    )

    relationships: list[str] = Field(
        default_factory=list
    )

    dramatic_tension: str = ""

    speech_patterns: list[str] = Field(
        default_factory=list
    )

    preferred_vocabulary: list[str] = Field(
        default_factory=list
    )

    communication_style: list[str] = Field(
        default_factory=list
    )

    behavioral_constraints: list[str] = Field(
        default_factory=list
    )

    era_language_notes: list[str] = Field(
        default_factory=list
    )

    out_of_character_risks: list[str] = Field(
        default_factory=list
    )


class FakeClaimAudit(BaseModel):
    marker: str


class FakeCharacterAudit(BaseModel):
    grounded: bool
    reason: str = ""


class FakeWriterRoomCharacterResult(BaseModel):
    character_id: str
    character_name: str
    research_plan: FakeResearchPlan

    canon_research: list[str] = Field(
        default_factory=list
    )

    voice_research: list[str] = Field(
        default_factory=list
    )

    analysis: FakeCharacterAnalysis
    claim_audit: FakeClaimAudit
    audit: FakeCharacterAudit


class FakeCharacterProductionResult(BaseModel):
    research_plan: FakeResearchPlan

    combined_canon_research: list[str] = Field(
        default_factory=list
    )

    voice_research: list[str] = Field(
        default_factory=list
    )

    character_analysis: FakeCharacterAnalysis
    claim_audit: FakeClaimAudit
    analysis_audit: FakeCharacterAudit


class FakeLoreResult(BaseModel):
    research_id: str
    marker: str


class FakeScriptResult(BaseModel):
    marker: str


class FakeContinuityResult(BaseModel):
    marker: str


class FakeEvaluationResult(BaseModel):
    marker: str


class FakeLedger(BaseModel):
    entries: list[str] = Field(
        default_factory=list
    )


class FakeWriterRoomState(BaseModel):
    story_plan: str

    lore_results: list[
        FakeLoreResult
    ]

    character_results: list[
        FakeWriterRoomCharacterResult
    ]

    script_result: (
        FakeScriptResult
        | None
    )

    story_state_ledger: FakeLedger

    continuity_result: (
        FakeContinuityResult
        | None
    )

    evaluation_result: (
        FakeEvaluationResult
        | None
    )


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------


def assert_true(
    condition: bool,
    message: str,
) -> None:
    if not condition:
        raise AssertionError(
            message
        )


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


def character_target() -> EvaluationResolutionTarget:
    return EvaluationResolutionTarget(
        criterion_id=(
            "character_agent_guidance_"
            "elim_garak"
        ),
        category="character_consistency",
        owner="character",
        action="refresh_character_evidence",
        artifact_id="character_000",
        subject_name="Elim Garak",
        reason=(
            "Regression Character blocker."
        ),
    )


def base_analysis(
    *,
    character_name: str = "Elim Garak",
    likely_behavior: list[str] | None = None,
) -> FakeCharacterAnalysis:
    return FakeCharacterAnalysis(
        character_name=character_name,
        story_role="Information broker",
        motivations=[
            "Protect his interests.",
        ],
        likely_behavior=(
            likely_behavior
            if likely_behavior is not None
            else [
                (
                    "Evaluate uncertain information "
                    "before acting."
                )
            ]
        ),
        unlikely_behavior=[
            (
                "Treat an allegation as proven "
                "without corroboration."
            )
        ],
        relationships=[
            "Maintains layered trust with allies.",
        ],
        dramatic_tension=(
            "He must judge whether uncertain "
            "information deserves action."
        ),
        speech_patterns=[
            "Uses implication and understatement.",
        ],
        preferred_vocabulary=[
            "my dear doctor",
        ],
        communication_style=[
            "Indirect but precise.",
        ],
        behavioral_constraints=[
            (
                "Do not make him needlessly "
                "transparent."
            )
        ],
        era_language_notes=[
            "Keep dialogue consistent with DS9.",
        ],
        out_of_character_risks=[
            "Blunt exposition without subtext.",
        ],
    )


def character_artifact(
    *,
    character_id: str,
    character_name: str,
    story_context: str,
    story_era: str,
    analysis: FakeCharacterAnalysis,
    marker: str,
) -> FakeWriterRoomCharacterResult:
    return FakeWriterRoomCharacterResult(
        character_id=character_id,
        character_name=character_name,
        research_plan=FakeResearchPlan(
            character_name=character_name,
            story_context=story_context,
            story_era=story_era,
            canon_questions=[
                f"{marker} canon question",
            ],
            voice_questions=[
                f"{marker} voice question",
            ],
        ),
        canon_research=[
            f"{marker} canon research",
        ],
        voice_research=[
            f"{marker} voice research",
        ],
        analysis=analysis,
        claim_audit=FakeClaimAudit(
            marker=f"{marker}-claim-audit"
        ),
        audit=FakeCharacterAudit(
            grounded=True,
            reason=f"{marker} aggregate audit",
        ),
    )


def build_state() -> FakeWriterRoomState:
    era = (
        "Deep Space Nine, pre-Dominion War"
    )

    return FakeWriterRoomState(
        story_plan="StoryPlan marker",
        lore_results=[
            FakeLoreResult(
                research_id="lore_000",
                marker="lore-zero",
            ),
            FakeLoreResult(
                research_id="lore_001",
                marker="lore-one",
            ),
        ],
        character_results=[
            character_artifact(
                character_id="character_000",
                character_name="Elim Garak",
                story_context=(
                    "Garak evaluates a damaged data rod "
                    "containing an uncertain warning."
                ),
                story_era=era,
                analysis=base_analysis(),
                marker="existing-garak",
            ),
            character_artifact(
                character_id="character_001",
                character_name="Benjamin Sisko",
                story_context=(
                    "Sisko evaluates station security."
                ),
                story_era=era,
                analysis=base_analysis(
                    character_name=(
                        "Benjamin Sisko"
                    )
                ),
                marker="existing-sisko",
            ),
        ],
        script_result=FakeScriptResult(
            marker="script"
        ),
        story_state_ledger=FakeLedger(
            entries=[
                "door open",
                "rod on table",
            ]
        ),
        continuity_result=FakeContinuityResult(
            marker="continuity"
        ),
        evaluation_result=FakeEvaluationResult(
            marker="evaluation"
        ),
    )


def production_result(
    *,
    character_name: str,
    story_context: str,
    story_era: str,
    analysis: FakeCharacterAnalysis,
) -> FakeCharacterProductionResult:
    return FakeCharacterProductionResult(
        research_plan=FakeResearchPlan(
            character_name=character_name,
            story_context=story_context,
            story_era=story_era,
            canon_questions=[
                (
                    "Fresh canon question about "
                    "uncertain intelligence."
                )
            ],
            voice_questions=[
                (
                    "Fresh voice question about "
                    "signaling doubt."
                )
            ],
        ),
        combined_canon_research=[
            "Fresh canon research.",
        ],
        voice_research=[
            "Fresh voice research.",
        ],
        character_analysis=analysis,
        claim_audit=FakeClaimAudit(
            marker="fresh-claim-audit"
        ),
        analysis_audit=FakeCharacterAudit(
            grounded=True,
            reason="Fresh aggregate audit.",
        ),
    )


def serialize_state(
    state: FakeWriterRoomState,
) -> bytes:
    return (
        state.model_dump_json(
            indent=2
        )
        + "\n"
    ).encode(
        "utf-8"
    )


def write_checkpoint(
    checkpoint_path: Path,
    state: FakeWriterRoomState,
) -> bytes:
    checkpoint_bytes = serialize_state(
        state
    )

    checkpoint_path.write_bytes(
        checkpoint_bytes
    )

    return checkpoint_bytes


def build_unchanged_commit(
    state: FakeWriterRoomState,
):
    current = state.character_results[0]

    def fake_runner(
        *,
        character_name: str,
        story_context: str,
        story_era: str,
        debug: bool,
    ) -> FakeCharacterProductionResult:
        return production_result(
            character_name=character_name,
            story_context=story_context,
            story_era=story_era,
            analysis=base_analysis(
                character_name=character_name
            ),
        )

    refresh = refresh_character_evidence(
        target=character_target(),
        current_character_result=current,
        character_runner=fake_runner,
    )

    return commit_character_refresh_to_state(
        state=state,
        refresh_result=refresh,
    )


def build_material_commit(
    state: FakeWriterRoomState,
):
    current = state.character_results[0]

    changed_analysis = base_analysis(
        character_name=(
            current.character_name
        ),
        likely_behavior=[
            (
                "Evaluate uncertain information "
                "before acting."
            ),
            (
                "Seek corroboration before trusting "
                "a strategically important warning."
            ),
        ],
    )

    def fake_runner(
        *,
        character_name: str,
        story_context: str,
        story_era: str,
        debug: bool,
    ) -> FakeCharacterProductionResult:
        return production_result(
            character_name=character_name,
            story_context=story_context,
            story_era=story_era,
            analysis=changed_analysis,
        )

    refresh = refresh_character_evidence(
        target=character_target(),
        current_character_result=current,
        character_runner=fake_runner,
    )

    return commit_character_refresh_to_state(
        state=state,
        refresh_result=refresh,
    )


# ---------------------------------------------------------
# Regression Cases
# ---------------------------------------------------------


def test_unchanged_commit_persists_atomically() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(
            temp_dir
        )

        checkpoint_path = (
            temp_path
            / "writers_room.json"
        )

        state = build_state()

        original_bytes = write_checkpoint(
            checkpoint_path,
            state,
        )

        commit_result = (
            build_unchanged_commit(
                state
            )
        )

        persistence = (
            persist_character_refresh_commit(
                checkpoint_path=(
                    checkpoint_path
                ),
                original_checkpoint_bytes=(
                    original_bytes
                ),
                commit_result=(
                    commit_result
                ),
            )
        )

        backup_path = Path(
            persistence.backup_path
        )

        assert_true(
            persistence.persistence_status
            == "saved",
            "Unchanged Character commit was not "
            "persisted.",
        )

        assert_true(
            persistence.commit_status
            == (
                "committed_without_invalidation"
            ),
            "Unchanged Character persistence lost "
            "commit status.",
        )

        assert_true(
            persistence.invalidated_artifacts
            == [],
            "Unchanged Character persistence "
            "incorrectly recorded invalidations.",
        )

        assert_true(
            persistence.character_id
            == "character_000",
            "Character persistence lost character_id.",
        )

        assert_true(
            persistence.character_name
            == "Elim Garak",
            "Character persistence lost character_name.",
        )

        assert_true(
            backup_path.exists(),
            "Character checkpoint backup was not "
            "created.",
        )

        assert_true(
            backup_path.read_bytes()
            == original_bytes,
            "Character checkpoint backup does not "
            "preserve exact original bytes.",
        )

        saved_state = (
            FakeWriterRoomState
            .model_validate_json(
                checkpoint_path.read_text(
                    encoding="utf-8"
                )
            )
        )

        assert_true(
            saved_state.character_results[0]
            .research_plan
            .canon_questions
            != state.character_results[0]
            .research_plan
            .canon_questions,
            "Refreshed Character provenance was "
            "not saved.",
        )

        assert_true(
            saved_state.script_result
            == state.script_result,
            "Unchanged Character persistence changed "
            "script_result.",
        )

        assert_true(
            saved_state.continuity_result
            == state.continuity_result,
            "Unchanged Character persistence changed "
            "continuity_result.",
        )

        assert_true(
            saved_state.story_state_ledger
            == state.story_state_ledger,
            "Unchanged Character persistence changed "
            "story-state ledger.",
        )

        assert_true(
            saved_state.evaluation_result
            == state.evaluation_result,
            "Unchanged Character persistence changed "
            "evaluation_result.",
        )

        assert_true(
            saved_state.lore_results
            == state.lore_results,
            "Unchanged Character persistence changed "
            "Lore artifacts.",
        )

        assert_true(
            saved_state.character_results[1]
            == state.character_results[1],
            "Unchanged Character persistence changed "
            "sibling Character artifact.",
        )

        assert_true(
            persistence.original_sha256
            == sha256_bytes(
                original_bytes
            ),
            "Original Character checkpoint hash "
            "is wrong.",
        )

        assert_true(
            persistence.saved_sha256
            == sha256_bytes(
                checkpoint_path.read_bytes()
            ),
            "Saved Character checkpoint hash "
            "is wrong.",
        )

    print(
        "Unchanged Character checkpoint "
        "persistence: PASS"
    )


def test_material_commit_persists_invalidations() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(
            temp_dir
        )

        checkpoint_path = (
            temp_path
            / "writers_room.json"
        )

        state = build_state()

        original_bytes = write_checkpoint(
            checkpoint_path,
            state,
        )

        commit_result = (
            build_material_commit(
                state
            )
        )

        persistence = (
            persist_character_refresh_commit(
                checkpoint_path=(
                    checkpoint_path
                ),
                original_checkpoint_bytes=(
                    original_bytes
                ),
                commit_result=(
                    commit_result
                ),
            )
        )

        saved_state = (
            persistence.saved_state
        )

        assert_true(
            persistence.commit_status
            == "committed_with_invalidation",
            "Material Character persistence lost "
            "commit status.",
        )

        assert_true(
            persistence.invalidated_artifacts
            == [
                "script_result",
                "continuity_result",
                "story_state_ledger",
                "evaluation_result",
            ],
            "Material Character persistence "
            "recorded the wrong invalidation set.",
        )

        assert_true(
            saved_state.character_results[0]
            .analysis
            .likely_behavior
            != state.character_results[0]
            .analysis
            .likely_behavior,
            "Material Character candidate was "
            "not saved.",
        )

        assert_true(
            saved_state.script_result is None,
            "Material Character persistence did "
            "not clear script_result.",
        )

        assert_true(
            saved_state.continuity_result is None,
            "Material Character persistence did "
            "not clear continuity_result.",
        )

        assert_true(
            saved_state.evaluation_result is None,
            "Material Character persistence did "
            "not clear evaluation_result.",
        )

        assert_true(
            saved_state.story_state_ledger.entries
            == [],
            "Material Character persistence did "
            "not reset story-state ledger.",
        )

        assert_true(
            saved_state.story_plan
            == state.story_plan,
            "Material Character persistence changed "
            "StoryPlan.",
        )

        assert_true(
            saved_state.lore_results
            == state.lore_results,
            "Material Character persistence changed "
            "Lore artifacts.",
        )

        assert_true(
            saved_state.character_results[1]
            == state.character_results[1],
            "Material Character persistence changed "
            "sibling Character artifact.",
        )

        backup_path = Path(
            persistence.backup_path
        )

        assert_true(
            backup_path.read_bytes()
            == original_bytes,
            "Material Character persistence backup "
            "does not match original checkpoint.",
        )

    print(
        "Material Character checkpoint "
        "persistence: PASS"
    )


def test_stale_checkpoint_fails_closed() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(
            temp_dir
        )

        checkpoint_path = (
            temp_path
            / "writers_room.json"
        )

        state = build_state()

        original_bytes = write_checkpoint(
            checkpoint_path,
            state,
        )

        commit_result = (
            build_material_commit(
                state
            )
        )

        newer_state = state.model_copy(
            update={
                "story_plan": (
                    "Newer StoryPlan state"
                )
            }
        )

        newer_bytes = serialize_state(
            newer_state
        )

        checkpoint_path.write_bytes(
            newer_bytes
        )

        assert_raises(
            ValueError,
            lambda: persist_character_refresh_commit(
                checkpoint_path=(
                    checkpoint_path
                ),
                original_checkpoint_bytes=(
                    original_bytes
                ),
                commit_result=(
                    commit_result
                ),
            ),
            (
                "changed after the Character "
                "evidence refresh began"
            ),
        )

        assert_true(
            checkpoint_path.read_bytes()
            == newer_bytes,
            "Stale Character persistence attempt "
            "overwrote newer checkpoint state.",
        )

    print(
        "Stale Character checkpoint fail-closed: PASS"
    )


def test_missing_checkpoint_fails_closed() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(
            temp_dir
        )

        checkpoint_path = (
            temp_path
            / "missing.json"
        )

        state = build_state()

        original_bytes = serialize_state(
            state
        )

        commit_result = (
            build_material_commit(
                state
            )
        )

        assert_raises(
            FileNotFoundError,
            lambda: persist_character_refresh_commit(
                checkpoint_path=(
                    checkpoint_path
                ),
                original_checkpoint_bytes=(
                    original_bytes
                ),
                commit_result=(
                    commit_result
                ),
            ),
            "checkpoint disappeared",
        )

    print(
        "Missing Character checkpoint "
        "fail-closed: PASS"
    )


def test_non_path_checkpoint_fails_closed() -> None:
    state = build_state()

    commit_result = (
        build_material_commit(
            state
        )
    )

    assert_raises(
        TypeError,
        lambda: persist_character_refresh_commit(
            checkpoint_path="writers_room.json",
            original_checkpoint_bytes=(
                serialize_state(
                    state
                )
            ),
            commit_result=(
                commit_result
            ),
        ),
        "pathlib.Path",
    )

    print(
        "Non-Path Character checkpoint "
        "fail-closed: PASS"
    )


def test_empty_original_bytes_fail_closed() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(
            temp_dir
        )

        checkpoint_path = (
            temp_path
            / "writers_room.json"
        )

        state = build_state()

        write_checkpoint(
            checkpoint_path,
            state,
        )

        commit_result = (
            build_material_commit(
                state
            )
        )

        assert_raises(
            ValueError,
            lambda: persist_character_refresh_commit(
                checkpoint_path=(
                    checkpoint_path
                ),
                original_checkpoint_bytes=b"",
                commit_result=(
                    commit_result
                ),
            ),
            "cannot be empty",
        )

    print(
        "Empty Character checkpoint bytes "
        "fail-closed: PASS"
    )


def test_wrong_commit_result_type_fails_closed() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(
            temp_dir
        )

        checkpoint_path = (
            temp_path
            / "writers_room.json"
        )

        state = build_state()

        original_bytes = write_checkpoint(
            checkpoint_path,
            state,
        )

        assert_raises(
            TypeError,
            lambda: persist_character_refresh_commit(
                checkpoint_path=(
                    checkpoint_path
                ),
                original_checkpoint_bytes=(
                    original_bytes
                ),
                commit_result=object(),
            ),
            "CharacterRefreshCommitResult",
        )

    print(
        "Wrong Character commit-result type "
        "fail-closed: PASS"
    )


def test_tampered_commit_contract_fails_closed() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(
            temp_dir
        )

        checkpoint_path = (
            temp_path
            / "writers_room.json"
        )

        state = build_state()

        original_bytes = write_checkpoint(
            checkpoint_path,
            state,
        )

        commit_result = (
            build_material_commit(
                state
            )
        )

        tampered = (
            commit_result.model_copy(
                update={
                    "invalidated_artifacts": [
                        "script_result",
                    ]
                }
            )
        )

        assert_raises(
            ValueError,
            lambda: persist_character_refresh_commit(
                checkpoint_path=(
                    checkpoint_path
                ),
                original_checkpoint_bytes=(
                    original_bytes
                ),
                commit_result=(
                    tampered
                ),
            ),
            "unexpected invalidation set",
        )

        assert_true(
            checkpoint_path.read_bytes()
            == original_bytes,
            "Tampered Character commit changed "
            "checkpoint.",
        )

    print(
        "Tampered Character commit contract "
        "fail-closed: PASS"
    )


def test_material_saved_hash_changes() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(
            temp_dir
        )

        checkpoint_path = (
            temp_path
            / "writers_room.json"
        )

        state = build_state()

        original_bytes = write_checkpoint(
            checkpoint_path,
            state,
        )

        commit_result = (
            build_material_commit(
                state
            )
        )

        persistence = (
            persist_character_refresh_commit(
                checkpoint_path=(
                    checkpoint_path
                ),
                original_checkpoint_bytes=(
                    original_bytes
                ),
                commit_result=(
                    commit_result
                ),
            )
        )

        assert_true(
            persistence.original_sha256
            != persistence.saved_sha256,
            "Material Character commit should "
            "change checkpoint content hash.",
        )

    print(
        "Material Character checkpoint hash "
        "change: PASS"
    )


def main() -> None:
    print(
        "\n--- Character Evidence Refresh Checkpoint "
        "Regression Test ---"
    )

    test_unchanged_commit_persists_atomically()
    test_material_commit_persists_invalidations()
    test_stale_checkpoint_fails_closed()
    test_missing_checkpoint_fails_closed()
    test_non_path_checkpoint_fails_closed()
    test_empty_original_bytes_fail_closed()
    test_wrong_commit_result_type_fails_closed()
    test_tampered_commit_contract_fails_closed()
    test_material_saved_hash_changes()

    print(
        "\n============================================"
    )
    print(
        "CHARACTER EVIDENCE REFRESH CHECKPOINT "
        "REGRESSION: PASS"
    )
    print(
        "============================================"
    )


if __name__ == "__main__":
    main()
