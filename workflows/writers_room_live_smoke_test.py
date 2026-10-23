from pathlib import Path

from workflows.evidence_resolution_live_bridge import (
    resolve_live_evaluation_checkpoint,
)
from workflows.writers_room import (
    WritersRoomCharacterRequest,
    WritersRoomScriptCharacterRequest,
    WritersRoomScriptRequest,
    WritersRoomState,
    run_character_stage,
    run_continuity_stage,
    run_evaluation_stage,
    run_lore_stage,
    run_scriptwriter_stage,
    run_story_architect_stage,
)


# ---------------------------------------------------------
# Live Smoke-Test Helpers
# ---------------------------------------------------------


def print_stage(title: str):
    print("\n")
    print("=" * 60)
    print(title)
    print("=" * 60)


# ---------------------------------------------------------
# Live Smoke-Test Checkpointing
# ---------------------------------------------------------


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "runtime"
    / "checkpoints"
    / "writers_room_live_smoke.json"
)


def save_checkpoint(
    state: WritersRoomState,
    stage_name: str,
) -> None:
    """
    Persist the latest fully completed Writer's Room state.

    The checkpoint is written to a temporary file first and
    then atomically moved into place so an interrupted write
    cannot leave behind a partially written production state.
    """

    CHECKPOINT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = (
        CHECKPOINT_PATH.with_suffix(".tmp")
    )

    temporary_path.write_text(
        state.model_dump_json(
            indent=2
        ),
        encoding="utf-8",
    )

    temporary_path.replace(
        CHECKPOINT_PATH
    )

    print(
        "\n[Checkpoint] Saved after:",
        stage_name,
    )

    print(
        "[Checkpoint] Path:",
        CHECKPOINT_PATH,
    )


def load_checkpoint(
    expected_story_idea: str,
) -> WritersRoomState | None:
    """
    Load the most recent live-flight checkpoint.

    A checkpoint belonging to a different story idea is
    rejected rather than silently reused.
    """

    if not CHECKPOINT_PATH.exists():
        return None

    checkpoint_text = (
        CHECKPOINT_PATH.read_text(
            encoding="utf-8"
        )
    )

    try:
        state = (
            WritersRoomState.model_validate_json(
                checkpoint_text
            )
        )

    except Exception as error:
        raise ValueError(
            "Live Writer's Room checkpoint could not "
            "be loaded. The checkpoint may be damaged."
        ) from error

    if state.story_idea != expected_story_idea:
        raise ValueError(
            "Existing live Writer's Room checkpoint "
            "belongs to a different story idea. "
            "Delete the checkpoint before starting "
            "a new live scenario."
        )

    print(
        "\n[Checkpoint] Existing Writer's Room "
        "state loaded."
    )

    print(
        "[Checkpoint] Path:",
        CHECKPOINT_PATH,
    )

    return state


def checkpoint_stage(
    state: WritersRoomState,
) -> str:
    """
    Describe the latest fully completed production stage.
    """

    if state.evaluation_result is not None:
        return "Evaluation"

    if state.continuity_result is not None:
        return "Continuity"

    if state.script_result is not None:
        return "Scriptwriter"

    if state.character_results:
        return "Character"

    if state.lore_results:
        return "Lore"

    if state.story_plan is not None:
        return "Story Architect"

    return "Initial State"


def evaluation_assessment(
    state: WritersRoomState,
):
    """
    Return the completed Evaluation assessment.
    """

    if state.evaluation_result is None:
        raise ValueError(
            "Evaluation assessment requested before "
            "Evaluation completed."
        )

    return (
        state
        .evaluation_result
        .production_review
        .review
        .assessment
    )


def print_evaluation_summary(
    state: WritersRoomState,
) -> None:
    """
    Print the current completed Evaluation verdict.
    """

    assessment = evaluation_assessment(
        state
    )

    print(
        "FINAL STATUS:",
        assessment.status.upper(),
    )

    print(
        "Pass count:",
        assessment.pass_count,
    )

    print(
        "Fail count:",
        assessment.fail_count,
    )

    print(
        "Insufficient evidence:",
        assessment.insufficient_evidence_count,
    )

    print(
        "Blocking criteria:",
        assessment.blocking_criterion_ids,
    )


# ---------------------------------------------------------
# Main Live Flight
# ---------------------------------------------------------


def main():
    # -----------------------------------------------------
    # Initial User Story Idea
    #
    # Keep the first live flight intentionally narrow.
    # We want to test orchestration, not stress every
    # creative boundary simultaneously.
    # -----------------------------------------------------

    story_idea = (
        "During Star Trek: Deep Space Nine, after Garak has "
        "already encountered the Founders but before the "
        "Dominion War begins, a fictional Cardassian courier "
        "secretly approaches him on Deep Space Nine. "
        "The courier possesses a damaged data rod allegedly "
        "containing partial Obsidian Order records related to "
        "possible Founder activity. Garak must decide whether "
        "the information is credible enough to bring to "
        "Benjamin Sisko. Do not assume Garak has current "
        "contacts inside Cardassian Central Command or the "
        "Obsidian Order."
    )

    # -----------------------------------------------------
    # Resume Previous Live Flight If Available
    # -----------------------------------------------------

    state = load_checkpoint(
        expected_story_idea=story_idea,
    )

    if state is None:
        state = WritersRoomState(
            story_idea=story_idea,
        )

        print(
            "\n[Checkpoint] No existing checkpoint. "
            "Starting a fresh live flight."
        )

    else:
        print(
            "[Checkpoint] Resuming after:",
            checkpoint_stage(state),
        )

    # =====================================================
    # STAGE 1 — STORY ARCHITECT
    # =====================================================

    print_stage(
        "LIVE STAGE 1/6 — STORY ARCHITECT"
    )

    if state.story_plan is None:
        state = run_story_architect_stage(
            state
        )

        save_checkpoint(
            state=state,
            stage_name="Story Architect",
        )

    else:
        print(
            "[Resume] Story Architect already complete. "
            "Using checkpointed artifact."
        )

    print(
        "Working title:",
        state.story_plan.working_title,
    )

    print(
        "Lore questions:",
        len(
            state.story_plan.lore_questions
        ),
    )

    for index, question in enumerate(
        state.story_plan.lore_questions
    ):
        print(
            f"  [{index}] {question}"
        )

    # =====================================================
    # STAGE 2 — LORE
    # =====================================================

    print_stage(
        "LIVE STAGE 2/6 — LORE"
    )

    if not state.lore_results:
        state = run_lore_stage(
            state
        )

        save_checkpoint(
            state=state,
            stage_name="Lore",
        )

    else:
        print(
            "[Resume] Lore already complete. "
            "Using checkpointed artifacts."
        )

    print(
        "Lore results:",
        len(state.lore_results),
    )

    for result in state.lore_results:
        print(
            "\n",
            result.research_id,
            sep="",
        )

        print(
            "Question:",
            result.research_question,
        )

        print(
            "Evidence supported:",
            result.response.evidence_supported,
        )

        print(
            "Premise supported:",
            result.response.premise_supported,
        )

        print(
            "Summary:",
            result.response.summary,
        )

    # =====================================================
    # STAGE 3 — CHARACTER
    #
    # Explicitly research Garak only.
    #
    # The fictional courier does NOT enter canon Character
    # research.
    # =====================================================

    print_stage(
        "LIVE STAGE 3/6 — CHARACTER"
    )

    character_request = (
        WritersRoomCharacterRequest(
            character_name="Elim Garak",
            story_context=(
                state.story_plan.premise
                + "\n\n"
                + state.story_plan.central_conflict
            ),
            story_era=(
                state.story_plan.timeframe
            ),
        )
    )

    if not state.character_results:
        state = run_character_stage(
            state=state,
            requests=[
                character_request,
            ],
        )

        save_checkpoint(
            state=state,
            stage_name="Character",
        )

    else:
        print(
            "[Resume] Character already complete. "
            "Using checkpointed artifact."
        )

    garak_result = (
        state.character_results[0]
    )

    print(
        "Character:",
        garak_result.character_name,
    )

    print(
        "Character ID:",
        garak_result.character_id,
    )

    print(
        "Grounded:",
        garak_result.audit.grounded,
    )

    if garak_result.audit.issues:
        print(
            "\nCharacter grounding issues:"
        )

        for issue in garak_result.audit.issues:
            print(
                "-",
                issue,
            )

    # -----------------------------------------------------
    # Important production stop
    #
    # Scriptwriter is intentionally forbidden from treating
    # an ungrounded Character result as authoritative.
    #
    # If Character is not grounded, the live flight should
    # stop here rather than weakening that gate.
    # -----------------------------------------------------

    if not garak_result.audit.grounded:
        raise ValueError(
            "LIVE FLIGHT STOPPED: Character Agent "
            "returned an ungrounded Garak analysis. "
            "The Writer's Room correctly refuses to "
            "promote it into Scriptwriter guidance."
        )

    # =====================================================
    # STAGE 4 — SCRIPTWRITER
    # =====================================================

    print_stage(
        "LIVE STAGE 4/6 — SCRIPTWRITER"
    )

    script_request = WritersRoomScriptRequest(
        story_context=(
            state.story_plan.premise
            + "\n\nCentral conflict:\n"
            + state.story_plan.central_conflict
            + "\n\nStakes:\n"
            + state.story_plan.stakes
        ),
        scene_objective=(
            "The fictional Cardassian courier privately "
            "presents the damaged data rod and its allegation "
            "to Garak. Garak must assess whether the claim "
            "deserves further investigation without the scene "
            "assuming facts not established by supplied canon "
            "or Character evidence."
        ),
        story_era=(
            state.story_plan.timeframe
        ),
        characters=[
            WritersRoomScriptCharacterRequest(
                character_name="Elim Garak",
                character_result_id=(
                    garak_result.character_id
                ),
            ),
            WritersRoomScriptCharacterRequest(
                character_name=(
                    "Cardassian Courier"
                ),
                character_result_id=None,
            ),
        ],
        era_style_constraints=[
            (
                "Use restrained dramatic pacing and "
                "screenplay conventions appropriate to "
                "Star Trek: Deep Space Nine."
            ),
            (
                "Keep the scene focused on conversation, "
                "uncertainty, and character tension rather "
                "than action spectacle."
            ),
        ],
        scene_requirements=[
            (
                "The Cardassian Courier must present "
                "the damaged data rod to Garak."
            ),
            (
                "The courier must claim that the rod "
                "contains partial Obsidian Order records "
                "related to possible Founder activity."
            ),
            (
                "Garak must decide whether the information "
                "deserves further investigation."
            ),
        ],
        prohibited_assumptions=[
            (
                "Garak has current contacts inside "
                "Cardassian Central Command."
            ),
            (
                "Garak has current access to the "
                "Obsidian Order."
            ),
            (
                "The courier's allegation is proven true."
            ),
            (
                "The data rod's contents have already "
                "been independently verified."
            ),
        ],
    )

    if state.script_result is None:
        state = run_scriptwriter_stage(
            state=state,
            request=script_request,
        )

        save_checkpoint(
            state=state,
            stage_name="Scriptwriter",
        )

    else:
        print(
            "[Resume] Scriptwriter already complete. "
            "Using checkpointed scene."
        )

    print(
        "Scriptwriter grounding:",
        state.script_result.audit.grounded,
    )

    print(
        "\n--- GENERATED SCENE ---\n"
    )

    print(
        state.script_result.scene.scene_heading
    )

    print()

    print(
        state.script_result.scene.setting_description
    )

    print()

    print(
        state.script_result.scene.script_text
    )

    print(
        "\n--- END GENERATED SCENE ---"
    )

    # =====================================================
    # STAGE 5 — CONTINUITY
    # =====================================================

    print_stage(
        "LIVE STAGE 5/6 — CONTINUITY"
    )

    if state.continuity_result is None:
        state = run_continuity_stage(
            state=state,
            scene_index=0,
        )

        save_checkpoint(
            state=state,
            stage_name="Continuity",
        )

    else:
        print(
            "[Resume] Continuity already complete. "
            "Using checkpointed result and ledger."
        )

    continuity_status = (
        state
        .continuity_result
        .review
        .assessment
        .status
    )

    print(
        "Continuity status:",
        continuity_status,
    )

    print(
        "Continuity issues:",
        len(
            state
            .continuity_result
            .review
            .assessment
            .issues
        ),
    )

    print(
        "Persistent story states:",
        len(
            state
            .story_state_ledger
            .entries
        ),
    )

    for entry in (
        state.story_state_ledger.entries
    ):
        print(
            "-",
            entry.normalized_state,
        )

    # =====================================================
    # STAGE 6 — EVALUATION
    # =====================================================

    print_stage(
        "LIVE STAGE 6/6 — EVALUATION"
    )

    if state.evaluation_result is None:
        state = run_evaluation_stage(
            state
        )

        save_checkpoint(
            state=state,
            stage_name="Evaluation",
        )

    else:
        print(
            "[Resume] Evaluation already complete. "
            "Using checkpointed final review."
        )

    print_evaluation_summary(
        state
    )

    # =====================================================
    # BOUNDED EVIDENCE-RESOLUTION PASS
    #
    # Important:
    # - Runs at most once per process invocation.
    # - Supports the proven Lore + Character blocker
    #   families only.
    # - Resolves all upstream evidence first.
    # - Persists that combined upstream transaction once.
    # - If authority changed materially, downstream stages
    #   resume with their EXISTING per-stage checkpoints.
    # - The newly produced Evaluation is reported and the
    #   process stops; no recursive second resolution pass
    #   occurs in this invocation.
    # =====================================================

    assessment = evaluation_assessment(
        state
    )

    evidence_resolution_attempted = False
    evidence_resolution_outcome = None

    if assessment.status == "needs_evidence":
        evidence_resolution_attempted = True

        print_stage(
            "LIVE EVIDENCE RESOLUTION"
        )

        print(
            "Blocking criteria:",
            assessment.blocking_criterion_ids,
        )

        print(
            "\n[Evidence Resolution] Resolving all "
            "supported upstream blockers before any "
            "downstream stage is regenerated."
        )

        resolution = (
            resolve_live_evaluation_checkpoint(
                state=state,
                checkpoint_path=(
                    CHECKPOINT_PATH
                ),
                debug=False,
            )
        )

        evidence_resolution_outcome = (
            resolution.outcome
        )

        print(
            "[Evidence Resolution] Outcome:",
            resolution.outcome,
        )

        print(
            "[Evidence Resolution] Checkpoint changed:",
            resolution.checkpoint_changed,
        )

        if resolution.backup_path:
            print(
                "[Evidence Resolution] Backup:",
                resolution.backup_path,
            )

        state = resolution.final_state

        if (
            resolution.outcome
            == "resolved_material_change"
        ):
            stale_stage = (
                resolution
                .earliest_stale_stage
            )

            print(
                "[Evidence Resolution] Earliest stale stage:",
                stale_stage,
            )

            # ---------------------------------------------
            # SCRIPTWRITER RESUME
            # ---------------------------------------------

            if stale_stage == "scriptwriter":
                print_stage(
                    "EVIDENCE RESUME — SCRIPTWRITER"
                )

                if state.script_result is not None:
                    raise ValueError(
                        "Evidence resolution reported "
                        "Scriptwriter stale, but "
                        "script_result is still present."
                    )

                state = run_scriptwriter_stage(
                    state=state,
                    request=script_request,
                )

                save_checkpoint(
                    state=state,
                    stage_name=(
                        "Scriptwriter "
                        "(Evidence Resume)"
                    ),
                )

                print(
                    "Scriptwriter grounding:",
                    state.script_result.audit.grounded,
                )

            # ---------------------------------------------
            # CONTINUITY RESUME
            #
            # Run if Scriptwriter was stale or if a future
            # supported resolver starts directly here.
            # ---------------------------------------------

            if stale_stage in {
                "scriptwriter",
                "continuity",
            }:
                print_stage(
                    "EVIDENCE RESUME — CONTINUITY"
                )

                if state.continuity_result is not None:
                    raise ValueError(
                        "Evidence resolution reported "
                        "Continuity stale, but "
                        "continuity_result is still present."
                    )

                state = run_continuity_stage(
                    state=state,
                    scene_index=0,
                )

                save_checkpoint(
                    state=state,
                    stage_name=(
                        "Continuity "
                        "(Evidence Resume)"
                    ),
                )

                print(
                    "Continuity status:",
                    (
                        state
                        .continuity_result
                        .review
                        .assessment
                        .status
                    ),
                )

            # ---------------------------------------------
            # EVALUATION RESUME
            #
            # Every supported stale suffix terminates in a
            # fresh Evaluation.
            # ---------------------------------------------

            if stale_stage in {
                "scriptwriter",
                "continuity",
                "evaluation",
            }:
                print_stage(
                    "EVIDENCE RESUME — EVALUATION"
                )

                if state.evaluation_result is not None:
                    raise ValueError(
                        "Evidence resolution reported "
                        "Evaluation stale, but "
                        "evaluation_result is still present."
                    )

                state = run_evaluation_stage(
                    state
                )

                save_checkpoint(
                    state=state,
                    stage_name=(
                        "Evaluation "
                        "(Evidence Resume)"
                    ),
                )

                print_evaluation_summary(
                    state
                )

            else:
                raise ValueError(
                    "Evidence resolution returned an "
                    "unsupported stale stage: "
                    f"{stale_stage!r}"
                )

            print(
                "\n[Evidence Resolution] Downstream "
                "resume completed. This process will "
                "not perform another evidence-resolution "
                "pass automatically."
            )

        elif (
            resolution.outcome
            == "resolved_no_material_change"
        ):
            print(
                "[Evidence Resolution] Refreshed "
                "research provenance was saved, but "
                "authoritative downstream guidance did "
                "not materially change."
            )

            print(
                "[Evidence Resolution] Existing "
                "downstream artifacts remain valid. "
                "No automatic repeat resolution or "
                "regeneration will run."
            )

        elif resolution.outcome == "no_action":
            print(
                "[Evidence Resolution] Evaluation no "
                "longer requires evidence resolution."
            )

        elif (
            resolution.outcome
            == "not_applicable"
        ):
            print(
                "[Evidence Resolution] Evaluation is "
                "outside the evidence-refresh recovery "
                "path."
            )

        else:
            raise ValueError(
                "Unknown live evidence-resolution "
                f"outcome: {resolution.outcome!r}"
            )

    # =====================================================
    # FINAL PROVENANCE SUMMARY
    # =====================================================

    assessment = evaluation_assessment(
        state
    )

    print_stage(
        "LIVE WRITER'S ROOM FLIGHT COMPLETE"
    )

    print(
        "StoryPlan:",
        "PRESENT",
    )

    print(
        "Lore artifacts:",
        len(state.lore_results),
    )

    print(
        "Character artifacts:",
        len(state.character_results),
    )

    print(
        "Scriptwriter result:",
        (
            "PRESENT"
            if state.script_result
            else "MISSING"
        ),
    )

    print(
        "Continuity result:",
        (
            "PRESENT"
            if state.continuity_result
            else "MISSING"
        ),
    )

    print(
        "Evaluation result:",
        (
            "PRESENT"
            if state.evaluation_result
            else "MISSING"
        ),
    )

    print(
        "Final evaluation status:",
        assessment.status,
    )

    print(
        "Evidence resolution attempted:",
        evidence_resolution_attempted,
    )

    if evidence_resolution_outcome is not None:
        print(
            "Evidence resolution outcome:",
            evidence_resolution_outcome,
        )

    print(
        "\n============================================"
    )

    print(
        "LIVE WRITER'S ROOM FLIGHT: COMPLETE"
    )

    print(
        "============================================"
    )


if __name__ == "__main__":
    main()
