from agents.evaluation_agent import (
    CharacterEvaluationSource,
    LoreEvaluationSource,
    build_production_evaluation_manifest,
)

from agents.character_agent import (
    CharacterAnalysis,
    CharacterAnalysisAudit,
)

from agents.continuity_agent import (
    ContinuitySceneAssessment,
    ContinuityScriptSceneAudit,
    ContinuitySceneReview,
)

from agents.lore_agent import (
    LoreResponseModel,
)

from agents.scriptwriter_agent import (
    ScriptCharacterContext,
    ScriptSceneAudit,
    ScriptwriterInput,
)


def main() -> None:
    print(
        "Running Evaluation Agent v0.1L "
        "production-manifest builder test."
    )

    # =====================================================
    # Scriptwriter input
    # =====================================================

    script_input = ScriptwriterInput(
        story_context=(
            "A Cardassian defector meets "
            "Garak privately."
        ),
        scene_objective=(
            "Garak evaluates the defector's "
            "intelligence."
        ),
        story_era=(
            "Deep Space Nine, shortly before "
            "the Dominion War."
        ),
        characters=[
            ScriptCharacterContext(
                character_name="Elim Garak",
                grounded_character_guidance=[],
                grounded_voice_guidance=[],
            ),
        ],
        verified_canon_facts=[],
        era_style_constraints=[
            (
                "Use restrained DS9-era "
                "dialogue without modern "
                "internet slang."
            ),
        ],
        scene_requirements=[],
        prohibited_assumptions=[],
    )

    # =====================================================
    # Frozen Scriptwriter audit
    # =====================================================

    scriptwriter_audit = ScriptSceneAudit(
        grounded=True,
        issues=[],
        reason=(
            "All screenplay segments are "
            "grounded or permitted."
        ),
    )

    # =====================================================
    # Frozen Continuity review
    # =====================================================

    continuity_review = ContinuitySceneReview(
        audit=ContinuityScriptSceneAudit(
            segment_audits=[],
            has_contradiction=False,
            contradiction_count=0,
        ),
        assessment=(
            ContinuitySceneAssessment(
                status="clear",
                issues=[],
                contradiction_count=0,
                insufficient_context_count=0,
            )
        ),
    )

    # =====================================================
    # Frozen Character source
    # =====================================================

    character_analysis = CharacterAnalysis(
        character_name="Elim Garak",
        story_role=(
            "Evaluate the defector's claim."
        ),
        motivations=[],
        likely_behavior=[
            (
                "Evaluate the claim before "
                "committing to action."
            ),
        ],
        unlikely_behavior=[],
        relationships=[],
        dramatic_tension="",
        remaining_canon_questions=[],
        remaining_voice_questions=[],
        speech_patterns=[
            (
                "Uses indirect and carefully "
                "qualified phrasing."
            ),
        ],
        preferred_vocabulary=[],
        communication_style=[],
        behavioral_constraints=[],
        era_language_notes=[],
        out_of_character_risks=[],
    )

    character_audit = (
        CharacterAnalysisAudit(
            grounded=True,
            issues=[],
            reason=(
                "All CharacterAnalysis claims "
                "are grounded."
            ),
        )
    )

    # =====================================================
    # Frozen Lore source
    # =====================================================

    lore_response = LoreResponseModel(
        topic="Elim Garak",
        summary=(
            "Garak participated in a joint "
            "Obsidian Order and Tal Shiar "
            "operation against the Founders."
        ),
        sources=[],
        tools_used=[],
        evidence_supported=True,
        premise_supported=True,
    )

    # =====================================================
    # Production manifest
    # =====================================================

    manifest = (
        build_production_evaluation_manifest(
            script_input=script_input,
            scriptwriter_audit=(
                scriptwriter_audit
            ),
            continuity_review=(
                continuity_review
            ),
            character_sources=[
                CharacterEvaluationSource(
                    character_analysis=(
                        character_analysis
                    ),
                    audit=character_audit,
                ),
            ],
            lore_sources=[
                LoreEvaluationSource(
                    research_id=(
                        "garak_founders"
                    ),
                    research_question=(
                        "What does canon establish "
                        "about Garak's prior "
                        "involvement with the "
                        "Founders?"
                    ),
                    lore_response=(
                        lore_response
                    ),
                ),
            ],
        )
    )

    print("\n--- LLM CRITERIA ---")

    for criterion in manifest.criteria:
        print(
            f"{criterion.criterion_id} "
            f"[{criterion.category}]"
        )

    print("\n--- PRECOMPUTED RESULTS ---")

    for result in (
        manifest.precomputed_results
    ):
        print(
            f"{result.criterion_id} "
            f"[{result.category}] "
            f"-> {result.verdict}"
        )

    # -----------------------------------------------------
    # Expected composition
    # -----------------------------------------------------

    criterion_ids = {
        criterion.criterion_id
        for criterion in manifest.criteria
    }

    precomputed_ids = {
        result.criterion_id
        for result
        in manifest.precomputed_results
    }

    assert (
        "era_style_review"
        in criterion_ids
    )

    assert (
        "character_agent_guidance_elim_garak"
        in criterion_ids
    )

    assert (
        "lore_agent_evidence_garak_founders"
        in criterion_ids
    )

    assert (
        "scriptwriter_grounding_review"
        in precomputed_ids
    )

    assert (
        "continuity_agent_review"
        in precomputed_ids
    )

    assert len(manifest.criteria) == 3

    assert (
        len(
            manifest.precomputed_results
        )
        == 2
    )

    # -----------------------------------------------------
    # Missing Character source fails closed
    # -----------------------------------------------------

    try:
        build_production_evaluation_manifest(
            script_input=script_input,
            scriptwriter_audit=(
                scriptwriter_audit
            ),
            continuity_review=(
                continuity_review
            ),
            character_sources=[],
            lore_sources=[
                LoreEvaluationSource(
                    research_id="test",
                    research_question="Test.",
                    lore_response=lore_response,
                ),
            ],
        )

    except ValueError as exc:
        print(
            "\n--- EXPECTED CHARACTER "
            "SOURCE FAILURE ---"
        )
        print(str(exc))

    else:
        raise AssertionError(
            "Production manifest without "
            "Character input was incorrectly "
            "accepted."
        )

    # -----------------------------------------------------
    # Missing Lore source fails closed
    # -----------------------------------------------------

    try:
        build_production_evaluation_manifest(
            script_input=script_input,
            scriptwriter_audit=(
                scriptwriter_audit
            ),
            continuity_review=(
                continuity_review
            ),
            character_sources=[
                CharacterEvaluationSource(
                    character_analysis=(
                        character_analysis
                    ),
                    audit=character_audit,
                ),
            ],
            lore_sources=[],
        )

    except ValueError as exc:
        print(
            "\n--- EXPECTED LORE "
            "SOURCE FAILURE ---"
        )
        print(str(exc))

    else:
        raise AssertionError(
            "Production manifest without "
            "Lore input was incorrectly "
            "accepted."
        )

    print(
        "\nPASS: Python assembled all five "
        "Evaluation quality dimensions from "
        "real frozen upstream object types, "
        "preserved precomputed quality gates, "
        "and rejected incomplete production "
        "inputs before final review."
    )


if __name__ == "__main__":
    main()