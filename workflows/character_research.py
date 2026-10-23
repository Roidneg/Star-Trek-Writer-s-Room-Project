from tools.lore_evidence import authoritative_lore_text
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

from agents.character_agent import (
    CharacterAnalysis,
    CharacterAnalysisAudit,
    CharacterClaimAuditBatch,
    CharacterResearchPlan,
    analyze_character,
    audit_character_claims,
    build_character_analysis_audit,
    create_character_research_plan,
    validate_character_claim_audit_coverage,
)

from agents.lore_agent import (
    run_lore_query,
)


# ---------------------------------------------------------
# Character Research Constants
# ---------------------------------------------------------


VOICE_RESEARCH_PLACEHOLDER = (
    "No dialogue or voice research has been performed yet. "
    "Treat all voice-related questions as unresolved."
)


# ---------------------------------------------------------
# Production Result
# ---------------------------------------------------------


class CharacterResearchProductionResult(BaseModel):
    """
    Complete Character research package for downstream
    production orchestration.

    The Character research workflow owns:
    - research planning
    - Character-specific Lore routing
    - canon evidence bucketing
    - Character analysis
    - Character grounding audit

    Downstream workflows should preserve these artifacts
    rather than reinterpreting them.
    """

    research_plan: CharacterResearchPlan

    canon_answers: list[dict]

    combined_canon_research: str

    voice_research: str

    character_analysis: CharacterAnalysis

    claim_audit: CharacterClaimAuditBatch

    analysis_audit: CharacterAnalysisAudit


# ---------------------------------------------------------
# Character Canon Research Workflow
# ---------------------------------------------------------


def research_character_canon(
    character_name: str,
    story_context: str,
    story_era: str,
    debug: bool = False,
):
    """
    Create a Character Research Plan and send its canon
    questions to the Lore Agent.

    Voice questions are preserved for the future
    dialogue-research layer.

    This function preserves the original Character research
    workflow contract and returns:

    - research_plan
    - canon_answers
    - combined_canon_research
    - character_analysis
    """

    # -----------------------------------------------------
    # Stage 1: Create the Character Research Plan
    # -----------------------------------------------------

    research_plan = create_character_research_plan(
        character_name=character_name,
        story_context=story_context,
        story_era=story_era,
    )

    if debug:
        print("\n--- CHARACTER RESEARCH PLAN ---")
        print(
            "Character:",
            research_plan.character_name,
        )
        print(
            "Story Context:",
            research_plan.story_context,
        )
        print(
            "Story Era:",
            research_plan.story_era,
        )

        print("\nCanon Questions:")

        for question in research_plan.canon_questions:
            print("-", question)

        print("\nVoice Questions:")

        for question in research_plan.voice_questions:
            print("-", question)

        print("--- END RESEARCH PLAN ---\n")

    # -----------------------------------------------------
    # Stage 2: Research Canon Questions
    # -----------------------------------------------------

    canon_answers = []

    for question in research_plan.canon_questions:

        if debug:
            print(
                "\n--- RESEARCHING CHARACTER CANON ---"
            )
            print(question)
            print("---\n")

        try:
            answer = run_lore_query(
                query=question,
                debug=debug,
            )

        except ValueError as error:
            canon_answers.append(
                {
                    "question": question,
                    "answer": None,
                    "error": str(error),
                }
            )

            continue

        canon_answers.append(
            {
                "question": question,
                "answer": answer,
            }
        )

    # -----------------------------------------------------
    # Stage 3: Format Canon Research
    # -----------------------------------------------------

    verified_canon_notes = []
    unresolved_canon_notes = []
    unsupported_premise_notes = []

    for item in canon_answers:
        answer = item["answer"]

        # -------------------------------------------------
        # Retrieval / research failure remains unresolved.
        # It is not evidence that the premise is false.
        # -------------------------------------------------

        if answer is None:
            note = f"""
Research Question:
{item["question"]}

Canon Answer:
Research could not be completed.

Research Status:
UNRESOLVED - retrieval or research failure.

Research Error:
{item["error"]}
            """.strip()

            unresolved_canon_notes.append(
                note
            )

            continue

        sources = "\n".join(
            f"- {source}"
            for source in answer.sources
        )

        note = f"""
Research Question:
{item["question"]}

Canon Answer:
{answer.summary}

Evidence Supported:
{answer.evidence_supported}

Premise Supported:
{answer.premise_supported}

Sources:
{sources or "No source URL returned."}
        """.strip()

        # -------------------------------------------------
        # Frozen Character evidence routing
        # -------------------------------------------------

        if answer.evidence_supported:
            # Keep research questions, status flags, and metadata out of the
            # auditor's factual lane. canon_answers retains full provenance.
            verified_canon_notes.append(authoritative_lore_text(answer))

        elif answer.premise_supported:
            unresolved_canon_notes.append(
                note
            )

        else:
            unsupported_premise_notes.append(
                note
            )

    # -----------------------------------------------------
    # Stage 4: Build Research Buckets
    # -----------------------------------------------------

    verified_canon_research = (
        "\n\n---\n\n".join(
            verified_canon_notes
        )
    )

    unresolved_canon_research = (
        "\n\n---\n\n".join(
            unresolved_canon_notes
        )
    )

    unsupported_premise_research = (
        "\n\n---\n\n".join(
            unsupported_premise_notes
        )
    )

    # IMPORTANT: Leave the VERIFIED CANON EVIDENCE section blank when
    # no verified evidence exists. A textual placeholder such as
    # "No verified canon evidence found." can be misread downstream
    # as supporting evidence for negative character claims.
    combined_canon_research = f"""
VERIFIED CANON EVIDENCE

{verified_canon_research}


UNRESOLVED CANON QUESTIONS

{unresolved_canon_research or "No unresolved canon questions."}


UNSUPPORTED CANON PREMISES

{unsupported_premise_research or "No unsupported canon premises found."}
    """.strip()

    # -----------------------------------------------------
    # Voice Research
    #
    # Dedicated dialogue / voice research does not yet
    # exist. Preserve that fact explicitly rather than
    # allowing model familiarity to fill the gap.
    # -----------------------------------------------------

    voice_research = (
        VOICE_RESEARCH_PLACEHOLDER
    )

    # -----------------------------------------------------
    # Stage 5: Character Analysis
    # -----------------------------------------------------

    character_analysis = analyze_character(
        research_plan=research_plan,
        canon_research=combined_canon_research,
        voice_research=voice_research,
    )

    # -----------------------------------------------------
    # Stage 6: Debug Output
    # -----------------------------------------------------

    if debug:
        print(
            "\n--- COMBINED CHARACTER CANON RESEARCH ---"
        )
        print(
            combined_canon_research
        )
        print(
            "--- END CHARACTER CANON RESEARCH ---\n"
        )

        print(
            "\n--- CHARACTER ANALYSIS ---"
        )
        print(
            character_analysis.model_dump_json(
                indent=2
            )
        )
        print(
            "--- END CHARACTER ANALYSIS ---\n"
        )

    # -----------------------------------------------------
    # Stage 7: Preserve Existing Return Contract
    # -----------------------------------------------------

    return (
        research_plan,
        canon_answers,
        combined_canon_research,
        character_analysis,
    )

def project_grounded_character_analysis(
    character_analysis: CharacterAnalysis,
    claim_audit: CharacterClaimAuditBatch,
) -> CharacterAnalysis:
    """
    Build a conservative CharacterAnalysis containing only claims
    that the frozen Character grounding audit already supported.

    Python does not reinterpret or rewrite claims here.
    Unsupported claims are simply removed.
    """

    validate_character_claim_audit_coverage(
        character_analysis=character_analysis,
        audit_batch=claim_audit,
    )

    audited_by_field = {
        audited.field: audited
        for audited in claim_audit.claims
    }

    scalar_fields = [
        "story_role",
        "dramatic_tension",
    ]

    list_fields = [
        "motivations",
        "likely_behavior",
        "unlikely_behavior",
        "relationships",
        "remaining_canon_questions",
        "remaining_voice_questions",
        "speech_patterns",
        "preferred_vocabulary",
        "communication_style",
        "behavioral_constraints",
        "era_language_notes",
        "out_of_character_risks",
    ]

    updates = {}

    for field_name in scalar_fields:
        value = getattr(
            character_analysis,
            field_name,
        )

        if not value.strip():
            updates[field_name] = value
            continue

        audited = audited_by_field.get(
            field_name
        )

        if audited is None:
            raise ValueError(
                "Character grounded projection is "
                "missing an audited scalar claim: "
                f"{field_name!r}"
            )

        updates[field_name] = (
            value
            if audited.supported
            else ""
        )

    for field_name in list_fields:
        values = getattr(
            character_analysis,
            field_name,
        )

        grounded_values = []

        for index, value in enumerate(values):
            if not value.strip():
                continue

            audit_field = (
                f"{field_name}[{index}]"
            )

            audited = audited_by_field.get(
                audit_field
            )

            if audited is None:
                raise ValueError(
                    "Character grounded projection is "
                    "missing an audited list claim: "
                    f"{audit_field!r}"
                )

            if audited.supported:
                grounded_values.append(
                    value
                )

        updates[field_name] = (
            grounded_values
        )

    return character_analysis.model_copy(
        update=updates
    )

# ---------------------------------------------------------
# Production Character Research Wrapper
# ---------------------------------------------------------


def research_character_for_production(
    character_name: str,
    story_context: str,
    story_era: str,
    debug: bool = False,
) -> CharacterResearchProductionResult:
    """
    Run the complete frozen Character research and grounding
    path required for downstream production.

    This wrapper does not replace or reinterpret the frozen
    Character Agent.

    It coordinates:

    Character research workflow
        ->
    CharacterAnalysis
        ->
    claim-by-claim grounding audit
        ->
    deterministic audit coverage validation
        ->
    deterministic CharacterAnalysisAudit

    The legacy whole-analysis auditor is intentionally not
    used here.
    """

    (
        research_plan,
        canon_answers,
        combined_canon_research,
        character_analysis,
    ) = research_character_canon(
        character_name=character_name,
        story_context=story_context,
        story_era=story_era,
        debug=debug,
    )

    # -----------------------------------------------------
    # Stage 1: Claim-by-Claim Character Grounding Audit
    # -----------------------------------------------------

    claim_audit = audit_character_claims(
        research_plan=research_plan,
        canon_research=combined_canon_research,
        voice_research=VOICE_RESEARCH_PLACEHOLDER,
        character_analysis=character_analysis,
    )

    # -----------------------------------------------------
    # Stage 2: Deterministic Coverage Validation
    #
    # Python confirms that every CharacterAnalysis claim
    # was audited exactly once and in the correct order.
    # -----------------------------------------------------

    validate_character_claim_audit_coverage(
        character_analysis=character_analysis,
        audit_batch=claim_audit,
    )

    # -----------------------------------------------------
    # Stage 3: Deterministic Aggregate Audit
    #
    # Python, not another LLM vote, owns the final grounded
    # status of the CharacterAnalysis.
    # -----------------------------------------------------

    # -----------------------------------------------------
    # Bounded Grounded-Subset Convergence
    #
    # A later audit may reject a claim that an earlier audit
    # accepted. Projection is conservative and monotonic:
    # it may only remove unsupported claims, never rewrite
    # or add claims.
    #
    # Bound the number of repair passes so semantic
    # instability can never cause an unbounded loop.
    # -----------------------------------------------------

    max_grounded_subset_repairs = 3
    grounded_subset_repairs = 0

    while True:
        analysis_audit = (
            build_character_analysis_audit(
                claim_audit
            )
        )

        if analysis_audit.grounded:
            break

        if (
            grounded_subset_repairs
            >= max_grounded_subset_repairs
        ):
            issue_summary = "\n".join(
                (
                    f"- {issue.field}: "
                    f"{issue.claim}\n"
                    f"  Reason: {issue.reason}"
                )
                for issue in analysis_audit.issues
            )

            raise ValueError(
                "Character production remained "
                "ungrounded after bounded conservative "
                "grounded-subset repair passes.\n"
                + issue_summary
            )

        projected_analysis = (
            project_grounded_character_analysis(
                character_analysis=character_analysis,
                claim_audit=claim_audit,
            )
        )

        if (
            projected_analysis.model_dump()
            == character_analysis.model_dump()
        ):
            issue_summary = "\n".join(
                (
                    f"- {issue.field}: "
                    f"{issue.claim}\n"
                    f"  Reason: {issue.reason}"
                )
                for issue in analysis_audit.issues
            )

            raise ValueError(
                "Character grounded-subset projection "
                "made no progress while unsupported "
                "claims remained.\n"
                + issue_summary
            )

        character_analysis = projected_analysis
        grounded_subset_repairs += 1

        claim_audit = audit_character_claims(
            research_plan=research_plan,
            canon_research=combined_canon_research,
            voice_research=VOICE_RESEARCH_PLACEHOLDER,
            character_analysis=character_analysis,
        )

        validate_character_claim_audit_coverage(
            character_analysis=character_analysis,
            audit_batch=claim_audit,
        )

    if debug:
        print(
            "\n--- CHARACTER CLAIM AUDIT ---"
        )

        print(
            claim_audit.model_dump_json(
                indent=2
            )
        )

        print(
            "--- END CHARACTER CLAIM AUDIT ---\n"
        )

        print(
            "\n--- CHARACTER ANALYSIS AUDIT ---"
        )

        print(
            analysis_audit.model_dump_json(
                indent=2
            )
        )

        print(
            "--- END CHARACTER ANALYSIS AUDIT ---\n"
        )

    # -----------------------------------------------------
    # Stage 4: Return Complete Production Package
    # -----------------------------------------------------

    return CharacterResearchProductionResult(
        research_plan=research_plan,
        canon_answers=canon_answers,
        combined_canon_research=(
            combined_canon_research
        ),
        voice_research=(
            VOICE_RESEARCH_PLACEHOLDER
        ),
        character_analysis=(
            character_analysis
        ),
        claim_audit=claim_audit,
        analysis_audit=analysis_audit,
    )


# ---------------------------------------------------------
# Direct Test
# ---------------------------------------------------------


if __name__ == "__main__":

    production_result = (
        research_character_for_production(
            character_name="Elim Garak",
            story_context=(
                "A Cardassian defector secretly approaches "
                "Garak with intelligence claiming that the "
                "Founders have infiltrated the Cardassian "
                "government. Garak must decide how seriously "
                "to take the claim and whether to involve "
                "the DS9 command staff."
            ),
            story_era=(
                "Star Trek: Deep Space Nine, shortly before "
                "the Dominion War"
            ),
            debug=True,
        )
    )

    print(
        "\n================================"
    )
    print(
        "CHARACTER PRODUCTION RESEARCH COMPLETE"
    )
    print(
        "================================\n"
    )

    print(
        "Character:",
        production_result
        .research_plan
        .character_name,
    )

    print(
        "Canon Questions Researched:",
        len(
            production_result
            .canon_answers
        ),
    )

    print(
        "Character Analysis Grounded:",
        production_result
        .analysis_audit
        .grounded,
    )

    print(
        "\nVoice Questions Reserved For Later:"
    )

    for question in (
        production_result
        .research_plan
        .voice_questions
    ):
        print(
            "-",
            question,
        )