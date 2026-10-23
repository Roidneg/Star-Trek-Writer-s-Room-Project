from agents.character_agent import (
    CharacterResearchPlan,
    analyze_character,
    audit_character_analysis,
    audit_character_claims,
    validate_character_claim_audit_coverage,
    build_character_analysis_audit,
)

research_plan = CharacterResearchPlan(
    character_name="Elim Garak",
    story_context=(
        "A Cardassian defector secretly approaches Garak with "
        "intelligence claiming that the Founders have infiltrated "
        "the Cardassian government. Garak must decide how seriously "
        "to take the claim and whether to involve the DS9 command staff."
    ),
    story_era=(
        "Star Trek: Deep Space Nine, shortly before the Dominion War"
    ),
    canon_questions=[
        "What does canon establish about Garak's relationship with Cardassia before the Dominion War?",
        "What does canon establish about Garak's knowledge of the Founders or Dominion before the Dominion War?",
    ],
    voice_questions=[
        "How does Garak typically address Starfleet officers in DS9 dialogue?",
        "What recurring rhetorical patterns appear in Garak's DS9 dialogue?",
    ],
)
canon_research = """
VERIFIED CANON EVIDENCE

Research Question:
What does canon establish about Garak's knowledge of the Founders or Dominion before the Dominion War?

Canon Answer:
Canon shows that Garak had knowledge of the Founders and the Dominion before the Dominion War. He joined Enabran Tain on a joint Obsidian Order/Tal Shiar mission to destroy the Founders' homeworld and later received information directly from a Female Changeling.

Evidence Supported:
True

Premise Supported:
True


UNRESOLVED CANON QUESTIONS

Research Question:
What does canon establish about Garak's exact official status with Cardassia shortly before the Dominion War?

Canon Answer:
The available evidence establishes that Garak was exiled and retained Cardassian contacts, but the exact requested timeframe and official status remain unresolved.


UNSUPPORTED CANON PREMISES

Research Question:
Did a Cardassian defector canonically approach Garak with intelligence about Founder infiltration?

Canon Answer:
No supplied research establishes that this specific event occurred in canon.
""".strip()

voice_research = (
    "No dialogue or voice research has been performed yet. "
    "Treat all voice-related questions as unresolved."
)

character_analysis = analyze_character(
    research_plan=research_plan,
    canon_research=canon_research,
    voice_research=voice_research,
)
print(character_analysis.model_dump_json(indent=2))

# audit = audit_character_analysis(
#    research_plan=research_plan,
#    canon_research=canon_research,
#    voice_research=voice_research,
#    character_analysis=character_analysis,
# )

claim_audit = audit_character_claims(
    research_plan=research_plan,
    canon_research=canon_research,
    voice_research=voice_research,
    character_analysis=character_analysis,
)

validate_character_claim_audit_coverage(
    character_analysis=character_analysis,
    audit_batch=claim_audit,
)

analysis_audit = build_character_analysis_audit(
    claim_audit
)

print("\n--- CHARACTER ANALYSIS AUDIT ---")
print(
    analysis_audit.model_dump_json(
        indent=2
    )
)
print("--- END CHARACTER ANALYSIS AUDIT ---")

print("\n--- CHARACTER CLAIM AUDIT ---")
print(claim_audit.model_dump_json(indent=2))
print("--- END CHARACTER CLAIM AUDIT ---")
