import re
from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.exceptions import OutputParserException
from openai import LengthFinishReasonError
from pydantic import BaseModel, Field
from tools.model_provider import create_chat_model
from dotenv import load_dotenv
from tools.rate_limit import safe_invoke

from config import CHARACTER_MODEL


load_dotenv()


# ---------------------------------------------------------
# Character Analysis Output
# ---------------------------------------------------------


class CharacterAnalysis(BaseModel):
    character_name: str = Field(
        description=(
            "The exact character being analyzed."
        )
    )

    story_role: str = Field(
        description=(
            "The character's dramatic role in the proposed story."
        )
    )

    motivations: list[str] = Field(
        description=(
            "The motivations that are likely driving the "
            "character within the story."
        )
    )

    likely_behavior: list[str] = Field(
        description=(
            "Behavior that would be consistent with the character "
            "based on supplied and established canon research."
        )
    )

    unlikely_behavior: list[str] = Field(
        description=(
            "Behavior that would conflict with the character's "
            "established personality and/or history."
        )
    )

    relationships: list[str] = Field(
        description=(
            "Relevant relationships that may affect the "
            "character's decisions in the story."
        )
    )

    dramatic_tension: str = Field(
        description=(
            "The internal or interpersonal tension the character "
            "can contribute to the story."
        )
    )

    remaining_canon_questions: list[str] = Field(
        description=(
            "Important factual canon questions that remain unresolved "
            "after reviewing the supplied research."
        )
    )

    remaining_voice_questions: list[str] = Field(
        description=(
            "Important character-voice or era-language questions that "
            "remain unresolved after reviewing the supplied dialogue research."
        )
    )

    speech_patterns: list[str] = Field(
        description=(
            "Recurring speech habits, sentence structures, rhetorical "
            "patterns, and communication tendencies supported by the "
            "supplied character research."
        )
    )

    preferred_vocabulary: list[str] = Field(
        description=(
            "Words, terminology, forms of address, and expressions that "
            "fit the character and the story's Star Trek era."
        )
    )

    communication_style: list[str] = Field(
        description=(
            "Guidance for how the character typically communicates, "
            "including formality, directness, humor, emotional restraint, "
            "technical language, and interpersonal style."
        )
    )

    behavioral_constraints: list[str] = Field(
        description=(
            "Important boundaries the writer should respect so that the "
            "character does not behave inconsistently with established "
            "personality and history."
        )
    )

    era_language_notes: list[str] = Field(
        description=(
            "Language and terminology constraints specific to the Star Trek "
            "era in which the story takes place."
        )
    )

    out_of_character_risks: list[str] = Field(
        description=(
            "Specific writing choices that could make the character's "
            "dialogue or behavior feel inconsistent with the supplied "
            "canon evidence."
        )
    )

class CharacterAnalysisIssue(BaseModel):
    field: str = Field(
        description=(
            "The CharacterAnalysis field containing the unsupported claim."
        )
    )

    claim: str = Field(
        description=(
            "The specific unsupported or insufficiently grounded claim."
        )
    )

    reason: str = Field(
        description=(
            "Why the claim is not supported by the supplied evidence."
        )
    )
class CharacterClaimAudit(BaseModel):
    field: str = Field(
        description=(
            "The exact CharacterAnalysis field or indexed field "
            "being audited."
        )
    )

    claim: str = Field(
        description=(
            "The exact character-specific claim being audited."
        )
    )

    supported: bool = Field(
        description=(
            "True only if the claim is supported by VERIFIED CANON "
            "EVIDENCE, supplied voice research, or explicit story context."
        )
    )

    evidence_excerpt: str = Field(
        description=(
            "An exact verbatim excerpt from VERIFIED CANON EVIDENCE, "
            "supplied voice research, or explicit story context that "
            "supports the claim. Use an empty string when the claim is "
            "unsupported."
        )
    )

    reason: str = Field(
        description=(
            "A concise explanation of why the claim is supported "
            "or unsupported."
        )
    )

class CharacterClaimAuditBatch(BaseModel):
    claims: list[CharacterClaimAudit] = Field(
        description=(
            "One audit result for every CharacterAnalysis claim "
            "submitted for verification."
        )
    )

class CharacterAnalysisAudit(BaseModel):
    grounded: bool = Field(
        description=(
            "True only if all character-specific claims in the analysis "
            "are supported by verified canon evidence, supplied voice "
            "research, or the current fictional story context."
        )
    )

    issues: list[CharacterAnalysisIssue] = Field(
        default_factory=list,
        description=(
            "Specific claims that are unsupported, derived from unresolved "
            "or unsupported research, or imported from internal model knowledge."
        ),
    )

    reason: str = Field(
        description=(
            "A concise explanation of why the character analysis is or is not grounded."
        )
    )

class CharacterEvidenceEntailment(BaseModel):
    entailed: bool = Field(
        description=(
            "True only when the supplied evidence excerpt alone "
            "supports the entire material claim."
        )
    )

    reason: str = Field(
        description=(
            "A concise explanation of whether the supplied "
            "evidence supports the entire claim."
        )
    )

class CharacterInterpretationVerification(BaseModel):
    field: str = Field(
        description=(
            "The exact CharacterAnalysis field being verified."
        )
    )

    claim: str = Field(
        description=(
            "The exact interpretive CharacterAnalysis claim being verified."
        )
    )

    evidence_excerpt: str = Field(
        description=(
            "The exact evidence excerpt supplied as the basis "
            "for the interpretation."
        )
    )

    unsupported_additions: list[str] = Field(
        description=(
            "Specific material details introduced by the claim that "
            "are not supported by the supplied evidence excerpt. "
            "Return an empty list when the claim introduces no "
            "unsupported material details."
        )
    )

    valid_interpretation: bool = Field(
        description=(
            "True only when the claim is a bounded interpretation "
            "of the supplied evidence without introducing unsupported "
            "character facts, motives, traits, relationships, or behavior."
        )
    )

    reason: str = Field(
        description=(
            "A concise explanation of why the interpretation "
            "is or is not justified by the supplied evidence."
        )
    )

# ---------------------------------------------------------
# Character Research Plan
# ---------------------------------------------------------


class CharacterInterpretationCheck(BaseModel):
    status: Literal["supported", "unsupported", "not_applicable"]
    claim_excerpt: str
    source: Literal["evidence", "story_context", "none"]
    support_excerpt: str
    reason: str = Field(min_length=1)


class CharacterInterpretationReview(BaseModel):
    field: str
    claim: str
    evidence_excerpt: str
    action: CharacterInterpretationCheck
    target: CharacterInterpretationCheck
    manner: CharacterInterpretationCheck
    sequence: CharacterInterpretationCheck
    scope: CharacterInterpretationCheck
    character_facts: CharacterInterpretationCheck


INTERPRETATION_CHECK_DIMENSIONS = (
    "action", "target", "manner", "sequence", "scope", "character_facts",
)


def finalize_character_interpretation_review(
    audited_claim: CharacterClaimAudit,
    story_context: str,
    review: CharacterInterpretationReview,
) -> CharacterInterpretationVerification:
    """Validate review accounting; Python aggregates the component verdicts.

    Exact excerpts establish provenance, not semantic entailment. The model
    still owns each component's semantic judgment.
    """
    for name in ("field", "claim", "evidence_excerpt"):
        if getattr(review, name) != getattr(audited_claim, name):
            raise OutputParserException(f"Component review changed {name}.")

    sources = {
        "evidence": audited_claim.evidence_excerpt,
        "story_context": story_context,
    }
    covered = set()
    unsupported = []
    applicable = 0
    for dimension in INTERPRETATION_CHECK_DIMENSIONS:
        check = getattr(review, dimension)
        if not check.reason.strip():
            raise OutputParserException("Component review has an empty reason.")
        if check.status == "not_applicable":
            if check.claim_excerpt or check.support_excerpt or check.source != "none":
                raise OutputParserException("Inapplicable component supplied excerpts.")
            continue
        applicable += 1
        if not check.claim_excerpt.strip() or check.claim_excerpt not in audited_claim.claim:
            raise OutputParserException("Component changed or omitted its claim excerpt.")
        start = 0
        while True:
            index = audited_claim.claim.find(check.claim_excerpt, start)
            if index < 0:
                break
            covered.update(range(index, index + len(check.claim_excerpt)))
            start = index + 1
        if check.status == "supported":
            if (
                check.source not in sources
                or not check.support_excerpt.strip()
                or check.support_excerpt not in sources[check.source]
            ):
                raise OutputParserException("Supported component has no exact source excerpt.")
        else:
            if check.source != "none" or check.support_excerpt:
                raise OutputParserException("Unsupported component claimed supporting evidence.")
            unsupported.append(f"{dimension}: {check.claim_excerpt} -- {check.reason}")

    if not applicable or any(
        character.isalnum() and index not in covered
        for index, character in enumerate(audited_claim.claim)
    ):
        raise OutputParserException("Component review did not cover the complete claim.")

    return CharacterInterpretationVerification(
        field=audited_claim.field,
        claim=audited_claim.claim,
        evidence_excerpt=audited_claim.evidence_excerpt,
        unsupported_additions=unsupported,
        valid_interpretation=not unsupported,
        reason=(
            "; ".join(unsupported)
            if unsupported else
            "All applicable components were judged supported; exact excerpts and claim coverage were validated."
        ),
    )


class CharacterResearchPlan(BaseModel):
    character_name: str = Field(
        description=(
            "The exact Star Trek character who needs to be researched."
        )
    )

    story_context: str = Field(
        description=(
            "A concise description of the situation the character "
            "will face in the proposed story."
        )
    )

    canon_questions: list[str] = Field(
        description=(
            "Specific factual canon questions that must be researched "
            "before making behavioral or characterization judgments."
        )
    )

    voice_questions: list[str] = Field(
        description=(
            "Specific questions about the character's dialogue, "
            "speech patterns, terminology, rhetorical habits, and "
            "communication style that should be researched from "
            "appropriate-era dialogue evidence."
        )
    )

    story_era: str = Field(
        description=(
            "The specific Star Trek series, era, or story period "
            "that constrains canon and dialogue research."
        )
    )


# ---------------------------------------------------------
# Character Research Guardrails
# ---------------------------------------------------------


MIN_CANON_QUESTIONS = 2
MAX_CANON_QUESTIONS = 5

MIN_VOICE_QUESTIONS = 2
MAX_VOICE_QUESTIONS = 5

CHARACTER_CLAIM_AUDIT_BATCH_SIZE = 1

STRICT_CHARACTER_CLAIM_FIELDS = {
    "story_role",
    "relationships",
    "speech_patterns",
    "preferred_vocabulary",
    "communication_style",
    "era_language_notes",
}

INTERPRETIVE_CHARACTER_CLAIM_FIELDS = {
    "motivations",
    "likely_behavior",
    "unlikely_behavior",
    "dramatic_tension",
    "behavioral_constraints",
    "out_of_character_risks",
}

def get_character_claim_base_field(
    field: str,
) -> str:
    """
    Return the CharacterAnalysis field name without
    an optional list index.
    """

    return field.split(
        "[",
        1,
    )[0]

AMBIGUOUS_RESEARCH_PHRASES = [
    "during this timeframe",
    "during this time period",
    "during this era",
    "during the same timeframe",
    "during the same time period",
    "during the same era",
    "in this timeframe",
    "in this time period",
    "in this era",
    "at this point",
    "in this situation",
    "in that situation",
    "current state",
    "currently",
    "at the time",
    "current relationship",
    "current status",
    "current position",
    "current standing",
    "current",
    "this period",
    "that timeframe",
    "that time period",
    "that era",
    "that period",
    "the same era",
]


SPECULATIVE_CANON_PHRASES = [
    "how might",
    "how could",
    "could influence",
    "might influence",
    "could affect",
    "might affect",
    "could impact",
    "might impact",
    "decision-making",
    "reaction to",
    "what are the stakes",
    "what would happen",
    "what could happen",
    "what might happen",
    "might employ",
    "how does it affect",
    "how does this affect",
    "what effect",
    "what prior instances",
    "what instances",
]


SPECULATIVE_VOICE_PHRASES = [
    "how might",
    "how could",
    "might employ",
    "could employ",
    "might use",
    "could use",
    "would use",
    "would respond",
    "would react",
    "what does this reveal",
    "what do these",
    "suggest about",
    "reveal about",
    "reflect his",
    "reflect hers",
    "reflect their",
    "demonstrate cultural influences",
]


SPECULATIVE_LORE_PHRASES = [
    "can be inferred",
    "it can be inferred",
    "likely",
    "probably",
    "suggests that",
    "may indicate",
    "could indicate",
    "what does this reveal",
    "what do these",
    "suggest about",
    "reveal about",
]


UNVERIFIED_CHARACTER_ATTRIBUTE_PATTERNS = [
    r"\b(?:his|her|their)\b.{0,30}\bdefection\b",
    r"\b(?:his|her|their)\b.{0,30}\binfiltration\b",
    r"\b(?:his|her|their)\b.{0,30}\bbetrayal\b",
    r"\bas a defector\b",
    r"\bas an infiltrator\b",
    r"\bofficial rank and position\b",
    r"\bofficial rank within\b",
]


def validate_lore_summary(
    summary: str,
) -> None:
    """
    Reject Lore summaries containing common speculative
    or inferential phrasing.
    """

    normalized_summary = summary.casefold()

    for phrase in SPECULATIVE_LORE_PHRASES:
        if phrase in normalized_summary:
            raise ValueError(
                "Lore summary contains unsupported "
                f"inference ('{phrase}'):\n"
                f"{summary}"
            )


def validate_voice_question_content(
    questions: list[str],
) -> None:
    """
    Reject voice research questions that ask the research
    layer to predict character behavior in the fictional story.
    """

    for question in questions:
        normalized_question = question.casefold()

        for phrase in SPECULATIVE_VOICE_PHRASES:
            if phrase in normalized_question:
                raise ValueError(
                    "Voice question contains speculative "
                    f"language ('{phrase}'):\n"
                    f"{question}"
                )


def validate_canon_question_content(
    questions: list[str],
) -> None:
    """
    Ensure canon questions ask for factual research
    rather than characterization or speculation.
    """

    for question in questions:
        lowered_question = question.casefold()

        # Example lists often turn one research target into several
        # implicit claims, sometimes with incorrect group membership.
        if re.search(
            r"\(\s*(?:e\.g\.\s*,?|for example\b|such as\b)",
            question,
            flags=re.IGNORECASE,
        ):
            raise ValueError(
                "Canon question bundles illustrative examples; "
                "ask about one named counterpart or one coherent group:\n"
                f"{question}"
            )

        # A single Lore answer should not have to prove separate
        # role, action, contact, and response claims at once.
        compound_targets = (
            r"\b(?:official\s+)?(?:position|rank|role)\s+and\s+"
            r"(?:activities|actions|interactions)\b",
            r"\bpast\s+as\b.+\band\s+any\b",
            r"\band\s+how\b",
        )
        if any(
            re.search(pattern, lowered_question)
            for pattern in compound_targets
        ):
            raise ValueError(
                "Canon question combines distinct factual targets; "
                "ask one factual thing per question:\n"
                f"{question}"
            )

        for phrase in SPECULATIVE_CANON_PHRASES:
            if phrase in lowered_question:
                raise ValueError(
                    "Canon question contains speculative or "
                    "character-analysis language "
                    f"('{phrase}'):\n"
                    f"{question}"
                )


def validate_question_set(
    questions: list[str],
    question_type: str,
    minimum: int,
    maximum: int,
) -> None:
    """
    Validate one set of Character Agent research questions.
    """

    question_count = len(questions)

    if question_count < minimum:
        raise ValueError(
            f"Character Agent generated too few "
            f"{question_type} questions. "
            f"Expected at least {minimum}, "
            f"but received {question_count}."
        )

    if question_count > maximum:
        raise ValueError(
            f"Character Agent generated too many "
            f"{question_type} questions. "
            f"Expected no more than {maximum}, "
            f"but received {question_count}."
        )

    for question in questions:
        cleaned_question = question.strip()

        if not cleaned_question:
            raise ValueError(
                f"Character Agent generated an empty "
                f"{question_type} question."
            )

        if not cleaned_question.endswith("?"):
            raise ValueError(
                f"{question_type.title()} question must end "
                f"with a question mark: {cleaned_question}"
            )

        lowered_question = cleaned_question.casefold()

        for phrase in AMBIGUOUS_RESEARCH_PHRASES:
            if phrase in lowered_question:
                raise ValueError(
                    f"{question_type.title()} question contains "
                    f"an ambiguous reference ('{phrase}'):\n"
                    f"{cleaned_question}"
                )


def validate_unverified_character_attributes(
    questions: list[str],
) -> None:
    """
    Reject canon questions that embed an unverified role,
    event, or status as though it were already established.
    """

    for question in questions:
        normalized_question = question.casefold()

        for pattern in UNVERIFIED_CHARACTER_ATTRIBUTE_PATTERNS:
            if re.search(
                pattern,
                normalized_question,
            ):
                raise ValueError(
                    "Canon question assumes an unverified "
                    "character attribute or event:\n"
                    f"{question}"
                )


def validate_character_research_plan(
    research_plan: CharacterResearchPlan,
) -> None:
    """
    Validate both Character Agent research lanes.
    """

    validate_question_set(
        questions=research_plan.canon_questions,
        question_type="canon",
        minimum=MIN_CANON_QUESTIONS,
        maximum=MAX_CANON_QUESTIONS,
    )

    validate_canon_question_content(
        research_plan.canon_questions
    )

    validate_unverified_character_attributes(
        research_plan.canon_questions
    )

    validate_question_set(
        questions=research_plan.voice_questions,
        question_type="voice",
        minimum=MIN_VOICE_QUESTIONS,
        maximum=MAX_VOICE_QUESTIONS,
    )

    validate_voice_question_content(
        research_plan.voice_questions
    )


def enumerate_character_analysis_claims(
    character_analysis: CharacterAnalysis,
) -> list[dict[str, str]]:
    """
    Convert every auditable CharacterAnalysis value into
    an explicit field/claim pair.

    Python owns audit coverage.
    The LLM only judges whether each submitted claim is supported.
    """

    claims: list[dict[str, str]] = []

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

    for field_name in scalar_fields:
        value = getattr(character_analysis, field_name)

        if value.strip():
            claims.append(
                {
                    "field": field_name,
                    "claim": value,
                }
            )

    for field_name in list_fields:
        values = getattr(character_analysis, field_name)

        for index, value in enumerate(values):
            if value.strip():
                claims.append(
                    {
                        "field": f"{field_name}[{index}]",
                        "claim": value,
                    }
                )

    return claims

def extract_verified_canon_evidence(
    canon_research: str,
) -> str:
    """
    Extract only the VERIFIED CANON EVIDENCE section.

    Unresolved and unsupported canon content must not be exposed
    to the character claim auditor as supporting evidence.
    """

    verified_marker = "VERIFIED CANON EVIDENCE"
    unresolved_marker = "UNRESOLVED CANON QUESTIONS"
    unsupported_marker = "UNSUPPORTED CANON PREMISES"

    if verified_marker not in canon_research:
        return ""

    verified_section = canon_research.split(
        verified_marker,
        1,
    )[1]

    end_positions = []

    for marker in (
        unresolved_marker,
        unsupported_marker,
    ):
        position = verified_section.find(marker)

        if position != -1:
            end_positions.append(position)

    if end_positions:
        verified_section = verified_section[
            :min(end_positions)
        ]

    return verified_section.strip()

def normalize_research_question_text(
    question: str,
) -> str:
    """
    Normalize whitespace only for deterministic comparison
    of research-question text.
    """

    return re.sub(
        r"\s+",
        " ",
        question,
    ).strip()


def extract_unresolved_canon_questions(
    canon_research: str,
) -> set[str]:
    """
    Extract research questions explicitly classified by the
    Lore workflow as UNRESOLVED CANON QUESTIONS.

    These questions are workflow-state metadata, not
    supporting canon evidence.
    """

    unresolved_marker = "UNRESOLVED CANON QUESTIONS"
    unsupported_marker = "UNSUPPORTED CANON PREMISES"

    if unresolved_marker not in canon_research:
        return set()

    unresolved_section = canon_research.split(
        unresolved_marker,
        1,
    )[1]

    if unsupported_marker in unresolved_section:
        unresolved_section = unresolved_section.split(
            unsupported_marker,
            1,
        )[0]

    matches = re.findall(
        r"Research Question:\s*(.*?)\s*Canon Answer:",
        unresolved_section,
        flags=re.DOTALL,
    )

    return {
        normalize_research_question_text(question)
        for question in matches
        if question.strip()
    }

def audit_character_claims(
    research_plan: CharacterResearchPlan,
    canon_research: str,
    voice_research: str,
    character_analysis: CharacterAnalysis,
) -> CharacterClaimAuditBatch:
    """
    Audit every CharacterAnalysis claim individually.

    Python determines which claims must be audited.
    The LLM only determines whether each claim is supported.
    """

    claims = enumerate_character_analysis_claims(
        character_analysis
    )

    verified_canon_research = extract_verified_canon_evidence(
        canon_research
    )

    unresolved_canon_questions = (
        extract_unresolved_canon_questions(
            canon_research
        )
    )

    audit_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are a strict claim-by-claim grounding auditor for a
Star Trek character analysis.

Python has already determined every claim that must be audited.

You MUST return one verdict for EVERY submitted claim.

Do not omit claims.
Do not add claims.
Do not merge claims.
Do not split claims.
Do not reorder claims.

For every result:

- Copy "field" exactly as supplied.
- Copy "claim" exactly as supplied.
- Set "supported" to true or false.
- If "supported" is true, copy an exact verbatim excerpt from the
  allowed evidence into "evidence_excerpt".
- The evidence excerpt must actually appear word-for-word in the
  supplied VERIFIED CANON EVIDENCE, voice research, or story context.
- Never paraphrase, summarize, reconstruct, or invent an evidence excerpt.
- If no exact allowed evidence excerpt supports the claim, set
  "supported" to false.
- If "supported" is false, set "evidence_excerpt" to an empty string.
- Explain the verdict in "reason".

CONTIGUOUS EVIDENCE RULE:

The evidence excerpt must be one single contiguous substring copied
from one allowed source exactly as written.

Do not:

- combine two separate excerpts
- concatenate sentences from different locations
- remove text that appears between supporting statements
- normalize or rewrite wording
- reconstruct a passage from multiple facts
- join evidence from canon research and story context into one excerpt

If the exact complete evidence excerpt cannot be copied as one
contiguous substring from an allowed source, do not manufacture one.

For strict factual fields, set "supported" to false if no single
allowed excerpt supports the material claim under the current schema.

For interpretive fields, select one exact contiguous excerpt that
establishes the factual basis for the interpretation.

IMPORTANT:

STRICT FACTUAL FIELDS:

For these fields:

- story_role
- relationships
- speech_patterns
- preferred_vocabulary
- communication_style
- era_language_notes

the evidence excerpt must support the entire material claim, not merely
one related fact inside it.

STORY_ROLE MEANING:
story_role is the character's dramatic task in the proposed fictional
story, not necessarily a canon occupation, formal title, or rank.
An explicit present task or decision assigned to the character in STORY
CONTEXT can support the complete story_role claim. Quote that context.
Do not reject a task merely because it is an action rather than a title,
or because VERIFIED CANON EVIDENCE does not repeat the fictional task.
Strict factual support still applies to every added attribute: a context
assigning a decision does not establish an official post, personality,
ongoing relationship, or a decision outcome that the context leaves open.

INTERPRETIVE FIELDS:

For these fields:

- motivations
- likely_behavior
- unlikely_behavior
- dramatic_tension
- behavioral_constraints
- out_of_character_risks

the evidence excerpt does NOT need to state the complete interpretive
claim literally.

Instead, identify an exact evidence excerpt that establishes the
factual basis from which the interpretation is being made.

Mark the claim supported at this stage when:

- a real factual basis exists in the allowed evidence, and
- the excerpt directly relates to the interpretation being proposed.

Do NOT decide whether the interpretation itself goes too far.
A separate bounded-interpretation verifier performs that judgment.

Example:

CLAIM:
"Portraying Garak as unaware of the Founders before the war."

EVIDENCE:
"Garak had knowledge of the Founders before the Dominion War."

At this stage:
supported = true

The evidence establishes the factual basis.
A later verifier determines whether the writing guidance is a valid
bounded interpretation.

However, do not supply merely tangential evidence. The excerpt must
establish the actual factual premise on which the interpretation depends.

ALLOWED SUPPORTING EVIDENCE:

1. VERIFIED CANON EVIDENCE
2. Supplied voice research
3. Explicit facts in the current fictional story context

PROHIBITED SUPPORTING EVIDENCE:

- UNRESOLVED CANON QUESTIONS
- UNSUPPORTED CANON PREMISES
- Internal Star Trek knowledge
- Character reputation
- Plausibility alone
- Assumptions based on biography
- Related facts that do not establish the actual claim

A fact appearing only in an unresolved bucket remains unresolved.

A canonically correct claim is still unsupported if the supplied
allowed evidence does not establish it.

BOUNDED INTERPRETATION IS ALLOWED.

The story context may support an immediate task or decision explicitly
assigned to the character.

Example:

Story context:
"The character must decide how seriously to take the claim."

Supported interpretation:
"Assess the credibility of the claim."

Unsupported embellishment:
"Assess the claim cautiously because of his secretive personality."

Verified canon evidence may be applied to the fictional situation when
the connection is direct and introduces no new motive, relationship,
personality trait, status, skill, behavioral tendency, or historical fact.

SPECIAL RULE FOR remaining_canon_questions AND remaining_voice_questions:

These fields contain unresolved research questions rather than
established factual claims.

Mark such a question supported when:

- it accurately preserves a genuine unresolved research need, and
- its wording does not convert an unresolved fact into an established fact.

Do not require an unresolved question itself to be established canon.

EVIDENCE EXCERPT EXCEPTION FOR UNRESOLVED QUESTIONS:

For fields beginning with "remaining_canon_questions[" or
"remaining_voice_questions[":

- "supported" means the item is a legitimate unresolved research question.
- Do not require factual evidence proving the question itself.
- Set "evidence_excerpt" to an empty string.
- The empty evidence excerpt does NOT make an otherwise valid unresolved
  research question unsupported.

VOICE RULE:

If supplied voice research says no voice research has been performed,
character-specific speech or language claims are unsupported.

Return a JSON object with this exact top-level structure:

{{
    "claims": [
        {{
            "field": "string",
            "claim": "string",
            "supported": true,
            "evidence_excerpt": "exact verbatim supporting evidence",
            "reason": "string"
        }}
    ]
}}

Return only valid JSON.
""",
            ),
            (
                "human",
                """
CHARACTER:
{character_name}

STORY CONTEXT:
{story_context}

STORY ERA:
{story_era}

CANON RESEARCH:
{canon_research}

VOICE RESEARCH:
{voice_research}

CLAIMS TO AUDIT:
{claims}
""",
            ),
        ]
    )

    audit_chain = audit_prompt | character_claim_auditor

    all_audited_claims: list[CharacterClaimAudit] = []

    for start_index in range(
            0,
            len(claims),
            CHARACTER_CLAIM_AUDIT_BATCH_SIZE,
    ):
        submitted_batch = claims[
            start_index:
            start_index + CHARACTER_CLAIM_AUDIT_BATCH_SIZE
        ]

        try:
            batch_result = safe_invoke(
                audit_chain,
                {
                    "character_name": research_plan.character_name,
                    "story_context": research_plan.story_context,
                    "story_era": research_plan.story_era,
                    "canon_research": verified_canon_research,
                    "voice_research": voice_research,
                    "claims": submitted_batch,
                },
                label="Character claim audit batch",
            )
        except (OutputParserException, LengthFinishReasonError) as error:
            # No complete batch verdict exists. Preserve exact Python-owned
            # claim identities and coverage, but grant no factual support.
            failure_reason = (
                "Character claim audit batch exceeded the completion limit."
                if isinstance(error, LengthFinishReasonError)
                else "Character claim audit batch verdict could not be parsed."
            )
            batch_result = CharacterClaimAuditBatch(
                claims=[
                    CharacterClaimAudit(
                        field=claim["field"],
                        claim=claim["claim"],
                        supported=False,
                        evidence_excerpt="",
                        reason=failure_reason,
                    )
                    for claim in submitted_batch
                ]
            )

        validate_character_claim_audit_batch(
            submitted_claims=submitted_batch,
            audit_batch=batch_result,
        )

        # ---------------------------------------------------------
        # Python owns unresolved-canon workflow state
        # ---------------------------------------------------------

        for index, audited_claim in enumerate(
                batch_result.claims
        ):
            if not audited_claim.field.startswith(
                    "remaining_canon_questions["
            ):
                continue

            normalized_question = (
                normalize_research_question_text(
                    audited_claim.claim
                )
            )

            is_known_unresolved = (
                    normalized_question
                    in unresolved_canon_questions
            )

            batch_result.claims[index] = (
                audited_claim.model_copy(
                    update={
                        "supported": is_known_unresolved,
                        "evidence_excerpt": "",
                        "reason": (
                            "Python confirmed that this question "
                            "is explicitly classified under "
                            "UNRESOLVED CANON QUESTIONS."
                            if is_known_unresolved
                            else
                            "Python could not match this question "
                            "to an explicitly unresolved Lore "
                            "research question."
                        ),
                    }
                )
            )

        validate_character_claim_evidence(
            audit_batch=batch_result,
            verified_canon_research=verified_canon_research,
            voice_research=voice_research,
            story_context=research_plan.story_context,
        )

        for audited_claim in batch_result.claims:
            is_remaining_question = (
                    audited_claim.field.startswith(
                        "remaining_canon_questions["
                    )
                    or audited_claim.field.startswith(
                "remaining_voice_questions["
            )
            )

            if (
                    audited_claim.supported
                    and not is_remaining_question
            ):
                base_field = get_character_claim_base_field(
                    audited_claim.field
                )

                if base_field in INTERPRETIVE_CHARACTER_CLAIM_FIELDS:
                    try:
                        interpretation_result = (
                            verify_character_interpretation(
                                audited_claim=audited_claim,
                                character_name=research_plan.character_name,
                                story_context=research_plan.story_context,
                            )
                        )
                    except (OutputParserException, LengthFinishReasonError) as error:
                        # No valid verdict exists for this claim. Keep the
                        # failure explicit so projection can remove it.
                        audited_claim = audited_claim.model_copy(
                            update={
                                "supported": False,
                                "evidence_excerpt": "",
                                "reason": (
                                    "Bounded interpretation verdict exceeded "
                                    "the completion limit."
                                    if isinstance(error, LengthFinishReasonError)
                                    else "Bounded interpretation verdict could "
                                    "not be parsed."
                                ),
                            }
                        )
                        all_audited_claims.append(audited_claim)
                        continue

                    validate_character_interpretation_verification(
                        audited_claim=audited_claim,
                        verification=interpretation_result,
                    )

                    if not interpretation_result.valid_interpretation:
                        audited_claim = audited_claim.model_copy(
                            update={
                                "supported": False,
                                "evidence_excerpt": "",
                                "reason": (
                                    "Bounded interpretation verification failed: "
                                    f"{interpretation_result.reason}"
                                ),
                            }
                        )

                else:
                    try:
                        entailment_result = (
                            verify_character_evidence_entailment(
                                audited_claim
                            )
                        )
                    except (OutputParserException, LengthFinishReasonError) as error:
                        audited_claim = audited_claim.model_copy(
                            update={
                                "supported": False,
                                "evidence_excerpt": "",
                                "reason": (
                                    "Evidence entailment verdict exceeded "
                                    "the completion limit."
                                    if isinstance(error, LengthFinishReasonError)
                                    else "Evidence entailment verdict could "
                                    "not be parsed."
                                ),
                            }
                        )
                        all_audited_claims.append(audited_claim)
                        continue

                    if not entailment_result.entailed:
                        audited_claim = audited_claim.model_copy(
                            update={
                                "supported": False,
                                "evidence_excerpt": "",
                                "reason": (
                                    "Evidence entailment verification failed: "
                                    f"{entailment_result.reason}"
                                ),
                            }
                        )

            all_audited_claims.append(
                audited_claim
            )


    return CharacterClaimAuditBatch(
        claims=all_audited_claims
    )

def validate_character_claim_audit_batch(
    submitted_claims: list[dict[str, str]],
    audit_batch: CharacterClaimAuditBatch,
) -> None:
    """
    Ensure one audit batch returned exactly one verdict
    for every submitted claim, in the same order,
    with unchanged field and claim text.
    """

    audited_claims = audit_batch.claims

    if len(audited_claims) != len(submitted_claims):
        raise ValueError(
            "Character claim audit batch returned the wrong number "
            "of claims. "
            f"Expected {len(submitted_claims)}, "
            f"received {len(audited_claims)}."
        )

    for index, expected in enumerate(submitted_claims):
        audited = audited_claims[index]

        if audited.field != expected["field"]:
            raise ValueError(
                "Character claim audit batch changed or reordered "
                "a field.\n"
                f"Expected: {expected['field']}\n"
                f"Received: {audited.field}"
            )

        if audited.claim != expected["claim"]:
            audit_batch.claims[index] = audited.model_copy(
                update={
                    "claim": expected["claim"],
                }
            )

def verify_character_evidence_entailment(
    audited_claim: CharacterClaimAudit,
) -> CharacterEvidenceEntailment:
    """
    Determine whether an exact evidence excerpt, by itself,
    supports the entire character-analysis claim.

    This verifier receives no broader canon or story context.
    """

    entailment_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are a strict textual entailment verifier.

Your only job is to determine whether the supplied EVIDENCE EXCERPT,
by itself, supports the entire material CLAIM.

Do not use outside knowledge.
Do not use internal Star Trek knowledge.
Do not assume unstated facts.
Do not repair or reinterpret the claim.

Judge only the relationship between:

1. CLAIM
2. EVIDENCE EXCERPT

Set "entailed" to true only when the evidence excerpt alone supports
the entire material meaning of the claim.

A related fact is not enough.

If the claim introduces any additional material detail that is not
contained in, or necessarily implied by, the evidence excerpt,
set "entailed" to false.

Examples of unsupported additions include:

- rank
- title
- profession
- hierarchy
- mentor or superior status
- motive
- loyalty
- personality trait
- secrecy
- behavioral tendency
- current access
- current relationship
- skill or training
- political status
- causal explanation

Example:

CLAIM:
"Former operative under Enabran Tain during the mission."

EVIDENCE:
"Joined Enabran Tain on a joint mission."

VERDICT:
false

The evidence establishes joint participation.
It does not establish that one person served under the other.

Return only the judgment.

Always return BOTH top-level fields:

- "entailed"
- "reason"

"reason" is REQUIRED for both true and false verdicts.
Never omit "reason".

Return a JSON object with exactly this structure:

{{
    "entailed": false,
    "reason": "concise explanation of the entailment verdict"
}}

Return only valid JSON.
""",
            ),
            (
                "human",
                """
FIELD:
{field}

CLAIM:
{claim}

EVIDENCE EXCERPT:
{evidence_excerpt}
""",
            ),
        ]
    )

    entailment_chain = (
        entailment_prompt
        | character_evidence_entailment_verifier
    )

    return safe_invoke(
        entailment_chain,
        {
            "field": audited_claim.field,
            "claim": audited_claim.claim,
            "evidence_excerpt": audited_claim.evidence_excerpt,
        },
        label="Character evidence entailment",
    )
def build_character_analysis_audit(
    claim_audit: CharacterClaimAuditBatch,
) -> CharacterAnalysisAudit:
    """
    Build the final CharacterAnalysisAudit deterministically
    from the completed claim-by-claim audit.

    Python owns the final grounded decision.
    """

    unsupported_claims = [
        claim
        for claim in claim_audit.claims
        if not claim.supported
    ]

    issues = [
        CharacterAnalysisIssue(
            field=claim.field,
            claim=claim.claim,
            reason=claim.reason,
        )
        for claim in unsupported_claims
    ]

    if issues:
        reason = (
            f"{len(issues)} CharacterAnalysis claim(s) "
            "remain unsupported."
        )
    else:
        reason = (
            "All audited CharacterAnalysis claims are grounded "
            "in allowed evidence or valid bounded interpretation."
        )

    return CharacterAnalysisAudit(
        grounded=not issues,
        issues=issues,
        reason=reason,
    )

def verify_character_interpretation(
    audited_claim: CharacterClaimAudit,
    story_context: str,
    character_name: str = "",
) -> CharacterInterpretationVerification:
    """Review exact claim components, then aggregate their judgments in Python."""
    interpretation_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """
Review the EXACT claim against the evidence excerpt and story context.
CHARACTER TO ANALYZE identifies the acting subject of implicit instructions.
Do not transfer another person's actions, motives, or cooperation to that
character merely because both appear in the same evidence excerpt.
Participation or presence does not establish awareness or voluntary cooperation.
In target, check the acting subject as well as any recipient or object.
If an action is proposed, the acting subject is applicable even when implicit.
A reason about a different actor cannot support this character's action.

For likely_behavior and unlikely_behavior, scope must assess the predicted
behavior in the fictional scene or the general tendency being proposed.
Do not mark scope inapplicable merely because there is no explicit quantifier.
A dated example supports that past event; similarity alone does not establish
future willingness, repetition, or refusal. Identify whether any supplied
basis establishes the proposed future behavior for THIS character.
Return a component review, not an overall approval. Do not rewrite the
claim into a safer instruction in your reasoning. No outside knowledge.

Always return field, claim, evidence_excerpt copied exactly, and ALL six
checks: action, target, manner, sequence, scope, character_facts.
These six checks are TOP-LEVEL keys alongside field, claim, and
evidence_excerpt. Do not wrap them in a "checks" object or an array.
Use this exact top-level layout (replace the placeholder check objects):
{{
  "field": "exact supplied field",
  "claim": "exact supplied claim",
  "evidence_excerpt": "exact supplied evidence excerpt",
  "action": {{"status": "unsupported", "claim_excerpt": "exact claim substring", "source": "none", "support_excerpt": "", "reason": "brief reason"}},
  "target": {{"status": "not_applicable", "claim_excerpt": "", "source": "none", "support_excerpt": "", "reason": "brief reason"}},
  "manner": {{"status": "not_applicable", "claim_excerpt": "", "source": "none", "support_excerpt": "", "reason": "brief reason"}},
  "sequence": {{"status": "not_applicable", "claim_excerpt": "", "source": "none", "support_excerpt": "", "reason": "brief reason"}},
  "scope": {{"status": "not_applicable", "claim_excerpt": "", "source": "none", "support_excerpt": "", "reason": "brief reason"}},
  "character_facts": {{"status": "not_applicable", "claim_excerpt": "", "source": "none", "support_excerpt": "", "reason": "brief reason"}}
}}
The placeholder statuses above are layout examples, not verdicts.
Every applicable check needs a nonempty claim_excerpt copied from EXACT
CLAIM, never from evidence or context. If a dimension is absent, mark it
not_applicable; do not invent a component or mark an empty one unsupported.
All words of the exact claim must appear in the applicable claim excerpts.
Each check has status (supported/unsupported/not_applicable), claim_excerpt,
source (evidence/story_context/none), support_excerpt, and a brief reason.
Use exact contiguous claim excerpts; together applicable checks must cover
the entire claim. Overlapping excerpts are allowed. Do not omit modifiers.
For supported checks, copy an exact support excerpt from the named source.
Unsupported checks use source=none and support_excerpt="".
Not-applicable checks use source=none and both excerpts="".

Judge these dimensions independently:
- action: the exact proposed task or behavior, including its polarity.
- target: the people, groups, objects, and pronoun referents involved.
- manner: method, attitude, emotion, and modifiers such as carefully.
- sequence: prerequisites and temporal or causal order between actions.
- scope: obligation, certainty, quantifiers, and selected versus open outcomes.
- character_facts: biography, traits, status, relationships, skills, access.

Evidence of an evaluation or open decision supports that task itself.
It does not select an outcome, prescribe a method, or order an external
action. Assessing before deciding whether to act can preserve a decision;
assessing before involving someone adds an external-action prerequisite.
Do not replace "before involving" with "before deciding whether to involve".
Neither "and", sentence order, nor a serious allegation establishes order.
If a component has no supplied basis, mark it unsupported, not inapplicable.

A historical event may support avoiding its contradiction. It does not
require mentioning that event in every depiction, establish a recurring
behavior, or establish ongoing access to a past contact. Fictional context
establishes explicit present tasks and circumstances, not unstated canon.
Direct bounded applications of facts are allowed; verbatim repetition of
the full claim in evidence is not required. Plausibility alone is inadequate.

Return only one valid JSON object. No final valid_interpretation field.
"""),
            ("human", """
CHARACTER TO ANALYZE: {character_name}
FIELD: {field}
EXACT CLAIM: {claim}
EVIDENCE EXCERPT: {evidence_excerpt}
CURRENT FICTIONAL STORY CONTEXT: {story_context}
"""),
        ]
    )
    review = safe_invoke(
        interpretation_prompt | character_interpretation_verifier,
        {
            "character_name": character_name or "Not supplied; do not guess an actor from prominence in the evidence.",
            "field": audited_claim.field,
            "claim": audited_claim.claim,
            "evidence_excerpt": audited_claim.evidence_excerpt,
            "story_context": story_context,
        },
        label="Character bounded interpretation",
    )
    return finalize_character_interpretation_review(
        audited_claim, story_context, review,
    )


def validate_character_interpretation_verification(
    audited_claim: CharacterClaimAudit,
    verification: CharacterInterpretationVerification,
) -> None:
    """
    Ensure the interpretation verifier judged exactly
    the claim and evidence it was given.
    """

    if verification.field != audited_claim.field:
        raise ValueError(
            "Interpretation verifier changed the claim field."
        )

    if verification.claim != audited_claim.claim:
        raise ValueError(
            "Interpretation verifier changed the claim text."
        )

    if verification.evidence_excerpt != audited_claim.evidence_excerpt:
        raise ValueError(
            "Interpretation verifier changed the evidence excerpt."
        )

    if (
        verification.valid_interpretation
        and verification.unsupported_additions
    ):
        raise ValueError(
            "Interpretation verifier returned a contradictory verdict: "
            "valid_interpretation=True while unsupported_additions "
            "contains unsupported material.\n"
            f"Field: {verification.field}\n"
            f"Unsupported additions: "
            f"{verification.unsupported_additions}"
        )

    if (
        not verification.valid_interpretation
        and not verification.unsupported_additions
    ):
        raise ValueError(
            "Interpretation verifier returned a contradictory verdict: "
            "valid_interpretation=False but identified no unsupported "
            "material additions.\n"
            f"Field: {verification.field}\n"
            f"Claim: {verification.claim}\n"
            f"Reason: {verification.reason}"
        )


def find_whitespace_equivalent_evidence_excerpt(
    evidence_excerpt: str,
    allowed_sources: list[str],
) -> str | None:
    """
    Restore the exact source substring when differences are limited
    to whitespace or capitalization of the first ASCII letter.
    """
    text = evidence_excerpt.strip()
    if not text:
        return None

    pattern = r"\s+".join(
        re.escape(token)
        for token in re.split(r"\s+", text)
    )

    # Prefer matches that preserve all capitalization.
    for source in allowed_sources:
        match = re.search(pattern, source)
        if match:
            return match.group(0)

    # Permit only the excerpt's first letter to change case.
    if text[0].isascii() and text[0].isalpha():
        pattern = (
            "[" + text[0].lower() + text[0].upper() + "]"
            + pattern[1:]
        )
        for source in allowed_sources:
            match = re.search(pattern, source)
            if match:
                return match.group(0)

    return None

def validate_character_claim_evidence(
    audit_batch: CharacterClaimAuditBatch,
    verified_canon_research: str,
    voice_research: str,
    story_context: str,
) -> None:
    """
    Verify that every supported factual claim includes an exact
    evidence excerpt that actually exists in an allowed source.

    Remaining research questions are allowed to be supported
    without an evidence excerpt.
    """

    allowed_sources = [
        verified_canon_research,
        voice_research,
        story_context,
    ]

    for index, audited_claim in enumerate(
            audit_batch.claims
    ):
        is_remaining_question = (
            audited_claim.field.startswith(
                "remaining_canon_questions["
            )
            or audited_claim.field.startswith(
                "remaining_voice_questions["
            )
        )

        if is_remaining_question:
            if audited_claim.evidence_excerpt:
                raise ValueError(
                    "Remaining research question unexpectedly "
                    "contained an evidence excerpt.\n"
                    f"Field: {audited_claim.field}\n"
                    f"Excerpt: {audited_claim.evidence_excerpt}"
                )

            continue

        if not audited_claim.supported:
            if audited_claim.evidence_excerpt:
                raise ValueError(
                    "Unsupported character claim unexpectedly "
                    "contained an evidence excerpt.\n"
                    f"Field: {audited_claim.field}\n"
                    f"Excerpt: {audited_claim.evidence_excerpt}"
                )

            continue

        if not audited_claim.evidence_excerpt:
            raise ValueError(
                "Supported character claim did not provide "
                "an evidence excerpt.\n"
                f"Field: {audited_claim.field}\n"
                f"Claim: {audited_claim.claim}"
            )

        excerpt_exists = any(
            audited_claim.evidence_excerpt in source
            for source in allowed_sources
        )

        if not excerpt_exists:
            canonical_excerpt = (
                find_whitespace_equivalent_evidence_excerpt(
                    evidence_excerpt=audited_claim.evidence_excerpt,
                    allowed_sources=allowed_sources,
                )
            )

            if canonical_excerpt is None:
                audit_batch.claims[index] = audited_claim.model_copy(
                    update={
                        "supported": False,
                        "evidence_excerpt": "",
                        "reason": (
                            "Claim auditor supplied an evidence excerpt "
                            "that does not exist in any allowed source."
                        ),
                    }
                )
                continue

            audit_batch.claims[index] = audited_claim.model_copy(
                update={
                    "evidence_excerpt": canonical_excerpt,
                }
            )

def validate_character_claim_audit_coverage(
    character_analysis: CharacterAnalysis,
    audit_batch: CharacterClaimAuditBatch,
) -> None:
    """
    Ensure the claim auditor returned exactly one verdict
    for every Python-enumerated claim, in the same order,
    with the same field and claim text.
    """

    expected_claims = enumerate_character_analysis_claims(
        character_analysis
    )

    audited_claims = audit_batch.claims

    if len(audited_claims) != len(expected_claims):
        raise ValueError(
            "Character claim audit returned the wrong number of claims. "
            f"Expected {len(expected_claims)}, "
            f"received {len(audited_claims)}."
        )

    for index, expected in enumerate(expected_claims):
        audited = audited_claims[index]

        if audited.field != expected["field"]:
            raise ValueError(
                "Character claim audit changed or reordered a field.\n"
                f"Expected: {expected['field']}\n"
                f"Received: {audited.field}"
            )

        if audited.claim != expected["claim"]:
            raise ValueError(
                "Character claim audit changed or reordered a claim.\n"
                f"Field: {expected['field']}\n"
                f"Expected: {expected['claim']}\n"
                f"Received: {audited.claim}"
            )

def validate_remaining_character_questions(
    character_analysis: CharacterAnalysis,
) -> None:
    """
    Validate unresolved research questions generated during
    CharacterAnalysis using the same semantic rules as the
    original research planner.
    """

    validate_canon_question_content(
        character_analysis.remaining_canon_questions
    )

    validate_unverified_character_attributes(
        character_analysis.remaining_canon_questions
    )

    validate_voice_question_content(
        character_analysis.remaining_voice_questions
    )

# ---------------------------------------------------------
# Model Configuration
# ---------------------------------------------------------


model = create_chat_model(
    model_name=CHARACTER_MODEL,
    temperature=0.3,
    timeout=30,
    max_retries=0,
)

character_planner_model = create_chat_model(
    model_name=CHARACTER_MODEL,
    temperature=0.3,
    timeout=30,
    max_retries=0,
    reasoning_effort="low",
)

character_repair_model = create_chat_model(
    model_name=CHARACTER_MODEL,
    temperature=0.0,
    timeout=30,
    max_retries=0,
    max_tokens=2400,
    reasoning_effort="low",
)

character_grounding_model = create_chat_model(
    model_name=CHARACTER_MODEL,
    temperature=0.0,
    timeout=30,
    max_retries=0,
    max_tokens=2400,
    reasoning_effort="low",
)

character_claim_audit_model = create_chat_model(
    model_name=CHARACTER_MODEL,
    temperature=0.0,
    timeout=30,
    max_retries=0,
    max_tokens=4096,
    reasoning_effort="low",
)

character_research_planner = (
    character_planner_model.with_structured_output(
        CharacterResearchPlan,
        method="json_mode",
    )
)

character_research_repairer = (
    character_repair_model.with_structured_output(
        CharacterResearchPlan,
        method="json_mode",
    )
)

character_analyzer = model.with_structured_output(
    CharacterAnalysis,
    method="json_mode",
)

character_analysis_repairer = model.with_structured_output(
    CharacterAnalysis,
    method="json_mode",
)

character_analysis_auditor = model.with_structured_output(
    CharacterAnalysisAudit,
    method="json_mode",
)

character_claim_auditor = character_claim_audit_model.with_structured_output(
    CharacterClaimAuditBatch,
    method="json_mode",
)

character_evidence_entailment_verifier = character_grounding_model.with_structured_output(
    CharacterEvidenceEntailment,
    method="json_mode",
)

character_interpretation_verifier = character_grounding_model.with_structured_output(
    CharacterInterpretationReview,
    method="json_mode",
)


# ---------------------------------------------------------
# Character Analysis
# ---------------------------------------------------------


def analyze_character(
    research_plan: CharacterResearchPlan,
    canon_research: str,
    voice_research: str,
) -> CharacterAnalysis:
    """
    Analyze a Star Trek character using supplied canon evidence
    and era-appropriate dialogue research.

    This stage interprets evidence but does not perform research.
    """

    response = safe_invoke(
        character_analyzer,
        [
            (
                "system",
                """
You are a Star Trek character analysis specialist.

Your job is to determine how a character should behave and
communicate within a proposed story using supplied research.

You are NOT performing canon research yourself.

You will receive two separate evidence sources:

1. Canon research
   This establishes factual information about history,
   relationships, allegiances, past behavior, political context,
   and relevant events.

2. Voice research
   This establishes evidence about dialogue style, syntax,
   vocabulary, rhetorical habits, formality, humor, emotional
   expression, forms of address, and era-specific language.

EVIDENCE BUCKET RULES:

The canon research may contain three evidence buckets:

1. VERIFIED CANON EVIDENCE
   - This is the ONLY bucket that may be treated as established canon.
   - Factual claims in motivations, likely_behavior, unlikely_behavior,
     relationships, behavioral_constraints, and out_of_character_risks
     must be grounded in this evidence or clearly framed as interpretation
     of this evidence.

2. UNRESOLVED CANON QUESTIONS
   - This is NOT established canon.
   - Do not convert claims from this bucket into character facts.
   - Preserve relevant uncertainty in remaining_canon_questions.

3. UNSUPPORTED CANON PREMISES
   - Treat these premises as unestablished.
   - Never incorporate them into motivations, relationships, behavior,
     history, or other factual character claims.

If a claim appears only in an unresolved or unsupported bucket,
omit it from factual character fields.

STRICT EVIDENCE DERIVATION RULE:

Do not use your internal knowledge of the character to complete,
embellish, or infer character traits that are absent from VERIFIED
CANON EVIDENCE.

Every factual or character-specific claim placed in:

- motivations
- likely_behavior
- unlikely_behavior
- relationships
- behavioral_constraints
- out_of_character_risks

must be directly supported by VERIFIED CANON EVIDENCE or be a
clearly necessary interpretation of that verified evidence.

A claim is not permitted merely because it is canonically true,
widely known about the character, plausible, or consistent with
your prior knowledge.

If VERIFIED CANON EVIDENCE is insufficient to support a useful
claim for one of these fields, return fewer items or an empty list.

ATOMIC CLAIM CONSTRUCTION RULE:

Every CharacterAnalysis claim must contain only ONE material assertion.

Do not combine:

- a verified fact with an unsupported interpretation
- an established event with a relationship label
- an observed action with a personality trait
- a past interaction with an ongoing relationship
- a fact with an inferred motive
- a fact with an inferred professional or political status
- a supported observation with a causal explanation not present
  in the supplied evidence

If only part of a possible sentence is supported, output only the
supported part or omit the item entirely.

STRICT FACTUAL FIELDS:

For:

- story_role
- relationships
- speech_patterns
- preferred_vocabulary
- communication_style
- era_language_notes

use wording that stays as close as practical to the supplied evidence.

Do not add descriptive relationship labels such as:

- cooperative
- secretive
- pragmatic
- uneasy
- trusted
- adversarial
- loyal
- collaborative

unless that relationship quality itself is explicitly established
by the supplied evidence.

A single interaction does not establish an ongoing relationship.

For example:

NOT ALLOWED:
"Covertly cooperative relationship with Captain Sisko,
demonstrated by the 2372 suit-fitting cover story."

ALLOWED WHEN SUPPORTED:
"In 2372, Sisko used the pretense of Garak measuring him
for a suit as cover while Worf delivered Sisko an
intelligence report."

The second wording preserves the established interaction without
inventing a broader relationship characterization.

STORY ROLE RULE:

story_role may describe only the immediate role explicitly established
by the current fictional story context or VERIFIED CANON EVIDENCE.

Do not add biography, profession, intelligence status, affiliation,
rank, or historical identity merely because it would explain why the
character is suited to the role.

For example:

STORY CONTEXT:
"Garak must decide how seriously to take the claim and whether
to involve the DS9 command staff."

SUPPORTED STORY ROLE:
"Evaluate the defector's claim and decide whether to involve
the DS9 command staff."

UNSUPPORTED STORY ROLE:
"Former intelligence operative evaluating a defector's claim."

INTERPRETIVE FIELD RULE:

For:

- motivations
- likely_behavior
- unlikely_behavior
- dramatic_tension
- behavioral_constraints
- out_of_character_risks

each item must express only one bounded interpretation.

Do not attach unsupported personality traits, motives, loyalties,
professional history, access, secrecy, political goals, or behavioral
patterns to an otherwise supported interpretation.

If no narrow bounded interpretation is justified, leave the field
empty instead of creating richer characterization.

Prefer sparse, defensible analysis over complete but speculative
characterization.

Do not promote facts found only in UNRESOLVED CANON QUESTIONS,
UNSUPPORTED CANON PREMISES, or your own background knowledge.

Story context may establish the fictional situation the character
is currently facing, but it must not be treated as evidence of the
character's pre-existing history, relationships, personality,
skills, motives, or behavioral tendencies.

VOICE EVIDENCE RULES:

Voice-related conclusions must come only from supplied voice research.

If the voice research states that no dialogue or voice research has been performed:

- speech_patterns must be an empty list
- preferred_vocabulary must be an empty list
- communication_style must be an empty list
- era_language_notes must not invent character-specific speech habits,
  vocabulary, formality, titles, idioms, cadence, sarcasm, humor,
  rhetorical devices, or other dialogue characteristics
- preserve unresolved voice needs in remaining_voice_questions

Do not use general Star Trek knowledge, internal model knowledge,
stereotypes about Cardassians, or remembered Garak dialogue to fill
missing voice evidence.

Behavioral observations are not voice evidence.
Do not convert personality traits into speech-style conclusions.

Rules:
- Treat supplied research as the factual basis for your analysis.
- Do not invent canon facts that are absent from the supplied research.
- Do not silently rely on internal Star Trek knowledge.
- Clearly distinguish supplied evidence from your interpretation.
- Behavioral judgments may be inferred from supplied evidence,
  but they must remain consistent with that evidence.
- Do not turn possibilities, allegations, or unresolved questions
  into established facts.
- Do not write scenes or dialogue.

- Treat allegations in story_context as fictional story claims unless
  the supplied canon research independently establishes them.
- Do not convert an unsupported story allegation into character
  knowledge, established history, or established canon.
- Respect unresolved canon questions as unresolved.
- Respect unsupported canon premises as unsupported.
- Do not fill research gaps using internal knowledge.

- Do not transfer roles, labels, relationships, occupations, actions,
  or attributes from other people described in story_context onto
  the character being analyzed.
- A role such as "defector", "informant", "agent", "traitor",
  "infiltrator", or similar description belongs only to the person
  to whom story_context explicitly assigns it.
- Do not assign the analyzed character a rank, title, position,
  relationship, affiliation, or historical status unless it is
  established by the supplied research or explicitly supplied
  by the user.

- likely_behavior should describe behavior supported by the evidence.
- unlikely_behavior should identify behavior that would conflict
  with the evidence.
- motivations may interpret supplied evidence, but must not invent
  motives unsupported by that evidence.
- relationships must be based on relationships established in the
  supplied research.
- behavioral_constraints should give practical boundaries for a
  future scriptwriter.
- out_of_character_risks should identify specific ways a writer
  could accidentally make the character inconsistent with the
  supplied evidence.
- If canon evidence is insufficient, preserve the uncertainty in
  remaining_canon_questions rather than guessing.

- Do not imitate later Star Trek production conventions when the
  supplied story era predates them.
- Keep terminology, forms of address, character voice, and speech
  conventions appropriate to the supplied story era.
- Do not treat all Star Trek series or eras as linguistically identical.
- Do not infer speech patterns from biography or canon history alone.
- Base speech_patterns, preferred_vocabulary,
  communication_style, and era_language_notes primarily on
  supplied voice research.
- If voice evidence is insufficient, preserve the uncertainty in
  remaining_voice_questions rather than inventing a style rule.
- Do not use the fictional scene in story_context as evidence for
  how the character speaks.
- Apply researched voice patterns to characterization only after
  those patterns have been established by supplied dialogue evidence.

Return valid JSON matching this exact structure:

{
    "character_name": "string",
    "story_role": "string",
    "motivations": [
        "string"
    ],
    "likely_behavior": [
        "string"
    ],
    "unlikely_behavior": [
        "string"
    ],
    "relationships": [
        "string"
    ],
    "dramatic_tension": "string",
    "remaining_canon_questions": [
        "question"
    ],
    "remaining_voice_questions": [
        "question"
    ],
    "speech_patterns": [
        "string"
    ],
    "preferred_vocabulary": [
        "string"
    ],
    "communication_style": [
        "string"
    ],
    "behavioral_constraints": [
        "string"
    ],
    "era_language_notes": [
        "string"
    ],
    "out_of_character_risks": [
        "string"
    ]
}

Output requirements:
- Return only the JSON object.
- Do not use markdown.
- Do not include commentary outside the JSON.
- Use valid JSON syntax.
- Use double quotes for keys and string values.
                """,
            ),
            (
                "human",
                f"""
Character:
{research_plan.character_name}

Story context:
{research_plan.story_context}

Story era:
{research_plan.story_era}

Canon research:
{canon_research}

Voice research:
{voice_research}

Analyze the character for use in the proposed story.
                """,
            ),
        ],
        label="Character analysis",
    )

    try:
        validate_remaining_character_questions(
            response
        )

    except ValueError as error:
        repaired_response = repair_character_analysis_questions(
            character_analysis=response,
            validation_error=str(error),
        )

        # Repair gets exactly one chance.
        validate_remaining_character_questions(
            repaired_response
        )

        return repaired_response

    return response

#Audit character analysis for grounding in supplied evidence
def audit_character_analysis(
    research_plan: CharacterResearchPlan,
    canon_research: str,
    voice_research: str,
    character_analysis: CharacterAnalysis,
) -> CharacterAnalysisAudit:

    audit_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are a strict grounding auditor for a Star Trek character analysis.

Your job is NOT to improve the analysis.
Your job is NOT to add canon knowledge.
Your job is only to determine whether every character-specific claim
is supported by the supplied evidence.

Allowed evidence sources:

1. VERIFIED CANON EVIDENCE
2. Supplied voice research
3. The current fictional story context

UNRESOLVED CANON QUESTIONS are not established facts.
UNSUPPORTED CANON PREMISES are not established facts.

STRICT SOURCE RULE:

UNRESOLVED CANON QUESTIONS are prohibited as supporting evidence.

Do not use any statement, subclaim, implication, or factual-sounding
language contained inside the UNRESOLVED CANON QUESTIONS section to
validate a CharacterAnalysis claim.

A statement appearing in an unresolved bucket remains unresolved even
if that statement itself sounds factual or canonically correct.

AUDIT COVERAGE AND INTERPRETATION RULES:

Audit EVERY CharacterAnalysis field, including:

- story_role
- motivations
- likely_behavior
- unlikely_behavior
- relationships
- dramatic_tension
- remaining_canon_questions
- remaining_voice_questions
- speech_patterns
- preferred_vocabulary
- communication_style
- behavioral_constraints
- era_language_notes
- out_of_character_risks

Do not skip a field merely because it is descriptive, interpretive,
or phrased as a question.

For remaining_canon_questions and remaining_voice_questions:
- unresolved subjects may be preserved as questions
- the wording must not silently convert an unresolved fact into an
  established fact

BOUNDED INTERPRETATION IS ALLOWED.

The fictional story context may directly support immediate story tasks
or decisions explicitly assigned to the character.

For example, if story_context says the character must decide how
seriously to take a claim, an interpretation such as "assess the
credibility of the claim" is supported.

Verified canon evidence may also be applied to the current fictional
situation when the connection is direct and introduces no new factual
claim, relationship, motive, personality trait, or behavioral tendency.

However, do not allow extra characterization to be smuggled into an
otherwise supported interpretation.

For example:

SUPPORTED:
"Assess the credibility of the claim."

NOT SUPPORTED without additional evidence:
"Discreetly assess the claim because of his characteristic caution."

The first follows directly from story_context.
The second invents a behavioral trait.

UNSUPPORTED CANON PREMISES are also prohibited as supporting evidence.

For every character-specific claim, ask:

"Can I point to support for this claim specifically within VERIFIED
CANON EVIDENCE, supplied voice research, or an explicit fact in the
current fictional story context?"

If the answer is no, flag the claim.

Do not validate a claim merely because:
- it is plausible
- it is canonically true from your own knowledge
- it follows from the character's reputation or background
- a related fact is verified
- it appears inside an unresolved section
- the story context makes the behavior seem reasonable

A verified relationship or interaction does not automatically establish
trust, loyalty, current access, motive, secrecy, behavioral tendency,
or the ability to contact that person again.

Do not use your own internal Star Trek knowledge to validate a claim.

Flag any claim that:

- comes only from unresolved canon research
- comes only from an unsupported premise
- appears to come from internal model knowledge
- invents a relationship, motive, skill, personality trait, status,
  affiliation, behavioral tendency, or dialogue characteristic
- turns the fictional story premise into pre-existing canon history

A plausible or canonically correct claim must still be flagged if the
supplied evidence does not support it.

Return the audit as a JSON object matching the required structured output schema.

Set grounded=True only when all character-specific claims are supported.
Always include the top-level fields "grounded", "issues", and "reason".
Return the audit as a JSON object matching the required structured output schema.

Always include the top-level fields "grounded", "issues", and "reason".

Every object in "issues" must contain exactly these three fields:

- "field": the CharacterAnalysis field containing the problem,
  using an indexed path when appropriate, such as
  "motivations[0]", "likely_behavior[1]", or "story_role"

- "claim": the exact or concise unsupported claim being audited

- "reason": why that claim is not supported by the allowed evidence

Never omit "field".
Do not place the field name inside "claim".

If there are no issues, return "issues": [].

Set grounded=True only when all character-specific claims are supported.

""",
            ),
            (
                "human",
                """
CHARACTER:
{character_name}

STORY CONTEXT:
{story_context}

STORY ERA:
{story_era}

CANON RESEARCH:
{canon_research}

VOICE RESEARCH:
{voice_research}

CHARACTER ANALYSIS TO AUDIT:
{character_analysis}
""",
            ),
        ]
    )

    audit_chain = audit_prompt | character_analysis_auditor

    return safe_invoke(
        audit_chain,
        {
            "character_name": research_plan.character_name,
            "story_context": research_plan.story_context,
            "story_era": research_plan.story_era,
            "canon_research": canon_research,
            "voice_research": voice_research,
            "character_analysis": character_analysis.model_dump_json(indent=2),
        },
        label="Character analysis audit",
    )


def format_forbidden_phrases(
    phrases: list[str],
) -> str:
    """
    Format Python validation phrases so the repair model
    receives the exact same restrictions as the validators.
    """

    return "\n".join(
        f'- "{phrase}"'
        for phrase in phrases
    )

def repair_character_analysis_questions(
    character_analysis: CharacterAnalysis,
    validation_error: str,
) -> CharacterAnalysis:
    """
    Repair invalid unresolved research questions generated
    during CharacterAnalysis.

    Only remaining_canon_questions and remaining_voice_questions
    may change. Python preserves every other analysis field.
    """

    forbidden_ambiguous_phrases = format_forbidden_phrases(
        AMBIGUOUS_RESEARCH_PHRASES
    )

    forbidden_canon_phrases = format_forbidden_phrases(
        SPECULATIVE_CANON_PHRASES
    )

    forbidden_voice_phrases = format_forbidden_phrases(
        SPECULATIVE_VOICE_PHRASES
    )

    repaired_analysis = safe_invoke(
        character_analysis_repairer,
        [
            (
                "system",
                """
    You are repairing invalid unresolved research questions inside
    an existing Star Trek CharacterAnalysis.

    Your job is NOT to redo the character analysis.

    You may repair only:

    - remaining_canon_questions
    - remaining_voice_questions

    Rules:

    - Preserve every other CharacterAnalysis field exactly.
    - Do not add new character facts.
    - Do not answer the research questions.
    - Do not use internal Star Trek knowledge to fill research gaps.
    - Questions may be removed if they are no longer valid or useful.
    - It is acceptable for either remaining-question list to be empty.

    For remaining_canon_questions:

    - Ask only factual canon-research questions.
    - Do not predict behavior, motives, reactions, or consequences.
    - Do not assume an uncertain fact is already true.
    - Use neutral wording when existence or status is uncertain.

    For remaining_voice_questions:

    - Ask only for observable dialogue or linguistic evidence.
    - Research syntax, vocabulary, forms of address, rhetorical patterns,
      formality, cadence, terminology, or similar observable features.
    - Do not ask what dialogue reflects, reveals, implies, or suggests
      about biography, personality, motives, training, or background.
    - Do not predict how the character would speak in the proposed story.

    Return valid JSON matching the complete CharacterAnalysis schema.
    Return only JSON.
    """,
            ),
            (
                "human",
                f"""
    The CharacterAnalysis failed Python validation.

    Validation error:
    {validation_error}

    Current CharacterAnalysis:
    {character_analysis.model_dump_json(indent=2)}

    PYTHON VALIDATION RULES

    Forbidden ambiguous phrases:
    {forbidden_ambiguous_phrases}

    Forbidden canon-question phrases:
    {forbidden_canon_phrases}

    Forbidden voice-question phrases:
    {forbidden_voice_phrases}

    Repair all invalid remaining research questions in one response.
    """,
            ),
        ],
        label="Character analysis question repair",
    )


    # -----------------------------------------------------
    # Python preserves every non-question analysis field
    # -----------------------------------------------------

    preserved_fields = character_analysis.model_dump(
        exclude={
            "remaining_canon_questions",
            "remaining_voice_questions",
        }
    )

    repaired_analysis = repaired_analysis.model_copy(
        update=preserved_fields
    )

    return repaired_analysis

def repair_character_research_plan(
    research_plan: CharacterResearchPlan,
    validation_error: str,
) -> CharacterResearchPlan:
    """
    Give the Character Agent one opportunity to repair
    invalid research questions.

    Preserve the rest of the research plan.
    """

    # -----------------------------------------------------
    # Share exact Python validation rules with repair model
    # -----------------------------------------------------

    forbidden_ambiguous_phrases = format_forbidden_phrases(
        AMBIGUOUS_RESEARCH_PHRASES
    )
    detected_violations: list[str] = []

    for index, question in enumerate(
            research_plan.canon_questions
    ):
        normalized = question.casefold()

        for phrase in AMBIGUOUS_RESEARCH_PHRASES:
            if phrase in normalized:
                detected_violations.append(
                    f"canon_questions[{index}] contains "
                    f"ambiguous phrase '{phrase}': {question}"
                )

        for phrase in SPECULATIVE_CANON_PHRASES:
            if phrase in normalized:
                detected_violations.append(
                    f"canon_questions[{index}] contains "
                    f"speculative phrase '{phrase}': {question}"
                )

        if re.search(
            r"\(\s*(?:e\.g\.\s*,?|for example\b|such as\b)",
            question,
            flags=re.IGNORECASE,
        ):
            detected_violations.append(
                f"canon_questions[{index}] bundles illustrative "
                f"examples: {question}"
            )

        compound_targets = (
            r"\b(?:official\s+)?(?:position|rank|role)\s+and\s+"
            r"(?:activities|actions|interactions)\b",
            r"\bpast\s+as\b.+\band\s+any\b",
            r"\band\s+how\b",
        )
        if any(
            re.search(pattern, normalized)
            for pattern in compound_targets
        ):
            detected_violations.append(
                f"canon_questions[{index}] combines distinct "
                f"factual targets: {question}"
            )

        for pattern in UNVERIFIED_CHARACTER_ATTRIBUTE_PATTERNS:
            if re.search(pattern, normalized):
                detected_violations.append(
                    f"canon_questions[{index}] contains an "
                    f"unverified character attribute: {question}"
                )

        if not question.strip().endswith("?"):
            detected_violations.append(
                f"canon_questions[{index}] does not end "
                f"with a question mark: {question}"
            )

    for index, question in enumerate(
            research_plan.voice_questions
    ):
        normalized = question.casefold()

        for phrase in AMBIGUOUS_RESEARCH_PHRASES:
            if phrase in normalized:
                detected_violations.append(
                    f"voice_questions[{index}] contains "
                    f"ambiguous phrase '{phrase}': {question}"
                )

        for phrase in SPECULATIVE_VOICE_PHRASES:
            if phrase in normalized:
                detected_violations.append(
                    f"voice_questions[{index}] contains "
                    f"speculative phrase '{phrase}': {question}"
                )

        if not question.strip().endswith("?"):
            detected_violations.append(
                f"voice_questions[{index}] does not end "
                f"with a question mark: {question}"
            )

    detected_violation_report = (
            "\n".join(
                f"- {violation}"
                for violation in detected_violations
            )
            or "- No phrase-level violations were detected."
    )

    forbidden_canon_phrases = format_forbidden_phrases(
        SPECULATIVE_CANON_PHRASES
    )

    forbidden_voice_phrases = format_forbidden_phrases(
        SPECULATIVE_VOICE_PHRASES
    )

    repaired_plan = safe_invoke(
        character_research_repairer,
        [
            (
                "system",
                """
You are correcting a previously generated Star Trek
character research plan.

The plan failed Python validation because one or more
research questions were not suitable for independent
research by another agent.

Your job is to repair the research questions only.

Rules:
- Preserve character_name exactly.
- Preserve story_context exactly.
- Preserve story_era exactly.
- Do not paraphrase, summarize, normalize, reinterpret, or expand
  these three fields.
- Do not convert relative era wording into a specific year unless
  that year was supplied in story_era.

- Treat the supplied validation error as evidence of at least one
  problem, not necessarily the only problem in the plan.
- Before returning the repaired plan, review EVERY canon question
  and EVERY voice question against ALL rules in this prompt.
- Repair every invalid question in a single response.
- Do not stop after correcting only the question mentioned in the
  validation error.
- Preserve questions verbatim only when they already satisfy all
  rules.

- Keep between 2 and 5 canon_questions.
- Keep a useful research baseline: at least two distinct canon
  questions must probe the named character's documented history,
  actions, knowledge, or relationships relevant to story_era.
- At most one canon question may probe whether the particular new
  allegation or encounter from story_context has a canon precedent.
- Ask one factual thing per question plus a necessary era boundary;
  do not combine a rank with an affiliation, an offer with a later
  mission, or an interaction with a level of trust.
- Do not put illustrative lists (e.g., for example, such as) inside
  canon questions. For interactions, ask about one named counterpart
  or one coherent group. Do not assign a person to a group unless
  the user's input established that membership.
- Prefer one documented action or interaction with an identifiable
  point in the requested era over a broad relationship or status
  that the source may not date. Ask about an affiliation, activity,
  contact, or response separately when each needs its own evidence.
- Do not impose a period earlier than story_era unless that earlier
  history directly informs the proposed story.
- Keep between 2 and 5 voice_questions.
- Do not answer the questions.
- Do not invent canon facts.

- Every research question must be independently understandable.
- Include the relevant character, faction, institution, event,
  series, or timeframe directly when required for clarity.
- Time-sensitive questions must explicitly identify the relevant
  historical period or event.

- canon_questions must request established factual information only.
- Canon questions must concern facts such as history, relationships,
  affiliations, allegiances, professional roles, institutions,
  political circumstances, conditions, and established events.
- Do not ask Lore to predict behavior, reactions, decisions,
  consequences, stakes, outcomes, or what might happen if a
  fictional premise is true.
- Questions about what canon evidence means for characterization
  belong to the later Character Analysis stage.

- Canon questions must be written so that "no canon evidence exists"
  is always a logically valid possible answer.
- Do not phrase a canon question in a way that presupposes the
  existence of a rank, role, relationship, interaction, procedure,
  event, policy, infiltration, defection, affiliation, or historical
  example.
- When existence is uncertain, use neutral formulations such as:
  "Does canon document..."
  "What does canon establish about..."
  "Is there canon evidence that..."
  "What, if anything, does canon establish about..."
- Avoid wording such as "Which..." or "What instances..." when it
  assumes that examples necessarily exist.

- Treat allegations in story_context as allegations rather than
  established canon facts.
- If story_context contains a claim involving infiltration,
  conspiracy, betrayal, defection, deception, or secret activity,
  research established canon surrounding the claim rather than
  assuming the claim is true.

- Do not transfer roles, labels, relationships, occupations,
  affiliations, actions, or attributes from another person in
  story_context onto the researched character.
- A fictional person's role such as "defector", "informant",
  "agent", "traitor", or "infiltrator" must not be attributed to
  the researched character unless the user explicitly assigned it.

- Do not assign a rank, title, position, affiliation, or honorific
  unless that information was explicitly supplied by the user.
- If rank, title, position, or affiliation matters, ask canon
  research to establish it using neutral wording.

- Institutional research must concern established institutions,
  policies, practices, or historical conditions generally.
- Do not construct a supposed canon procedure around the specific
  fictional encounter described in story_context.

- voice_questions must request observable dialogue evidence from
  the supplied Star Trek era.
- Keep voice research specific to the supplied series and period.
- Do not treat all Star Trek series or eras as stylistically identical.
- Do not infer dialogue style from biography or personality alone.

- Voice questions must not assume that a fictional interaction in
  story_context occurred in canon.
- Voice questions must not reproduce or presuppose the specific
  fictional decision, encounter, informant, allegation, or dilemma
  supplied in story_context.
- Research comparable established dialogue contexts instead.
- Ask how the character speaks in established categories of
  interactions rather than how the character speaks while performing
  the proposed fictional action.

- Voice questions must not predict how the character might speak,
  react, behave, or use a rhetorical technique in the proposed scene.
- Do not ask how the character "might", "could", or "would" employ
  a speech pattern in story_context.
- Voice questions must request observable linguistic evidence only.
- Do not ask what dialogue patterns reveal, imply, suggest, or
  indicate about motives, intentions, attitudes, or personality.
- Interpretation belongs to the later Character Analysis stage.

- The Python validation lists supplied in the human message are
  authoritative.
- Do not use any phrase from those lists in a repaired question.
- Review all questions against those lists before returning JSON.

Return valid JSON matching this exact structure:

{
    "character_name": "string",
    "story_context": "string",
    "story_era": "string",
    "canon_questions": [
        "question",
        "question"
    ],
    "voice_questions": [
        "question",
        "question"
    ]
}

Return only valid JSON.
Do not use markdown.
Do not include commentary before or after the JSON.
                """,
            ),
            (
                "human",
                f"""
The previous Character Research Plan failed validation.

Validation error:
{validation_error}


PYTHON DETECTED VIOLATIONS ACROSS THE FULL PLAN

Python scanned every canon question and every voice question
before this repair attempt and detected the following violations:

{detected_violation_report}

Every violation listed above MUST be repaired in this single response.
Do not return the plan until none of the listed violations remain.


Character:
{research_plan.character_name}

Story context:
{research_plan.story_context}

Story era:
{research_plan.story_era}

Canon questions:
{research_plan.canon_questions}

Voice questions:
{research_plan.voice_questions}


IMPORTANT PYTHON VALIDATION RULES

Python will reject ANY canon or voice question containing
any of these ambiguous phrases:

{forbidden_ambiguous_phrases}


Python will reject ANY canon question containing
any of these speculative phrases:

{forbidden_canon_phrases}


Python will reject ANY voice question containing
any of these speculative phrases:

{forbidden_voice_phrases}


Before returning the repaired plan:

1. Review every canon question.
2. Review every voice question.
3. Ensure none contains any forbidden phrase listed above.
4. Ensure all questions satisfy the semantic rules in the
   system instructions.
5. Repair all violations in this single response.
6. Preserve character_name, story_context, and story_era exactly.

Return the corrected research plan.
                """,
            ),
        ],
        label="Character research plan repair",
    )

    return repaired_plan


# ---------------------------------------------------------
# Immutable User Input Enforcement
# ---------------------------------------------------------


def enforce_research_plan_inputs(
    research_plan: CharacterResearchPlan,
    character_name: str,
    story_context: str,
    story_era: str,
) -> CharacterResearchPlan:
    """
    Force user-supplied Character Research Plan fields
    to remain exactly as provided.

    The model controls research-question generation.
    Python controls immutable user input.
    """

    return research_plan.model_copy(
        update={
            "character_name": character_name,
            "story_context": story_context,
            "story_era": story_era,
        }
    )


# ---------------------------------------------------------
# Character Agent Workflow - Version 0.2
# ---------------------------------------------------------


def create_character_research_plan(
    character_name: str,
    story_context: str,
    story_era: str,
) -> CharacterResearchPlan:
    """
    Identify the canon and dialogue research needed before
    making characterization decisions about a Star Trek character.
    """

    response = safe_invoke(
        character_research_planner,
        [
            (
                "system",
                """
You are a Star Trek character research specialist.

Your job is NOT to decide how the character should behave yet.

Your job is to identify the factual canon research and
era-appropriate dialogue research needed before another stage
performs character analysis.

You must distinguish between two types of research:

1. canon_questions
   These concern established facts such as history, relationships,
   allegiances, past behavior, political circumstances, professional
   roles, and relevant events.

2. voice_questions
   These concern the character's actual dialogue and communication
   patterns, including syntax, vocabulary, formality, rhetorical
   habits, humor, emotional directness, technical terminology,
   forms of address, and recurring speech patterns.

Rules:
- Copy character_name, story_context, and story_era exactly as
  supplied by the user.
- Do not paraphrase, summarize, normalize, reinterpret, or expand
  these three fields.
- Do not convert relative era wording into a specific year unless
  that year was supplied by the user.

- Do not answer any research questions yourself.
- Do not make final characterization judgments.
- Do not invent canon facts.
- Do not assume uncertain facts are true.

- Every research question must be independently understandable.
- Include the character name directly when needed for clarity.
- Include the relevant Star Trek era or story period directly
  when necessary.
- Time-sensitive questions must explicitly identify the relevant
  historical period or event.
- Do not use ambiguous references such as:
  "this timeframe,"
  "that timeframe,"
  "this time period,"
  "that time period,"
  "this era,"
  "that era,"
  "this period,"
  "that period,"
  "at this point,"
  "in this situation,"
  "currently,"
  "at the time,"
  or similar wording that depends on outside context.

- Generate between 2 and 5 focused canon_questions.
- Build a useful baseline before probing uncertain story details: include
  at least two distinct canon questions about the named character's
  documented history, actions, knowledge, or relationships relevant
  to story_era. These questions must remain useful even if the new
  allegation or encounter in story_context is entirely fictional.
- Prefer concrete, source-locatable actions or interactions over an
  assumed rank, organizational standing, or abstract level of trust.
- At most one canon question should ask whether the particular
  allegation, new encounter, or procedure in story_context has a
  canon precedent. Do not let it displace character-history research.
- Give each canon question one factual target plus any necessary era
  boundary; do not bundle a rank with an affiliation, an offer with
  a later mission, or an interaction with a level of trust.
- Do not put illustrative lists (e.g., for example, such as) inside
  canon questions. For interactions, ask about one named counterpart
  or one coherent group. Do not assign a person to a group unless
  the user's input established that membership.
- Prefer one documented action or interaction with an identifiable
  point in the requested era over a broad relationship or status
  that the source may not date. Ask about an affiliation, activity,
  contact, or response separately when each needs its own evidence.
- Do not shift research to a period earlier than story_era unless
  that earlier history directly informs the proposed story.
- Generate between 2 and 5 focused voice_questions.

- canon_questions must request established factual information only.
- Canon questions must concern facts such as history, relationships,
  allegiances, affiliations, past behavior, political circumstances,
  professional roles, institutions, and relevant events.
- Do not ask the canon researcher to predict how a fact might affect,
  influence, motivate, or change the character's behavior.
- Questions about what canon evidence means for characterization
  belong to the later Character Analysis stage.
- Do not ask Lore to predict stakes, consequences, reactions,
  decisions, outcomes, or what might happen if a fictional premise
  is true.

- Canon questions must be written so that "no canon evidence exists"
  is always a logically valid possible answer.
- Do not phrase a canon question in a way that presupposes the
  existence of a rank, role, relationship, interaction, procedure,
  event, policy, infiltration, defection, affiliation, or historical
  example.
- When existence is uncertain, prefer neutral formulations such as:
  "Does canon document..."
  "What does canon establish about..."
  "Is there canon evidence that..."
  "What, if anything, does canon establish about..."
- Do not use wording such as "Which..." or "What instances..."
  when doing so assumes that examples necessarily exist.

- Treat allegations in story_context as allegations, not established
  canon facts.
- If story_context contains a claim involving infiltration,
  conspiracy, betrayal, defection, deception, or secret activity,
  research the established canon surrounding that claim rather than
  assuming the claim is true.

- Do not transfer roles, labels, relationships, occupations,
  affiliations, actions, or attributes from another person described
  in story_context onto the character being researched.
- A fictional person's role in story_context, such as "defector",
  "informant", "agent", "traitor", or "infiltrator", must not be
  attributed to the researched character unless the user explicitly
  assigned that role to the character.

- Do not assign a rank, title, position, affiliation, or honorific to
  a character unless that exact information was supplied by the user.
- When rank, title, position, or affiliation matters, formulate a
  neutral canon question that asks research to establish it.

- Institutional research must concern established institutions,
  policies, practices, or historical conditions generally.
- Do not construct a supposed canon procedure around the specific
  fictional encounter described in story_context.

- voice_questions must ask about observable dialogue evidence.
- Voice questions should investigate syntax, vocabulary, formality,
  rhetorical habits, humor, emotional directness, technical
  terminology, forms of address, idioms, and recurring speech patterns.
- Keep voice research specific to the supplied Star Trek era.
- Do not treat all Star Trek series or eras as stylistically identical.
- Do not import terminology, speech conventions, characterization,
  or stylistic assumptions from later Star Trek productions unless
  the supplied story era requires them.
- Do not infer speech style merely from biography or personality.

- Voice questions must research observed canon dialogue rather than
  assume that a fictional interaction from story_context already
  exists in canon.
- Voice questions must not reproduce or presuppose the specific
  fictional decision, encounter, informant, allegation, or dilemma
  supplied in story_context.
- Do not ask how the character speaks specifically while performing
  an action created by the fictional story premise.
- Research comparable observable canon dialogue instead.
- Ask how the character speaks in established categories of
  interactions rather than how the character would speak in the
  proposed scene.
- Comparable categories may include conversations with allies,
  adversaries, Cardassians, Starfleet personnel, intelligence
  contacts, political figures, or other established groups when
  relevant.

- Voice questions must not predict how the character might speak,
  react, behave, or use a technique in the proposed fictional story.
- Do not ask how the character "might", "could", or "would" employ
  a speech pattern in the fictional context.
- Do not ask what dialogue patterns reveal, imply, suggest, or
  indicate about the character's motives or personality.
- Interpretation belongs to the later Character Analysis stage.

Return valid JSON matching this exact structure:

{
    "character_name": "string",
    "story_context": "string",
    "story_era": "string",
    "canon_questions": [
        "question",
        "question"
    ],
    "voice_questions": [
        "question",
        "question"
    ]
}

Output requirements:
- Return only the JSON object.
- Do not use markdown.
- Do not wrap the JSON in triple backticks.
- Do not include commentary before or after the JSON.
- Use valid JSON syntax.
- Use double quotes for all keys and string values.
                """,
            ),
            (
                "human",
                f"""
            Character:
            {character_name}

            Story era:
            {story_era}

            Story situation:
            {story_context}

            Create the research plan needed before analyzing this character.
                            """,
            ),
        ],
        label="Character research plan",
    )

    # -----------------------------------------------------
    # Python owns immutable user-supplied fields
    # -----------------------------------------------------

    response = enforce_research_plan_inputs(
        research_plan=response,
        character_name=character_name,
        story_context=story_context,
        story_era=story_era,
    )

    # -----------------------------------------------------
    # Validate generated research questions
    # -----------------------------------------------------

    try:
        validate_character_research_plan(
            response
        )

    except ValueError as error:
        repaired_response = repair_character_research_plan(
            research_plan=response,
            validation_error=str(error),
        )

        # -------------------------------------------------
        # Re-enforce immutable fields after LLM repair
        # -------------------------------------------------

        repaired_response = enforce_research_plan_inputs(
            research_plan=repaired_response,
            character_name=character_name,
            story_context=story_context,
            story_era=story_era,
        )

        # -------------------------------------------------
        # Repair gets one chance
        # -------------------------------------------------

        validate_character_research_plan(
            repaired_response
        )

        return repaired_response

    return response


# ---------------------------------------------------------
# Direct Test
# ---------------------------------------------------------


if __name__ == "__main__":
    test_plan = create_character_research_plan(
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
    )

    print("\n--- CHARACTER RESEARCH PLAN ---\n")

    print(
        "Character:",
        test_plan.character_name,
    )

    print(
        "Story Context:",
        test_plan.story_context,
    )

    print(
        "Story Era:",
        test_plan.story_era,
    )

    print("\n--- CANON QUESTIONS ---")

    for question in test_plan.canon_questions:
        print("-", question)

    print("\n--- VOICE QUESTIONS ---")

    for question in test_plan.voice_questions:
        print("-", question)
