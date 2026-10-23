from tools.lore_evidence import authoritative_lore_text, LoreEvidenceContractError
from dotenv import load_dotenv
from tools.model_provider import create_chat_model
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import Literal

from config import EVALUATION_MODEL
from tools.rate_limit import safe_invoke

from agents.continuity_agent import (
    ContinuitySceneReview,
)

from agents.scriptwriter_agent import (
    ScriptSceneAudit,
    ScriptwriterInput,
)

from agents.character_agent import (
    CharacterAnalysis,
    CharacterAnalysisAudit,
)

from agents.lore_agent import (
    LoreResponseModel,
)

load_dotenv()


# ---------------------------------------------------------
# Models
# ---------------------------------------------------------


class EvaluationCheckInput(BaseModel):
    criterion: str = Field(
        min_length=1,
        description=(
            "The explicit quality criterion that "
            "the supplied scene text must satisfy."
        ),
    )

    subject_text: str = Field(
        min_length=1,
        description=(
            "The exact scene text being evaluated."
        ),
    )

    evaluation_context: list[str] = Field(
        default_factory=list,
        description=(
            "Authoritative context supplied by "
            "upstream agents or Python."
        ),
    )

class EvaluationJudgment(BaseModel):
    verdict: Literal[
        "pass",
        "fail",
        "insufficient_evidence",
    ] = Field(
        description=(
            "Whether the supplied text satisfies "
            "the explicit evaluation criterion."
        ),
    )

    reason: str = Field(
        min_length=1,
        description=(
            "A concise explanation grounded only "
            "in the supplied text and context."
        ),
    )

class EvaluationCriterion(BaseModel):
    criterion_id: str = Field(
        min_length=1,
        description=(
            "Stable Python-owned identifier "
            "for this evaluation criterion."
        ),
    )

    category: Literal[
        "canon_grounding",
        "character_consistency",
        "era_style",
        "upstream_constraint",
        "continuity",
    ] = Field(
        description=(
            "The quality dimension owned by "
            "this evaluation criterion."
        ),
    )

    criterion: str = Field(
        min_length=1,
        description=(
            "The exact rule the scene "
            "must satisfy."
        ),
    )

    evaluation_context: list[str] = Field(
        default_factory=list,
        description=(
            "Authoritative evidence supplied "
            "for this criterion."
        ),
    )


class EvaluationCriterionResult(BaseModel):
    criterion_id: str

    category: Literal[
        "canon_grounding",
        "character_consistency",
        "era_style",
        "upstream_constraint",
        "continuity",
    ]

    criterion: str

    verdict: Literal[
        "pass",
        "fail",
        "insufficient_evidence",
    ]

    reason: str


class CharacterEvaluationPreparation(BaseModel):
    criterion: EvaluationCriterion | None = None

    precomputed_result: (
        EvaluationCriterionResult | None
    ) = None

class EraStyleEvaluationPreparation(BaseModel):
    criterion: EvaluationCriterion | None = None

    precomputed_result: (
        EvaluationCriterionResult | None
    ) = None

class LoreEvaluationPreparation(BaseModel):
    criterion: EvaluationCriterion | None = None

    precomputed_result: (
        EvaluationCriterionResult | None
    ) = None


class EvaluationSuiteResult(BaseModel):
    results: list[EvaluationCriterionResult]

    pass_count: int

    fail_count: int

    insufficient_evidence_count: int

    has_failure: bool

class EvaluationAssessment(BaseModel):
    status: Literal[
        "approved",
        "needs_evidence",
        "rejected",
    ]

    pass_count: int
    fail_count: int
    insufficient_evidence_count: int

    blocking_criterion_ids: list[str]

class EvaluationReview(BaseModel):
    suite: EvaluationSuiteResult
    assessment: EvaluationAssessment


class EvaluationManifest(BaseModel):
    criteria: list[EvaluationCriterion] = Field(
        default_factory=list,
        description=(
            "Criteria that still require semantic "
            "judgment by the Evaluation Agent."
        ),
    )

    precomputed_results: list[
        EvaluationCriterionResult
    ] = Field(
        default_factory=list,
        description=(
            "Authoritative evaluation results "
            "already produced by frozen upstream "
            "quality gates."
        ),
    )

class CharacterEvaluationSource(BaseModel):
    character_analysis: CharacterAnalysis
    audit: CharacterAnalysisAudit

class EvaluationProductionReview(BaseModel):
    manifest: EvaluationManifest
    review: EvaluationReview

class LoreEvaluationSource(BaseModel):
    research_id: str = Field(
        min_length=1,
    )

    research_question: str = Field(
        min_length=1,
    )

    lore_response: LoreResponseModel

# ---------------------------------------------------------
# Model Configuration
# ---------------------------------------------------------


evaluation_model = create_chat_model(
    model_name=EVALUATION_MODEL,
    temperature=0.1,
    timeout=30,
    max_retries=0,
    max_tokens=800,
    reasoning_effort="low",
)


structured_evaluation_model = (
    evaluation_model.with_structured_output(
        EvaluationJudgment,
        method="json_mode",
    )
)


# ---------------------------------------------------------
# Prompt
# ---------------------------------------------------------


EVALUATION_SYSTEM_PROMPT = """
You are the Evaluation Agent in a multi-agent Star Trek
screenwriting system.

Your current task is deliberately narrow.

You will receive:

1. one explicit evaluation criterion;
2. one exact piece of scene text;
3. zero or more authoritative context statements.

Determine whether the supplied scene text satisfies the
evaluation criterion.

Use ONLY:

- the explicit criterion;
- the supplied scene text;
- the supplied evaluation context.

Do not use your own Star Trek canon knowledge.
Do not research missing information.
Do not invent missing evidence.
Do not repair or rewrite the scene.
Do not infer facts that the supplied context does not establish.

The criterion itself is authoritative.

The evaluation context is authoritative when supplied.


VERDICTS


pass

Use when the supplied text satisfies the criterion and no supplied
context establishes a violation.

Example:

CRITERION:
The scene must not claim that the Cardassian defector is dead.

SCENE TEXT:
The Cardassian defector leaves the room.

CONTEXT:
The Cardassian defector is alive.

VERDICT:
pass


fail

Use when the supplied text clearly violates the criterion using the
criterion itself or the supplied authoritative context.

Example:

CRITERION:
The scene must not use a destroyed data rod as though it were intact.

SCENE TEXT:
Garak inserts the intact data rod into the console.

CONTEXT:
The data rod is destroyed.

VERDICT:
fail


insufficient_evidence

Use when the criterion requires evidence that has not been supplied,
so a reliable pass/fail judgment cannot be made.

Example:

CRITERION:
Garak's dialogue must match his established speech patterns.

SCENE TEXT:
Interesting.

CONTEXT:
No character voice evidence is supplied.

VERDICT:
insufficient_evidence


CORE RULES

- Evaluate only the requested criterion.
- Do not broaden the task into a general scene review.
- Do not use unstated franchise knowledge.
- Do not assume missing evidence supports either pass or fail.
- Missing required evidence means insufficient_evidence.
- Unrelated context must not affect the verdict.
- A scene is not automatically good because no violation is obvious.
- A scene is not automatically bad because context is incomplete.
- Evaluation is not canon research.
- Evaluation is not continuity state management.
- Evaluation is not script repair.


OUTPUT CONTRACT

Return exactly one JSON object:

{{
  "verdict": "pass | fail | insufficient_evidence",
  "reason": "A concise explanation grounded only in the supplied material."
}}

Both keys are required.
Never omit "reason".
Do not return additional keys.
"""


evaluation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            EVALUATION_SYSTEM_PROMPT,
        ),
        (
            "human",
            """
EVALUATION CRITERION:

{criterion}


SCENE TEXT:

{subject_text}


AUTHORITATIVE EVALUATION CONTEXT:

{evaluation_context}


Return the evaluation judgment as JSON.

The JSON object must contain both:
- "verdict"
- "reason"

Do not omit either field.
""",
        ),
    ]
)


evaluation_chain = (
    evaluation_prompt
    | structured_evaluation_model
)


# ---------------------------------------------------------
# Validation
# ---------------------------------------------------------


def validate_evaluation_check_input(
    evaluation_input: EvaluationCheckInput,
) -> None:
    errors: list[str] = []

    if not evaluation_input.criterion.strip():
        errors.append(
            "criterion cannot be empty."
        )

    if not evaluation_input.subject_text.strip():
        errors.append(
            "subject_text cannot be empty."
        )

    for index, context_statement in enumerate(
        evaluation_input.evaluation_context
    ):
        if not context_statement.strip():
            errors.append(
                "evaluation_context entries cannot "
                f"be empty. Invalid index: {index}."
            )

    if errors:
        raise ValueError(
            "Invalid EvaluationCheckInput:\n- "
            + "\n- ".join(errors)
        )


def build_evaluation_manifest(
    criteria: list[EvaluationCriterion],
    precomputed_results: (
        list[EvaluationCriterionResult] | None
    ) = None,
) -> EvaluationManifest:
    precomputed_results = (
        list(precomputed_results)
        if precomputed_results is not None
        else []
    )

    if not criteria and not precomputed_results:
        raise ValueError(
            "Evaluation manifest requires "
            "at least one criterion or "
            "precomputed result."
        )

    criterion_ids = [
        criterion.criterion_id
        for criterion in criteria
    ]

    precomputed_ids = [
        result.criterion_id
        for result in precomputed_results
    ]

    all_ids = [
        *criterion_ids,
        *precomputed_ids,
    ]

    if len(all_ids) != len(set(all_ids)):
        raise ValueError(
            "Evaluation manifest criterion "
            "IDs must be unique across "
            "criteria and precomputed results."
        )

    required_categories = {
        "canon_grounding",
        "character_consistency",
        "era_style",
        "upstream_constraint",
        "continuity",
    }

    present_categories = {
        criterion.category
        for criterion in criteria
    }

    present_categories.update(
        result.category
        for result in precomputed_results
    )

    missing_categories = (
        required_categories
        - present_categories
    )

    if missing_categories:
        raise ValueError(
            "Evaluation manifest is missing "
            "required categories: "
            + ", ".join(
                sorted(missing_categories)
            )
        )

    return EvaluationManifest(
        criteria=list(criteria),
        precomputed_results=(
            precomputed_results
        ),
    )

def build_evaluation_suite_from_results(
    results: list[EvaluationCriterionResult],
) -> EvaluationSuiteResult:
    if not results:
        raise ValueError(
            "Evaluation suite requires "
            "at least one result."
        )

    criterion_ids = [
        result.criterion_id
        for result in results
    ]

    if len(criterion_ids) != len(
        set(criterion_ids)
    ):
        raise ValueError(
            "Evaluation result criterion "
            "IDs must be unique."
        )

    pass_count = sum(
        1
        for result in results
        if result.verdict == "pass"
    )

    fail_count = sum(
        1
        for result in results
        if result.verdict == "fail"
    )

    insufficient_evidence_count = sum(
        1
        for result in results
        if (
            result.verdict
            == "insufficient_evidence"
        )
    )

    return EvaluationSuiteResult(
        results=list(results),
        pass_count=pass_count,
        fail_count=fail_count,
        insufficient_evidence_count=(
            insufficient_evidence_count
        ),
        has_failure=(fail_count > 0),
    )

def review_subject_with_manifest(
    subject_text: str,
    manifest: EvaluationManifest,
) -> EvaluationReview:
    if not subject_text.strip():
        raise ValueError(
            "subject_text cannot be empty."
        )

    evaluated_results: list[
        EvaluationCriterionResult
    ] = []

    # Only criteria that still require semantic
    # judgment are sent through the Evaluation LLM.
    if manifest.criteria:
        evaluated_suite = (
            evaluate_criterion_suite(
                subject_text=subject_text,
                criteria=manifest.criteria,
            )
        )

        evaluated_results.extend(
            evaluated_suite.results
        )

    # Frozen upstream results are accepted
    # directly. They are NOT semantically
    # re-evaluated here.
    combined_results = [
        *manifest.precomputed_results,
        *evaluated_results,
    ]

    suite = (
        build_evaluation_suite_from_results(
            combined_results
        )
    )

    assessment = (
        build_evaluation_assessment(
            suite
        )
    )

    return EvaluationReview(
        suite=suite,
        assessment=assessment,
    )



def adapt_continuity_review(
    review: ContinuitySceneReview,
) -> EvaluationCriterionResult:
    assessment = review.assessment

    issue_contradiction_count = sum(
        1
        for issue in assessment.issues
        if issue.issue_type == "contradiction"
    )

    issue_insufficient_count = sum(
        1
        for issue in assessment.issues
        if (
            issue.issue_type
            == "insufficient_context"
        )
    )

    # -----------------------------------------------------
    # Integrity checks
    # -----------------------------------------------------

    if (
        issue_contradiction_count
        != assessment.contradiction_count
    ):
        raise ValueError(
            "Continuity assessment contradiction "
            "count does not match its issues."
        )

    if (
        issue_insufficient_count
        != assessment.insufficient_context_count
    ):
        raise ValueError(
            "Continuity assessment insufficient-"
            "context count does not match "
            "its issues."
        )

    if (
        review.audit.contradiction_count
        != assessment.contradiction_count
    ):
        raise ValueError(
            "Continuity audit and assessment "
            "contradiction counts do not match."
        )

    if (
        review.audit.has_contradiction
        != (
            assessment.contradiction_count > 0
        )
    ):
        raise ValueError(
            "Continuity audit contradiction flag "
            "does not match the assessment."
        )

    # -----------------------------------------------------
    # Deterministic verdict mapping
    # -----------------------------------------------------

    if assessment.status == "contradiction":
        if assessment.contradiction_count <= 0:
            raise ValueError(
                "Continuity contradiction status "
                "requires at least one "
                "contradiction."
            )

        verdict = "fail"

        segment_ids = [
            issue.segment_id
            for issue in assessment.issues
            if issue.issue_type
            == "contradiction"
        ]

        reason = (
            "Frozen Continuity Agent reported "
            f"{assessment.contradiction_count} "
            "continuity contradiction(s)"
        )

        if segment_ids:
            reason += (
                " at: "
                + ", ".join(segment_ids)
            )

        reason += "."

    elif assessment.status == "needs_context":
        if assessment.contradiction_count != 0:
            raise ValueError(
                "Continuity needs-context status "
                "cannot contain contradictions."
            )

        if (
            assessment.insufficient_context_count
            <= 0
        ):
            raise ValueError(
                "Continuity needs-context status "
                "requires unresolved context."
            )

        verdict = "insufficient_evidence"

        segment_ids = [
            issue.segment_id
            for issue in assessment.issues
            if (
                issue.issue_type
                == "insufficient_context"
            )
        ]

        reason = (
            "Frozen Continuity Agent reported "
            f"{assessment.insufficient_context_count} "
            "unresolved continuity context "
            "issue(s)"
        )

        if segment_ids:
            reason += (
                " at: "
                + ", ".join(segment_ids)
            )

        reason += "."

    elif assessment.status == "clear":
        if (
            assessment.contradiction_count != 0
            or assessment.insufficient_context_count
            != 0
        ):
            raise ValueError(
                "Continuity clear status cannot "
                "contain blocking issues."
            )

        verdict = "pass"

        reason = (
            "Frozen Continuity Agent reported "
            "the scene clear with no continuity "
            "contradictions or unresolved context."
        )

    else:
        raise ValueError(
            "Unsupported Continuity assessment "
            f"status: {assessment.status!r}"
        )

    return EvaluationCriterionResult(
        criterion_id="continuity_agent_review",
        category="continuity",
        criterion=(
            "The scene must pass the frozen "
            "Continuity Agent review without "
            "contradictions or unresolved "
            "continuity context."
        ),
        verdict=verdict,
        reason=reason,
    )

# ---------------------------------------------------------
# Upstream Agent Adapters
# ---------------------------------------------------------


def build_character_evaluation_criterion_id(
    character_name: str,
) -> str:
    if not character_name.strip():
        raise ValueError(
            "character_name cannot be empty."
        )

    normalized_name = "".join(
        character.lower()
        if character.isalnum()
        else "_"
        for character in character_name
    ).strip("_")

    if not normalized_name:
        raise ValueError(
            "character_name does not contain "
            "usable identifier characters."
        )

    return (
        "character_agent_guidance_"
        f"{normalized_name}"
    )


def build_lore_evaluation_criterion_id(
    research_id: str,
) -> str:
    if not research_id.strip():
        raise ValueError(
            "research_id cannot be empty."
        )

    normalized_id = "".join(
        character.lower()
        if character.isalnum()
        else "_"
        for character in research_id
    ).strip("_")

    if not normalized_id:
        raise ValueError(
            "research_id does not contain "
            "usable identifier characters."
        )

    return (
        "lore_agent_evidence_"
        f"{normalized_id}"
    )


def prepare_character_evaluation(
    character_analysis: CharacterAnalysis,
    audit: CharacterAnalysisAudit,
) -> CharacterEvaluationPreparation:
    if not audit.reason.strip():
        raise ValueError(
            "Character analysis audit reason "
            "cannot be empty."
        )

    # -----------------------------------------------------
    # Audit integrity
    # -----------------------------------------------------

    if audit.grounded:
        if audit.issues:
            raise ValueError(
                "Grounded CharacterAnalysis audit "
                "cannot contain unsupported issues."
            )

    else:
        if not audit.issues:
            raise ValueError(
                "Ungrounded CharacterAnalysis audit "
                "must contain at least one issue."
            )

    for issue in audit.issues:
        if not issue.field.strip():
            raise ValueError(
                "CharacterAnalysis issue field "
                "cannot be empty."
            )

        if not issue.claim.strip():
            raise ValueError(
                "CharacterAnalysis issue claim "
                "cannot be empty."
            )

        if not issue.reason.strip():
            raise ValueError(
                "CharacterAnalysis issue reason "
                "cannot be empty."
            )

    criterion_id = (
        build_character_evaluation_criterion_id(
            character_analysis.character_name
        )
    )

    # -----------------------------------------------------
    # Ungrounded guidance must never reach Evaluation
    # as authoritative character context.
    # -----------------------------------------------------

    if not audit.grounded:
        return CharacterEvaluationPreparation(
            criterion=None,
            precomputed_result=(
                EvaluationCriterionResult(
                    criterion_id=criterion_id,
                    category=(
                        "character_consistency"
                    ),
                    criterion=(
                        "The scene must be evaluable "
                        "against grounded Character "
                        "Agent guidance."
                    ),
                    verdict=(
                        "insufficient_evidence"
                    ),
                    reason=(
                        "Frozen Character Agent "
                        f"reported {len(audit.issues)} "
                        "unsupported CharacterAnalysis "
                        "claim(s), so the analysis "
                        "cannot be treated as "
                        "authoritative character "
                        "guidance."
                    ),
                )
            ),
        )

    # -----------------------------------------------------
    # Build context ONLY from grounded guidance.
    #
    # Remaining research questions are deliberately
    # excluded because unresolved questions are not facts.
    #
    # era_language_notes are also excluded here because
    # era/style receives its own Evaluation category.
    # -----------------------------------------------------

    context: list[str] = []

    if character_analysis.story_role.strip():
        context.append(
            "Story role: "
            + character_analysis.story_role.strip()
        )

    for value in character_analysis.motivations:
        if value.strip():
            context.append(
                "Motivation: "
                + value.strip()
            )

    for value in character_analysis.likely_behavior:
        if value.strip():
            context.append(
                "Likely behavior: "
                + value.strip()
            )

    for value in character_analysis.unlikely_behavior:
        if value.strip():
            context.append(
                "Unlikely behavior: "
                + value.strip()
            )

    for value in character_analysis.relationships:
        if value.strip():
            context.append(
                "Relationship guidance: "
                + value.strip()
            )

    if character_analysis.dramatic_tension.strip():
        context.append(
            "Dramatic tension: "
            + character_analysis.dramatic_tension.strip()
        )

    for value in character_analysis.speech_patterns:
        if value.strip():
            context.append(
                "Speech pattern: "
                + value.strip()
            )

    for value in character_analysis.preferred_vocabulary:
        if value.strip():
            context.append(
                "Preferred vocabulary: "
                + value.strip()
            )

    for value in character_analysis.communication_style:
        if value.strip():
            context.append(
                "Communication style: "
                + value.strip()
            )

    for value in character_analysis.behavioral_constraints:
        if value.strip():
            context.append(
                "Behavioral constraint: "
                + value.strip()
            )

    for value in character_analysis.out_of_character_risks:
        if value.strip():
            context.append(
                "Out-of-character risk: "
                + value.strip()
            )

    # A technically grounded but empty analysis still
    # cannot support a meaningful consistency judgment.
    if not context:
        return CharacterEvaluationPreparation(
            criterion=None,
            precomputed_result=(
                EvaluationCriterionResult(
                    criterion_id=criterion_id,
                    category=(
                        "character_consistency"
                    ),
                    criterion=(
                        "The scene must be evaluable "
                        "against grounded Character "
                        "Agent guidance."
                    ),
                    verdict=(
                        "insufficient_evidence"
                    ),
                    reason=(
                        "The CharacterAnalysis passed "
                        "its grounding audit but "
                        "contains no usable character "
                        "guidance for Evaluation."
                    ),
                )
            ),
        )

    criterion = EvaluationCriterion(
        criterion_id=criterion_id,
        category="character_consistency",
        criterion=(
            f"The scene's portrayal of "
            f"{character_analysis.character_name} "
            "must remain compatible with the "
            "supplied grounded Character Agent "
            "guidance. Do not require traits, "
            "motives, behavior, relationships, "
            "or voice patterns that are not "
            "present in that guidance."
        ),
        evaluation_context=context,
    )

    return CharacterEvaluationPreparation(
        criterion=criterion,
        precomputed_result=None,
    )

def prepare_lore_evaluation(
    research_id: str,
    research_question: str,
    lore_response: LoreResponseModel,
) -> LoreEvaluationPreparation:
    """
    Convert one frozen Lore response into Evaluation input.

    Supported flags and a valid source-passage package are both
    required before source text can become canon context. Historical
    generated summaries remain insufficient evidence until refreshed.
    """

    if not research_question.strip():
        raise ValueError(
            "research_question cannot be empty."
        )

    criterion_id = (
        build_lore_evaluation_criterion_id(
            research_id
        )
    )

    # -----------------------------------------------------
    # Frozen Lore output integrity
    #
    # Lore computes:
    # evidence_supported = usable
    # usable requires the premise to be supported.
    # Therefore evidence_supported=True with
    # premise_supported=False is an impossible upstream
    # state and must fail closed.
    # -----------------------------------------------------

    if (
        lore_response.evidence_supported
        and not lore_response.premise_supported
    ):
        raise ValueError(
            "Lore response cannot report "
            "evidence_supported=True while "
            "premise_supported=False."
        )

    # -----------------------------------------------------
    # Unsupported premise -> unresolved canon evidence
    # -----------------------------------------------------

    if not lore_response.premise_supported:
        return LoreEvaluationPreparation(
            criterion=None,
            precomputed_result=(
                EvaluationCriterionResult(
                    criterion_id=criterion_id,
                    category="canon_grounding",
                    criterion=(
                        "The scene must be evaluable "
                        "against supplied canon "
                        "evidence for the requested "
                        "factual premise."
                    ),
                    verdict=(
                        "insufficient_evidence"
                    ),
                    reason=(
                        "Frozen Lore Agent reported "
                        "that the retrieved evidence "
                        "does not establish the "
                        "requested factual premise."
                    ),
                )
            ),
        )

    # -----------------------------------------------------
    # Premise supported but requested constraints were
    # not fully established.
    # -----------------------------------------------------

    if not lore_response.evidence_supported:
        return LoreEvaluationPreparation(
            criterion=None,
            precomputed_result=(
                EvaluationCriterionResult(
                    criterion_id=criterion_id,
                    category="canon_grounding",
                    criterion=(
                        "The scene must be evaluable "
                        "against canon evidence that "
                        "satisfies the research "
                        "question and its constraints."
                    ),
                    verdict=(
                        "insufficient_evidence"
                    ),
                    reason=(
                        "Frozen Lore Agent established "
                        "the underlying premise but "
                        "did not establish all evidence "
                        "required by the research "
                        "question and its constraints."
                    ),
                )
            ),
        )

    # -----------------------------------------------------
    # Revalidate the source package before promoting its copied text.
    # -----------------------------------------------------

    try:
        source_text = authoritative_lore_text(lore_response)
    except LoreEvidenceContractError as error:
        return LoreEvaluationPreparation(
            criterion=None,
            precomputed_result=EvaluationCriterionResult(
                criterion_id=criterion_id, category="canon_grounding",
                criterion="The scene requires current source-backed Lore evidence.",
                verdict="insufficient_evidence", reason=str(error),
            ),
        )

    if not lore_response.summary.strip():
        raise ValueError(
            "Supported Lore response cannot "
            "contain an empty summary."
        )

    criterion = EvaluationCriterion(
        criterion_id=criterion_id,
        category="canon_grounding",
        criterion=(
            "The scene must remain compatible "
            "with the canon evidence established "
            "for this research question: "
            f"{research_question.strip()}"
        ),
        evaluation_context=[
            (
                "Grounded Lore evidence: "
                + source_text
            )
        ],
    )

    return LoreEvaluationPreparation(
        criterion=criterion,
        precomputed_result=None,
    )


def build_evaluation_assessment(
    suite_result: EvaluationSuiteResult,
) -> EvaluationAssessment:
    pass_count = sum(
        1
        for result in suite_result.results
        if result.verdict == "pass"
    )

    fail_count = sum(
        1
        for result in suite_result.results
        if result.verdict == "fail"
    )

    insufficient_evidence_count = sum(
        1
        for result in suite_result.results
        if (
            result.verdict
            == "insufficient_evidence"
        )
    )

    if pass_count != suite_result.pass_count:
        raise ValueError(
            "Evaluation pass count does not "
            "match suite results."
        )

    if fail_count != suite_result.fail_count:
        raise ValueError(
            "Evaluation fail count does not "
            "match suite results."
        )

    if (
        insufficient_evidence_count
        != suite_result.insufficient_evidence_count
    ):
        raise ValueError(
            "Evaluation insufficient-evidence "
            "count does not match suite results."
        )

    if (
        suite_result.has_failure
        != (fail_count > 0)
    ):
        raise ValueError(
            "Evaluation failure flag does not "
            "match suite results."
        )

    if fail_count > 0:
        status = "rejected"

        blocking_criterion_ids = [
            result.criterion_id
            for result in suite_result.results
            if result.verdict == "fail"
        ]

    elif insufficient_evidence_count > 0:
        status = "needs_evidence"

        blocking_criterion_ids = [
            result.criterion_id
            for result in suite_result.results
            if (
                result.verdict
                == "insufficient_evidence"
            )
        ]

    else:
        status = "approved"
        blocking_criterion_ids = []

    return EvaluationAssessment(
        status=status,
        pass_count=pass_count,
        fail_count=fail_count,
        insufficient_evidence_count=(
            insufficient_evidence_count
        ),
        blocking_criterion_ids=(
            blocking_criterion_ids
        ),
    )

def prepare_era_style_evaluation(
    script_input: ScriptwriterInput,
) -> EraStyleEvaluationPreparation:
    story_era = script_input.story_era.strip()

    if not story_era:
        raise ValueError(
            "ScriptwriterInput story_era "
            "cannot be empty."
        )

    style_constraints = [
        constraint.strip()
        for constraint
        in script_input.era_style_constraints
        if constraint.strip()
    ]

    # -----------------------------------------------------
    # Era alone identifies the target period,
    # but without explicit stylistic constraints
    # Evaluation does not have enough bounded
    # evidence to perform a reliable style gate.
    # -----------------------------------------------------

    if not style_constraints:
        return EraStyleEvaluationPreparation(
            criterion=None,
            precomputed_result=(
                EvaluationCriterionResult(
                    criterion_id=(
                        "era_style_review"
                    ),
                    category="era_style",
                    criterion=(
                        "The scene must be evaluable "
                        "against explicit era/style "
                        "constraints."
                    ),
                    verdict=(
                        "insufficient_evidence"
                    ),
                    reason=(
                        "The ScriptwriterInput "
                        f"identifies the story era as "
                        f"{story_era!r} but supplies "
                        "no explicit era/style "
                        "constraints for final "
                        "Evaluation."
                    ),
                )
            ),
        )

    context = [
        f"Story era: {story_era}",
        *[
            (
                "Era/style constraint: "
                + constraint
            )
            for constraint
            in style_constraints
        ],
    ]

    criterion = EvaluationCriterion(
        criterion_id="era_style_review",
        category="era_style",
        criterion=(
            "The scene must remain compatible "
            "with the supplied Star Trek story "
            "era and every explicit era/style "
            "constraint. Judge linguistic, "
            "technological, dramatic, production, "
            "and screenwriting conventions only "
            "where the supplied constraints "
            "establish them. Do not import "
            "additional franchise style knowledge."
        ),
        evaluation_context=context,
    )

    return EraStyleEvaluationPreparation(
        criterion=criterion,
        precomputed_result=None,
    )

def evaluate_criterion_suite(
    subject_text: str,
    criteria: list[EvaluationCriterion],
) -> EvaluationSuiteResult:
    if not subject_text.strip():
        raise ValueError(
            "subject_text cannot be empty."
        )

    if not criteria:
        raise ValueError(
            "Evaluation criterion suite "
            "requires at least one criterion."
        )

    criterion_ids = [
        criterion.criterion_id
        for criterion in criteria
    ]

    if len(criterion_ids) != len(
        set(criterion_ids)
    ):
        raise ValueError(
            "Evaluation criterion IDs "
            "must be unique."
        )

    results: list[
        EvaluationCriterionResult
    ] = []

    for criterion in criteria:
        judgment = evaluate_criterion(
            EvaluationCheckInput(
                criterion=criterion.criterion,
                subject_text=subject_text,
                evaluation_context=(
                    criterion.evaluation_context
                ),
            )
        )

        results.append(
            EvaluationCriterionResult(
                criterion_id=(
                    criterion.criterion_id
                ),
                category=criterion.category,
                criterion=criterion.criterion,
                verdict=judgment.verdict,
                reason=judgment.reason,
            )
        )

    pass_count = sum(
        1
        for result in results
        if result.verdict == "pass"
    )

    fail_count = sum(
        1
        for result in results
        if result.verdict == "fail"
    )

    insufficient_evidence_count = sum(
        1
        for result in results
        if (
            result.verdict
            == "insufficient_evidence"
        )
    )

    return EvaluationSuiteResult(
        results=results,
        pass_count=pass_count,
        fail_count=fail_count,
        insufficient_evidence_count=(
            insufficient_evidence_count
        ),
        has_failure=(fail_count > 0),
    )

def review_subject_against_criteria(
    subject_text: str,
    criteria: list[EvaluationCriterion],
) -> EvaluationReview:
    suite = evaluate_criterion_suite(
        subject_text=subject_text,
        criteria=criteria,
    )

    assessment = build_evaluation_assessment(
        suite
    )

    return EvaluationReview(
        suite=suite,
        assessment=assessment,
    )

def build_production_evaluation_manifest(
    script_input: ScriptwriterInput,
    scriptwriter_audit: ScriptSceneAudit,
    continuity_review: ContinuitySceneReview,
    character_sources: list[
        CharacterEvaluationSource
    ],
    lore_sources: list[
        LoreEvaluationSource
    ],
) -> EvaluationManifest:
    criteria: list[EvaluationCriterion] = []

    precomputed_results: list[
        EvaluationCriterionResult
    ] = []

    # -----------------------------------------------------
    # Scriptwriter frozen grounding gate
    # -----------------------------------------------------

    precomputed_results.append(
        adapt_scriptwriter_audit(
            scriptwriter_audit
        )
    )

    # -----------------------------------------------------
    # Continuity frozen quality gate
    # -----------------------------------------------------

    precomputed_results.append(
        adapt_continuity_review(
            continuity_review
        )
    )

    # -----------------------------------------------------
    # Era/style requirements
    # -----------------------------------------------------

    era_style = (
        prepare_era_style_evaluation(
            script_input
        )
    )

    if era_style.criterion is not None:
        criteria.append(
            era_style.criterion
        )

    if (
        era_style.precomputed_result
        is not None
    ):
        precomputed_results.append(
            era_style.precomputed_result
        )

    # -----------------------------------------------------
    # Character guidance
    # -----------------------------------------------------

    if not character_sources:
        raise ValueError(
            "Production Evaluation requires "
            "at least one Character Agent source."
        )

    for source in character_sources:
        prepared = (
            prepare_character_evaluation(
                character_analysis=(
                    source.character_analysis
                ),
                audit=source.audit,
            )
        )

        if prepared.criterion is not None:
            criteria.append(
                prepared.criterion
            )

        if (
            prepared.precomputed_result
            is not None
        ):
            precomputed_results.append(
                prepared.precomputed_result
            )

    # -----------------------------------------------------
    # Lore/canon evidence
    # -----------------------------------------------------

    if not lore_sources:
        raise ValueError(
            "Production Evaluation requires "
            "at least one Lore Agent source."
        )

    for source in lore_sources:
        prepared = prepare_lore_evaluation(
            research_id=source.research_id,
            research_question=(
                source.research_question
            ),
            lore_response=(
                source.lore_response
            ),
        )

        if prepared.criterion is not None:
            criteria.append(
                prepared.criterion
            )

        if (
            prepared.precomputed_result
            is not None
        ):
            precomputed_results.append(
                prepared.precomputed_result
            )

    # -----------------------------------------------------
    # Final Python coverage validation
    # -----------------------------------------------------

    return build_evaluation_manifest(
        criteria=criteria,
        precomputed_results=(
            precomputed_results
        ),
    )

def review_production_subject(
    subject_text: str,
    script_input: ScriptwriterInput,
    scriptwriter_audit: ScriptSceneAudit,
    continuity_review: ContinuitySceneReview,
    character_sources: list[
        CharacterEvaluationSource
    ],
    lore_sources: list[
        LoreEvaluationSource
    ],
) -> EvaluationProductionReview:
    if not subject_text.strip():
        raise ValueError(
            "subject_text cannot be empty."
        )

    manifest = (
        build_production_evaluation_manifest(
            script_input=script_input,
            scriptwriter_audit=(
                scriptwriter_audit
            ),
            continuity_review=(
                continuity_review
            ),
            character_sources=(
                character_sources
            ),
            lore_sources=lore_sources,
        )
    )

    review = review_subject_with_manifest(
        subject_text=subject_text,
        manifest=manifest,
    )

    return EvaluationProductionReview(
        manifest=manifest,
        review=review,
    )

def adapt_scriptwriter_audit(
    audit: ScriptSceneAudit,
) -> EvaluationCriterionResult:
    if not audit.reason.strip():
        raise ValueError(
            "Scriptwriter audit reason "
            "cannot be empty."
        )

    unsupported_issues = [
        issue
        for issue in audit.issues
        if issue.verdict == "unsupported"
    ]

    prohibited_issues = [
        issue
        for issue in audit.issues
        if issue.verdict == "prohibited"
    ]

    issue_count = len(audit.issues)

    # -----------------------------------------------------
    # Integrity checks
    # -----------------------------------------------------

    for issue in audit.issues:
        if not issue.segment_id.strip():
            raise ValueError(
                "Scriptwriter audit issue "
                "segment_id cannot be empty."
            )

        if not issue.reason.strip():
            raise ValueError(
                "Scriptwriter audit issue "
                "reason cannot be empty."
            )

    if audit.grounded:
        if issue_count != 0:
            raise ValueError(
                "Grounded Scriptwriter audit "
                "cannot contain audit issues."
            )

    else:
        if issue_count == 0:
            raise ValueError(
                "Ungrounded Scriptwriter audit "
                "must contain at least one issue."
            )

    # -----------------------------------------------------
    # Deterministic verdict mapping
    # -----------------------------------------------------

    if audit.grounded:
        verdict = "pass"

        reason = (
            "Frozen Scriptwriter Agent reported "
            "the scene grounded: all audited "
            "segments are grounded or permitted "
            "story inventions."
        )

    else:
        verdict = "fail"

        segment_ids = [
            issue.segment_id
            for issue in audit.issues
        ]

        reason_parts = [
            "Frozen Scriptwriter Agent reported "
            f"{issue_count} blocking grounding "
            "issue(s)."
        ]

        if unsupported_issues:
            reason_parts.append(
                f"{len(unsupported_issues)} unsupported."
            )

        if prohibited_issues:
            reason_parts.append(
                f"{len(prohibited_issues)} prohibited."
            )

        if segment_ids:
            reason_parts.append(
                "Affected segments: "
                + ", ".join(segment_ids)
                + "."
            )

        reason = " ".join(reason_parts)

    return EvaluationCriterionResult(
        criterion_id="scriptwriter_grounding_review",
        category="upstream_constraint",
        criterion=(
            "The scene must pass the frozen "
            "Scriptwriter grounding gate with "
            "no unsupported or prohibited "
            "screenplay segments."
        ),
        verdict=verdict,
        reason=reason,
    )

# ---------------------------------------------------------
# Evaluation Check
# ---------------------------------------------------------


def evaluate_criterion(
    evaluation_input: EvaluationCheckInput,
) -> EvaluationJudgment:
    validate_evaluation_check_input(
        evaluation_input
    )

    context_text = (
        "\n".join(
            f"- {statement}"
            for statement
            in evaluation_input.evaluation_context
        )
        if evaluation_input.evaluation_context
        else "No evaluation context was supplied."
    )

    result = safe_invoke(
        evaluation_chain,
        {
            "criterion": (
                evaluation_input.criterion
            ),
            "subject_text": (
                evaluation_input.subject_text
            ),
            "evaluation_context": context_text,
        },
        label="Evaluation criterion check",
    )

    if not isinstance(
        result,
        EvaluationJudgment,
    ):
        raise TypeError(
            "Evaluation check did not return "
            "an EvaluationJudgment."
        )

    return result
