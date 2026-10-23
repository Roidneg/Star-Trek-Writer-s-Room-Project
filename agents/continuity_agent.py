from dotenv import load_dotenv
from tools.model_provider import create_chat_model
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import Literal

from config import CONTINUITY_MODEL
from tools.rate_limit import safe_invoke

load_dotenv()

# ---------------------------------------------------------
# Scriptwriter Agent Imports
# ---------------------------------------------------------

from agents.scriptwriter_agent import (
    ScriptScene,
    enumerate_script_scene_segments,
)


# ---------------------------------------------------------
# Models
# ---------------------------------------------------------


class ContinuityCheckInput(BaseModel):
    established_state: str = Field(
        description=(
            "A story-state statement that has already been "
            "established by an earlier scene."
        )
    )

    new_statement: str = Field(
        description=(
            "A statement from a later scene that must be checked "
            "against the established story state."
        )
    )


class ContinuityJudgment(BaseModel):
    verdict: Literal[
        "consistent",
        "state_progression",
        "contradiction",
        "insufficient_context",
    ] = Field(
        description=(
            "The continuity relationship between the established "
            "state and the later statement."
        )
    )

    reason: str = Field(
        description=(
            "A concise explanation of the continuity judgment."
        )
    )


class ContinuityStateCommitPlan(BaseModel):
    segment_id: str

    statement: str

    normalized_state: str | None = None

    supersedes_state_id: str | None = None


class StoryStateEntry(BaseModel):
    state_id: str = Field(
        description=(
            "A stable Python-owned identifier for this "
            "established story-state entry."
        )
    )

    scene_index: int = Field(
        ge=0,
        description=(
            "The zero-based story scene index where this "
            "state became established."
        )
    )

    statement: str = Field(
        min_length=1,
        description=(
            "The exact established story-state statement."
        )
    )

    supersedes_state_id: str | None = Field(
        default=None,
        description=(
            "The earlier story-state entry replaced by this "
            "state, if this entry represents a legitimate "
            "state progression."
        ),
    )

    normalized_state: str | None = Field(
        default=None,
        description=(
            "A normalized persistent-state representation "
            "derived from the source statement, when available."
        ),
    )


class StoryStateLedger(BaseModel):
    entries: list[StoryStateEntry] = Field(
        default_factory=list,
        description=(
            "Ordered established story-state history."
        )
    )


class ContinuityStateAuditResult(BaseModel):
    state_id: str

    established_state: str

    verdict: Literal[
        "consistent",
        "state_progression",
        "contradiction",
        "insufficient_context",
    ]

    reason: str

class ContinuityStatementAudit(BaseModel):
    new_statement: str

    results: list[ContinuityStateAuditResult]

    has_contradiction: bool


class ContinuitySceneAudit(BaseModel):
    statement_audits: list[ContinuityStatementAudit]

    has_contradiction: bool

    contradiction_count: int


class ContinuitySceneStatement(BaseModel):
    segment_id: str

    statement_type: Literal[
        "scene_metadata",
        "objective_story_state",
        "character_assertion",
    ]

    text: str

    speaker: str | None = None


class ContinuityScriptSegmentAudit(BaseModel):
    segment_id: str

    statement: str

    audit: ContinuityStatementAudit


class ContinuityScriptSceneAudit(BaseModel):
    segment_audits: list[
        ContinuityScriptSegmentAudit
    ]

    has_contradiction: bool

    contradiction_count: int


class ContinuitySceneIssue(BaseModel):
    segment_id: str
    statement: str
    state_id: str
    established_state: str

    issue_type: Literal[
        "contradiction",
        "insufficient_context",
    ]

    reason: str


class ContinuitySceneAssessment(BaseModel):
    status: Literal[
        "clear",
        "needs_context",
        "contradiction",
    ]

    issues: list[ContinuitySceneIssue]
    contradiction_count: int
    insufficient_context_count: int


class ContinuitySceneReview(BaseModel):
    audit: ContinuityScriptSceneAudit
    assessment: ContinuitySceneAssessment


class ContinuityStateCandidate(BaseModel):
    segment_id: str
    statement: str


class ContinuityStateCandidateJudgment(BaseModel):
    verdict: Literal[
        "durable_state",
        "transient_event",
        "insufficient_context",
    ]

    resulting_state: str = ""

    reason: str


class ContinuityStateLearningResult(BaseModel):
    segment_id: str
    statement: str
    candidate_judgment: ContinuityStateCandidateJudgment
    commit_plan: ContinuityStateCommitPlan
    committed_state_id: str
    ledger: StoryStateLedger

class ContinuitySceneLearningResult(BaseModel):
    committed_state_ids: list[str]

    transient_segment_ids: list[str]

    ledger: StoryStateLedger


# ---------------------------------------------------------
# Model Configuration
# ---------------------------------------------------------


continuity_model = create_chat_model(
    model_name=CONTINUITY_MODEL,
    temperature=0.1,
    timeout=30,
    max_retries=0,
    max_tokens=800,
    reasoning_effort="low",
)

structured_continuity_model = (
    continuity_model.with_structured_output(
        ContinuityJudgment,
        method="json_mode",
    )
)

structured_state_candidate_model = (
    continuity_model.with_structured_output(
        ContinuityStateCandidateJudgment,
        method="json_mode",
    )
)
# ---------------------------------------------------------
# Prompt
# ---------------------------------------------------------


CONTINUITY_SYSTEM_PROMPT = """
You are the Continuity Agent in a multi-agent Star Trek
screenwriting system.

Your current task is deliberately narrow.

You will receive:

1. one story-state statement already established by an earlier scene;
2. one statement from a later scene.

Determine the continuity relationship between them.

Use ONLY the supplied statements.

Do not use your own Star Trek knowledge.
Do not invent missing events in order to reconcile a conflict.
Do not perform canon research.
Do not rewrite the story.

VERDICTS

consistent

Use when both statements can be true at the same time and the later
statement does not materially change the established state.

Example:

ESTABLISHED:
The defector is carrying a data rod.

NEW:
The defector speaks privately with Garak.

VERDICT:
consistent


state_progression

Use when the later statement explicitly describes a reasonable change
from the previously established state.

The transition itself must be present or necessarily contained in the
new statement. Do not invent an unseen transition.

Example:

ESTABLISHED:
The defector possesses the data rod.

NEW:
The defector hands the data rod to Garak.

VERDICT:
state_progression


contradiction

Use when the later statement cannot be true without violating the
established state, and no explicit transition in the supplied text
resolves the conflict.

Example:

ESTABLISHED:
The data rod was destroyed.

NEW:
Garak inserts the same data rod into a console.

VERDICT:
contradiction


insufficient_context

Use when the two supplied statements do not contain enough information
to determine whether a continuity conflict exists.

Do not assume missing travel, recovery, repair, resurrection,
communication, possession transfer, relationship change, or other
unstated bridging events.

STATE DIMENSION RULE

Use "state_progression" only when the later statement changes,
replaces, or advances the SAME material story-state property
established by the earlier statement.

Examples of the same state dimension:

- possession → possession
- location → location
- alive/dead condition → alive/dead condition
- intact/damaged/destroyed condition → condition
- knowledge → knowledge
- relationship status → relationship status

A later event involving the same character or object is NOT
automatically a state progression.

Example:

ESTABLISHED:
The Cardassian defector is alive.

NEW:
The Cardassian defector leaves the room.

VERDICT:
consistent

Reason:
Leaving the room changes location, not the established alive/dead state.

CORE RULES

- Treat established_state as authoritative story history.
- Judge only the supplied text.
- Past story state may change, but the later statement must establish
  the relevant transition when one is required.
- Do not silently invent an off-screen event to repair continuity.
- A new event is not automatically a contradiction.
- A change of possession, location, condition, knowledge, relationship,
  or other story state may be valid progression when the transition is
  explicitly established.
- Continuity judgment is not canon judgment.
- Do not reject a statement merely because you personally believe it
  conflicts with Star Trek canon.

OUTPUT CONTRACT

Return exactly one JSON object with BOTH of these keys:

{{
  "verdict": "consistent | state_progression | contradiction | insufficient_context",
  "reason": "A concise explanation of why this verdict applies."
}}

Both keys are required.
Never omit "reason".
"reason" must be a non-empty string.
Do not return any keys other than "verdict" and "reason".

"""

continuity_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            CONTINUITY_SYSTEM_PROMPT,
        ),
        (
            "human",
            """
ESTABLISHED STORY STATE:

{established_state}


LATER STORY STATEMENT:

{new_statement}


Return the continuity judgment as JSON.

The JSON object must contain both:
- "verdict"
- "reason"

Do not omit either field.
""",
        ),
    ]
)

continuity_chain = (
        continuity_prompt
        | structured_continuity_model
)

STATE_CANDIDATE_SYSTEM_PROMPT = """
You are evaluating objective screenplay events for a
cross-scene continuity system.

You will receive one objective story statement from a scene
that has already passed continuity review.

Decide whether the statement establishes durable story state.

Use ONLY the supplied statement.

Do not use Star Trek canon knowledge.
Do not invent missing context.
Do not rewrite the statement.


VERDICTS


durable_state

Use when the statement establishes a material fact that would
normally remain true across later scenes until something changes it.

Examples include:

- possession or custody;
- explicit location;
- alive/dead status;
- physical condition such as intact, damaged, or destroyed;
- knowledge explicitly gained;
- relationship or role status;
- access or availability;
- another persistent material story condition.

Example:

STATEMENT:
Garak takes the data rod.

VERDICT:
durable_state

RESULTING STATE:
Garak possesses the data rod.

Reason:
The statement establishes Garak's possession of the data rod.


transient_event

Use when the statement describes something that happens but does
not establish a material state that needs to remain active across
later scenes.

Examples include:

- gestures;
- facial expressions;
- momentary movements;
- looking at something;
- speaking;
- other brief actions without a persistent resulting condition.

Example:

STATEMENT:
Garak smiles.

VERDICT:
transient_event

RESULTING STATE:


insufficient_context

Use when the supplied statement is too vague or incomplete to
determine whether it establishes durable state.

Do not invent missing referents or outcomes.

Example:

STATEMENT:
Garak does something to the data rod.

VERDICT:
insufficient_context

RESULTING STATE:


RESULTING STATE RULE

When verdict is "durable_state", resulting_state must contain a concise
present-state statement describing the persistent condition created by
the event.

Example:

STATEMENT:
The Cardassian defector hands the data rod to Garak.

RESULTING STATE:
Garak possesses the data rod.

Example:

STATEMENT:
The console is destroyed.

RESULTING STATE:
The console is destroyed.

Do not add information that is not entailed by the supplied statement.

For "transient_event" and "insufficient_context",
resulting_state must be an empty string.


CORE RULES

- Durable state must describe a material resulting condition.
- An event is not durable merely because it occurred.
- Do not infer possession merely from touching or examining an object.
- Do not infer destruction merely from damage.
- Do not infer a location unless the statement establishes one.
- Do not infer knowledge unless the statement establishes that
  information was learned or received.
- Do not invent a persistent result that the statement does not state.


OUTPUT CONTRACT

Return exactly one JSON object:

{{
  "verdict": "durable_state | transient_event | insufficient_context",
  "resulting_state": "Persistent state, or an empty string.",
  "reason": "A concise explanation of the judgment."
}}

All three keys are required.
Never omit "resulting_state".
Never omit "reason".
Do not return additional keys.

"""
state_candidate_prompt = (
    ChatPromptTemplate.from_messages(
        [
            (
                "system",
                STATE_CANDIDATE_SYSTEM_PROMPT,
            ),
            (
                "human",
                """
OBJECTIVE STORY STATEMENT:

{statement}


Return the classification as JSON.

The JSON object must contain all three:
- "verdict"
- "resulting_state"
- "reason"
""",
            ),
        ]
    )
)

state_candidate_chain = (
        state_candidate_prompt
        | structured_state_candidate_model
)


# ---------------------------------------------------------
# Validation
# ---------------------------------------------------------


def validate_continuity_check_input(
        continuity_input: ContinuityCheckInput,
) -> None:
    errors: list[str] = []

    if not continuity_input.established_state.strip():
        errors.append(
            "established_state cannot be empty."
        )

    if not continuity_input.new_statement.strip():
        errors.append(
            "new_statement cannot be empty."
        )

    if errors:
        raise ValueError(
            "Invalid ContinuityCheckInput:\n- "
            + "\n- ".join(errors)
        )


def add_story_state(
        ledger: StoryStateLedger,
        entry: StoryStateEntry,
) -> StoryStateLedger:
    existing_ids = {
        existing.state_id
        for existing in ledger.entries
    }

    if entry.state_id in existing_ids:
        raise ValueError(
            f"Duplicate story-state ID: "
            f"{entry.state_id!r}"
        )

    if ledger.entries:
        latest_scene_index = max(
            existing.scene_index
            for existing in ledger.entries
        )

        if entry.scene_index < latest_scene_index:
            raise ValueError(
                "Story-state entries cannot be added "
                "out of scene order."
            )
    if entry.supersedes_state_id is not None:
        if entry.supersedes_state_id == entry.state_id:
            raise ValueError(
                "A story-state entry cannot "
                "supersede itself."
            )

        superseded_entry = next(
            (
                existing
                for existing in ledger.entries
                if existing.state_id
                   == entry.supersedes_state_id
            ),
            None,
        )

        if superseded_entry is None:
            raise ValueError(
                "Cannot supersede unknown "
                f"story-state ID: "
                f"{entry.supersedes_state_id!r}"
            )

        already_superseded_ids = {
            existing.supersedes_state_id
            for existing in ledger.entries
            if existing.supersedes_state_id
               is not None
        }

        if (
                entry.supersedes_state_id
                in already_superseded_ids
        ):
            raise ValueError(
                "Story-state entry has already "
                "been superseded."
            )
    return ledger.model_copy(
        update={
            "entries": [
                *ledger.entries,
                entry,
            ]
        }
    )


def get_active_story_states(
        ledger: StoryStateLedger,
) -> list[StoryStateEntry]:
    superseded_ids = {
        entry.supersedes_state_id
        for entry in ledger.entries
        if entry.supersedes_state_id is not None
    }

    return [
        entry
        for entry in ledger.entries
        if entry.state_id not in superseded_ids
    ]


def get_story_state_by_id(
        ledger: StoryStateLedger,
        state_id: str,
) -> StoryStateEntry:
    for entry in ledger.entries:
        if entry.state_id == state_id:
            return entry

    raise ValueError(
        f"Unknown story-state ID: {state_id!r}"
    )



def get_story_state_semantic_statement(
    entry: StoryStateEntry,
) -> str:
    if (
        entry.normalized_state is not None
        and entry.normalized_state.strip()
    ):
        return entry.normalized_state.strip()

    return entry.statement


def learn_scene_states_in_order(
    ledger: StoryStateLedger,
    scene: ScriptScene,
    scene_index: int,
) -> ContinuitySceneLearningResult:
    if scene_index < 0:
        raise ValueError(
            "scene_index cannot be negative."
        )

    continuity_statements = (
        extract_continuity_scene_statements(
            scene
        )
    )

    objective_statements = [
        statement
        for statement in continuity_statements
        if statement.statement_type
        == "objective_story_state"
    ]

    working_ledger = ledger

    committed_state_ids: list[str] = []
    transient_segment_ids: list[str] = []

    for statement in objective_statements:
        statement_audit = (
            audit_statement_against_active_states(
                ledger=working_ledger,
                new_statement=statement.text,
            )
        )

        contradiction_results = [
            result
            for result in statement_audit.results
            if result.verdict == "contradiction"
        ]

        if contradiction_results:
            raise ValueError(
                "Sequential scene learning found "
                "a continuity contradiction at "
                f"{statement.segment_id!r}."
            )

        unresolved_results = [
            result
            for result in statement_audit.results
            if (
                result.verdict
                == "insufficient_context"
            )
        ]

        if unresolved_results:
            raise ValueError(
                "Sequential scene learning requires "
                "additional continuity context at "
                f"{statement.segment_id!r}."
            )

        candidate = ContinuityStateCandidate(
            segment_id=statement.segment_id,
            statement=statement.text,
        )

        candidate_judgment = (
            classify_state_candidate(
                candidate
            )
        )

        if (
            candidate_judgment.verdict
            == "transient_event"
        ):
            transient_segment_ids.append(
                statement.segment_id
            )
            continue

        if (
            candidate_judgment.verdict
            == "insufficient_context"
        ):
            raise ValueError(
                "Objective scene statement could "
                "not be classified as durable or "
                "transient: "
                f"{statement.segment_id!r}"
            )

        segment_audit = (
            ContinuityScriptSegmentAudit(
                segment_id=statement.segment_id,
                statement=statement.text,
                audit=statement_audit,
            )
        )

        commit_plan = build_state_commit_plan(
            candidate=candidate,
            candidate_judgment=(
                candidate_judgment
            ),
            segment_audit=segment_audit,
        )

        new_state_id = build_learned_state_id(
            scene_index=scene_index,
            segment_id=statement.segment_id,
        )

        working_ledger = commit_state_plan(
            ledger=working_ledger,
            plan=commit_plan,
            new_state_id=new_state_id,
            scene_index=scene_index,
        )

        committed_state_ids.append(
            new_state_id
        )

    return ContinuitySceneLearningResult(
        committed_state_ids=(
            committed_state_ids
        ),
        transient_segment_ids=(
            transient_segment_ids
        ),
        ledger=working_ledger,
    )

def advance_story_state(
        ledger: StoryStateLedger,
        established_state_id: str,
        new_state_id: str,
        scene_index: int,
        new_statement: str,
) -> tuple[
    StoryStateLedger,
    ContinuityJudgment,
]:
    active_state_ids = {
        entry.state_id
        for entry in get_active_story_states(
            ledger
        )
    }

    if established_state_id not in active_state_ids:
        raise ValueError(
            "State advancement requires an "
            "active story-state ID. "
            f"Received: {established_state_id!r}"
        )

    judgment = check_ledger_continuity(
        ledger=ledger,
        state_id=established_state_id,
        new_statement=new_statement,
    )

    if judgment.verdict != "state_progression":
        return ledger, judgment

    updated_ledger = add_story_state(
        ledger,
        StoryStateEntry(
            state_id=new_state_id,
            scene_index=scene_index,
            statement=new_statement,
            supersedes_state_id=(
                established_state_id
            ),
        ),
    )

    return updated_ledger, judgment


def build_learned_state_id(
        scene_index: int,
        segment_id: str,
) -> str:
    if scene_index < 0:
        raise ValueError(
            "scene_index cannot be negative."
        )

    if not segment_id.strip():
        raise ValueError(
            "segment_id cannot be empty."
        )

    normalized_segment_id = "".join(
        character
        if character.isalnum()
        else "_"
        for character in segment_id
    ).strip("_")

    if not normalized_segment_id:
        raise ValueError(
            "segment_id does not contain "
            "usable identifier characters."
        )

    return (
        f"scene_{scene_index:04d}_"
        f"{normalized_segment_id}"
    )


def build_continuity_scene_assessment(
        scene_audit: ContinuityScriptSceneAudit,
) -> ContinuitySceneAssessment:
    issues: list[ContinuitySceneIssue] = []

    contradiction_count = 0
    insufficient_context_count = 0

    for segment_audit in (
            scene_audit.segment_audits
    ):
        for result in (
                segment_audit.audit.results
        ):
            if result.verdict == "contradiction":
                contradiction_count += 1

            elif (
                    result.verdict
                    == "insufficient_context"
            ):
                insufficient_context_count += 1

            else:
                continue

            issues.append(
                ContinuitySceneIssue(
                    segment_id=(
                        segment_audit.segment_id
                    ),
                    statement=(
                        segment_audit.statement
                    ),
                    state_id=result.state_id,
                    established_state=(
                        result.established_state
                    ),
                    issue_type=result.verdict,
                    reason=result.reason,
                )
            )

    # Cross-check the earlier deterministic audit.
    if (
            contradiction_count
            != scene_audit.contradiction_count
    ):
        raise ValueError(
            "Continuity contradiction count "
            "does not match the ScriptScene audit."
        )

    if (
            scene_audit.has_contradiction
            != (contradiction_count > 0)
    ):
        raise ValueError(
            "Continuity contradiction flag "
            "does not match the ScriptScene audit."
        )

    if contradiction_count > 0:
        status = "contradiction"

    elif insufficient_context_count > 0:
        status = "needs_context"

    else:
        status = "clear"

    return ContinuitySceneAssessment(
        status=status,
        issues=issues,
        contradiction_count=(
            contradiction_count
        ),
        insufficient_context_count=(
            insufficient_context_count
        ),
    )


def extract_continuity_scene_statements(
        scene: ScriptScene,
) -> list[ContinuitySceneStatement]:
    segments = enumerate_script_scene_segments(
        scene
    )

    statements: list[
        ContinuitySceneStatement
    ] = []

    for segment in segments:
        # Speaker cues identify who is talking,
        # but do not themselves establish story state.
        if segment.segment_type == "speaker_cue":
            continue

        # Parentheticals are performance directions,
        # not independent continuity facts.
        if segment.segment_type == "parenthetical":
            continue

        if segment.segment_type in {
            "scene_heading",
            "setting_description",
        }:
            statement_type = "scene_metadata"

        elif segment.segment_type == "action":
            statement_type = (
                "objective_story_state"
            )

        elif segment.segment_type == "dialogue":
            statement_type = (
                "character_assertion"
            )

        else:
            raise ValueError(
                "Unsupported ScriptScene segment "
                f"type for Continuity: "
                f"{segment.segment_type!r}"
            )

        statements.append(
            ContinuitySceneStatement(
                segment_id=segment.segment_id,
                statement_type=statement_type,
                text=segment.text,
                speaker=segment.speaker,
            )
        )

    return statements


def check_ledger_continuity(
    ledger: StoryStateLedger,
    state_id: str,
    new_statement: str,
) -> ContinuityJudgment:
    established_entry = get_story_state_by_id(
        ledger,
        state_id,
    )

    established_state = (
        get_story_state_semantic_statement(
            established_entry
        )
    )

    return check_continuity(
        ContinuityCheckInput(
            established_state=established_state,
            new_statement=new_statement,
        )
    )


def get_established_states(
        ledger: StoryStateLedger,
) -> list[StoryStateEntry]:
    return list(ledger.entries)


def collect_state_candidates_from_review(
        review: ContinuitySceneReview,
) -> list[ContinuityStateCandidate]:
    if review.assessment.status != "clear":
        raise ValueError(
            "Only a continuity-clear scene may "
            "produce new story-state candidates."
        )

    return [
        ContinuityStateCandidate(
            segment_id=segment_audit.segment_id,
            statement=segment_audit.statement,
        )
        for segment_audit
        in review.audit.segment_audits
    ]


def build_state_commit_plan(
        candidate: ContinuityStateCandidate,
        candidate_judgment: ContinuityStateCandidateJudgment,
        segment_audit: ContinuityScriptSegmentAudit,
) -> ContinuityStateCommitPlan:
    if candidate.segment_id != segment_audit.segment_id:
        raise ValueError(
            "State candidate segment ID does not "
            "match the ScriptScene audit segment."
        )

    if candidate.statement != segment_audit.statement:
        raise ValueError(
            "State candidate statement does not "
            "match the ScriptScene audit statement."
        )

    if candidate_judgment.verdict != "durable_state":
        raise ValueError(
            "Only a durable_state candidate may "
            "produce a ledger commit plan."
        )

    progression_state_ids = [
        result.state_id
        for result in segment_audit.audit.results
        if result.verdict == "state_progression"
    ]

    if len(progression_state_ids) > 1:
        raise ValueError(
            "Durable state candidate ambiguously "
            "progresses more than one active state."
        )

    supersedes_state_id = (
        progression_state_ids[0]
        if progression_state_ids
        else None
    )

    return ContinuityStateCommitPlan(
        segment_id=candidate.segment_id,
        statement=candidate.statement,
        normalized_state=(
            candidate_judgment.resulting_state.strip()
        ),
        supersedes_state_id=(
            supersedes_state_id
        ),
    )


def learn_state_from_clear_scene(
        ledger: StoryStateLedger,
        scene: ScriptScene,
        scene_index: int,
        segment_id: str,
) -> ContinuityStateLearningResult:
    review = review_script_scene_continuity(
        ledger=ledger,
        scene=scene,
    )

    if review.assessment.status != "clear":
        raise ValueError(
            "Continuity may learn story state "
            "only from a clear ScriptScene."
        )

    candidates = (
        collect_state_candidates_from_review(
            review
        )
    )

    candidate = next(
        (
            item
            for item in candidates
            if item.segment_id == segment_id
        ),
        None,
    )

    if candidate is None:
        raise ValueError(
            "Requested segment is not an "
            "objective state candidate: "
            f"{segment_id!r}"
        )

    candidate_judgment = (
        classify_state_candidate(
            candidate
        )
    )

    if (
            candidate_judgment.verdict
            != "durable_state"
    ):
        raise ValueError(
            "Requested scene segment does not "
            "establish durable story state. "
            f"Verdict: "
            f"{candidate_judgment.verdict!r}"
        )

    segment_audit = next(
        (
            item
            for item in review.audit.segment_audits
            if item.segment_id == segment_id
        ),
        None,
    )

    if segment_audit is None:
        raise ValueError(
            "Continuity review is missing the "
            "requested segment audit."
        )

    commit_plan = build_state_commit_plan(
        candidate=candidate,
        candidate_judgment=(
            candidate_judgment
        ),
        segment_audit=segment_audit,
    )

    new_state_id = build_learned_state_id(
        scene_index=scene_index,
        segment_id=segment_id,
    )

    updated_ledger = commit_state_plan(
        ledger=ledger,
        plan=commit_plan,
        new_state_id=new_state_id,
        scene_index=scene_index,
    )

    return ContinuityStateLearningResult(
        segment_id=segment_id,
        statement=candidate.statement,
        candidate_judgment=(
            candidate_judgment
        ),
        commit_plan=commit_plan,
        committed_state_id=new_state_id,
        ledger=updated_ledger,
    )


def audit_scene_statements(
        ledger: StoryStateLedger,
        statements: list[str],
) -> ContinuitySceneAudit:
    if not statements:
        raise ValueError(
            "Scene continuity audit requires "
            "at least one statement."
        )

    cleaned_statements: list[str] = []

    for statement in statements:
        cleaned = statement.strip()

        if not cleaned:
            raise ValueError(
                "Scene statements cannot be empty."
            )

        cleaned_statements.append(cleaned)

    statement_audits: list[
        ContinuityStatementAudit
    ] = []

    for statement in cleaned_statements:
        audit = (
            audit_statement_against_active_states(
                ledger=ledger,
                new_statement=statement,
            )
        )

        statement_audits.append(audit)

    contradiction_count = sum(
        1
        for audit in statement_audits
        for result in audit.results
        if result.verdict == "contradiction"
    )

    return ContinuitySceneAudit(
        statement_audits=statement_audits,
        has_contradiction=(
                contradiction_count > 0
        ),
        contradiction_count=(
            contradiction_count
        ),
    )


def classify_state_candidate(
    candidate: ContinuityStateCandidate,
) -> ContinuityStateCandidateJudgment:
    if not candidate.statement.strip():
        raise ValueError(
            "State candidate statement "
            "cannot be empty."
        )

    result = safe_invoke(
        state_candidate_chain,
        {
            "statement": candidate.statement,
        },
        label=(
            "Continuity state candidate "
            "classification"
        ),
    )

    if not isinstance(
        result,
        ContinuityStateCandidateJudgment,
    ):
        raise TypeError(
            "State candidate classification "
            "did not return the expected model."
        )

    if (
        result.verdict == "durable_state"
        and not result.resulting_state.strip()
    ):
        raise ValueError(
            "durable_state classification must "
            "include a resulting_state."
        )

    if (
        result.verdict != "durable_state"
        and result.resulting_state.strip()
    ):
        raise ValueError(
            "Only durable_state classifications "
            "may include a resulting_state."
        )

    return result


def audit_script_scene_continuity(
        ledger: StoryStateLedger,
        scene: ScriptScene,
) -> ContinuityScriptSceneAudit:
    continuity_statements = (
        extract_continuity_scene_statements(
            scene
        )
    )

    objective_statements = [
        statement
        for statement in continuity_statements
        if statement.statement_type
           == "objective_story_state"
    ]

    segment_audits: list[
        ContinuityScriptSegmentAudit
    ] = []

    for statement in objective_statements:
        audit = (
            audit_statement_against_active_states(
                ledger=ledger,
                new_statement=statement.text,
            )
        )

        segment_audits.append(
            ContinuityScriptSegmentAudit(
                segment_id=statement.segment_id,
                statement=statement.text,
                audit=audit,
            )
        )

    contradiction_count = sum(
        1
        for segment_audit
        in segment_audits
        for result
        in segment_audit.audit.results
        if result.verdict
        == "contradiction"
    )

    return ContinuityScriptSceneAudit(
        segment_audits=segment_audits,
        has_contradiction=(
                contradiction_count > 0
        ),
        contradiction_count=(
            contradiction_count
        ),
    )


def review_script_scene_continuity(
        ledger: StoryStateLedger,
        scene: ScriptScene,
) -> ContinuitySceneReview:
    audit = audit_script_scene_continuity(
        ledger=ledger,
        scene=scene,
    )

    assessment = (
        build_continuity_scene_assessment(
            audit
        )
    )

    return ContinuitySceneReview(
        audit=audit,
        assessment=assessment,
    )


def commit_state_plan(
        ledger: StoryStateLedger,
        plan: ContinuityStateCommitPlan,
        new_state_id: str,
        scene_index: int,
) -> StoryStateLedger:
    if not new_state_id.strip():
        raise ValueError(
            "new_state_id cannot be empty."
        )

    if scene_index < 0:
        raise ValueError(
            "scene_index cannot be negative."
        )

    if plan.supersedes_state_id is not None:
        active_state_ids = {
            entry.state_id
            for entry in get_active_story_states(
                ledger
            )
        }

        if (
                plan.supersedes_state_id
                not in active_state_ids
        ):
            raise ValueError(
                "Commit plan references a "
                "story state that is no "
                "longer active."
            )

    new_entry = StoryStateEntry(
        state_id=new_state_id,
        scene_index=scene_index,
        statement=plan.statement,
        normalized_state=plan.normalized_state,
        supersedes_state_id=(
            plan.supersedes_state_id
        ),
    )

    return add_story_state(
        ledger,
        new_entry,
    )


def audit_statement_against_active_states(
    ledger: StoryStateLedger,
    new_statement: str,
) -> ContinuityStatementAudit:
    if not new_statement.strip():
        raise ValueError(
            "new_statement cannot be empty."
        )

    active_states = get_active_story_states(
        ledger
    )

    results: list[
        ContinuityStateAuditResult
    ] = []

    for state in active_states:
        established_state = (
            get_story_state_semantic_statement(
                state
            )
        )

        judgment = check_continuity(
            ContinuityCheckInput(
                established_state=(
                    established_state
                ),
                new_statement=new_statement,
            )
        )

        results.append(
            ContinuityStateAuditResult(
                state_id=state.state_id,
                established_state=(
                    established_state
                ),
                verdict=judgment.verdict,
                reason=judgment.reason,
            )
        )

    has_contradiction = any(
        result.verdict == "contradiction"
        for result in results
    )

    return ContinuityStatementAudit(
        new_statement=new_statement,
        results=results,
        has_contradiction=has_contradiction,
    )

# ---------------------------------------------------------
# Continuity Check
# ---------------------------------------------------------


def check_continuity(
        continuity_input: ContinuityCheckInput,
) -> ContinuityJudgment:
    validate_continuity_check_input(
        continuity_input
    )

    result = safe_invoke(
        continuity_chain,
        {
            "established_state": (
                continuity_input.established_state
            ),
            "new_statement": (
                continuity_input.new_statement
            ),
        },
        label="Continuity semantic check",
    )

    if not isinstance(
            result,
            ContinuityJudgment,
    ):
        raise TypeError(
            "Continuity check did not return "
            "a ContinuityJudgment."
        )

    return result