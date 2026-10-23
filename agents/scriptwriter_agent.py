from dotenv import load_dotenv
from tools.model_provider import create_chat_model
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import Literal
import json
from config import SCRIPTWRITER_MODEL
from tools.rate_limit import safe_invoke
import re

load_dotenv()

scriptwriter_model = create_chat_model(
    model_name=SCRIPTWRITER_MODEL,
    temperature=0.6,
    timeout=30,
    max_retries=0,
)

SCRIPT_SEGMENT_AUDIT_BATCH_SIZE = 8

script_audit_model = create_chat_model(
    model_name=SCRIPTWRITER_MODEL,
    temperature=0.1,
    timeout=30,
    max_retries=0,
    max_tokens=1800,
    reasoning_effort="low",
)

script_repair_model = create_chat_model(
    model_name=SCRIPTWRITER_MODEL,
    temperature=0.2,
    timeout=30,
    max_retries=0,
    max_tokens=2600,
    reasoning_effort="low",
)

SCRIPTWRITER_SYSTEM_PROMPT = """
You are the Scriptwriter Agent in a multi-agent Star Trek
screenwriting system.

Your job is to write a compelling screenplay-style scene using only
the story situation, verified canon facts, grounded character guidance,
grounded voice guidance, era/style constraints, scene requirements,
and prohibited assumptions supplied to you.

CORE RULE:

You may invent new fictional STORY EVENTS.
You may not invent new CANON FACTS.

You may create:
- fictional dialogue
- scene blocking
- physical actions
- dramatic reactions
- new story-specific interactions
- pacing and transitions
- fictional circumstances required by the supplied scene
- reasonable connective action that does not establish new canon

You may NOT invent or assume:
- canon history not supplied to you
- canon relationships not supplied to you
- ranks, titles, positions, or institutional status not supplied
- current contacts, sources, networks, or access for an established
  canon character unless supplied
- political circumstances not supplied
- technological capabilities not supplied
- character biography not supplied
- historical events not supplied
- motives, personality traits, habits, or behavioral tendencies
  that are not supported by grounded character guidance
- character-specific dialogue habits that are not supported by
  grounded voice guidance

STORY CONTEXT RULE:

Story context establishes the current fictional situation.
It does not establish missing canon history, biography, relationships,
personality, motives, skills, status, or current institutional access.

CHARACTER RULE:

Use only grounded_character_guidance when making claims about a
character's behavior, relationships, motivations, tendencies,
constraints, or established knowledge.

Do not recover rejected or unresolved characterization from your own
Star Trek knowledge.

VOICE RULE:

Use grounded_voice_guidance for character-specific speech patterns,
vocabulary, cadence, rhetorical habits, or other voice traits.

If grounded_voice_guidance is empty, write natural dialogue appropriate
to the supplied Star Trek era and situation, but do not introduce or
claim specific researched speech habits for that character.

TEMPORAL ACCESS RULE:

Past contact, employment, membership, association, or interaction does
not establish present-day access, communication, authority, or an
active network.

PROHIBITED ASSUMPTIONS:

Anything listed under prohibited_assumptions is unavailable to the
scene. Do not state it, imply it, or create story action that requires
it to be true.

SUBJECT-BOUND PROHIBITION RULE

A prohibited assumption applies only to the person, character,
organization, object, relationship, resource, or other subject that
the supplied prohibition actually names or necessarily describes.

Do not broaden a prohibition from one subject to another.

For example:

PROHIBITED ASSUMPTION:
"Garak has current contacts inside Cardassian Central Command."

DIALOGUE SPOKEN BY GARAK:
"My longtime source inside Central Command confirmed it."

→ prohibited

because the dialogue attributes the forbidden current contact
directly to Garak.

But:

DIALOGUE SPOKEN BY THE CARDASSIAN DEFECTOR:
"My longtime source inside Central Command confirmed it."

→ NOT prohibited merely because Garak is forbidden from having such
a contact.

The prohibition concerns Garak, not the Cardassian Defector.

If the defector's claimed source or relationship is not otherwise
established by supplied evidence, classify that claim according to
the normal grounding rules, typically unsupported.

Likewise, a prohibition about:
- Character A's access
- Character A's relationship
- Character A's possession
- Character A's status
- Character A's knowledge

must not automatically prohibit the same access, relationship,
possession, status, or knowledge for Character B.

A prohibited verdict requires the submitted segment itself to state,
imply, or require the PARTICULAR prohibited assumption cited in
prohibited_assumptions.

Semantic similarity alone is not sufficient.

Use the supplied segment_type and speaker metadata when determining
which character a dialogue claim belongs to.

ERA FIDELITY:

Follow the supplied story era and era_style_constraints.
Prefer the linguistic, dramatic, technological, and production
conventions appropriate to that Star Trek era rather than generic
modern science-fiction dialogue.

OUTPUT ACCOUNTING:

canon_facts_used must contain only supplied verified canon facts that
the generated scene actually relies upon.

invented_story_elements must identify significant new fictional
actions, events, interactions, or circumstances created for this scene.

These two fields are descriptive accounting only. Do not treat your own
classification of an element as proof that it is canonically valid.

When evidence is missing, write around the gap rather than filling it
with remembered Star Trek knowledge.
"""

scriptwriter_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            SCRIPTWRITER_SYSTEM_PROMPT,
        ),
        (
            "human",
            """
Write the requested Star Trek scene from the following
ScriptwriterInput.

Return ONLY one valid JSON object.

The JSON object must use exactly these top-level keys:

{{
  "scene_heading": "string",
  "setting_description": "string",
  "characters_present": ["string"],
  "script_text": "string",
  "canon_facts_used": ["string"],
  "invented_story_elements": ["string"]
}}

Do not use alternative field names such as "scene_text".
Do not place canon_facts_used or invented_story_elements inside
script_text.
Do not wrap the JSON in Markdown or code fences.
Do not include text before or after the JSON object.

{scriptwriter_input}
""",
        ),
    ]
)


class ScriptScene(BaseModel):
    scene_heading: str = Field(
        description="Era-appropriate screenplay scene heading."
    )

    setting_description: str = Field(
        description=(
            "Brief description of the location, atmosphere, and immediately "
            "observable circumstances of the scene."
        )
    )

    characters_present: list[str] = Field(
        description="Characters physically present in the scene."
    )

    script_text: str = Field(
        description=(
            "The complete screenplay-style scene, including action and dialogue."
        )
    )

    canon_facts_used: list[str] = Field(
        default_factory=list,
        description=(
            "Canon facts from supplied verified evidence that the scene relies on."
        )
    )

    invented_story_elements: list[str] = Field(
        default_factory=list,
        description=(
            "New fictional actions, events, interactions, or circumstances "
            "created specifically for this story."
        )
    )


class ScriptSceneSegment(BaseModel):
    segment_id: str
    field: str

    segment_type: Literal[
        "scene_heading",
        "setting_description",
        "speaker_cue",
        "parenthetical",
        "dialogue",
        "action",
    ]

    speaker: str | None = None

    text: str


class ScriptSegmentJudgment(BaseModel):
    verdict: Literal[
        "grounded",
        "permitted_story_invention",
        "unsupported",
        "prohibited",
    ]

    evidence_source: Literal[
        "verified_canon_facts",
        "story_context",
        "scene_objective",
        "story_era",
        "characters",
        "scene_requirements",
        "character_guidance",
        "voice_guidance",
        "era_style_constraints",
        "prohibited_assumptions",
        "none",
    ] = "none"

    evidence_index: int | None = None

    # Python owns the exact evidence text. The LLM selects
    # evidence_source + evidence_index; Python resolves this field.
    evidence_excerpt: str = ""

    reason: str


class ScriptSegmentAuditBatch(BaseModel):
    judgments: list[ScriptSegmentJudgment]


def normalize_script_segment_audit_batch(
        audit_batch: ScriptSegmentAuditBatch,
) -> ScriptSegmentAuditBatch:
    normalized_judgments: list[
        ScriptSegmentJudgment
    ] = []

    for judgment in audit_batch.judgments:
        if judgment.verdict in {
            "permitted_story_invention",
            "unsupported",
        }:
            judgment = judgment.model_copy(
                update={
                    "evidence_source": "none",
                    "evidence_index": None,
                    "evidence_excerpt": "",
                }
            )

        normalized_judgments.append(judgment)

    return audit_batch.model_copy(
        update={
            "judgments": normalized_judgments,
        }
    )


def split_script_line_for_audit(
        line: str,
) -> list[str]:
    stripped_line = line.strip()

    if not stripped_line:
        return []

    parts = re.split(
        r"(?<=[.!?])\s+",
        stripped_line,
    )

    return [
        part.strip()
        for part in parts
        if part.strip()
    ]


INLINE_SPEAKER_PATTERN = re.compile(
    r"^([A-Z][A-Z0-9 .'\-()]+):\s+(.+)$"
)

PRODUCTION_CUES = (
    "FADE IN",
    "FADE OUT",
    "CUT TO",
    "DISSOLVE TO",
    "SMASH CUT TO",
    "MATCH CUT TO",
)


# Recognize standard sluglines before speaker parsing. Keep the entire
# heading intact so its location claims remain available to the audit.
SCENE_HEADING_PATTERN = re.compile(
    r"^(?:INT\.?\s*/\s*EXT\.?|EXT\.?\s*/\s*INT\.?|"
    r"INT\.|EXT\.|I/E\.)(?=\s|$)",
    re.IGNORECASE,
)


def is_script_scene_heading(line: str) -> bool:
    return SCENE_HEADING_PATTERN.match(line.strip()) is not None

SPEAKER_CONTINUATION_PATTERN = re.compile(
    r"\s*\((?:CONT\.?|CONT'D|CONTINUED)\)\s*$",
    re.IGNORECASE,
)


def strip_speaker_continuation_marker(
        speaker: str,
) -> str:
    """
    Remove a standard screenplay continuation marker
    from a speaker cue while preserving the underlying
    speaker name.

    Examples:

    GARAK (cont.) -> GARAK
    GARAK (CONT.) -> GARAK
    GARAK (CONT'D) -> GARAK
    GARAK (CONTINUED) -> GARAK
    """

    return SPEAKER_CONTINUATION_PATTERN.sub(
        "",
        speaker.strip(),
    ).strip()

def is_standalone_speaker_cue(
        line: str,
) -> bool:
    stripped_line = line.strip()

    if not stripped_line:
        return False

    if not any(
            character.isalpha()
            for character in stripped_line
    ):
        return False

    speaker_name = strip_speaker_continuation_marker(
        stripped_line
    )

    if not speaker_name:
        return False

    upper_speaker_name = speaker_name.upper()

    if any(
            upper_speaker_name.startswith(cue)
            for cue in PRODUCTION_CUES
    ):
        return False

    # Ordinary sentences and production labels are not
    # standalone screenplay speaker cues.
    if speaker_name[-1] in ".?!:":
        return False

    return speaker_name == upper_speaker_name


def enumerate_script_scene_segments(
        scene: ScriptScene,
) -> list[ScriptSceneSegment]:
    segments: list[ScriptSceneSegment] = []

    if scene.scene_heading.strip():
        segments.append(
            ScriptSceneSegment(
                segment_id="scene_heading",
                field="scene_heading",
                segment_type="scene_heading",
                speaker=None,
                text=scene.scene_heading.strip(),
            )
        )

    if scene.setting_description.strip():
        segments.append(
            ScriptSceneSegment(
                segment_id="setting_description",
                field="setting_description",
                segment_type="setting_description",
                speaker=None,
                text=scene.setting_description.strip(),
            )
        )

    script_segment_index = 0
    current_speaker: str | None = None

    # Consecutive physical screenplay lines belonging to
    # the same speaker are buffered here before sentence
    # segmentation. This prevents visual line wrapping from
    # becoming false semantic claim boundaries.
    dialogue_buffer: list[str] = []
    action_buffer: list[str] = []

    def flush_dialogue_buffer() -> None:
        nonlocal script_segment_index
        nonlocal dialogue_buffer

        if not dialogue_buffer:
            return

        if current_speaker is None:
            raise ValueError(
                "Dialogue buffer contains text without "
                "an active speaker."
            )

        reconstructed_dialogue = " ".join(
            dialogue_buffer
        ).strip()

        for auditable_part in split_script_line_for_audit(
                reconstructed_dialogue
        ):
            segments.append(
                ScriptSceneSegment(
                    segment_id=(
                        f"script_text["
                        f"{script_segment_index}]"
                    ),
                    field="script_text",
                    segment_type="dialogue",
                    speaker=current_speaker,
                    text=auditable_part,
                )
            )

            script_segment_index += 1

        dialogue_buffer = []

    def flush_action_buffer() -> None:
        nonlocal script_segment_index
        nonlocal action_buffer

        if not action_buffer:
            return

        reconstructed_action = " ".join(
            action_buffer
        ).strip()

        for auditable_part in split_script_line_for_audit(
                reconstructed_action
        ):
            segments.append(
                ScriptSceneSegment(
                    segment_id=(
                        f"script_text["
                        f"{script_segment_index}]"
                    ),
                    field="script_text",
                    segment_type="action",
                    speaker=None,
                    text=auditable_part,
                )
            )

            script_segment_index += 1

        action_buffer = []

    for line in scene.script_text.splitlines():
        stripped_line = line.strip()

        # A blank screenplay line ends the current
        # dialogue block.
        if not stripped_line:
            flush_dialogue_buffer()
            flush_action_buffer()
            current_speaker = None
            continue

        # Support compact screenplay syntax such as:
        #
        # GARAK: You have my attention.
        #
        # Python separates the speaker cue from the dialogue.
        inline_speaker_match = INLINE_SPEAKER_PATTERN.match(
            stripped_line
        )

        if inline_speaker_match:
            # Finish any preceding standard dialogue block
            # before processing the inline speaker.
            flush_dialogue_buffer()
            flush_action_buffer()

            speaker = inline_speaker_match.group(1).strip()
            dialogue_text = inline_speaker_match.group(2).strip()

            segments.append(
                ScriptSceneSegment(
                    segment_id=(
                        f"script_text[{script_segment_index}]"
                    ),
                    field="script_text",
                    segment_type="speaker_cue",
                    speaker=speaker,
                    text=speaker,
                )
            )

            script_segment_index += 1

            for auditable_part in split_script_line_for_audit(
                    dialogue_text
            ):
                segments.append(
                    ScriptSceneSegment(
                        segment_id=(
                            f"script_text["
                            f"{script_segment_index}]"
                        ),
                        field="script_text",
                        segment_type="dialogue",
                        speaker=speaker,
                        text=auditable_part,
                    )
                )

                script_segment_index += 1

            # Inline speaker syntax is self-contained.
            current_speaker = None
            continue

        # Standard screenplay speaker cue:
        #
        # GARAK
        # (quietly)
        # Dialogue...
        if is_standalone_speaker_cue(
                stripped_line
        ):
            flush_dialogue_buffer()
            flush_action_buffer()

            current_speaker = stripped_line

            segments.append(
                ScriptSceneSegment(
                    segment_id=(
                        f"script_text[{script_segment_index}]"
                    ),
                    field="script_text",
                    segment_type="speaker_cue",
                    speaker=current_speaker,
                    text=stripped_line,
                )
            )

            script_segment_index += 1
            continue

        # Parenthetical dialogue direction belongs to the
        # active speaker. Flush any dialogue preceding it so
        # the parenthetical remains correctly ordered.
        if (
                current_speaker is not None
                and stripped_line.startswith("(")
                and stripped_line.endswith(")")
        ):
            flush_dialogue_buffer()

            segments.append(
                ScriptSceneSegment(
                    segment_id=(
                        f"script_text[{script_segment_index}]"
                    ),
                    field="script_text",
                    segment_type="parenthetical",
                    speaker=current_speaker,
                    text=stripped_line,
                )
            )

            script_segment_index += 1
            continue

        # With an active speaker, accumulate physical lines
        # until the dialogue block ends. Sentence splitting
        # happens only after those lines are reconstructed.
        if current_speaker is not None:
            dialogue_buffer.append(
                stripped_line
            )
            continue

        # Without an active speaker, the line is objective
        # screenplay action/narration. Action continues to
        # use its existing per-line semantic segmentation.
        # Without an active speaker, accumulate physical
        # screenplay action lines into one semantic block.
        #
        # Physical screenplay wrapping must not create
        # artificial story-state boundaries for Continuity.
        action_buffer.append(
            stripped_line
        )
    flush_dialogue_buffer()
    flush_action_buffer()

    return segments


SCRIPT_SEGMENT_AUDIT_PROMPT = """
You are the semantic grounding auditor for a Star Trek
screenwriting system.

You will receive:
1. authoritative evidence sources supplied upstream;
2. an ordered batch of generated screenplay segments.

Python owns the segment identities and ordering.
Return exactly one judgment for each submitted segment,
in the same order.

Do not repeat or rewrite the segment text.

VERDICTS

Use exactly one of these verdicts:

grounded
- The material claim is established by an allowed evidence source.

permitted_story_invention
- The segment creates new fictional story material.
- The invention does not establish an unsupported fact about canon,
  an established character, an established relationship, historical
  access, institutional status, technology capability, biography,
  motive, personality, behavioral tendency, or voice trait.

unsupported
- The segment introduces or embellishes a factual, canon-like,
  character, relational, historical, technological, psychological,
  behavioral, or voice claim that the supplied evidence does not
  establish.

prohibited
- The segment states, implies, or requires something listed in
  prohibited_assumptions.

VERDICT PRECEDENCE AND SCOPE RULE

Apply the verdict rules with the following semantic priorities.

1. PROHIBITED requires actual entailment of a supplied prohibition.

A segment is prohibited only when the segment itself states, implies,
or requires the PARTICULAR prohibited assumption cited.

The prohibited assumption must match the relevant:
- subject;
- relationship;
- resource;
- organization;
- access;
- status;
- capability;
- or other material scope.

Semantic similarity is not enough.

Do not broaden a narrow prohibition into a larger category.

Example:

PROHIBITED:
"Garak has a current Cardassian intelligence network."

GARAK:
"My contact within the Tal Shiar gave me this information."

This is NOT prohibited merely because both involve intelligence
contacts. A Tal Shiar contact does not entail a current Cardassian
intelligence network.

If no supplied prohibition actually covers the claim, classify it
under the remaining grounding rules, typically unsupported.

2. GROUNDED takes precedence over permitted_story_invention when the
material proposition is already established upstream.

Dialogue may generally contain fictional allegations or beliefs.

However, when dialogue restates or faithfully paraphrases an
allegation, belief, claim, event, or circumstance already established
by story_context, scene_requirements, or another authoritative source,
classify the segment as grounded rather than permitted_story_invention.

Example:

STORY CONTEXT:
"A defector claims that the Founders have infiltrated the
Cardassian government."

DEFECTOR:
"The Founders have placed agents within the Cardassian government."

→ grounded

The dialogue is dramatizing an already supplied fictional premise.

3. PERMITTED STORY INVENTION applies when the material is genuinely
new, not already grounded, and falls inside an allowed creative
category.

For a newly invented fictional character, this includes a new
story-specific contact, source, ordinary resource, possession,
relationship, or clue when:

- it belongs to that fictional character;
- it does not establish a new fact about an established canon
  character;
- it does not require an unsupported technological capability,
  canon history, or political circumstance; and
- any canon-specific institution or organization named by the line
  is already established somewhere in the supplied evidence.

Example:

Supplied evidence establishes that the Tal Shiar exists.

FICTIONAL CARDASSIAN DEFECTOR:
"A contact within the Tal Shiar fed me intercepted transmissions."

→ permitted_story_invention

This invents a source for the fictional defector. It does not establish
that Garak has current Tal Shiar access.

4. UNSUPPORTED is the fallback for an ungrounded factual or canon-like
claim that is neither an allowed story invention nor actually covered
by a supplied prohibition.

Do not use prohibited as a stronger synonym for unsupported.

SOURCE AUTHORITY RULES

verified_canon_facts:
May establish only canon facts actually contained in those facts.

story_context:
May establish the current fictional situation.
It may not establish missing canon history, biography, personality,
relationships, motives, skills, institutional access, or status.

scene_objective:
May establish what the scene is intended to accomplish.
It does not establish canon or character traits.

story_era:
May establish only the supplied series, era, and timeframe.
Do not use it to infer unsupplied locations, technology, politics,
relationships, or biography.

characters:
May establish that the supplied characters are available to the scene.
A character name alone establishes no biography or characterization.

scene_requirements:
May establish required fictional events and circumstances.
They do not establish unsupplied canon history or character traits.

character_guidance:
May establish only character behavior, motivations, knowledge,
relationships, or constraints actually stated in that guidance.

voice_guidance:
May establish only speech and dialogue traits actually stated there.
If voice_guidance is empty, do not treat model familiarity with a
character as evidence of that character's voice.

era_style_constraints:
May establish stylistic and production constraints only.
They do not establish story facts.

prohibited_assumptions:
Anything supported by or dependent upon these assumptions must receive
the verdict prohibited.

CREATIVE INVENTION RULE

The Scriptwriter is allowed to invent story-specific details about new
fictional people, objects, actions, and circumstances when doing so does
not silently create new canon facts.

Example categories of generally permissible invention:
- appearance of a newly invented fictional character;
- a fictional character carrying an ordinary object;
- blocking and movement;
- newly invented dialogue;
- temporary story-specific circumstances;
- new evidence or clues created for the fictional plot.

However, an invented detail about an established canon character may
still be unsupported if it asserts biography, personality, habits,
relationships, possessions, residence, status, abilities, motives,
behavioral tendencies, or historical facts not supplied upstream.

FICTIONAL CHARACTER RESOURCE RULE

A newly invented fictional character may be given new story-specific
contacts, sources, possessions, information, ordinary resources, or
relationships when those inventions exist only for the current story
and do not establish new facts about an established canon character.

This includes fictional relationships with canon institutions or
organizations whose existence is already established by the supplied
input.

For example, if "Tal Shiar" is already established by supplied
evidence:

A fictional Cardassian defector saying:

"A contact within the Tal Shiar gave me these transmissions."

may be permitted_story_invention.

This invents a fictional source belonging to a fictional character.
It does not establish that Garak, or any other established canon
character, has current Tal Shiar access.

By contrast:

Garak saying:

"My contact within the Tal Shiar gave me these transmissions."

requires supplied evidence establishing Garak's current contact or
access and must otherwise be unsupported or prohibited.

Do not transfer a fictional character's invented resource, contact,
relationship, or access to an established canon character.

This rule does NOT permit invention of unsupported technological
capabilities, canon history, political circumstances, or facts about
existing canon characters merely because a fictional character
mentions them.

CURRENT-SCENE DECISIONS AND OBJECT HANDLING

A character's new decision, offer, instruction, or action involving an
object introduced in this scene is permitted_story_invention when it
does not depend on prohibited access or unsupported canon capabilities.
Do not require prior evidence for a fictional event the scene creates.
For example, after receiving a newly introduced object, "Until then, it
stays here" or "I will keep this for now" expresses a present decision;
it does not assert historical ownership or a researched character trait.
This does not authorize claims of longstanding possession, institutional
access, or the ability to decode the object using unsupported technology.

HYPOTHETICAL AND SPECULATIVE DIALOGUE RULE

A fictional character may express uncertainty, possibility,
prediction, fear, suspicion, speculation, or a hypothetical
consequence without establishing the proposition as canon fact.

Pay attention to linguistic markers such as:
- if
- may
- might
- could
- perhaps
- possibly
- I think
- I believe
- I suspect
- I fear
- what if

A conditional statement does not assert that its condition is true.

For example:

"If the Dominion moves through Cardassian channels, war may follow."

does not establish that the Dominion is currently moving through
Cardassian channels, nor that war will definitely follow.

It is a fictional hypothetical prediction and may be classified as
permitted_story_invention when it does not depend on an explicitly
prohibited assumption.

By contrast:

"The Dominion is moving through Cardassian channels."

asserts a current factual condition and requires support from an
allowed evidence source or an explicitly supplied fictional premise.

Likewise:

"I fear they may already be here."

may be fictional speculation.

"They are already here."

is a factual assertion and requires grounding unless the current
fictional story context explicitly establishes it.

Do not convert uncertainty, conditional language, or character belief
into an objective factual assertion during audit.

SET DRESSING AND BLOCKING RULE

Non-material set dressing and one-off physical actions may be
permitted_story_invention when they exist only to stage the scene
and do not establish a meaningful canon or character fact.

Examples that may be permitted:
- ordinary lighting conditions;
- generic furniture or containers;
- doors opening or closing;
- a character walking, sitting, turning, or handling a generic
  story-specific object;
- a generic fictional prop whose existence does not establish a
  special technological capability or canon fact.

However, set dressing is not permission to invent facts about an
established canon character.

For an established canon character, do not invent:
- residence, quarters, office, workplace, or property ownership;
- personally owned canon-specific objects;
- established possessions;
- recurring habits or preferences;
- specialized equipment or technological capability.

For example:
"Garak's quarters are dimly lit"
is unsupported unless supplied evidence establishes that the
location is Garak's quarters.

"A lamp flickers beside Garak"
may be permitted story-specific set dressing because it does not
establish ownership, biography, or capability.

DIALOGUE EPISTEMIC RULE:

Dialogue is not automatically an objective statement of world fact.

A character may state a fictional allegation, belief, suspicion,
theory, lie, prediction, interpretation, or uncertain claim without
that statement becoming established canon or objective story truth.

When auditing dialogue, distinguish between:

1. OBJECTIVE WORLD ASSERTION
   Narration, action description, scene description, or other text
   that presents a proposition as objectively true.

2. GROUNDED CHARACTER ASSERTION
   Dialogue that expresses an allegation, belief, claim, or other
   proposition already established as part of the supplied current
   story situation.

3. PERMITTED FICTIONAL CHARACTER ASSERTION
   Dialogue that invents a new story-specific allegation, belief,
   suspicion, theory, lie, prediction, or interpretation without
   asserting that it is established canon.

A character-attributed assertion does not need to be canonically true
merely because the character says it.

Example:

STORY CONTEXT:
A defector claims that the Founders have infiltrated the
Cardassian government.

DIALOGUE:
"The Founders have placed agents within the Cardassian government."

This is grounded because the dialogue dramatizes the allegation already
established by the supplied story context. It does NOT establish the
allegation itself as canonically true.

Example:

DIALOGUE:
"I think the government may already be compromised."

If this is a new fictional suspicion consistent with the supplied
scene, it may be permitted_story_invention. It does NOT become canon
merely because a character says it.

However, dialogue is NOT a loophole for inventing unsupported canon.

A character's statement must still be unsupported when understanding
or accepting the statement requires an unsupported preexisting canon
fact, including an unsupplied:

- canon person, place, organization, or institution
- historical event
- relationship
- rank, title, or official status
- current contact, source, or access
- technological capability
- political circumstance
- biography or established history
- character-specific knowledge, skill, motive, or behavioral tendency

Example:

"My longtime source inside Central Command confirmed it."

is unsupported unless the supplied evidence establishes the required
source/relationship/current access and the canon-specific institutional
reference is itself permitted by the supplied evidence.

EPISTEMIC CLASSIFICATION RULE:

Judge what the SEGMENT itself establishes.

Do not convert character dialogue into objective narration.
Do not convert an allegation into an established fact merely because
the dialogue states the allegation declaratively.
Do not require a fictional allegation to be objectively proven before
a character may express it.

CANON-SPECIFIC DETAIL RULE

Creative freedom does not authorize the use of unsupplied
franchise-specific lore.

A generic story-specific object, location description, or piece of
set dressing may be a permitted_story_invention when it does not
establish a canon fact.

However, a named Star Trek-specific item, technology, substance,
location, institution, species, title, vessel, organization,
historical reference, or other established-universe term must be
supported by an allowed evidence source unless that exact concept is
already established elsewhere in the supplied input.

Do not use your own Star Trek knowledge as evidence that such a term
exists or is appropriate.

For example:

"Garak holds a glass."
may be permitted_story_invention.

"A lamp flickers beside Garak."
may be permitted_story_invention.

"Garak holds a glass of Rakantha."
is unsupported unless Rakantha is established by supplied evidence.

A generic fictional name invented specifically for the current story
may still be permitted when it clearly represents a new fictional
story element rather than recovered canon lore.

Terms already established by the supplied input may be used normally.
For example, if "Cardassian", "Founders", "Dominion", or "Deep Space
Nine" occur in authoritative evidence, their mere use does not require
independent rejection.

CANON ENTITY AVAILABILITY RULE

A canon-specific person, place, organization, institution, species,
technology name, vessel, title, or other established-universe term is
AVAILABLE FOR REFERENCE when that exact named term already appears in
any authoritative supplied evidence source.

An exact occurrence of the term is sufficient to establish that the
named entity or concept is available to the current scene.

This does NOT establish additional facts about that entity.

For example:

VERIFIED CANON FACT:
"Garak joined Enabran Tain on a joint Obsidian Order/Tal Shiar
mission to destroy the Founders' homeworld."

This establishes that the named organization "Tal Shiar" is available
for reference.

It does NOT establish:
- that Garak currently has a Tal Shiar contact;
- that the Tal Shiar currently operates in a particular location;
- that any specific Tal Shiar member exists;
- that a supplied character currently has Tal Shiar access;
- any unsupplied Tal Shiar capability, history, status, or activity.

Therefore:

FICTIONAL CARDASSIAN DEFECTOR:
"A contact within the Tal Shiar fed me intercepted transmissions."

may be permitted_story_invention under the FICTIONAL CHARACTER RESOURCE
RULE because:
- "Tal Shiar" itself is already available from supplied evidence;
- the contact is newly invented for a fictional character;
- the statement does not establish current Tal Shiar access for Garak
  or another established canon character.

Likewise:

FICTIONAL CARDASSIAN DEFECTOR:
"A contact within the Tal Shiar, disillusioned, fed me intercepted
transmissions."

may also be permitted_story_invention. "Disillusioned" is a fictional
trait assigned to the newly invented fictional contact, not a claim
about an established canon character.

By contrast:

GARAK:
"My contact within the Tal Shiar fed me intercepted transmissions."

is unsupported unless supplied evidence establishes Garak's current
Tal Shiar contact or access.

When determining whether a canon-specific term is available, inspect
ALL authoritative evidence sources for the exact named term.

Do not reject a term as unsupplied when that exact term already appears
in an authoritative evidence source.

Mere entity availability must never be expanded into unsupplied facts
about that entity.

CURRENT RESOURCE RULE

Do not invent that an established canon character already possesses
story-relevant information, evidence, records, intelligence, files,
contacts, equipment, access, or other resources unless an allowed
evidence source establishes that possession or access.

A newly invented clue may be introduced as a new fictional story
event when the scene itself establishes how it enters the story.

However:
"I will examine the data you just gave me"
may be permitted if the fictional scene has just established that
the data was provided.

But:
"I will examine the data I already possess"
is unsupported unless supplied evidence establishes that the
character currently possesses that data.

Current possession must not be inferred from past knowledge,
experience, employment, contact, or association.

CLAIM EXPANSION RULE

Evidence must support the entire material claim.

A related fact is not sufficient.

For example:
- "joined Tain on a joint mission"
  does NOT establish
  "served under Tain";

- "received information from a Female Changeling"
  does NOT establish what that Changeling specifically said;

- "knows about the Founders"
  does NOT establish distrust, fear, caution, or a preferred
  verification procedure.

TEMPORAL ACCESS RULE

Past contact, employment, membership, association, or interaction
does not establish current access, communication, authority,
or an active network.

PRODUCTION FORMATTING

Pure screenplay formatting such as:
FADE IN, FADE OUT, CUT TO, speaker labels, INT., EXT., DAY, NIGHT,
and equivalent formatting may be classified as
permitted_story_invention when no factual claim is introduced.

Formatting does not make the factual content inside a screenplay
element exempt from audit.

A scene heading may contain both formatting and a location claim.

For example:

"INT. DIM ALCOVE - NIGHT"
may be permitted_story_invention when the alcove is a generic
story-specific location.

"INT. QUIET CORNER OF QUARK'S BAR - NIGHT"
contains screenplay formatting, but also asserts the named
Star Trek location "Quark's Bar."

If a scene heading introduces an unsupplied canon-specific location,
institution, vessel, facility, or other established-universe detail,
the segment is unsupported unless an allowed evidence source
establishes that detail.

Do not classify an entire scene heading as permitted merely because
it uses screenplay formatting.

EVIDENCE REFERENCE RULE

Python supplies each evidence source as an ordered list of items with
an explicit zero-based evidence_index.

For grounded:
- evidence_source must identify the source that supports the claim;
- evidence_index must identify the ONE supplied evidence item that
  supports the material claim;
- evidence_excerpt must be empty because Python owns exact evidence
  text resolution.

For prohibited:
- evidence_source must be prohibited_assumptions;
- evidence_index must identify the ONE supplied prohibited assumption
  that the segment states, implies, or requires;
- evidence_excerpt must be empty because Python owns exact evidence
  text resolution.

For permitted_story_invention or unsupported:
- evidence_source must be "none";
- evidence_index must be null;
- evidence_excerpt must be empty.

Do not paraphrase or reproduce evidence text.
Select the evidence source and evidence index only.

Return valid JSON matching the ScriptSegmentAuditBatch schema.
"""

SCRIPTWRITER_REPAIR_SYSTEM_PROMPT = """
You are the repair stage for the Scriptwriter Agent in a multi-agent
Star Trek screenwriting system.

You will receive:

1. the authoritative ScriptwriterInput;
2. the original generated ScriptScene;
3. deterministic grounding-audit issues identifying unsupported or
   prohibited scene segments.

Your task is to return one repaired ScriptScene.

CORE REPAIR RULE:

Repair every supplied audit issue without introducing new unsupported
canon facts, character facts, relationships, resources, capabilities,
history, motives, voice traits, or prohibited assumptions.

The audit issues are authoritative negative findings.

Do not argue with them.
Do not preserve a flagged claim merely because it sounds plausible.

PRESERVATION RULE:

Preserve all unaffected scene material whenever reasonably possible.

Do not rewrite the entire scene for style.
Do not change characters, plot direction, dramatic objective, or
supplied canon merely because alternative wording is possible.

Small connective changes are allowed when necessary to make a repaired
passage read naturally.

REPAIR STRATEGIES:

When a canon-specific location or detail is unsupported:
- replace it with a generic story-specific equivalent when possible.

When dialogue expands a supplied allegation beyond what is established:
- narrow the statement;
- phrase it as an allowed belief, suspicion, allegation, uncertainty,
  or story-specific interpretation when appropriate;
- or remove the unsupported expansion.

When dialogue invents an unsupported technological, analytical,
historical, or institutional capability:
- remove that capability;
- replace it with ordinary story-specific evidence or uncertainty;
- or write around the unsupported detail.

When an issue flags an unsupported institutional attitude, history,
or collection practice:
- remove the institutional assertion, or narrow it to the specific
  story allegation actually supplied;
- do not replace one broad institutional assertion with another;
- for example, replacing "The agency feared them" with "The agency
  collected whatever it could" still invents institutional behavior;
- a bounded alternative, when the input supplies an unverified document
  allegation, is "I claim these fragments mention possible activity.
  I cannot establish whether the claim is true."
Check each changed passage against the input before returning the scene.
Do not treat stylistic paraphrasing as evidence of grounding.

When a segment violates a prohibited assumption:
- remove the prohibited dependency entirely.
- Do not merely paraphrase the same prohibited fact.

CANON RULE:

The Scriptwriter may invent new fictional STORY EVENTS.
It may not invent new CANON FACTS.

Use only canon facts contained in verified_canon_facts.

Do not use your own Star Trek knowledge to repair a gap.

CHARACTER RULE:

Do not add new facts about an established canon character unless they
are supported by grounded character guidance or another authoritative
supplied source.

VOICE RULE:

Do not invent researched character-specific speech habits when
grounded_voice_guidance is empty.

STORY REQUIREMENT RULE:

The repaired scene must still satisfy the supplied story_context,
scene_objective, scene_requirements, story_era, and
era_style_constraints.

Do not solve a grounding problem simply by deleting a required
dramatic beat.

OUTPUT ACCOUNTING:

canon_facts_used must contain only verified canon facts supplied in
ScriptwriterInput that the repaired scene actually relies upon.

invented_story_elements must describe significant fictional story
elements still present after repair.

Return the COMPLETE repaired ScriptScene, not patches or instructions.
"""

scriptwriter_repair_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            SCRIPTWRITER_REPAIR_SYSTEM_PROMPT,
        ),
        (
            "human",
            """
Repair the supplied ScriptScene using the deterministic audit issues.

Return ONLY one valid JSON object with exactly these top-level keys:

{{
  "scene_heading": "string",
  "setting_description": "string",
  "characters_present": ["string"],
  "script_text": "string",
  "canon_facts_used": ["string"],
  "invented_story_elements": ["string"]
}}

AUTHORITATIVE SCRIPTWRITER INPUT:

{scriptwriter_input}

ORIGINAL SCRIPT SCENE:

{original_scene}

AUDIT ISSUES:

{audit_issues}

Do not return explanations.
Do not return patches.
Do not wrap the JSON in Markdown or code fences.
Do not include text before or after the JSON object.
""",
        ),
    ]
)

script_segment_audit_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            SCRIPT_SEGMENT_AUDIT_PROMPT,
        ),
        (
            "human",
            """
        Audit the following ordered screenplay segments.

        AUTHORITATIVE EVIDENCE SOURCES:

        {evidence_sources}

        ORDERED SEGMENTS:

        {segments}

        Return ONLY one valid JSON object.

        The JSON object must use exactly this structure:

        {{
          "judgments": [
            {{
              "verdict": "grounded | permitted_story_invention | unsupported | prohibited",
              "evidence_source": "verified_canon_facts | story_context | scene_objective | story_era | characters | scene_requirements | character_guidance | voice_guidance | era_style_constraints | prohibited_assumptions | none",
              "evidence_index": "integer or null",
              "evidence_excerpt": "",
              "reason": "string"
            }}
          ]
        }}

        Return exactly one judgment for every submitted segment,
        in the same order as the submitted segments.

        Evidence indexes are zero-based and are supplied explicitly
        in AUTHORITATIVE EVIDENCE SOURCES.

        For grounded/prohibited judgments, select exactly one valid
        evidence_index from the selected evidence_source.

        For permitted_story_invention/unsupported judgments, return
        evidence_index as null.

        Always return evidence_excerpt as an empty string. Python
        resolves the exact evidence text after your judgment.

        Do NOT return segment_id.
        Do NOT return field.
        Do NOT repeat the segment text.
        Do NOT reproduce or paraphrase evidence text.
        Do NOT return a bare JSON list.
        Do NOT omit reason.
        Do NOT wrap the JSON in Markdown or code fences.
        Do NOT include text before or after the JSON object.
        """,
        ),
    ]
)

structured_script_audit_model = (
    script_audit_model.with_structured_output(
        ScriptSegmentAuditBatch,
        method="json_mode",
    )
)

script_segment_audit_chain = (
        script_segment_audit_prompt
        | structured_script_audit_model
)


class ScriptSceneAuditIssue(BaseModel):
    segment_id: str
    field: str
    text: str
    verdict: Literal[
        "unsupported",
        "prohibited",
    ]
    reason: str


class ScriptSceneAudit(BaseModel):
    grounded: bool
    issues: list[ScriptSceneAuditIssue]
    reason: str


class ScriptCharacterContext(BaseModel):
    character_name: str = Field(
        description="The character this writing guidance applies to."
    )

    grounded_character_guidance: list[str] = Field(
        default_factory=list,
        description=(
            "Character behavior, relationships, motivations, or constraints "
            "that survived upstream grounding checks."
        )
    )

    grounded_voice_guidance: list[str] = Field(
        default_factory=list,
        description=(
            "Dialogue and voice characteristics supported by supplied "
            "voice research. Leave empty when voice research is unavailable."
        )
    )


class ScriptwriterInput(BaseModel):
    story_context: str = Field(
        description=(
            "The established fictional situation surrounding the scene."
        )
    )

    scene_objective: str = Field(
        description=(
            "What this particular scene must accomplish dramatically."
        )
    )

    story_era: str = Field(
        description=(
            "The Star Trek series, era, and timeframe governing the scene."
        )
    )

    characters: list[ScriptCharacterContext] = Field(
        description=(
            "Characters available to the scene and only the grounded "
            "character guidance that may be used for each."
        )
    )

    verified_canon_facts: list[str] = Field(
        default_factory=list,
        description=(
            "Canon facts verified upstream and permitted for use in the scene."
        )
    )

    era_style_constraints: list[str] = Field(
        default_factory=list,
        description=(
            "Era-specific linguistic, technological, production, and "
            "screenwriting constraints the scene must follow."
        )
    )

    scene_requirements: list[str] = Field(
        default_factory=list,
        description=(
            "Specific fictional events, characters, locations, beats, or "
            "other requirements that must appear in the scene."
        )
    )

    prohibited_assumptions: list[str] = Field(
        default_factory=list,
        description=(
            "Facts or character assumptions the Scriptwriter must not "
            "introduce because they are unresolved, unsupported, or "
            "otherwise unavailable to this scene."
        )
    )


def build_script_scene_audit(
        script_input: ScriptwriterInput,
        scene: ScriptScene,
        audit_batch: ScriptSegmentAuditBatch,
) -> ScriptSceneAudit:
    segments = enumerate_script_scene_segments(scene)

    validate_script_segment_audit(
        script_input,
        segments,
        audit_batch,
    )

    issues: list[ScriptSceneAuditIssue] = []

    for segment, judgment in zip(
            segments,
            audit_batch.judgments,
    ):
        if judgment.verdict not in {
            "unsupported",
            "prohibited",
        }:
            continue

        issues.append(
            ScriptSceneAuditIssue(
                segment_id=segment.segment_id,
                field=segment.field,
                text=segment.text,
                verdict=judgment.verdict,
                reason=judgment.reason,
            )
        )

    if issues:
        return ScriptSceneAudit(
            grounded=False,
            issues=issues,
            reason=(
                f"{len(issues)} ScriptScene segment(s) "
                "remain unsupported or prohibited."
            ),
        )

    return ScriptSceneAudit(
        grounded=True,
        issues=[],
        reason=(
            "All ScriptScene segments are grounded or "
            "permitted story inventions."
        ),
    )


def build_scriptwriter_evidence_sources(
        script_input: ScriptwriterInput,
) -> dict[str, tuple[str, ...]]:
    character_guidance = tuple(
        guidance
        for character in script_input.characters
        for guidance in character.grounded_character_guidance
    )

    voice_guidance = tuple(
        guidance
        for character in script_input.characters
        for guidance in character.grounded_voice_guidance
    )

    character_names = tuple(
        character.character_name
        for character in script_input.characters
    )

    return {
        "verified_canon_facts": tuple(
            script_input.verified_canon_facts
        ),
        "story_context": (
            script_input.story_context,
        ),
        "scene_objective": (
            script_input.scene_objective,
        ),
        "story_era": (
            script_input.story_era,
        ),
        "characters": character_names,
        "scene_requirements": tuple(
            script_input.scene_requirements
        ),
        "character_guidance": character_guidance,
        "voice_guidance": voice_guidance,
        "era_style_constraints": tuple(
            script_input.era_style_constraints
        ),
        "prohibited_assumptions": tuple(
            script_input.prohibited_assumptions
        ),
    }


def build_scriptwriter_evidence_reference_payload(
        script_input: ScriptwriterInput,
) -> dict[str, list[dict[str, int | str]]]:
    evidence_sources = build_scriptwriter_evidence_sources(
        script_input
    )

    return {
        source_name: [
            {
                "evidence_index": index,
                "text": source_text,
            }
            for index, source_text in enumerate(source_texts)
        ]
        for source_name, source_texts in evidence_sources.items()
    }


def resolve_script_segment_evidence_references(
        script_input: ScriptwriterInput,
        audit_batch: ScriptSegmentAuditBatch,
) -> ScriptSegmentAuditBatch:
    evidence_sources = build_scriptwriter_evidence_sources(
        script_input
    )

    resolved_judgments: list[
        ScriptSegmentJudgment
    ] = []

    for judgment in audit_batch.judgments:
        if judgment.verdict in {
            "permitted_story_invention",
            "unsupported",
        }:
            resolved_judgments.append(
                judgment.model_copy(
                    update={
                        "evidence_source": "none",
                        "evidence_index": None,
                        "evidence_excerpt": "",
                    }
                )
            )
            continue

        if (
                judgment.evidence_source == "none"
                or judgment.evidence_index is None
        ):
            # Leave the invalid reference unresolved so the
            # deterministic validator can report the contract error.
            resolved_judgments.append(
                judgment.model_copy(
                    update={
                        "evidence_excerpt": "",
                    }
                )
            )
            continue

        source_texts = evidence_sources[
            judgment.evidence_source
        ]

        evidence_index = judgment.evidence_index

        if (
                evidence_index < 0
                or evidence_index >= len(source_texts)
        ):
            # Preserve the invalid index for validation.
            resolved_judgments.append(
                judgment.model_copy(
                    update={
                        "evidence_excerpt": "",
                    }
                )
            )
            continue

        resolved_judgments.append(
            judgment.model_copy(
                update={
                    "evidence_excerpt": source_texts[
                        evidence_index
                    ],
                }
            )
        )

    return audit_batch.model_copy(
        update={
            "judgments": resolved_judgments,
        }
    )


def validate_script_segment_audit(
        script_input: ScriptwriterInput,
        segments: list[ScriptSceneSegment],
        audit_batch: ScriptSegmentAuditBatch,
) -> None:
    errors: list[str] = []

    evidence_sources = build_scriptwriter_evidence_sources(
        script_input
    )

    # Python owns coverage. The auditor must return exactly
    # one judgment for every submitted segment.
    if len(audit_batch.judgments) != len(segments):
        errors.append(
            "Audit judgment count does not match submitted "
            f"segment count: expected {len(segments)}, "
            f"received {len(audit_batch.judgments)}."
        )

    # Only validate pairs that actually exist.
    pair_count = min(
        len(segments),
        len(audit_batch.judgments),
    )

    for index in range(pair_count):
        segment = segments[index]
        judgment = audit_batch.judgments[index]

        if not judgment.reason.strip():
            errors.append(
                f"{segment.segment_id}: audit reason cannot be empty."
            )

        if judgment.verdict == "grounded":
            if judgment.evidence_source in {
                "none",
                "prohibited_assumptions",
            }:
                errors.append(
                    f"{segment.segment_id}: grounded verdict must "
                    "cite an allowed supporting evidence source."
                )
                continue

            if judgment.evidence_index is None:
                errors.append(
                    f"{segment.segment_id}: grounded verdict requires "
                    "an evidence_index."
                )
                continue

            allowed_source_texts = evidence_sources[
                judgment.evidence_source
            ]

            if (
                    judgment.evidence_index < 0
                    or judgment.evidence_index
                    >= len(allowed_source_texts)
            ):
                errors.append(
                    f"{segment.segment_id}: evidence_index "
                    f"{judgment.evidence_index} is out of range for "
                    f"source {judgment.evidence_source!r}."
                )
                continue

            expected_excerpt = allowed_source_texts[
                judgment.evidence_index
            ]

            if judgment.evidence_excerpt != expected_excerpt:
                errors.append(
                    f"{segment.segment_id}: evidence_excerpt was "
                    "not resolved exactly from the selected "
                    "evidence_source/evidence_index."
                )

        elif judgment.verdict == "prohibited":
            if (
                    judgment.evidence_source
                    != "prohibited_assumptions"
            ):
                errors.append(
                    f"{segment.segment_id}: prohibited verdict must "
                    "cite prohibited_assumptions."
                )
                continue

            if judgment.evidence_index is None:
                errors.append(
                    f"{segment.segment_id}: prohibited verdict "
                    "requires an evidence_index."
                )
                continue

            prohibited_sources = evidence_sources[
                "prohibited_assumptions"
            ]

            if (
                    judgment.evidence_index < 0
                    or judgment.evidence_index
                    >= len(prohibited_sources)
            ):
                errors.append(
                    f"{segment.segment_id}: prohibited "
                    f"evidence_index {judgment.evidence_index} is "
                    "out of range."
                )
                continue

            expected_excerpt = prohibited_sources[
                judgment.evidence_index
            ]

            if judgment.evidence_excerpt != expected_excerpt:
                errors.append(
                    f"{segment.segment_id}: prohibited "
                    "evidence_excerpt was not resolved exactly from "
                    "the selected prohibited assumption."
                )

        elif judgment.verdict in {
            "permitted_story_invention",
            "unsupported",
        }:
            if judgment.evidence_source != "none":
                errors.append(
                    f"{segment.segment_id}: "
                    f"{judgment.verdict} verdict must use "
                    "evidence_source='none'."
                )

            if judgment.evidence_index is not None:
                errors.append(
                    f"{segment.segment_id}: "
                    f"{judgment.verdict} verdict must have "
                    "evidence_index=None."
                )

            if judgment.evidence_excerpt != "":
                errors.append(
                    f"{segment.segment_id}: "
                    f"{judgment.verdict} verdict must have an "
                    "empty evidence_excerpt."
                )

    if errors:
        raise ValueError(
            "Invalid Scriptwriter semantic audit:\n- "
            + "\n- ".join(errors)
        )

def resolve_script_speaker_name(
    speaker: str,
    character_names: list[str],
) -> str:
    normalized_speaker = (
        strip_speaker_continuation_marker(
            speaker
        ).casefold()
    )

    exact_matches = [
        character_name
        for character_name in character_names
        if character_name.strip().casefold()
        == normalized_speaker
    ]

    if len(exact_matches) == 1:
        return exact_matches[0]

    if len(exact_matches) > 1:
        raise ValueError(
            f"Speaker cue {speaker!r} is ambiguous among "
            "the supplied characters."
        )

    final_name_matches = [
        character_name
        for character_name in character_names
        if character_name.strip()
        and character_name.strip().split()[-1].casefold()
        == normalized_speaker
    ]

    if len(final_name_matches) == 1:
        return final_name_matches[0]

    if len(final_name_matches) > 1:
        raise ValueError(
            f"Speaker cue {speaker!r} is ambiguous among "
            "the supplied characters."
        )

    raise ValueError(
        f"Speaker cue {speaker!r} does not match any "
        "supplied character."
    )

def build_speaker_cue_audit_judgment(
    script_input: ScriptwriterInput,
    scene: ScriptScene,
    segment: ScriptSceneSegment,
) -> ScriptSegmentJudgment:
    if segment.segment_type != "speaker_cue":
        raise ValueError(
            "Deterministic speaker-cue validation received "
            "a non-speaker-cue segment."
        )

    if segment.speaker is None:
        raise ValueError(
            f"{segment.segment_id}: speaker cue has no speaker."
        )

    supplied_character_names = [
        character.character_name
        for character in script_input.characters
    ]

    canonical_character_name = resolve_script_speaker_name(
        segment.speaker,
        supplied_character_names,
    )

    present_character_names = {
        character_name.strip().casefold()
        for character_name in scene.characters_present
    }

    if (
        canonical_character_name.strip().casefold()
        not in present_character_names
    ):
        raise ValueError(
            f"{segment.segment_id}: speaker cue "
            f"{segment.speaker!r} resolves to "
            f"{canonical_character_name!r}, but that character "
            "is not listed in scene.characters_present."
        )

    canonical_character_index = (
        supplied_character_names.index(
            canonical_character_name
        )
    )

    return ScriptSegmentJudgment(
        verdict="grounded",
        evidence_source="characters",
        evidence_index=canonical_character_index,
        evidence_excerpt=canonical_character_name,
        reason=(
            "Speaker cue deterministically matches a character "
            "supplied upstream and listed in "
            "scene.characters_present."
        ),
    )

def audit_script_scene(
        script_input: ScriptwriterInput,
        scene: ScriptScene,
) -> ScriptSegmentAuditBatch:
    segments = enumerate_script_scene_segments(scene)

    if not segments:
        raise ValueError(
            "Cannot audit a ScriptScene with no auditable segments."
        )

    semantic_segments = [
        segment
        for segment in segments
        if segment.segment_type != "speaker_cue"
    ]

    evidence_reference_payload = (
        build_scriptwriter_evidence_reference_payload(
            script_input
        )
    )

    serialized_evidence = json.dumps(
        evidence_reference_payload,
        indent=2,
        ensure_ascii=False,
    )

    judgment_by_segment_id: dict[
        str,
        ScriptSegmentJudgment,
    ] = {}

    # Speaker cues are structural metadata. Python validates
    # them deterministically rather than spending an LLM
    # judgment on each cue.
    for segment in segments:
        if segment.segment_type == "speaker_cue":
            judgment_by_segment_id[segment.segment_id] = (
                build_speaker_cue_audit_judgment(
                    script_input,
                    scene,
                    segment,
                )
            )

    # Only semantically meaningful segments are sent to
    # the LLM auditor.
    for batch_start in range(
            0,
            len(semantic_segments),
            SCRIPT_SEGMENT_AUDIT_BATCH_SIZE,
    ):
        segment_batch = semantic_segments[
            batch_start:
            batch_start + SCRIPT_SEGMENT_AUDIT_BATCH_SIZE
        ]

        serialized_segments = json.dumps(
            [
                {
                    "segment_id": segment.segment_id,
                    "field": segment.field,
                    "segment_type": segment.segment_type,
                    "speaker": segment.speaker,
                    "text": segment.text,
                }
                for segment in segment_batch
            ],
            indent=2,
            ensure_ascii=False,
        )

        audit_batch = safe_invoke(
            script_segment_audit_chain,
            {
                "evidence_sources": serialized_evidence,
                "segments": serialized_segments,
            },
            label="Scriptwriter semantic audit batch",
        )

        if not isinstance(
                audit_batch,
                ScriptSegmentAuditBatch,
        ):
            raise TypeError(
                "Scriptwriter semantic audit did not return "
                "a ScriptSegmentAuditBatch."
            )

        audit_batch = normalize_script_segment_audit_batch(
            audit_batch
        )

        audit_batch = resolve_script_segment_evidence_references(
            script_input,
            audit_batch,
        )

        # Validate exact batch coverage and evidence contracts
        # before associating positional judgments with segments.
        validate_script_segment_audit(
            script_input,
            segment_batch,
            audit_batch,
        )

        for segment, judgment in zip(
                segment_batch,
                audit_batch.judgments,
        ):
            judgment_by_segment_id[segment.segment_id] = (
                judgment
            )

    # Restore the complete scene audit to the original
    # deterministic segment order, including Python-owned
    # speaker-cue judgments.
    ordered_judgments = [
        judgment_by_segment_id[segment.segment_id]
        for segment in segments
    ]

    combined_audit = ScriptSegmentAuditBatch(
        judgments=ordered_judgments,
    )

    # Final whole-scene coverage and evidence-contract check.
    validate_script_segment_audit(
        script_input,
        segments,
        combined_audit,
    )

    return combined_audit

def validate_scriptwriter_input(
        script_input: ScriptwriterInput,
) -> None:
    errors: list[str] = []

    # Required scalar inputs must contain meaningful text.
    required_text_fields = {
        "story_context": script_input.story_context,
        "scene_objective": script_input.scene_objective,
        "story_era": script_input.story_era,
    }

    for field_name, value in required_text_fields.items():
        if not value.strip():
            errors.append(f"{field_name} cannot be empty.")

    # A scene cannot be written without at least one character.
    if not script_input.characters:
        errors.append("At least one character must be supplied.")

    # Character names must be meaningful and unique.
    seen_character_names: set[str] = set()

    for index, character in enumerate(script_input.characters):
        normalized_name = character.character_name.strip().casefold()

        if not normalized_name:
            errors.append(
                f"characters[{index}].character_name cannot be empty."
            )
            continue

        if normalized_name in seen_character_names:
            errors.append(
                f"Duplicate character supplied: "
                f"{character.character_name!r}."
            )

        seen_character_names.add(normalized_name)

        # Guidance lists may be empty, but their entries may not be.
        for field_name in (
                "grounded_character_guidance",
                "grounded_voice_guidance",
        ):
            values = getattr(character, field_name)

            for item_index, item in enumerate(values):
                if not item.strip():
                    errors.append(
                        f"characters[{index}].{field_name}"
                        f"[{item_index}] cannot be empty."
                    )

    # These lists may be empty, but blank entries are invalid.
    list_fields = {
        "verified_canon_facts": script_input.verified_canon_facts,
        "era_style_constraints": script_input.era_style_constraints,
        "scene_requirements": script_input.scene_requirements,
        "prohibited_assumptions": script_input.prohibited_assumptions,
    }

    for field_name, values in list_fields.items():
        for index, item in enumerate(values):
            if not item.strip():
                errors.append(
                    f"{field_name}[{index}] cannot be empty."
                )

    # An exact statement cannot simultaneously be verified and prohibited.
    verified_facts = {
        fact.strip().casefold()
        for fact in script_input.verified_canon_facts
    }

    prohibited_assumptions = {
        assumption.strip().casefold()
        for assumption in script_input.prohibited_assumptions
    }

    contradictions = verified_facts & prohibited_assumptions

    if contradictions:
        errors.append(
            "The same statement cannot appear in both "
            "verified_canon_facts and prohibited_assumptions."
        )

    if errors:
        raise ValueError(
            "Invalid ScriptwriterInput:\n- "
            + "\n- ".join(errors)
        )


structured_scriptwriter_model = (
    scriptwriter_model.with_structured_output(
        ScriptScene,
        method="json_mode",
    )
)

structured_script_repair_model = (
    script_repair_model.with_structured_output(
        ScriptScene,
        method="json_mode",
    )
)

scriptwriter_repair_chain = (
        scriptwriter_repair_prompt
        | structured_script_repair_model
)

scriptwriter_chain = (
        scriptwriter_prompt
        | structured_scriptwriter_model
)


def validate_script_scene_output(
        script_input: ScriptwriterInput,
        scene: ScriptScene,
) -> None:
    errors: list[str] = []

    # Required generated text must contain meaningful content.
    required_text_fields = {
        "scene_heading": scene.scene_heading,
        "setting_description": scene.setting_description,
        "script_text": scene.script_text,
    }

    for field_name, value in required_text_fields.items():
        if not value.strip():
            errors.append(f"{field_name} cannot be empty.")

    # Every reported character must have been supplied upstream.
    allowed_characters = {
        character.character_name.strip().casefold()
        for character in script_input.characters
    }

    for index, character_name in enumerate(scene.characters_present):
        normalized_name = character_name.strip().casefold()

        if not normalized_name:
            errors.append(
                f"characters_present[{index}] cannot be empty."
            )
            continue

        if normalized_name not in allowed_characters:
            errors.append(
                f"characters_present[{index}] contains an "
                f"unsupplied character: {character_name!r}."
            )

    # canon_facts_used may only reproduce facts Python actually supplied.
    allowed_canon_facts = {
        fact.strip().casefold()
        for fact in script_input.verified_canon_facts
    }

    for index, fact in enumerate(scene.canon_facts_used):
        normalized_fact = fact.strip().casefold()

        if not normalized_fact:
            errors.append(
                f"canon_facts_used[{index}] cannot be empty."
            )
            continue

        if normalized_fact not in allowed_canon_facts:
            errors.append(
                f"canon_facts_used[{index}] contains a canon fact "
                "that was not supplied in verified_canon_facts."
            )

    # Accounting entries themselves may not be blank.
    for index, element in enumerate(scene.invented_story_elements):
        if not element.strip():
            errors.append(
                f"invented_story_elements[{index}] cannot be empty."
            )

    if errors:
        raise ValueError(
            "Invalid ScriptScene output:\n- "
            + "\n- ".join(errors)
        )


def write_scene(
    script_input: ScriptwriterInput,
) -> ScriptScene:
    """Preserve available diagnostic artifacts on the original exception."""
    diagnostics = {
        "schema_version": 1,
        "script_input": script_input.model_dump(mode="json"),
    }
    try:
        return _write_scene_with_diagnostics(script_input, diagnostics)
    except Exception as error:
        # Keep the original exception type and traceback. The production
        # adapter owns persistence; failed scenes never become valid results.
        error.scriptwriter_diagnostics = diagnostics
        raise


def _write_scene_with_diagnostics(
    script_input: ScriptwriterInput,
    diagnostics: dict,
) -> ScriptScene:
    diagnostics["stage"] = "input_validation"
    validate_scriptwriter_input(
        script_input
    )

    serialized_input = script_input.model_dump_json(
        indent=2,
    )

    diagnostics["stage"] = "generation"
    result = safe_invoke(
        scriptwriter_chain,
        {
            "scriptwriter_input": serialized_input,
        },
        label="Scriptwriter scene generation",
    )

    if not isinstance(
        result,
        ScriptScene,
    ):
        raise TypeError(
            "Scriptwriter scene generation did not return "
            "a ScriptScene."
        )

    diagnostics["generated_scene"] = result.model_dump(mode="json")
    diagnostics["stage"] = "generated_output_validation"
    validate_script_scene_output(
        script_input,
        result,
    )

    diagnostics["stage"] = "initial_audit"
    # First semantic grounding pass.
    initial_audit_batch = audit_script_scene(
        script_input,
        result,
    )

    diagnostics["initial_audit_batch"] = initial_audit_batch.model_dump(mode="json")
    initial_scene_audit = build_script_scene_audit(
        script_input,
        result,
        initial_audit_batch,
    )

    diagnostics["initial_scene_audit"] = initial_scene_audit.model_dump(mode="json")
    # Clean generation: return immediately.
    if initial_scene_audit.grounded:
        return result

    diagnostics["stage"] = "repair"
    # One repair pass only.
    repaired_result = repair_script_scene(
        script_input,
        result,
        initial_scene_audit,
    )

    diagnostics["repaired_scene"] = repaired_result.model_dump(mode="json")
    diagnostics["stage"] = "repaired_output_validation"
    # Repair output must independently satisfy the
    # deterministic output validator.
    validate_script_scene_output(
        script_input,
        repaired_result,
    )

    diagnostics["stage"] = "repaired_audit"
    # Re-audit the repaired scene from scratch.
    repaired_audit_batch = audit_script_scene(
        script_input,
        repaired_result,
    )

    diagnostics["repaired_audit_batch"] = repaired_audit_batch.model_dump(mode="json")
    repaired_scene_audit = build_script_scene_audit(
        script_input,
        repaired_result,
        repaired_audit_batch,
    )

    diagnostics["repaired_scene_audit"] = repaired_scene_audit.model_dump(mode="json")
    if repaired_scene_audit.grounded:
        return repaired_result

    diagnostics["stage"] = "grounding_gate"
    issue_summary = "\n".join(
        (
            f"- {issue.segment_id} | "
            f"{issue.verdict}: {issue.reason}"
        )
        for issue in repaired_scene_audit.issues
    )

    raise ValueError(
        "Scriptwriter scene remained ungrounded after "
        "one repair pass.\n"
        + issue_summary
    )


def repair_script_scene(
        script_input: ScriptwriterInput,
        scene: ScriptScene,
        scene_audit: ScriptSceneAudit,
) -> ScriptScene:
    validate_scriptwriter_input(
        script_input
    )

    validate_script_scene_output(
        script_input,
        scene,
    )

    if scene_audit.grounded:
        return scene

    if not scene_audit.issues:
        raise ValueError(
            "Cannot repair an ungrounded ScriptScene "
            "without audit issues."
        )

    serialized_input = script_input.model_dump_json(
        indent=2,
    )

    serialized_scene = scene.model_dump_json(
        indent=2,
    )

    serialized_issues = json.dumps(
        [
            issue.model_dump()
            for issue in scene_audit.issues
        ],
        indent=2,
        ensure_ascii=False,
    )

    result = safe_invoke(
        scriptwriter_repair_chain,
        {
            "scriptwriter_input": serialized_input,
            "original_scene": serialized_scene,
            "audit_issues": serialized_issues,
        },
        label="Scriptwriter scene repair",
    )

    if not isinstance(
            result,
            ScriptScene,
    ):
        raise TypeError(
            "Scriptwriter scene repair did not return "
            "a ScriptScene."
        )

    validate_script_scene_output(
        script_input,
        result,
    )

    return result
