from pydantic import BaseModel, Field
from tools.model_provider import create_chat_model
from tools.rate_limit import safe_invoke
from dotenv import load_dotenv
from config import STORY_ARCHITECT_MODEL

load_dotenv()

# ---------
# Output Set up
# --------

class StoryPlan(BaseModel):
    working_title: str = Field(
        description=(
            "A short working title for the proposed Star Trek Story"
    )
 )

    premise: str = Field(
        description=(
            "A concise description of the core story idea"
        )
    )

    setting: str = Field(
        description=(
            "The primary location or locations where the story is occurring"
        )
    )

    timeframe: str = Field(
        description=(
            "When the story takes place in relation to established Star Trek events and eras"
        )
    )

    main_characters: list[str] = Field(
        description=(
            "The primary cast of characters in the story"
        )
    )

    central_conflict: str = Field(
        description=(
            "The main dramatic problem to drive the story forward"
        )
    )

    stakes: str = Field(
        description=(
            "What may be lost, gained, changed, and/or damaged if the conflict isn't resolved"
        )
    )

    lore_questions: list[str] = Field(
        description=(
            "Specific factual Star Trek canon questions that should "
            "be researched before the story is developed further."
        )
    )


    # ----
    # Model Configuration
    # ----

model = create_chat_model(
    model_name=STORY_ARCHITECT_MODEL,
    temperature=0.5,
    timeout=30,
    max_retries=0,
)

story_planner = model.with_structured_output(
    StoryPlan
)

def repair_story_plan_lore_questions(
        story_plan: StoryPlan,
        validation_error: str,
) -> StoryPlan:
    """
    Give the Story Architect one opportunity to repair an
    invalid lore question.

    All other parts of the story plan should remain the same.
    """

    repaired_plan = safe_invoke(
        story_planner,
        [
            (
                "system",
                """
You are correcting a previously generated Star Trek story plan.

The story plan failed a Python validation check because one or more
lore questions were not suitable for another agent.

Your only job is to repair the lore_questions.

Rules:
- Preserve the working title.
- Preserve the premise.
- Preserve the setting.
- Preserve the timeframe.
- Preserve the main characters.
- Preserve the central conflict.
- Preserve the stakes.
- Change only lore_questions that violate the validation rule.
- Every lore question must be independently understandable.
- Include necessary timeframe, character, faction, or event context
  directly in the question.
- Do not use ambiguous references such as "during this timeframe,"
  "during this time period," "at this point," "in this situation,"
  "at that time," "during that time," or "then."
- Do not answer the lore questions.
                """,
            ),
            (
                "human",
                f"""
    The previous story plan failed validation.

    Validation error:
    {validation_error}

    Previous story plan:

    Working title:
    {story_plan.working_title}

    Premise:
    {story_plan.premise}

    Setting:
    {story_plan.setting}

    Timeframe:
    {story_plan.timeframe}

    Main characters:
    {story_plan.main_characters}

    Central conflict:
    {story_plan.central_conflict}

    Stakes:
    {story_plan.stakes}

    Lore questions:
    {story_plan.lore_questions}

    Return the corrected story plan.
                    """,
            ),
        ]
    )

    return story_plan.model_copy(update={"lore_questions": repaired_plan.lore_questions})


# -----
# Agent's Workflow - Version 0.1 8/10/2026
# ----

def create_story_plan(
        idea: str,
        ) -> StoryPlan:
    """
    Convert a rough Star Trek story idea into a structured story-development plan.

    You're to identify questions that a lore specialist should research later
    """

    response = safe_invoke(
        story_planner,
        [
            (
                "system",
                """
                You are a story architect developing Star Trek fan-fiction concepts. 
                
                Your job is to turn a rough story idea into a structured dramatic plan.
                
                Do not write scenes or dialogue.
                
                Rules:
                - Preserve the user's core idea.
                - Do not silently replace the premise with a different story.
                - When the user places a story relative to a canon event, preserve that
                  relative timeframe.
                - Do not invent an exact year or stardate unless it is provided by the 
                  user or has been verified through lore research.
                - If the exact timeframe requires canon knowledge, add a question
                  to lore_questions instead.
                - Every lore question must be independently understandable without
                  access to the rest of the story plan.
                - Include the relevant era, timeframe, faction, character, or event
                  directly in each question.
                - Do not use ambiguous references such as "during this time period,"
                 "at this point," "in this situation," or "then."
                - Identify the central dramatic conflict.
                - Identify meaningful stakes.
                - Identify the main characters needed for the story.
                - Keep the concept consistent with the Star Trek era requested
                  by the user.
                - Do not assume uncertain canon facts are true.
                - When a canon fact needs verification, turn it into a specific
                  question in lore_questions.
                - lore_questions should contain factual research questions,
                  not creative-writing questions.
                - Prefer 2 to 5 focused lore questions.
                - Do not answer the lore questions yourself.
                """,
            ),
            (
                "human",
                f"""
                Develop structured story plan for this idea:
                {idea}
                """,
            ),
        ]
    )
    try:
        validate_lore_question(
            response.lore_questions
        )

    except ValueError as error:

        repaired_response = repair_story_plan_lore_questions(
            story_plan=response,
            validation_error=str(error),
        )

        validate_lore_question(
            repaired_response.lore_questions
        )

        return repaired_response

    return response

MIN_LORE_QUESTIONS = 2
MAX_LORE_QUESTIONS = 5

#
# Story Planning Guardrails
# * This rule was created due the model ignoring a prompt rule multiple times during testing
#

AMBIGUOUS_LORE_PHRASES = [
    "during this timeframe",
    "during this time period",
    "at this point",
    "in this situation",
    "at that time",
    "during that time",
    "then",
    "during the same timeframe",
    "during the same time period",
    "in the same timeframe",
    "in the same time period",
]

def validate_lore_question(
        lore_questions: list[str],
) -> None:
    """
    Validate that Story Architect lore questions
    are self-contained and suitable for another agent.
    """

    question_count = len(lore_questions)

    if question_count < MIN_LORE_QUESTIONS:
        raise ValueError(
            "Story Architect generated too few lore questions. "
            f"Expected at least {MIN_LORE_QUESTIONS}, "
            f"but received {question_count}."
        )

    if question_count > MAX_LORE_QUESTIONS:
        raise ValueError(
            "Story Architect generated too many lore questions. "
            f"Expected no more than {MAX_LORE_QUESTIONS}, "
            f"but received {question_count}."
        )
    for question in lore_questions:
        cleaned_question = question.strip()

        if not cleaned_question:
            raise ValueError(
                "Lore question must be written as a question: "
                f"{cleaned_question}"
            )

        lowered_question = cleaned_question.casefold()

        for phrase in AMBIGUOUS_LORE_PHRASES:
            if phrase in lowered_question:
                raise ValueError(
                    "Lore question contains an ambiguous reference"
                    f"('{phrase}'):\n{cleaned_question}"
                )


def revise_story_plan(
        original_plan: StoryPlan,
        lore_research: str,
) -> StoryPlan:
    """
    Revise an existing story plan using grounded canon research.

    The research should correct or refine factual assumptions
    without unnecessarily replacing the user's creative premise.
    """
    response = safe_invoke(
        story_planner,
        [
            (
                "system",
                """
You are a story architect revising a Star Trek fan-fiction
story plan after receiving canon research from a lore specialist.

Your job is to improve the existing story plan while preserving
the user's creative intent.

Rules:
- Treat the supplied lore research as the factual authority.
- Do not contradict the supplied lore research.
- Correct unsupported or inaccurate canon assumptions.
- Preserve the original premise whenever it can reasonably fit canon.
- If canon creates a problem, adapt the premise rather than silently
  abandoning it.
- Keep character behavior and political circumstances appropriate
  to the researched timeframe.
- Do not write scenes or dialogue.
- Do not invent new canon facts that are absent from the research.
- Clearly incorporate relevant facts from the research into the plan.
- Keep lore_questions only for important factual issues that remain
  unresolved after reviewing the supplied research.
- Do not repeat questions that the lore research already answered.
- Preserve uncertainty when the canon research does not establish a claim.
- Do not convert an allegation, suspicion, theory, or possibility into
  an established fact.
- If the story depends on an unverified claim, the revised conflict
  should preserve that uncertainty.
- Do not use ambiguous references such as "during this timeframe,"
"during the same timeframe," "during this time period,"
"during the same time period," "at this point,"
"in this situation," "at that time," "during that time,"
or "then."
                """,
            ),
            (
                "human",
                f"""
    Original story plan:

    Working title:
    {original_plan.working_title}

    Premise:
    {original_plan.premise}

    Setting:
    {original_plan.setting}

    Timeframe:
    {original_plan.timeframe}

    Main characters:
    {original_plan.main_characters}

    Central conflict:
    {original_plan.central_conflict}

    Stakes:
    {original_plan.stakes}

    Original lore questions:
    {original_plan.lore_questions}


    Canon research supplied by the Lore Agent:

    {lore_research}


    Revise the story plan using this research.
                    """,
            ),
        ]
    )

    return response
