from tools.lore_evidence import authoritative_lore_text
from agents.story_architect import (
    create_story_plan,
    revise_story_plan,
)

from agents.lore_agent import run_lore_query


def research_story_idea(
    idea: str,
    debug: bool = False,
):
    """
    Create a story plan, research each canon question,
    and revise the plan using grounded lore research.
    """

    # -----------------------------------------------------
    # Stage 1: Build the initial story plan
    # -----------------------------------------------------

    story_plan = create_story_plan(
        idea=idea
    )

    if debug:
        print("\n--- STORY PLAN ---")
        print("Title:", story_plan.working_title)
        print("Premise:", story_plan.premise)
        print(
            "Lore Questions:",
            story_plan.lore_questions,
        )
        print("--- END STORY PLAN ---\n")


    # -----------------------------------------------------
    # Stage 2: Send each canon question to the Lore Agent
    # -----------------------------------------------------

    lore_answers = []

    for question in story_plan.lore_questions:

        if debug:
            print("\n--- RESEARCHING ---")
            print(question)
            print("---\n")

        answer = run_lore_query(
            query=question,
            debug=debug,
        )

        lore_answers.append(
            {
                "question": question,
                "answer": answer,
            }
        )


    # -----------------------------------------------------
    # Stage 3: Format the Lore Agent research
    # -----------------------------------------------------

    research_notes = []

    for item in lore_answers:
        answer = item["answer"]

        sources = "\n".join(
            f"- {source}"
            for source in answer.sources
        )

        authority = authoritative_lore_text(answer) if answer.evidence_supported else ""
        research_notes.append(
            f"""
Research Question:
{item["question"]}

Lore Answer:
{authority or answer.summary}

Evidence Supported:
{answer.evidence_supported}

Premise Supported:
{answer.premise_supported}

Sources:
{sources}
            """.strip()
        )

    combined_lore_research = "\n\n---\n\n".join(
        research_notes
    )

    if debug:
        print("\n--- COMBINED LORE RESEARCH ---")
        print(combined_lore_research)
        print("--- END LORE RESEARCH ---\n")


    # -----------------------------------------------------
    # Stage 4: Revise the story using grounded lore
    # -----------------------------------------------------

    revised_story_plan = revise_story_plan(
        original_plan=story_plan,
        lore_research=combined_lore_research,
    )

    if debug:
        print("\n--- REVISED STORY PLAN ---")
        print(
            "Title:",
            revised_story_plan.working_title,
        )
        print(
            "Premise:",
            revised_story_plan.premise,
        )
        print(
            "Setting:",
            revised_story_plan.setting,
        )
        print(
            "Timeframe:",
            revised_story_plan.timeframe,
        )
        print(
            "Main Characters:",
            revised_story_plan.main_characters,
        )
        print(
            "Central Conflict:",
            revised_story_plan.central_conflict,
        )
        print(
            "Stakes:",
            revised_story_plan.stakes,
        )
        print(
            "Remaining Lore Questions:",
            revised_story_plan.lore_questions,
        )
        print("--- END REVISED STORY PLAN ---\n")


    return (
        story_plan,
        lore_answers,
        revised_story_plan,
    )

if __name__ == "__main__":
    test_idea = (
        "A DS9 story shortly before the Dominion War where Garak "
        "is secretly contacted by a Cardassian defector who claims "
        "the Founders have infiltrated the Cardassian government."
    )

    (
        original_plan,
        lore_answers,
        revised_plan,
    ) = research_story_idea(
        idea=test_idea,
        debug=True,
    )

    print("\n================================")
    print("FINAL CANON-INFORMED STORY PLAN")
    print("================================\n")

    print("Working Title:", revised_plan.working_title)
    print("Premise:", revised_plan.premise)
    print("Setting:", revised_plan.setting)
    print("Timeframe:", revised_plan.timeframe)
    print(
        "Main Characters:",
        revised_plan.main_characters,
    )
    print(
        "Central Conflict:",
        revised_plan.central_conflict,
    )
    print("Stakes:", revised_plan.stakes)

    print("\nRemaining Lore Questions:")

    if revised_plan.lore_questions:
        for question in revised_plan.lore_questions:
            print("-", question)
    else:
        print("None")