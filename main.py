# from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

from agents.lore_agent import run_lore_query


def main():

    query = input(
        "Let's talk about Star Trek! What would you like to know? "
    ).strip()

    if not query:
        print("Please enter a Star Trek question.")
        return

    print("\nConsulting Memory Alpha...")

    try:
        response = run_lore_query(
            query=query,
            debug=True,
        )

    except Exception as error:
        print(f"\nLore research failed: {error}")
        return

    print()
    print("Topic:", response.topic)
    print("Summary:", response.summary)
    print("Sources:", response.sources)
    print("Tools Used:", response.tools_used)


if __name__ == "__main__":
    main()