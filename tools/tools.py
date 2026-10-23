import requests
import re
import html
from ddgs import DDGS
from langchain_core.tools import tool
from bs4 import BeautifulSoup

## ----
# Tool for basic websearching using DuckDuckGo's free search API.
## -----

@tool
def search_tool(query: str) -> str:
    """
    Search the web for current or external information.

    Use this tool when information may not be available in the model's
    existing knowledge or when current information is required.
    """

    try:
        results = DDGS().text(
            query=query,
            max_results=5,
        )

    except Exception as error:
        return f"Web search failed: {error}"

    if not results:
        return f"No web-search results were found for: {query}"

    formatted_results = []

    for result in results:
        title = result.get("title", "Untitled result")
        url = result.get("href", "")
        summary = result.get("body", "No summary available.")

        formatted_results.append(
            f"Title: {title}\n"
            f"Source URL: {url}\n"
            f"Summary: {summary}"
        )

    return "\n\n---\n\n".join(formatted_results)



##----------------
# Memory Alpha lore-search tooling - The goal is to provide the agents with grounded
# information about Star Trek lore. This is a simple tool that uses the Memory Alpha API to
# retrieve information about Star Trek topics for fan scripts.
## ----------------


MEMORY_ALPHA_API_URL = "https://memory-alpha.fandom.com/api.php"

MEMORY_ALPHA_HEADERS = {
    "User-Agent": (
        "StarTrekLoreAgent/1.0 "
        "(personal educational portfolio project; "
        "not affiliated with Memory Alpha or Fandom)"
    )
}


def clean_search_snippet(snippet:str)->str:
    """Remove basic HTML formatting from MediaWiki search snippets"""

    without_tags =re.sub(r"<[^>]+>", "", snippet)
    return html.unescape(without_tags).strip()

@tool
def memory_alpha_sections_tool(title: str) -> str:
    """
    Return the section titles for a specific Memory Alpha article.

    Use this after selecting the correct Memory Alpha article.
    The results help determine which sections are relevant to the
    user's question.
    """

    params = {
        "action": "parse",
        "page": title,
        "prop": "tocdata",
        "redirects": 1,
        "format": "json",
        "formatversion": 2,
    }

    try:
        response = requests.get(
            MEMORY_ALPHA_API_URL,
            params=params,
            headers=MEMORY_ALPHA_HEADERS,
            timeout=15,
        )

        response.raise_for_status()
        data = response.json()

    except requests.Timeout:
        return "Memory Alpha section request timed out."

    except requests.RequestException as error:
        return f"Memory Alpha section request failed: {error}"

    except ValueError:
        return "Memory Alpha returned invalid JSON."

    parse_data = data.get("parse", {})
    toc_data = parse_data.get("tocdata", {})
    sections = toc_data.get("sections", [])

    if not sections:
        return f"No readable sections were found for: {title}"

    formatted_sections = []

    for section in sections:
        if not isinstance(section, dict):
            continue

        section_index = section.get("index", "")
        section_number = section.get("number", "")
        section_level = section.get("tocLevel", "")
        raw_title = section.get("line", "")

        clean_title = BeautifulSoup(
            raw_title,
            "html.parser",
        ).get_text(" ", strip=True)

        if not clean_title:
            continue

        formatted_sections.append(
            f"Section index: {section_index}\n"
            f"Section number: {section_number}\n"
            f"Section level: {section_level}\n"
            f"Section title: {clean_title}"
        )

    if not formatted_sections:
        return f"No usable sections were found for: {title}"

    return "\n\n---\n\n".join(formatted_sections)

@tool
def memory_alpha_article_tool(title:str)->str:
    """
    Retrieve the exact Memory Alpha article selected by title

    Use this only after memory_alpha_search_tool identifies possible
    article titles. Pass one exact candidate title as the input.
    """

    params={
        "action": "query",
        "prop": "extracts|info",
        "titles": title,
        "explaintext": True,
        "exchars": 1200,
        "inprop":"url",
        "redirects": 1,
        "format":"json",
        "formatversion":2,
    }

    try:
        response=requests.get(
            MEMORY_ALPHA_API_URL,
            params=params,
            headers=MEMORY_ALPHA_HEADERS,
            timeout=15,
        )

        response.raise_for_status()
        data=response.json()

    except requests.Timeout:
        return "Memory Alpha article request timed out."
    except requests.RequestException as error:
        return f"Memory Alpha article request failed: {error}"
    except ValueError:
        return f"Memory Alpha returned invalid JSON"

    pages = data.get("query", {}).get("pages", [])
    print("Memory Alpha article pages type:", type(pages))
    print("Memory Alpha article raw pages:", pages)



    # Normalize the different response shapes Fandom may return.
    if isinstance(pages, dict):
        pages = list(pages.values())

    if not isinstance(pages, list) or not pages:
        return f"Memory Alpha article not found: {title}"

    page = pages[0]

    # Some responses may contain another list layer.
    if isinstance(page, list):
        if not page:
            return f"Memory Alpha article not found: {title}"

        page = page[0]

    if not isinstance(page, dict):
        return (
            "Memory Alpha returned the article in an unexpected format: "
            f"{type(page).__name__}"
        )

    if "missing" in page:
        return f"Memory Alpha article not found: {title}"

    article_title = page.get("title", title)
    source_url = page.get("fullurl", "")
    extract = page.get("extract", "").strip()

    if not extract:
        return(
            f"Article: {article_title}\n"
            f"Source URL: {source_url}\n"
            "Extract: No readable article text was returned"
        )
    return (
        f"Article: {article_title}\n"
        f"Source URL: {source_url}\n"
        f"Extract: {extract}"
    )

def get_memory_alpha_section_title(
    title: str,
    section_index: str,
) -> str:
    """
    Look up the visible title for a Memory Alpha section index.
    """

    params = {
        "action": "parse",
        "page": title,
        "prop": "tocdata",
        "redirects": 1,
        "format": "json",
        "formatversion": 2,
    }

    try:
        response = requests.get(
            MEMORY_ALPHA_API_URL,
            params=params,
            headers=MEMORY_ALPHA_HEADERS,
            timeout=15,
        )

        response.raise_for_status()
        data = response.json()

    except (
        requests.Timeout,
        requests.RequestException,
        ValueError,
    ):
        return ""

    parse_data = data.get("parse", {})
    toc_data = parse_data.get("tocdata", {})
    sections = toc_data.get("sections", [])

    for section in sections:
        if not isinstance(section, dict):
            continue

        current_index = str(
            section.get("index", "")
        )

        if current_index != str(section_index):
            continue

        raw_title = section.get(
            "line",
            "",
        )

        return BeautifulSoup(
            raw_title,
            "html.parser",
        ).get_text(
            " ",
            strip=True,
        )

    return ""

def get_next_memory_alpha_section_title(
    title: str,
    section_index: str,
) -> str:
    """
    Return the title of the section immediately following
    the requested Memory Alpha section.

    This lets us stop retrieval before child or later
    sections are included.
    """

    params = {
        "action": "parse",
        "page": title,
        "prop": "tocdata",
        "redirects": 1,
        "format": "json",
        "formatversion": 2,
    }

    try:
        response = requests.get(
            MEMORY_ALPHA_API_URL,
            params=params,
            headers=MEMORY_ALPHA_HEADERS,
            timeout=15,
        )

        response.raise_for_status()
        data = response.json()

    except (
        requests.Timeout,
        requests.RequestException,
        ValueError,
    ):
        return ""

    parse_data = data.get("parse", {})
    toc_data = parse_data.get("tocdata", {})
    sections = toc_data.get("sections", [])

    for position, section in enumerate(sections):
        if not isinstance(section, dict):
            continue

        current_index = str(
            section.get("index", "")
        )

        if current_index != str(section_index):
            continue

        next_position = position + 1

        if next_position >= len(sections):
            return ""

        next_section = sections[next_position]

        if not isinstance(next_section, dict):
            return ""

        raw_title = next_section.get(
            "line",
            "",
        )

        return BeautifulSoup(
            raw_title,
            "html.parser",
        ).get_text(
            " ",
            strip=True,
        )

    return ""

def split_memory_alpha_lines_preserving_paragraphs(
    section_text: str,
) -> list[str]:
    lines = []

    for raw_line in section_text.splitlines():
        line = raw_line.strip()

        if line:
            lines.append(line)
        elif lines and lines[-1]:
            lines.append("")

    while lines and not lines[0]:
        lines.pop(0)

    while lines and not lines[-1]:
        lines.pop()

    return lines

def trim_memory_alpha_to_exact_section(
    section_text: str,
    section_title: str,
    next_section_title: str,
) -> str:
    """
    Keep only the requested Memory Alpha section.

    Memory Alpha may return child subsections together
    with a parent section. Stop when the next section
    heading begins.
    """

    lines = split_memory_alpha_lines_preserving_paragraphs(
        section_text
    )

    if not lines:
        return ""

    start_index = 0

    if section_title:
        for index, line in enumerate(lines):
            if (
                line.casefold()
                == section_title.casefold()
            ):
                start_index = index
                break

    end_index = len(lines)

    if next_section_title:
        for index in range(
            start_index + 1,
            len(lines),
        ):
            if (
                lines[index].casefold()
                == next_section_title.casefold()
            ):
                end_index = index
                break

    return "\n".join(
        lines[start_index:end_index]
    )

def clean_memory_alpha_section_text(
    section_text: str,
    section_title: str,
) -> str:
    """
    Remove Fandom-injected Quick Answers content from
    Memory Alpha section text.
    """

    lines = split_memory_alpha_lines_preserving_paragraphs(
        section_text
    )

    quick_answers_index = None

    for index, line in enumerate(lines):
        if line.casefold() == "quick answers":
            quick_answers_index = index
            break

    if quick_answers_index is not None:

        title_before_quick_answers = False
        title_after_quick_answers = None

        if section_title:
            for index, line in enumerate(lines):

                if (
                    line.casefold()
                    != section_title.casefold()
                ):
                    continue

                if index < quick_answers_index:
                    title_before_quick_answers = True

                elif index > quick_answers_index:
                    title_after_quick_answers = index
                    break

        # Example:
        #
        # Quick Answers
        # ...
        # Provided by: Fandom
        # Exile
        # [real article text]
        #
        # Keep everything beginning with the real section.
        if title_after_quick_answers is not None:
            lines = (
                lines[:quick_answers_index]
                + lines[title_after_quick_answers:]
            )

        # Some subsection responses contain the real article
        # first and append Quick Answers at the end.
        #
        # In that case, simply discard everything beginning
        # with Quick Answers.
        elif title_before_quick_answers:
            lines = lines[:quick_answers_index]

        else:
            # Fallback:
            # Find the final "Provided by:" marker.
            provider_indexes = [
                index
                for index, line in enumerate(lines)
                if (
                    index > quick_answers_index
                    and line.casefold().startswith(
                        "provided by:"
                    )
                )
            ]

            if provider_indexes:
                final_provider_index = (
                    provider_indexes[-1]
                )

                remaining_lines = lines[
                    final_provider_index + 1:
                ]

                if remaining_lines:
                    lines = (
                        lines[:quick_answers_index]
                        + remaining_lines
                    )
                else:
                    lines = lines[
                        :quick_answers_index
                    ]

    cleaned_lines = []

    for line in lines:
        if not line:
            if cleaned_lines and cleaned_lines[-1]:
                cleaned_lines.append("")
            continue

        if line.casefold().startswith("provided by:"):
            continue

        if line in {"[", "]"}:
            continue

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()

@tool
def memory_alpha_search_tool(query: str) -> str:
    """
    Search Memory Alpha and return several possible article titles.

    Use this tool first when researching Star Trek lore.
    Inspect the candidate titles before choosing an article.
    """

    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srnamespace": 0,
        "srlimit": 5,
        "srprop": "snippet",
        "format": "json",
        "formatversion": 2,
    }

    try:
        response = requests.get(
            MEMORY_ALPHA_API_URL,
            params=params,
            headers=MEMORY_ALPHA_HEADERS,
            timeout=15,
        )

        response.raise_for_status()
        data = response.json()

    except requests.Timeout:
        return "Memory Alpha search timed out."

    except requests.RequestException as error:
        return f"Memory Alpha search failed: {error}"

    except ValueError:
        return "Memory Alpha returned invalid JSON."

    raw_search_results = (
        data.get("query", {}).get("search", [])
    )

    if isinstance(raw_search_results, list):
        search_results = raw_search_results

    elif isinstance(raw_search_results, dict):
        search_results = list(raw_search_results.values())

    else:
        return (
            "Memory Alpha returned search results in an "
            f"unexpected format: {type(raw_search_results).__name__}"
        )

    if not search_results:
        return (
            f"No Memory Alpha articles were found for: {query}"
        )

    formatted_results = []

    for result in search_results:
        if not isinstance(result, dict):
            continue

        title = result.get(
            "title",
            "Unknown article",
        )

        snippet = clean_search_snippet(
            result.get(
                "snippet",
                "No search preview available.",
            )
        )

        formatted_results.append(
            f"Candidate title: {title}\n"
            f"Search preview: {snippet}"
        )

    if not formatted_results:
        return (
            "Memory Alpha returned results, but none contained "
            "usable article information."
        )

    return "\n\n---\n\n".join(formatted_results)

@tool
def memory_alpha_section_tool(
    title: str,
    section_index: str,
) -> str:
    """
    Retrieve the text of one specific section from a Memory Alpha article.

    Use this after memory_alpha_sections_tool identifies a relevant
    section. Pass the exact article title and exact section index.
    """

    params = {
        "action": "parse",
        "page": title,
        "prop": "text",
        "section": section_index,
        "redirects": 1,
        "format": "json",
        "formatversion": 2,
    }

    try:
        response = requests.get(
            MEMORY_ALPHA_API_URL,
            params=params,
            headers=MEMORY_ALPHA_HEADERS,
            timeout=15,
        )

        response.raise_for_status()
        data = response.json()

    except requests.Timeout:
        return "Memory Alpha section retrieval timed out."

    except requests.RequestException as error:
        return f"Memory Alpha section retrieval failed: {error}"

    except ValueError:
        return "Memory Alpha returned invalid JSON."

    parse_data = data.get("parse", {})
    raw_html = parse_data.get("text", "")

    # Be defensive in case the API returns an older nested structure.
    if isinstance(raw_html, dict):
        raw_html = raw_html.get("*", "")

    if not isinstance(raw_html, str) or not raw_html.strip():
        return (
            f"No readable text was returned for section "
            f"{section_index} of {title}."
        )

    soup = BeautifulSoup(
        raw_html,
        "html.parser",
    )

    # Remove content that adds noise to the language model.
    for unwanted in soup(
        [
            "script",
            "style",
            "table",
            "figure",
            "nav",
        ]
    ):
        unwanted.decompose()

    block_tags = [
        "h1", "h2", "h3", "h4", "h5", "h6",
        "p", "li", "dt", "dd", "blockquote",
    ]
    content_blocks = []

    for element in soup.find_all(block_tags):
        if element.find_parent(block_tags):
            continue

        text = element.get_text(" ", strip=True)
        if text:
            content_blocks.append(text)

    clean_text = "\n\n".join(content_blocks)

    # Look up the actual title of the requested section.
    section_title = get_memory_alpha_section_title(
        title=title,
        section_index=section_index,
    )

    # Remove Fandom-injected Quick Answers content.
    clean_text = clean_memory_alpha_section_text(
        section_text=clean_text,
        section_title=section_title,
    )

    next_section_title = (
        get_next_memory_alpha_section_title(
            title=title,
            section_index=section_index,
        )
    )

    clean_text = (
        trim_memory_alpha_to_exact_section(
            section_text=clean_text,
            section_title=section_title,
            next_section_title=next_section_title,
        )
    )

    if not clean_text:
        return (
            f"No readable text was extracted from section "
            f"{section_index} of {title}."
        )

    return (
        f"Article: {title}\n"
        f"Section index: {section_index}\n"
        f"Section text:\n{clean_text}"
    )

if __name__ == "__main__":
    garak_ds9 = memory_alpha_section_tool.invoke(
        {
            "title": "Elim Garak",
            "section_index": "4",
        }
    )

    print(garak_ds9)
