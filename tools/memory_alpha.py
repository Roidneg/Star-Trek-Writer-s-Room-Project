import html

import requests
from bs4 import BeautifulSoup
from langchain_core.tools import tool


# -----------------------------------------------------
# Memory Alpha Configuration
# -----------------------------------------------------

MEMORY_ALPHA_API_URL = (
    "https://memory-alpha.fandom.com/api.php"
)

REQUEST_TIMEOUT = 20

HEADERS = {
    "User-Agent": (
        "StarTrekLoreAgent/1.0 "
        "(personal AI research project)"
    )
}


# -----------------------------------------------------
# Internal Request Helper
# -----------------------------------------------------

def _memory_alpha_request(
    params: dict,
) -> dict:
    """
    Send a request to the Memory Alpha MediaWiki API
    and return parsed JSON.
    """

    response = requests.get(
        MEMORY_ALPHA_API_URL,
        params=params,
        headers=HEADERS,
        timeout=REQUEST_TIMEOUT,
    )

    response.raise_for_status()

    data = response.json()

    if "error" in data:
        raise RuntimeError(
            "Memory Alpha API returned an error: "
            f"{data['error']}"
        )

    return data


# -----------------------------------------------------
# Text Cleanup Helpers
# -----------------------------------------------------

def _clean_html_text(
    raw_html: str,
) -> str:
    """
    Convert small HTML fragments returned by the
    MediaWiki API into readable plain text.
    """

    if not raw_html:
        return ""

    soup = BeautifulSoup(
        raw_html,
        "html.parser",
    )

    text = soup.get_text(
        separator=" ",
        strip=True,
    )

    return html.unescape(text)


def clean_memory_alpha_section_text(
    section_text: str,
    section_title: str,
) -> str:
    """
    Remove Fandom-injected Quick Answers content from
    Memory Alpha section text.

    The actual article section normally begins again with
    the requested section title after the injected block.
    """

    lines = [
        line.strip()
        for line in section_text.splitlines()
        if line.strip()
    ]

    quick_answers_index = None

    for index, line in enumerate(lines):
        if line.casefold() == "quick answers":
            quick_answers_index = index
            break

    if quick_answers_index is None:
        return "\n".join(lines)

    actual_section_index = None

    for index in range(
        quick_answers_index + 1,
        len(lines),
    ):
        if (
            lines[index].casefold()
            == section_title.casefold()
        ):
            actual_section_index = index
            break

    if actual_section_index is not None:
        lines = (
            lines[:quick_answers_index]
            + lines[actual_section_index:]
        )

    cleaned_lines = []

    for line in lines:
        if (
            line.casefold()
            .startswith("provided by:")
        ):
            continue

        if line in {"[", "]"}:
            continue

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def _clean_section_html(
    raw_html: str,
    section_title: str,
) -> str:
    """
    Convert Memory Alpha section HTML into readable
    article text and remove non-article clutter.
    """

    soup = BeautifulSoup(
        raw_html,
        "html.parser",
    )

    # Remove elements that do not contribute useful
    # article evidence.
    for element in soup.find_all(
        ["script", "style"]
    ):
        element.decompose()

    # Remove citation markers such as [1].
    for element in soup.select(
        "sup.reference"
    ):
        element.decompose()

    # Remove MediaWiki edit controls.
    for element in soup.select(
        ".mw-editsection"
    ):
        element.decompose()

    section_text = soup.get_text(
        separator="\n",
        strip=True,
    )

    # Remove Fandom's injected Quick Answers blocks.
    section_text = (
        clean_memory_alpha_section_text(
            section_text=section_text,
            section_title=section_title,
        )
    )

    return section_text


# -----------------------------------------------------
# Memory Alpha Search Tool
# -----------------------------------------------------

@tool
def memory_alpha_search_tool(
    query: str,
) -> str:
    """
    Search Memory Alpha for articles related to a
    Star Trek canon research question.

    Returns up to five candidate article titles with
    short search previews.
    """

    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": 5,
        "format": "json",
        "formatversion": 2,
    }

    data = _memory_alpha_request(
        params
    )

    search_results = (
        data
        .get("query", {})
        .get("search", [])
    )

    if not search_results:
        return (
            "No Memory Alpha search results "
            f"were found for: {query}"
        )

    formatted_results = []

    for result in search_results:
        title = result.get(
            "title",
            "Unknown title",
        )

        preview = _clean_html_text(
            result.get(
                "snippet",
                "",
            )
        )

        formatted_results.append(
            "\n".join(
                [
                    f"Candidate title: {title}",
                    f"Search preview: {preview}",
                ]
            )
        )

    return "\n\n---\n\n".join(
        formatted_results
    )


# -----------------------------------------------------
# Memory Alpha Article Tool
# -----------------------------------------------------

@tool
def memory_alpha_article_tool(
    title: str,
) -> str:
    """
    Retrieve introductory text and the canonical URL
    for a Memory Alpha article.

    The introduction provides basic identifying context.
    Detailed canon research should rely primarily on
    retrieved article sections.
    """

    params = {
        "action": "query",
        "prop": "extracts|info",
        "titles": title,
        "explaintext": 1,
        "exchars": 1200,
        "inprop": "url",
        "redirects": 1,
        "format": "json",
        "formatversion": 2,
    }

    data = _memory_alpha_request(
        params
    )

    pages = (
        data
        .get("query", {})
        .get("pages", [])
    )

    if not pages:
        return (
            "Memory Alpha article not found: "
            f"{title}"
        )

    page = pages[0]

    if page.get("missing") is True:
        return (
            "Memory Alpha article not found: "
            f"{title}"
        )

    article_title = page.get(
        "title",
        title,
    )

    extract = page.get(
        "extract",
        "",
    )

    source_url = (
        page.get("canonicalurl")
        or page.get("fullurl")
        or ""
    )

    return "\n".join(
        [
            f"Article: {article_title}",
            f"Source URL: {source_url}",
            f"Extract: {extract}",
        ]
    )


# -----------------------------------------------------
# Section Normalization
# -----------------------------------------------------

def _normalize_section(
    section: dict,
) -> dict:
    """
    Normalize section metadata returned by different
    versions of the MediaWiki/Fandom API.
    """

    section_index = (
        section.get("index")
        or section.get("id")
        or ""
    )

    section_number = (
        section.get("number")
        or section.get("tocnumber")
        or ""
    )

    section_level = (
        section.get("level")
        or section.get("toclevel")
        or ""
    )

    section_title = (
        section.get("line")
        or section.get("title")
        or section.get("anchor")
        or ""
    )

    # Some API responses can contain HTML inside
    # section titles.
    section_title = _clean_html_text(
        str(section_title)
    )

    return {
        "index": str(section_index),
        "number": str(section_number),
        "level": str(section_level),
        "title": section_title,
    }


def _extract_sections_from_response(
    data: dict,
) -> list[dict]:
    """
    Extract section metadata from either Fandom's
    tocdata response or the standard MediaWiki
    sections response.
    """

    parse_data = data.get(
        "parse",
        {}
    )

    sections = []

    tocdata = parse_data.get(
        "tocdata"
    )

    if isinstance(tocdata, dict):
        sections = tocdata.get(
            "sections",
            []
        )

    elif isinstance(
        tocdata,
        list,
    ):
        sections = tocdata

    if not sections:
        sections = parse_data.get(
            "sections",
            [],
        )

    normalized_sections = []

    for section in sections:
        if not isinstance(
            section,
            dict,
        ):
            continue

        normalized = (
            _normalize_section(
                section
            )
        )

        if normalized["index"]:
            normalized_sections.append(
                normalized
            )

    return normalized_sections


# -----------------------------------------------------
# Memory Alpha Sections Tool
# -----------------------------------------------------

@tool
def memory_alpha_sections_tool(
    title: str,
) -> str:
    """
    List the available sections of a Memory Alpha
    article.

    The Lore Agent uses this tool to decide which
    sections are relevant before retrieving article
    text.
    """

    # Fandom supports tocdata on many pages.
    params = {
        "action": "parse",
        "page": title,
        "prop": "tocdata",
        "redirects": 1,
        "format": "json",
        "formatversion": 2,
    }

    try:
        data = _memory_alpha_request(
            params
        )

        sections = (
            _extract_sections_from_response(
                data
            )
        )

    except Exception:
        sections = []

    # Standard MediaWiki fallback if tocdata is
    # unavailable for a particular page/API response.
    if not sections:
        fallback_params = {
            "action": "parse",
            "page": title,
            "prop": "sections",
            "redirects": 1,
            "format": "json",
            "formatversion": 2,
        }

        data = _memory_alpha_request(
            fallback_params
        )

        sections = (
            _extract_sections_from_response(
                data
            )
        )

    if not sections:
        return (
            "No sections were found for "
            f"Memory Alpha article: {title}"
        )

    formatted_sections = []

    for section in sections:
        formatted_sections.append(
            "\n".join(
                [
                    (
                        "Section index: "
                        f"{section['index']}"
                    ),
                    (
                        "Section number: "
                        f"{section['number']}"
                    ),
                    (
                        "Section level: "
                        f"{section['level']}"
                    ),
                    (
                        "Section title: "
                        f"{section['title']}"
                    ),
                ]
            )
        )

    return "\n\n---\n\n".join(
        formatted_sections
    )


# -----------------------------------------------------
# Section Title Lookup
# -----------------------------------------------------

def _get_section_title(
    title: str,
    section_index: str,
) -> str:
    """
    Resolve a section index back to its visible title.

    This allows the cleanup function to identify where
    the real article section begins after Fandom's
    injected Quick Answers block.
    """

    params = {
        "action": "parse",
        "page": title,
        "prop": "sections",
        "redirects": 1,
        "format": "json",
        "formatversion": 2,
    }

    data = _memory_alpha_request(
        params
    )

    sections = (
        _extract_sections_from_response(
            data
        )
    )

    for section in sections:
        if (
            str(section["index"])
            == str(section_index)
        ):
            return section["title"]

    return ""


# -----------------------------------------------------
# Memory Alpha Section Tool
# -----------------------------------------------------

@tool
def memory_alpha_section_tool(
    title: str,
    section_index: str,
) -> str:
    """
    Retrieve the text of one specific section from a
    Memory Alpha article.

    Fandom-injected Quick Answers content is removed
    before the section is returned to the Lore Agent.
    """

    section_title = (
        _get_section_title(
            title=title,
            section_index=section_index,
        )
    )

    params = {
        "action": "parse",
        "page": title,
        "prop": "text",
        "section": section_index,
        "redirects": 1,
        "format": "json",
        "formatversion": 2,
    }

    data = _memory_alpha_request(
        params
    )

    parse_data = data.get(
        "parse",
        {}
    )

    raw_text = parse_data.get(
        "text",
        "",
    )

    # Depending on API response format, "text"
    # may be either a string or {"*": "..."}.
    if isinstance(
        raw_text,
        dict,
    ):
        raw_html = raw_text.get(
            "*",
            "",
        )
    else:
        raw_html = raw_text

    if not raw_html:
        return (
            "No section text was found for "
            f"'{title}', section "
            f"{section_index}."
        )

    section_text = _clean_section_html(
        raw_html=raw_html,
        section_title=section_title,
    )

    return "\n".join(
        [
            f"Article: {title}",
            (
                "Section index: "
                f"{section_index}"
            ),
            "Section text:",
            section_text,
        ]
    )