"""Python-owned Lore passages and the downstream authority contract.

This module makes no provider or retrieval calls. Exact reconstruction is
an integrity check, not a claim that a source or its interpretation is true.
"""

from __future__ import annotations

import hashlib
import json
import re
from urllib.parse import urlsplit

from pydantic import BaseModel, Field, model_validator


MAX_LORE_PASSAGES = 6
MAX_LORE_TEXT_CHARACTERS = 16000


class LoreEvidenceContractError(ValueError):
    """Evidence is incomplete, changed, or outside its retrieved scope."""


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _source_id(title: str, url: str, section: str | None, text: str) -> str:
    return _digest(json.dumps([title, url, section, text], ensure_ascii=False))


def _valid_source_url(url: str) -> bool:
    try:
        parsed = urlsplit(url)
    except ValueError:
        return False
    return (
        parsed.scheme == "https"
        and parsed.netloc == "memory-alpha.fandom.com"
        and parsed.path.startswith("/wiki/")
        and not any(character.isspace() for character in url)
    )


def source_paragraphs(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]


class LoreEvidenceSource(BaseModel):
    model_config = {"extra": "forbid", "frozen": True}
    source_id: str = Field(min_length=1)
    article_title: str = Field(min_length=1)
    source_url: str = Field(min_length=1)
    section_index: str | None
    text: str = Field(min_length=1)

    @model_validator(mode="after")
    def check_source(self):
        if not _valid_source_url(self.source_url):
            raise ValueError("Source URL is not an exact Memory Alpha article URL.")
        if self.source_id != _source_id(
            self.article_title, self.source_url, self.section_index, self.text
        ):
            raise ValueError("Source text or identity changed after retrieval.")
        return self


class LorePassageReference(BaseModel):
    model_config = {"extra": "forbid", "frozen": True}
    source_id: str = Field(min_length=1)
    paragraph_index: int = Field(ge=0, strict=True)


class LorePassageSelection(BaseModel):
    model_config = {"extra": "forbid"}
    passages: list[LorePassageReference] = Field(max_length=MAX_LORE_PASSAGES)


class LoreEvidencePassage(LorePassageReference):
    text: str = Field(min_length=1)


class LoreEvidencePackage(BaseModel):
    model_config = {"extra": "forbid", "frozen": True}
    version: int = Field(default=1, strict=True)
    sources: list[LoreEvidenceSource] = Field(min_length=1, max_length=MAX_LORE_PASSAGES)
    passages: list[LoreEvidencePassage] = Field(min_length=1, max_length=MAX_LORE_PASSAGES)

    @model_validator(mode="after")
    def check_package(self):
        if self.version != 1:
            raise ValueError("Unsupported Lore evidence contract version.")
        by_id = {source.source_id: source for source in self.sources}
        if len(by_id) != len(self.sources):
            raise ValueError("Duplicate source identities.")
        order = {source.source_id: index for index, source in enumerate(self.sources)}
        keys = []
        for passage in self.passages:
            source = by_id.get(passage.source_id)
            if source is None:
                raise ValueError("Passage refers to an unknown source.")
            paragraphs = source_paragraphs(source.text)
            if passage.paragraph_index >= len(paragraphs):
                raise ValueError("Passage index is outside the source.")
            if passage.text != paragraphs[passage.paragraph_index]:
                raise ValueError("Passage must copy a complete, unchanged source paragraph.")
            if passage.text.rstrip().endswith(("...", "…")):
                raise ValueError("Explicitly truncated passages cannot be authoritative.")
            keys.append((order[passage.source_id], passage.paragraph_index))
        if keys != sorted(set(keys)):
            raise ValueError("Passages must be unique and preserve source order.")
        if set(by_id) != {passage.source_id for passage in self.passages}:
            raise ValueError("Package contains unused source records.")
        if sum(len(passage.text) for passage in self.passages) > MAX_LORE_TEXT_CHARACTERS:
            raise ValueError("Selected Lore text exceeds its bounded context budget.")
        return self


def render_lore_evidence(package: LoreEvidencePackage) -> str:
    return "\n\n".join(passage.text for passage in package.passages)


def lore_evidence_urls(package: LoreEvidencePackage) -> list[str]:
    return list(dict.fromkeys(source.source_url for source in package.sources))


def authoritative_lore_text(response) -> str:
    """Keep unsupported and pre-contract summaries out of verified evidence."""
    if not response.evidence_supported:
        return ""
    if not response.premise_supported:
        raise LoreEvidenceContractError("Supported Lore has an unsupported premise.")
    package = getattr(response, "evidence_package", None)
    if package is None:
        raise LoreEvidenceContractError(
            "Supported Lore predates the source-passage contract; refresh this Lore artifact."
        )
    # Revalidation also catches nested list mutation after model construction.
    try:
        payload = package.model_dump() if isinstance(package, BaseModel) else package
        package = LoreEvidencePackage.model_validate(payload)
    except (TypeError, ValueError) as error:
        raise LoreEvidenceContractError("Lore evidence package is invalid.") from error
    text = render_lore_evidence(package)
    if response.summary != text or response.sources != lore_evidence_urls(package):
        raise LoreEvidenceContractError("Lore summary or URLs differ from the source package.")
    return text


def lore_evidence_fingerprint(response) -> str:
    authoritative_lore_text(response)
    if not response.evidence_supported:
        return ""
    package = response.evidence_package
    payload = package.model_dump() if isinstance(package, BaseModel) else package
    return _digest(json.dumps(payload, sort_keys=True, ensure_ascii=False))


class LoreRetrievalInventory:
    """Record successful tool returns without modifying tool or global state."""

    def __init__(self):
        self.sources: list[LoreEvidenceSource] = []
        self._articles: dict[str, tuple[str, str]] = {}

    def _add(self, title: str, url: str, section: str | None, text: str):
        text = text.strip()
        if not text or not _valid_source_url(url):
            return
        source = LoreEvidenceSource(
            source_id=_source_id(title, url, section, text),
            article_title=title, source_url=url, section_index=section, text=text,
        )
        if not any(item.source_id == source.source_id for item in self.sources):
            self.sources.append(source)

    def record_article(self, arguments: dict, result: str):
        match = re.fullmatch(
            r"Article: ([^\n]+)\nSource URL: ([^\n]+)\nExtract: (.+)", result, re.DOTALL
        )
        if not match:
            return
        title, url, text = match.groups()
        if not _valid_source_url(url):
            return
        self._articles[arguments["title"]] = (title, url)
        self._articles[title] = (title, url)
        if text.startswith("No readable article text"):
            return
        # Keep the returned source intact. Eligibility excludes its explicitly
        # truncated final paragraph while retaining earlier complete paragraphs.
        self._add(title, url, None, text)

    def record_section(self, arguments: dict, result: str):
        match = re.fullmatch(
            r"Article: ([^\n]+)\nSection index: ([^\n]+)\nSection text:\n(.+)",
            result, re.DOTALL,
        )
        if not match:
            return
        title, section, text = match.groups()
        article = self._articles.get(arguments["title"])
        if (article is None or title != arguments["title"]
                or section != arguments["section_index"]):
            return
        self._add(article[0], article[1], section, text)

    def eligible_passages(self, scope: str) -> list[tuple[LoreEvidenceSource, int, str]]:
        normalized_scope = " ".join(scope.split())
        return [
            (source, index, text)
            for source in self.sources
            for index, text in enumerate(source_paragraphs(source.text))
            if " ".join(text.split()) in normalized_scope
            and not text.rstrip().endswith(("...", "…"))
        ]


class RecordingLoreTool:
    def __init__(self, tool, recorder):
        self.tool, self.recorder = tool, recorder

    def invoke(self, arguments):
        result = self.tool.invoke(arguments)
        if isinstance(result, str):
            self.recorder(arguments, result)
        return result


def reconstruct_lore_package(
    candidates: list[tuple[LoreEvidenceSource, int, str]],
    selection: LorePassageSelection,
) -> LoreEvidencePackage:
    if not isinstance(selection, LorePassageSelection):
        raise LoreEvidenceContractError("Selector returned no complete parsed selection.")
    allowed = {(source.source_id, index): (source, text)
               for source, index, text in candidates}
    keys = [(item.source_id, item.paragraph_index) for item in selection.passages]
    if not keys or len(keys) != len(set(keys)) or any(key not in allowed for key in keys):
        raise LoreEvidenceContractError("Passage selection is empty, duplicated, or outside scope.")
    chosen = set(keys)
    ordered = [(source, index, text) for source, index, text in candidates
               if (source.source_id, index) in chosen]
    sources = list({source.source_id: source for source, _, _ in ordered}.values())
    try:
        return LoreEvidencePackage(
            sources=sources,
            passages=[LoreEvidencePassage(source_id=source.source_id,
                                         paragraph_index=index, text=text)
                      for source, index, text in ordered],
        )
    except ValueError as error:
        raise LoreEvidenceContractError("Selected evidence package is invalid.") from error
