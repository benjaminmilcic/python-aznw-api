"""Turns the knowledge base files into pages (for links) and chunks (for the search).

- Markdown files: YAML frontmatter (id, titles, route, source, ...) plus sections. Every "## " section becomes one
  chunk, prefixed with the page title, its link and source code links, so it makes sense on its own.
- portfolio.json + en.json: one chunk per visible portfolio project, linked so the website opens its dialog.
- The CV as PDF: split into chunks of about CV_CHUNK_SIZE characters.
"""

import hashlib
import io
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass, field

import yaml
from pypdf import PdfReader

CV_CHUNK_SIZE = 1200
# Language code in the CV's file name: benjamin_milcic_cv_en.pdf or benjamin_milcic_cv_en_2027.pdf
CV_LANGUAGE = re.compile(r"_en(?=[._])")
CATEGORY_LABELS = {
    "gimmick": "gimmick",
    "game": "game in the Games gimmick",
    "page": "page",
    "person": "about Benjamin",
    "project": "portfolio project",
}
FRONTMATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)


@dataclass
class Page:
    id: str
    titles: dict[str, str]
    route: str
    category: str
    summary: str
    subroutes: list[str] = field(default_factory=list)
    source: list[str] = field(default_factory=list)


@dataclass
class Chunk:
    text: str
    title: str
    route: str

    @property
    def key(self) -> str:
        return hashlib.sha1(self.text.encode()).hexdigest()


def as_list(value) -> list:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def parse_markdown(text: str) -> tuple[Page | None, list[Chunk]]:
    match = FRONTMATTER.match(text)
    if not match:
        return None, []
    meta = yaml.safe_load(match.group(1)) or {}
    body = text[match.end() :]
    page = Page(
        id=str(meta["id"]),
        titles={lang: meta.get(f"title_{lang}", "") for lang in ("en", "de", "hr")},
        route=str(meta["route"]),
        category=str(meta.get("category", "page")),
        summary=str(meta.get("summary", "")),
        subroutes=[str(s["path"]) for s in as_list(meta.get("subroutes")) if isinstance(s, dict) and s.get("path")],
        source=[str(s) for s in as_list(meta.get("source"))],
    )
    header = [f"[{page.titles['en']} – {CATEGORY_LABELS.get(page.category, page.category)}]", f"Link: {page.route}"]
    if page.titles["de"] and page.titles["de"] != page.titles["en"]:
        header.append(f"Name on the German website: {page.titles['de']}, Croatian: {page.titles['hr']}")
    if page.source:
        header.append("Source code: " + ", ".join(page.source))
    if meta.get("aliases"):
        header.append("Also known as: " + ", ".join(str(a) for a in meta["aliases"]))
    prefix = "\n".join(header)

    sections = re.split(r"^## ", body, flags=re.MULTILINE)[1:]
    chunks = [Chunk(f"{prefix}\n\n## {section.strip()}", page.titles["en"], page.route) for section in sections]
    if not chunks and page.summary:
        chunks = [Chunk(f"{prefix}\n\n{page.summary}", page.titles["en"], page.route)]
    return page, chunks


def translate(texts: dict, key: str) -> str:
    value = texts
    for part in key.split("."):
        value = value.get(part, {}) if isinstance(value, dict) else {}
    return value if isinstance(value, str) else ""


def portfolio_chunks(projects: list[dict], texts: dict) -> list[Chunk]:
    """One chunk per visible project, plus an overview of all projects. Texts are the English translations;
    hidden projects are left out.

    The overview is found by general questions ("which projects…?", "an example project") that match no single
    project – without it the model only sees project names (e.g. in the CV) and invents what they are about.
    """
    chunks, overview = [], []
    for project in projects:
        if project.get("hidden"):
            continue
        title = project.get("title") or translate(texts, project.get("titleKey", ""))
        link = project_link(project)
        overview_line = f"- {title} ({link}): {translate(texts, project['descKey'])} Technologies: {', '.join(project.get('stack', []))}."
        lines = [
            f"[Portfolio project: {title}]",
            f"Link: {link}",
            f"Short description: {translate(texts, project['descKey'])}",
            f"Technologies: {', '.join(project.get('stack', []))}",
        ]
        for block in project.get("dialog", []):
            if block["type"] == "website":
                lines.append(f"Website: {block['href']}")
            elif block["type"] == "text":
                parts = [translate(texts, p["key"]) if p.get("key") else p.get("text", "") for p in block["parts"]]
                description = re.sub(r"\s+([,.])", r"\1", " ".join(parts))
                lines.append(description)
                # The description names the client (e.g. Codext GmbH), so questions about employers find it too
                overview_line += f" {description}"
            elif block["type"] in ("section", "links"):
                if block["type"] == "section":
                    lines.append(translate(texts, block["titleKey"]))
                for button in block["links"]:
                    label = button.get("label") or translate(texts, button.get("labelKey", ""))
                    if "{lang}" not in button["href"]:
                        lines.append(f"{label}: {button['href']}")
            elif block["type"] == "storeBadge":
                lines.append(f"Google Play Store: {block['href']}")
        chunks.append(Chunk("\n".join(lines), f"Portfolio: {title}", link))
        overview.append(overview_line)
    if overview:
        text = "[Portfolio – all projects of Benjamin]\nLink: /#portfolio\n\n## Which projects are in Benjamin's portfolio?\n"
        chunks.append(Chunk(text + "\n".join(overview), "Portfolio overview", "/#portfolio"))
    return chunks


def project_link(project: dict) -> str:
    """The card's own route (Gimmicks), otherwise the portfolio section with the project's dialog opened."""
    return project.get("route") or f"/?project={project['id']}#portfolio"


def cv_links(paths: Iterable[str]) -> dict[str, str]:
    """Links to the CV PDF per language, from the English CV of the knowledge base (the others have the same name
    with their language code), e.g. {"de": "/assets/cv/benjamin_milcic_cv_de.pdf", ...}."""
    english = next((p for p in paths if p.endswith(".pdf") and CV_LANGUAGE.search(p)), None)
    if not english:
        return {}
    link = "/" + english.removeprefix("src/")
    return {lang: CV_LANGUAGE.sub(f"_{lang}", link) for lang in ("de", "en", "hr")}


def cv_chunks(pdf: bytes) -> list[Chunk]:
    text = "\n".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(pdf)).pages)
    text = re.sub(r"[ \t]+", " ", text).strip()
    parts, current = [], ""
    for line in text.splitlines():
        if len(current) + len(line) > CV_CHUNK_SIZE and current:
            parts.append(current)
            current = ""
        current += line + "\n"
    if current.strip():
        parts.append(current)
    prefix = "[Benjamin's CV (PDF) – link the PDF as the CV rule says, it is also in the About me section]\n\n"
    return [Chunk(prefix + part.strip(), "CV", "/#about") for part in parts]


def load_documents(files: dict[str, bytes]) -> tuple[list[Page], list[Chunk]]:
    """All pages and chunks from the raw files (relative path -> content)."""
    pages, chunks = [], []
    for path, content in sorted(files.items()):
        if path.endswith(".md"):
            page, page_chunks = parse_markdown(content.decode("utf-8"))
            if page:
                pages.append(page)
                chunks += page_chunks
        elif path.endswith(".pdf"):
            chunks += cv_chunks(content)
    portfolio = next((c for p, c in files.items() if p.endswith("portfolio.json")), None)
    texts = next((c for p, c in files.items() if p.endswith("i18n/en.json")), None)
    if portfolio and texts:
        chunks += portfolio_chunks(json.loads(portfolio), json.loads(texts))
    return pages, chunks
