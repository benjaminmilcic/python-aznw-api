"""Tests for parsing the knowledge base and fixing links – no network."""

from app.links import LinkFixer, Routes
from app.rag.documents import load_documents, parse_markdown, portfolio_chunks

PAGE = """---
id: weather
title_de: Wetter
title_en: Weather
title_hr: Vrijeme
route: /gimmicks/weather
category: gimmick
summary: Weather forecast.
aliases: [forecast, Wetter]
subroutes:
  - { path: /gimmicks/recipes/recipe/:id, title_en: Single recipe }
source:
  - https://github.com/benjaminmilcic/aznw-routes/tree/new/src/app/pages/gimmicks/weather
---

# Weather (Gimmick)

## What is the Weather gimmick?
It shows the weather.

## How is it built?
Open-Meteo.
"""


def test_markdown_becomes_page_and_one_chunk_per_section():
    page, chunks = parse_markdown(PAGE)
    assert page.id == "weather" and page.titles["de"] == "Wetter"
    assert page.subroutes == ["/gimmicks/recipes/recipe/:id"]
    assert len(chunks) == 2
    first = chunks[0].text
    assert "Link: /gimmicks/weather" in first
    assert "Source code: https://github.com/benjaminmilcic/aznw-routes/tree/new/src/app/pages/gimmicks/weather" in first
    assert "## What is the Weather gimmick?" in first and "Open-Meteo" not in first


def test_portfolio_skips_hidden_projects_and_resolves_texts():
    projects = [
        {"id": "a", "title": "Webaro", "descKey": "p.desc", "stack": ["Angular"], "dialog": [
            {"type": "text", "parts": [{"key": "p.t1"}, {"text": "Codext GmbH", "href": "x"}, {"text": ".", "glue": True}]},
        ]},
        {"id": "b", "title": "Hidden", "descKey": "p.desc", "hidden": True},
    ]
    chunks = portfolio_chunks(projects, {"p": {"desc": "Dashboard", "t1": "Built for"}})
    assert [c.title for c in chunks] == ["Portfolio: Webaro", "Portfolio overview"]
    assert "Built for Codext GmbH." in chunks[0].text
    # The overview lists every visible project with its description, so general questions find them
    assert "- Webaro (/?project=a#portfolio): Dashboard" in chunks[1].text and "Codext GmbH" in chunks[1].text
    assert "Hidden" not in chunks[1].text
    # The link opens the portfolio section with the project's dialog
    assert "Link: /?project=a#portfolio" in chunks[0].text


def test_cv_links_for_every_language():
    from app.rag.documents import cv_links

    links = cv_links(["ai-chat-knowledge-base/a.md", "src/assets/cv/benjamin_milcic_cv_en.pdf"])
    assert links == {
        "de": "/assets/cv/benjamin_milcic_cv_de.pdf",
        "en": "/assets/cv/benjamin_milcic_cv_en.pdf",
        "hr": "/assets/cv/benjamin_milcic_cv_hr.pdf",
    }
    assert cv_links(["ai-chat-knowledge-base/a.md"]) == {}


def test_load_documents_ignores_files_without_frontmatter():
    pages, chunks = load_documents({"ai-chat-knowledge-base/a.md": PAGE.encode(), "ai-chat-knowledge-base/b.md": b"# no frontmatter"})
    assert [p.id for p in pages] == ["weather"]


ROUTES = Routes(
    ["/", "/gimmicks/weather", "/gimmicks/recipes/recipe/:id", "/#about"],
    {"weather": "/gimmicks/weather", "wetter": "/gimmicks/weather"},
)


def test_project_and_cv_links_stay():
    routes = Routes(["/", "/assets/cv/benjamin_milcic_cv_de.pdf"], {})
    assert routes.fix("/?project=webaro#portfolio") == "/?project=webaro#portfolio"
    assert routes.fix("/assets/cv/benjamin_milcic_cv_de.pdf") == "/assets/cv/benjamin_milcic_cv_de.pdf"


def test_valid_links_stay():
    assert ROUTES.fix("/gimmicks/weather") == "/gimmicks/weather"
    assert ROUTES.fix("/gimmicks/recipes/recipe/7") == "/gimmicks/recipes/recipe/7"
    assert ROUTES.fix("/#contact") == "/#contact"
    assert ROUTES.fix("https://github.com/x") == "https://github.com/x"


def test_invented_links_are_repaired():
    assert ROUTES.fix("https://benjamin-milcic.dev/gimmicks/weather") == "/gimmicks/weather"
    assert ROUTES.fix("/gimmicks/wetter") == "/gimmicks/weather"
    assert ROUTES.fix("/#kontakt") == "/#contact"
    assert ROUTES.fix("/ueber-mich") == "/#about"
    assert ROUTES.fix("/nonsense") == "/"


def test_link_fixer_handles_links_split_over_tokens():
    fixer = LinkFixer(ROUTES)
    tokens = ["Siehe [Kontakt", "formular](/", "#kon", "takt) und [Wetter](/gimmicks/", "wetter).", " Ende"]
    text = "".join(fixer.feed(t) for t in tokens) + fixer.flush()
    assert text == "Siehe [Kontaktformular](/#contact) und [Wetter](/gimmicks/weather). Ende"


def test_language_fallback_is_the_website_language():
    from app.chat import language_reminder

    assert "answer in German" in language_reminder("de")
    assert "answer in Croatian" in language_reminder("hr")
    assert "no recognizable language" not in language_reminder(None)


async def test_files_without_any_page_are_reported_instead_of_a_silent_empty_index():
    import pytest

    from app.rag.index import KnowledgeIndex

    class OldBranch:
        """Like branch new before the knowledge base: files, but none with frontmatter"""

        def describe(self):
            return "github:test@old"

        async def list_files(self):
            return {"ai-chat-knowledge-base/about.md": "v1"}

        async def read(self, path):
            return b"# no frontmatter"

    index = KnowledgeIndex("../aznw-routes")
    index.source = OldBranch()
    with pytest.raises(ValueError, match="no page"):
        await index.refresh()
    # Not stored, so the next check reads the files again
    assert index.versions == {}
