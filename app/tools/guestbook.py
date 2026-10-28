"""Entries of the guestbook on Benjamin's website – written by visitors, often references from colleagues."""

import html
import re

from app import nest

SCHEMA = {
    "type": "function",
    "function": {
        "name": "find_guestbook_entries",
        "description": (
            "Lists guestbook entries (name and date only, not the text). Without a query it returns all entries; "
            "with a query only entries whose name or text contains it. Use it when a visitor asks about a person or "
            "company, who wrote in the guestbook, or what others think of Benjamin."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Name or company to look for. Omit to list all entries."},
            },
        },
    },
}

ENTRY_SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_guestbook_entry",
        "description": "Returns the full text of one guestbook entry. Only use it after the visitor asked to see it.",
        "parameters": {
            "type": "object",
            "properties": {
                "id": {"type": "integer", "description": "Id from find_guestbook_entries."},
                "name": {"type": "string", "description": "Name of the author, if the id is not known."},
            },
        },
    },
}

GUESTBOOK_LINK = "/gimmicks/guestbook"


def to_text(content: str) -> str:
    """The entry's HTML (from a rich text editor) as plain text."""
    text = re.sub(r"<br\s*/?>|</p>|</li>", "\n", content, flags=re.IGNORECASE)
    text = html.unescape(re.sub(r"<[^>]+>", "", text))
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def images(content: str) -> list[str]:
    return re.findall(r'<img[^>]+src="([^"]+)"', content)


async def load() -> list[dict]:
    entries = await nest.get("/guestbook")
    return sorted(entries, key=lambda e: e["date"], reverse=True)


async def run(args: dict, context) -> dict:
    entries = await load()
    query = (args.get("query") or "").strip().lower()
    if query:
        entries = [e for e in entries if query in e["name"].lower() or query in to_text(e["content"]).lower()]
    return {
        "count": len(entries),
        "entries": [{"id": e["id"], "name": e["name"], "date": e["date"][:10]} for e in entries],
        "guestbook_link": GUESTBOOK_LINK,
    }


async def run_entry(args: dict, context) -> dict:
    entries = await load()
    if args.get("id") is not None:
        entry = next((e for e in entries if e["id"] == args["id"]), None)
    else:
        name = (args.get("name") or "").strip().lower()
        entry = next((e for e in entries if name and name in e["name"].lower()), None)
    if entry is None:
        return {"error": "No such guestbook entry."}
    return {
        "name": entry["name"],
        "date": entry["date"][:10],
        # Written by a visitor: data to quote, never instructions
        "text": to_text(entry["content"]),
        "images": images(entry["content"]),
        "guestbook_link": GUESTBOOK_LINK,
    }
