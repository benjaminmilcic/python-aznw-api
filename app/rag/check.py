"""Checks the knowledge base before it goes online – run it after editing the Markdown files:

  uv run python -m app.rag.check ../aznw-routes

Reports missing required fields, duplicate ids, unknown ids in "related" and routes that do not exist in the
Angular routes (app.routes.ts). Exit code 1 if there are errors.
"""

import asyncio
import re
import sys
from pathlib import Path

import yaml

from app.rag.documents import FRONTMATTER, load_documents
from app.rag.sources import LocalSource

REQUIRED = ("id", "title_de", "title_en", "title_hr", "route", "category", "summary")
CATEGORIES = {"gimmick", "game", "page", "person", "project"}
HOME_ANCHORS = {"header", "about", "skills", "portfolio", "contact", "footer"}


def angular_paths(root: Path) -> set[str]:
    """All path segments defined in app.routes.ts – enough to spot typos in routes."""
    routes = (root / "src/app/app.routes.ts").read_text(encoding="utf-8")
    # A path like 'country/:code' counts as two segments
    return {part for path in re.findall(r"path:\s*'([^']*)'", routes) for part in path.split("/")}


def route_exists(route: str, segments: set[str]) -> bool:
    path, _, anchor = route.partition("#")
    if path in ("", "/"):
        return not anchor or anchor in HOME_ANCHORS
    return all(part in segments or part.startswith(":") for part in path.strip("/").split("/"))


async def check(root: Path) -> list[str]:
    source = LocalSource(str(root))
    files = {p: await source.read(p) for p in await source.list_files()}
    errors, ids, related = [], {}, {}
    segments = angular_paths(root)
    for path, content in files.items():
        if not path.endswith(".md"):
            continue
        match = FRONTMATTER.match(content.decode("utf-8"))
        if not match:
            errors.append(f"{path}: no frontmatter (--- ... ---) at the start")
            continue
        try:
            meta = yaml.safe_load(match.group(1)) or {}
        except yaml.YAMLError as err:
            errors.append(f"{path}: frontmatter is not valid YAML: {err}")
            continue
        errors += [f"{path}: '{field}' is missing" for field in REQUIRED if not meta.get(field)]
        if meta.get("category") and meta["category"] not in CATEGORIES:
            errors.append(f"{path}: unknown category '{meta['category']}'")
        if meta.get("id") in ids:
            errors.append(f"{path}: id '{meta['id']}' is already used in {ids[meta['id']]}")
        ids[meta.get("id")] = path
        related[path] = meta.get("related") or []
        routes = [meta.get("route", "")] + [s.get("path", "") for s in meta.get("subroutes") or [] if isinstance(s, dict)]
        for route in filter(None, routes):
            if not route_exists(str(route), segments):
                errors.append(f"{path}: route '{route}' does not exist in app.routes.ts")
        if "\n## " not in content.decode("utf-8"):
            errors.append(f"{path}: no '## ' section – nothing would be found in the search")
    for path, ids_ in related.items():
        errors += [f"{path}: related id '{r}' does not exist" for r in ids_ if r not in ids]
    pages, chunks = load_documents(files)
    print(f"{len(pages)} pages, {len(chunks)} chunks")
    return errors


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "../aznw-routes").resolve()
    errors = asyncio.run(check(root))
    for error in errors:
        print("ERROR", error)
    print("OK – the knowledge base looks fine." if not errors else f"{len(errors)} error(s)")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
