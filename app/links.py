"""Keeps links to the website valid.

Links to the website are relative paths (/gimmicks/weather, /#contact), so the frontend opens them with the Angular
router in the same tab. The model sometimes invents paths (/about, /gimmicks/wetter) or translates anchors (/#kontakt),
which would show the 404 page. LinkFixer corrects the targets of Markdown links directly in the token stream.
"""

import re
from collections.abc import Iterable

SITE_PATTERN = re.compile(r"^https?://(?:www\.)?benjamin-milcic\.dev(?=/|$)", re.IGNORECASE)
# Sections of the home page (known-fragment.guard.ts in aznw-routes) and words the model uses for them
HOME_SECTIONS = {
    "about": ("about", "ueber", "über", "uber", "o-meni", "omeni", "cv", "lebenslauf", "resume"),
    "skills": ("skill", "faehigkeit", "fähigkeit", "kenntnis", "vjestin", "vještin", "tools"),
    "portfolio": ("portfolio", "projekt", "project"),
    "contact": ("contact", "kontakt"),
    "footer": ("footer", "links", "social"),
    "header": ("header", "home", "start"),
}
# Start of a Markdown link target: "](" – the target ends at the next ")" or whitespace
LINK_TARGET = re.compile(r"\]\(([^)\s]*)\)")


class Routes:
    """The valid paths of the website, from the frontmatter of the knowledge base."""

    def __init__(self, routes: Iterable[str], names: dict[str, str]) -> None:
        self.exact: set[str] = set()
        self.patterns: list[re.Pattern] = []
        for route in routes:
            path = route.split("#")[0] or "/"
            if ":" in path:
                self.patterns.append(re.compile("^" + re.sub(r":\w+", r"[^/]+", re.escape(path).replace(r"\:", ":")) + "$"))
            else:
                self.exact.add(path.rstrip("/") or "/")
        # lowercase word (id, title) -> route, to repair invented paths like /gimmicks/wetter
        self.names = {k.lower(): v for k, v in names.items()}

    def valid(self, target: str) -> bool:
        path, _, anchor = target.partition("#")
        # The query only selects something on the page, e.g. /?project=webaro#portfolio opens a project
        path = path.split("?")[0].rstrip("/") or "/"
        if path == "/" and anchor:
            return anchor in HOME_SECTIONS
        return path in self.exact or any(p.match(path) for p in self.patterns)

    def fix(self, target: str) -> str:
        # The model sometimes escapes slashes like in JSON ("\/assets/...") – the browser can't use that
        target = target.replace("\\/", "/")
        if SITE_PATTERN.match(target):
            target = SITE_PATTERN.sub("", target) or "/"
        if not target.startswith("/") or target.startswith("//") or self.valid(target):
            return target
        words = [w for w in re.split(r"[/#?\-_]+", target.lower()) if w]
        for word in reversed(words):
            if word in self.names:
                return self.names[word]
        for word in words:
            for section, synonyms in HOME_SECTIONS.items():
                if any(s in word for s in synonyms):
                    return f"/#{section}"
        return "/"


class LinkFixer:
    """Corrects link targets in the token stream. A target can be split over several tokens, so text from an
    unfinished "](" on is held back until its closing bracket arrives."""

    def __init__(self, routes: Routes) -> None:
        self.routes = routes
        self.buffer = ""

    def fix_text(self, text: str) -> str:
        return LINK_TARGET.sub(lambda m: f"]({self.routes.fix(m.group(1))})", text)

    def feed(self, token: str) -> str:
        self.buffer += token
        start = self.pending_start()
        text, self.buffer = self.fix_text(self.buffer[:start]), self.buffer[start:]
        return text

    def flush(self) -> str:
        text, self.buffer = self.fix_text(self.buffer), ""
        return text

    def pending_start(self) -> int:
        """Index from which the buffer may still become a link target, or len(buffer) if it can't."""
        start = self.buffer.rfind("](")
        if start != -1 and not re.search(r"[)\s]", self.buffer[start + 2 :]):
            return start
        if self.buffer.endswith("]"):
            return len(self.buffer) - 1
        return len(self.buffer)
