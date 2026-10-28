"""Where the knowledge base files come from: a local checkout of aznw-routes or the repository on GitHub.

Both return the same relative paths, so everything after loading is identical in development and on the server.
"""

import fnmatch
import re
from pathlib import Path

import httpx

KNOWLEDGE_DIR = "ai-chat-knowledge-base"
# Files in the knowledge folder that are documentation, not knowledge
EXCLUDED = {"README.md", "_template.md", "tools.md", "test-questions.md"}
# Other files of the website that belong to the knowledge base
EXTRA_PATTERNS = [
    "src/app/pages/home/components/portfolio/portfolio.json",
    "src/assets/i18n/en.json",
    "src/assets/cv/benjamin_milcic_cv_en*.pdf",
]
TIMEOUT = httpx.Timeout(30.0)
HEADERS = {"User-Agent": "python-aznw-api (AI chat of benjamin-milcic.dev)"}


def wanted(path: str) -> bool:
    if path.startswith(f"{KNOWLEDGE_DIR}/") and path.endswith(".md"):
        return Path(path).name not in EXCLUDED
    return any(fnmatch.fnmatch(path, pattern) for pattern in EXTRA_PATTERNS)


def newest_cv(paths: list[str]) -> list[str]:
    """Only the newest CV, e.g. ..._en_2027.pdf instead of ..._en.pdf."""
    cvs = sorted(p for p in paths if p.endswith(".pdf"))
    return [p for p in paths if not p.endswith(".pdf")] + cvs[-1:]


class LocalSource:
    def __init__(self, root: str) -> None:
        self.root = Path(root).resolve()

    def describe(self) -> str:
        return str(self.root)

    async def list_files(self) -> dict[str, str]:
        """Relative path -> version (changes whenever the file changes)."""
        files = {}
        for file in self.root.rglob("*"):
            path = file.relative_to(self.root).as_posix()
            if file.is_file() and "node_modules" not in path and wanted(path):
                stat = file.stat()
                files[path] = f"{stat.st_mtime_ns}-{stat.st_size}"
        return {p: files[p] for p in newest_cv(sorted(files))}

    async def read(self, path: str) -> bytes:
        return (self.root / path).read_bytes()


class GitHubSource:
    """Reads a public GitHub repository: one tree request per check, raw downloads only for changed files."""

    def __init__(self, repo: str, branch: str) -> None:
        self.repo, self.branch = repo, branch
        self.commit = branch

    def describe(self) -> str:
        return f"github:{self.repo}@{self.branch}"

    async def list_files(self) -> dict[str, str]:
        async with httpx.AsyncClient(timeout=TIMEOUT, headers=HEADERS) as client:
            response = await client.get(f"https://api.github.com/repos/{self.repo}/commits/{self.branch}")
            response.raise_for_status()
            self.commit = response.json()["sha"]
            response = await client.get(f"https://api.github.com/repos/{self.repo}/git/trees/{self.commit}?recursive=1")
            response.raise_for_status()
        blobs = {e["path"]: e["sha"] for e in response.json()["tree"] if e["type"] == "blob" and wanted(e["path"])}
        return {p: blobs[p] for p in newest_cv(sorted(blobs))}

    async def read(self, path: str) -> bytes:
        async with httpx.AsyncClient(timeout=TIMEOUT, headers=HEADERS) as client:
            response = await client.get(f"https://raw.githubusercontent.com/{self.repo}/{self.commit}/{path}")
            response.raise_for_status()
            return response.content


def create_source(spec: str) -> LocalSource | GitHubSource:
    match = re.fullmatch(r"github:([\w.-]+/[\w.-]+)@([\w./-]+)", spec)
    if match:
        return GitHubSource(match.group(1), match.group(2))
    return LocalSource(spec)
