"""The searchable knowledge index: chunks with their embeddings, kept in memory.

The knowledge base is small (a few hundred chunks), so a vector database is not needed – a cosine similarity over
all chunks takes a few milliseconds. Embeddings are cached on disk by the hash of the chunk text, so a restart or a
change of one file only embeds the chunks that are new.

Embeddings come from the Cloudflare worker of benjamin-milcic.dev (POST /embed, @cf/baai/bge-m3): multilingual, so
a Croatian question also finds the English text.
"""

import asyncio
import json
import logging
import math

import httpx

from app.config import EMBED_URL, KNOWLEDGE_REFRESH_MINUTES, KNOWLEDGE_SOURCE, STATE_DIR, WORKER_SECRET
from app.rag.documents import Chunk, Page, cv_links, load_documents
from app.rag.sources import create_source

logger = logging.getLogger(__name__)

PARALLEL_REQUESTS = 8
CACHE_FILE = STATE_DIR / "embeddings.json"


async def embed(texts: list[str]) -> list[list[float]]:
    """Normalized embedding vectors – the worker takes one text per request, so they run in parallel."""
    limit = asyncio.Semaphore(PARALLEL_REQUESTS)
    async with httpx.AsyncClient(timeout=60, headers={"X-Worker-Secret": WORKER_SECRET}) as client:

        async def one(text: str) -> list[float]:
            async with limit:
                response = await client.post(EMBED_URL, json={"q": text})
                if response.status_code == 403:
                    raise RuntimeError("Embedding worker: 403 – WORKER_SECRET is missing or wrong")
                response.raise_for_status()
                return normalize(response.json()["vector"])

        return await asyncio.gather(*(one(text) for text in texts))


def normalize(vector: list[float]) -> list[float]:
    length = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / length for v in vector]


class KnowledgeIndex:
    def __init__(self, source_spec: str = KNOWLEDGE_SOURCE) -> None:
        self.source = create_source(source_spec)
        self.pages: list[Page] = []
        self.chunks: list[Chunk] = []
        self.vectors: list[list[float]] = []
        self.versions: dict[str, str] = {}
        self.files: dict[str, bytes] = {}
        self.cache: dict[str, list[float]] = self.load_cache()
        self.error: str | None = None

    @staticmethod
    def load_cache() -> dict[str, list[float]]:
        try:
            return json.loads(CACHE_FILE.read_text())
        except (OSError, ValueError):
            return {}

    def save_cache(self, keys: set[str]) -> None:
        # Only chunks that still exist, so the file does not grow forever
        self.cache = {k: v for k, v in self.cache.items() if k in keys}
        try:
            STATE_DIR.mkdir(parents=True, exist_ok=True)
            CACHE_FILE.write_text(json.dumps(self.cache))
        except OSError as err:
            logger.warning("Embedding cache not saved: %s", err)

    async def refresh(self) -> bool:
        """Reloads changed files and embeds new chunks. Returns True if something changed."""
        versions = await self.source.list_files()
        if versions == self.versions:
            return False
        changed = [p for p, v in versions.items() if self.versions.get(p) != v]
        contents = await asyncio.gather(*(self.source.read(p) for p in changed))
        files = {p: c for p, c in self.files.items() if p in versions} | dict(zip(changed, contents))
        pages, chunks = load_documents(files)
        if files and not pages:
            # Shown as knowledgeError in /health instead of a silent "0 pages" – e.g. the branch doesn't have the
            # knowledge base (yet). The versions are not stored, so the next check tries again.
            raise ValueError(f"{len(files)} files found, but no page with frontmatter in {self.source.describe()}")

        missing = [c for c in chunks if c.key not in self.cache]
        if missing:
            for chunk, vector in zip(missing, await embed([c.text for c in missing])):
                self.cache[chunk.key] = vector
        self.save_cache({c.key for c in chunks})

        self.files, self.versions = files, versions
        self.pages, self.chunks = pages, chunks
        self.vectors = [self.cache[c.key] for c in chunks]
        logger.info("Knowledge base %s: %d pages, %d chunks (%d embedded)", self.source.describe(), len(pages), len(chunks), len(missing))
        return True

    @property
    def cv_links(self) -> dict[str, str]:
        """Links to the CV PDF per language (de, en, hr)."""
        return cv_links(self.files)

    async def keep_fresh(self) -> None:
        """Background task: loads the knowledge base and checks for changes every KNOWLEDGE_REFRESH_MINUTES."""
        while True:
            try:
                await self.refresh()
                self.error = None
            except Exception as err:  # keep the old index and try again later
                self.error = f"{type(err).__name__}: {err}"
                logger.exception("Refreshing the knowledge base failed")
            await asyncio.sleep(KNOWLEDGE_REFRESH_MINUTES * 60)

    async def search(self, query: str, k: int) -> list[Chunk]:
        if not self.chunks or not query.strip():
            return []
        [vector] = await embed([query])
        scores = [sum(a * b for a, b in zip(vector, v)) for v in self.vectors]
        best = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)[:k]
        return [self.chunks[i] for i in best]


knowledge = KnowledgeIndex()
