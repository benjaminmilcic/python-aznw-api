"""LLM service for the AI chat on benjamin-milcic.dev.

Only the NestJS backend (nest-aznw-api) calls this API. Everything that talks to a model lives here (via LiteLLM).

Endpoints (all but /health require the header x-internal-api-key when INTERNAL_API_KEY is set):
  POST /chat/stream  JSON {message, history, location?}  ->  Server-Sent Events (see app.chat.stream_answer)
  POST /transcribe   multipart fields "audio" and "quality" (fast|quality)  ->  {"text": "..."}
  GET  /health
"""

import asyncio
import hmac
import logging
import warnings
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import litellm
from fastapi import FastAPI, Form, Header, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from app.chat import ChatRequest, stream_answer
from app.config import INTERNAL_API_KEY, KNOWLEDGE_SOURCE, LLM_MODEL, NEST_API_URL, TRANSCRIPTION_MODELS
from app.rag.index import knowledge
from app.tools import TOOLS

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
# LiteLLM answers GPT-5 models via OpenAI's Responses API and hands Pydantic a plain dict as "usage" – harmless, but
# it would log this warning with every answer
warnings.filterwarnings("ignore", message="Pydantic serializer warnings", category=UserWarning)

# Commit of the deployed code, written by the deploy workflow ("local" in development)
_build_file = Path(__file__).resolve().parent.parent / "BUILD_SHA"
BUILD = _build_file.read_text().strip() if _build_file.exists() else "local"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Loads the knowledge base in the background and keeps it up to date, so the API starts at once
    task = asyncio.create_task(knowledge.keep_fresh())
    yield
    task.cancel()


app = FastAPI(title="python-aznw-api", lifespan=lifespan)


def check_key(given: str | None) -> None:
    if INTERNAL_API_KEY and not hmac.compare_digest((given or "").encode(), INTERNAL_API_KEY.encode()):
        raise HTTPException(status_code=401, detail="invalid internal api key")


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest, x_internal_api_key: str | None = Header(default=None)) -> StreamingResponse:
    check_key(x_internal_api_key)
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="message is required")
    return StreamingResponse(stream_answer(request), media_type="text/event-stream")


@app.post("/transcribe")
async def transcribe(
    audio: UploadFile,
    quality: Literal["fast", "quality"] = Form("quality"),
    x_internal_api_key: str | None = Header(default=None),
) -> dict:
    """Turns a recorded audio clip into text. The language is detected automatically."""
    check_key(x_internal_api_key)
    content = await audio.read()
    if not content:
        raise HTTPException(status_code=400, detail="audio file is required")
    try:
        result = await litellm.atranscription(
            model=TRANSCRIPTION_MODELS[quality],
            file=(audio.filename or "audio.webm", content, audio.content_type or "audio/webm"),
        )
    except litellm.RateLimitError:
        raise HTTPException(status_code=429, detail="rate_limit")
    return {"text": result.text.strip()}


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "build": BUILD,
        "model": LLM_MODEL,
        "nestApi": NEST_API_URL,
        "knowledgeSource": KNOWLEDGE_SOURCE,
        "knowledgePages": len(knowledge.pages),
        "knowledgeChunks": len(knowledge.chunks),
        "knowledgeError": knowledge.error,
        "tools": [t["function"]["name"] for t in TOOLS],
    }
