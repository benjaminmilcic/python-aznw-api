"""Settings from the environment (.env locally, the service's EnvironmentFile on the server)."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent

LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-5.4-mini")
# Thinking before answering: "low" is enough for questions about the website and keeps answers fast and cheap.
# Reasoning tokens are billed as output.
REASONING_EFFORT = os.getenv("REASONING_EFFORT", "low")
# Upper limit per model call (answer plus reasoning), so a single answer can't run away – a full recipe still fits
MAX_COMPLETION_TOKENS = int(os.getenv("MAX_COMPLETION_TOKENS", "4000"))
# Costs of all visitors together per day (UTC) in US dollars, counted with LiteLLM's model prices. 0 switches it off.
DAILY_BUDGET_USD = float(os.getenv("DAILY_BUDGET_USD", "2"))
# Speech-to-text model per quality setting of the frontend: turbo is faster, large-v3 is more accurate
# for less common languages such as Croatian.
TRANSCRIPTION_MODELS = {
    "fast": os.getenv("TRANSCRIPTION_MODEL_FAST", "groq/whisper-large-v3-turbo"),
    "quality": os.getenv("TRANSCRIPTION_MODEL_QUALITY", "groq/whisper-large-v3"),
}

NEST_API_URL = os.getenv("NEST_API_URL", "https://api.benjamin-milcic.dev").rstrip("/")
INTERNAL_API_KEY = os.getenv("INTERNAL_API_KEY", "")
# Tool rounds per answer – protects against a model that keeps calling tools
MAX_TOOL_ROUNDS = int(os.getenv("MAX_TOOL_ROUNDS", "5"))

# Token budget of the conversation history alone. Every request also carries about 3,500 tokens of system prompt,
# tool descriptions and knowledge, and the history is sent again with every question and tool round – so it is
# summarized long before the model's real context window is reached, which keeps each question cheap. Once
# summarized, the history is far below this budget again, so the next summary only follows after several more questions.
HISTORY_TOKEN_BUDGET = int(os.getenv("HISTORY_TOKEN_BUDGET", "4000"))

# Knowledge base: a local checkout of aznw-routes ("../aznw-routes") or "github:<owner>/<repo>@<branch>"
KNOWLEDGE_SOURCE = os.getenv("KNOWLEDGE_SOURCE", "github:benjaminmilcic/aznw-routes@new")
# How often the knowledge base is checked for changes
KNOWLEDGE_REFRESH_MINUTES = float(os.getenv("KNOWLEDGE_REFRESH_MINUTES", "15"))
# Number of knowledge chunks added to the prompt per question
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "6"))

# Embeddings: Cloudflare worker of benjamin-milcic.dev (Workers AI, @cf/baai/bge-m3, multilingual)
EMBED_URL = os.getenv("EMBED_URL", "https://little-sky-725e.benjamin-milcic.workers.dev/embed")
WORKER_SECRET = os.getenv("WORKER_SECRET", "")

# Writable directory for the embedding cache – systemd's StateDirectory on the server
STATE_DIR = Path(os.getenv("STATE_DIRECTORY") or ROOT / ".state")
