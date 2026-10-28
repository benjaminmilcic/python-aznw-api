"""Tools the model can call while answering (OpenAI function calling format, used via LiteLLM).

Each tool module has a SCHEMA (what the model sees) and an async run(args, context).
Data is always fetched live, so new guestbook entries, recipes or high scores show up at once.
"""

import json
import logging
from dataclasses import dataclass, field

from app.tools import availability, guestbook, jokes, moorhuhn, recipes, weather

logger = logging.getLogger(__name__)


@dataclass
class ToolContext:
    """What a tool may know about the current request besides the model's arguments."""

    # Browser location of the visitor, only sent when they allowed it
    latitude: float | None = None
    longitude: float | None = None
    # Things the frontend should do after the answer, e.g. offer to share the location
    actions: set[str] = field(default_factory=set)


_MODULES = [availability, recipes, jokes, guestbook, weather, moorhuhn]
_RUNNERS = {m.SCHEMA["function"]["name"]: m.run for m in _MODULES}
# guestbook has two tools
_RUNNERS[guestbook.ENTRY_SCHEMA["function"]["name"]] = guestbook.run_entry

TOOLS = [m.SCHEMA for m in _MODULES] + [guestbook.ENTRY_SCHEMA]


async def run_tool(name: str, arguments: str, context: ToolContext) -> str:
    """Runs one tool call and returns its result as JSON text for the model.

    Errors are returned to the model as well, so it can tell the visitor instead of failing the answer.
    """
    runner = _RUNNERS.get(name)
    if runner is None:
        return json.dumps({"error": f"Unknown tool {name}"})
    try:
        args = json.loads(arguments or "{}")
        result = await runner(args, context)
    except Exception as err:
        logger.exception("Tool %s failed", name)
        result = {"error": f"The data is not available right now ({type(err).__name__})."}
    return json.dumps(result, ensure_ascii=False)
