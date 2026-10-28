"""High score list of the Moorhuhn game."""

import json

from app import nest

SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_moorhuhn_highscores",
        "description": "Returns the high score list of the Moorhuhn game on Benjamin's website, best first.",
        "parameters": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "description": "Number of places to return (default 10)."},
            },
        },
    },
}


async def run(args: dict, context) -> dict:
    data = await nest.get("/moorhuhn")
    # The endpoint returns the list as a JSON string
    scores = json.loads(data) if isinstance(data, str) else data
    limit = max(1, min(int(args.get("limit") or 10), 50))
    return {
        "highscores": [
            {"rank": s["rank"], "name": s["name"], "score": s["score"], "date": s["date"][:10]} for s in scores[:limit]
        ],
        "game_link": "/gimmicks/games/moorhuhn",
    }
