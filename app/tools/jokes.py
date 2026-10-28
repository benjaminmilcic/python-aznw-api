"""A random joke from Benjamin's joke book (German, English and Croatian versions of each joke)."""

from app import nest

SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_random_joke",
        "description": (
            "Returns a random joke from Benjamin's own joke book. Use it when a visitor asks for a joke, "
            "or after the visitor agreed to hear one."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "language": {
                    "type": "string",
                    "enum": ["de", "en", "hr"],
                    "description": "Language of the visitor's message. For any other language use 'en' and translate.",
                },
            },
            "required": ["language"],
        },
    },
}


async def run(args: dict, context) -> dict:
    language = args.get("language") if args.get("language") in ("de", "en", "hr") else "en"
    # Internal endpoint: on the website the jokes are only readable after login, the chat may always tell them
    joke = await nest.get("/jokes/random", params={"lang": language}, internal=True)
    return {"joke": joke["text"], "language": joke["language"], "jokes_gimmick_link": "/gimmicks/auth/main"}
