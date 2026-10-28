"""Whether Benjamin is available for new projects – the same data as the badge in the website header."""

from app import nest

SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_availability",
        "description": (
            "Returns whether Benjamin is currently available for new projects or jobs, and from which quarter. "
            "Use it whenever a visitor asks about his availability, hiring him, freelancing or starting a project."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
}


async def run(args: dict, context) -> dict:
    data = await nest.get("/availability")
    return {
        "available": data["available"],
        "quarter": f"Q{data['quarter']} {data['year']}",
        "contact": "Visitors can reach Benjamin via the contact form on the home page (link: /#contact).",
    }
