"""Recipes from the Recipes gimmick, with link and photo."""

import random

from app import nest

# Up to this many results come with ingredients and preparation, longer lists only with title, link and photo
FULL_RECIPE_LIMIT = 2

SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_recipes",
        "description": (
            "Searches the recipes of the Recipes gimmick on Benjamin's website. Use it when a visitor asks which "
            "recipes exist, wants a recipe suggestion or asks for a dish or an ingredient. "
            "The recipes are written in German, so pass search terms in German."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "German word to search in titles and ingredients, e.g. 'Lachs'. Omit to list all.",
                },
                "suggest": {
                    "type": "boolean",
                    "description": (
                        "true returns one random matching recipe. Always set it when the visitor wants a suggestion, "
                        "a recommendation or a random recipe."
                    ),
                },
            },
        },
    },
}


def recipe_link(recipe_id: int) -> str:
    return f"/gimmicks/recipes/recipe/{recipe_id}"


def matches(recipe: dict, query: str) -> bool:
    text = " ".join([recipe["title"], *recipe.get("ingredients", [])]).lower()
    return all(word in text for word in query.lower().split())


async def run(args: dict, context) -> dict:
    recipes = await nest.get("/recipes")
    query = (args.get("query") or "").strip()
    found = [r for r in recipes if matches(r, query)] if query else recipes
    if args.get("suggest") and found:
        found = [random.choice(found)]
    else:
        # In random order: a model that picks one itself would otherwise always take the first
        found = random.sample(found, len(found))
    full = len(found) <= FULL_RECIPE_LIMIT
    recipes = []
    for r in found:
        recipe = {"title": r["title"], "link": recipe_link(r["id"]), "image": r.get("imagePath") or None}
        if full:
            recipe["ingredients"] = r.get("ingredients", [])
            recipe["preparation"] = r.get("preparation") or ""
        recipes.append(recipe)
    result = {"count": len(found), "recipes": recipes, "all_recipes_link": "/gimmicks/recipes/list"}
    if not full:
        result["note"] = "List only. Search for one recipe to get its ingredients and preparation."
    return result
