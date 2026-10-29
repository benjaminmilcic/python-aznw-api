"""Tool tests with the NestJS API and Open-Meteo mocked (respx) – no network, no LLM."""

import json

import httpx
import pytest
import respx

from app.config import NEST_API_URL
from app.tools import ToolContext, run_tool
from app.tools.guestbook import to_text

GUESTBOOK = [
    {
        "id": 10,
        "name": "Timo",
        "date": "2022-10-04T09:22:33.000Z",
        "content": '<p><img src="https://example.org/a.jpg"><br></p><p>Als Gesch&auml;ftsf&uuml;hrer von ITM Webdesign &hellip;</p>',
    },
    {"id": 13, "name": "Lisa", "date": "2024-03-08T13:23:09.000Z", "content": "<p>Hallo!</p>"},
]
RECIPES = [
    {"id": 6, "title": "Cremiges Pilzrisotto", "ingredients": ["Risottoreis", "Champignons"], "imagePath": "https://example.org/r.png", "preparation": "..."},
    {"id": 7, "title": "Ofenlachs", "ingredients": ["Lachs", "Zitrone"], "imagePath": "", "preparation": "..."},
]


async def call(name: str, args: dict | None = None, context: ToolContext | None = None) -> dict:
    return json.loads(await run_tool(name, json.dumps(args or {}), context or ToolContext()))


@respx.mock
async def test_availability():
    respx.get(f"{NEST_API_URL}/availability").respond(json={"available": True, "year": 2026, "quarter": 4})
    result = await call("get_availability")
    assert result["available"] is True
    assert result["quarter"] == "Q4 2026"


@respx.mock
async def test_recipes_search_and_links():
    respx.get(f"{NEST_API_URL}/recipes").respond(json=RECIPES)
    result = await call("search_recipes", {"query": "lachs"})
    assert result["count"] == 1
    assert result["recipes"][0]["link"] == "/gimmicks/recipes/recipe/7"
    assert result["recipes"][0]["image"] is None
    assert result["recipes"][0]["preparation"] == "..."
    suggestion = await call("search_recipes", {"suggest": True})
    assert suggestion["count"] == 1
    assert suggestion["recipes"][0]["ingredients"]
    assert suggestion["recipes"][0]["preparation"]


@respx.mock
async def test_recipes_list_without_preparation():
    recipes = [{**RECIPES[0], "id": i, "title": f"Rezept {i}"} for i in range(5)]
    respx.get(f"{NEST_API_URL}/recipes").respond(json=recipes)
    result = await call("search_recipes")
    assert result["count"] == 5
    assert all("preparation" not in r and "ingredients" not in r for r in result["recipes"])
    assert "note" in result


@respx.mock
async def test_joke_sends_internal_key(monkeypatch):
    monkeypatch.setattr("app.nest.INTERNAL_API_KEY", "secret")
    route = respx.get(f"{NEST_API_URL}/jokes/random").respond(json={"language": "hr", "text": "Vic", "total": 1})
    result = await call("get_random_joke", {"language": "hr"})
    assert result["joke"] == "Vic"
    request = route.calls.last.request
    assert request.headers["x-internal-api-key"] == "secret"
    assert request.url.params["lang"] == "hr"


@respx.mock
async def test_guestbook_list_has_no_text_and_entry_has_images():
    respx.get(f"{NEST_API_URL}/guestbook").respond(json=GUESTBOOK)
    listing = await call("find_guestbook_entries", {"query": "itm"})
    assert listing["entries"] == [{"id": 10, "name": "Timo", "date": "2022-10-04"}]
    entry = await call("get_guestbook_entry", {"name": "timo"})
    assert entry["images"] == ["https://example.org/a.jpg"]
    assert "Geschäftsführer" in entry["text"]


def test_guestbook_html_to_text():
    assert to_text("<p>a</p><p>b&amp;c</p>") == "a\nb&c"


@respx.mock
async def test_moorhuhn_parses_json_string():
    scores = [{"name": "Hannah", "score": 840, "date": "2025-02-09T00:00:00.000Z", "rank": 1}]
    respx.get(f"{NEST_API_URL}/moorhuhn").respond(json=json.dumps(scores))
    result = await call("get_moorhuhn_highscores")
    assert result["highscores"][0] == {"rank": 1, "name": "Hannah", "score": 840, "date": "2025-02-09"}


async def test_weather_without_location_requests_it():
    context = ToolContext()
    result = await call("get_weather", {}, context)
    assert result == {"location_missing": True}
    assert context.actions == {"request_location"}


@respx.mock
async def test_weather_at_shared_location():
    respx.get("https://api.open-meteo.com/v1/forecast").respond(
        json={
            "current": {"temperature_2m": 18.2, "apparent_temperature": 17.0, "weather_code": 2, "wind_speed_10m": 9.0, "relative_humidity_2m": 60},
            "daily": {
                "time": ["2026-10-29", "2026-10-30"],
                "weather_code": [2, 61],
                "temperature_2m_min": [12.0, 10.5],
                "temperature_2m_max": [21.0, 16.0],
                "precipitation_probability_max": [10, 80],
            },
        }
    )
    respx.get(f"{NEST_API_URL}/geolocation/reverse-geocode").respond(
        json={"address": {"road": "Secret Street 1", "city": "Rijeka", "country": "Hrvatska"}}
    )
    result = await call("get_weather", {}, ToolContext(latitude=45.33, longitude=14.44))
    assert result["place"] == "Rijeka, Hrvatska"
    assert result["current"]["conditions"] == "partly cloudy"
    assert result["today"] == {"conditions": "partly cloudy", "min_c": 12.0, "max_c": 21.0, "precipitation_probability_percent": 10}
    assert result["next_days"] == [
        {"conditions": "light rain", "min_c": 10.5, "max_c": 16.0, "precipitation_probability_percent": 80, "date": "2026-10-30", "weekday": "Friday"}
    ]


@respx.mock
async def test_errors_are_returned_to_the_model():
    respx.get(f"{NEST_API_URL}/recipes").mock(side_effect=httpx.ConnectError("down"))
    result = await call("search_recipes")
    assert "error" in result


@pytest.mark.parametrize("name", ["unknown_tool"])
async def test_unknown_tool(name):
    assert "error" in await call(name)
