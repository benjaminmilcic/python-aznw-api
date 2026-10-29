"""Current weather at the visitor's location or in a named place (Open-Meteo, free and without a key)."""

from datetime import date

import httpx

from app import nest

SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": (
            "Returns the current weather, today's forecast and the forecast for the next three days. Without a "
            "city it uses the visitor's location, "
            "if they shared it. If the result says the location is missing, ask the visitor to share their "
            "location or to name a city."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                # null allowed: the model sometimes sends it, and some providers reject calls that don't match the schema
                "city": {"type": ["string", "null"], "description": "City name, only if the visitor named one."},
            },
        },
    },
}

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
TIMEOUT = httpx.Timeout(15.0)
FORECAST_DAYS = 4

# WMO weather codes as used by Open-Meteo
WEATHER_CODES = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "depositing rime fog",
    51: "light drizzle", 53: "drizzle", 55: "dense drizzle", 56: "freezing drizzle", 57: "dense freezing drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain", 66: "freezing rain", 67: "heavy freezing rain",
    71: "light snow", 73: "snow", 75: "heavy snow", 77: "snow grains",
    80: "light rain showers", 81: "rain showers", 82: "violent rain showers",
    85: "snow showers", 86: "heavy snow showers",
    95: "thunderstorm", 96: "thunderstorm with hail", 99: "thunderstorm with heavy hail",
}


async def find_city(client: httpx.AsyncClient, city: str) -> dict | None:
    response = await client.get(GEOCODING_URL, params={"name": city, "count": 1})
    response.raise_for_status()
    results = response.json().get("results") or []
    if not results:
        return None
    r = results[0]
    return {"latitude": r["latitude"], "longitude": r["longitude"], "place": f"{r['name']}, {r.get('country', '')}".strip(", ")}


def day(daily: dict, i: int) -> dict:
    """The forecast of one day from Open-Meteo's daily arrays."""
    codes = daily.get("weather_code") or []
    return {
        "conditions": WEATHER_CODES.get(codes[i], "unknown") if i < len(codes) else "unknown",
        "min_c": daily["temperature_2m_min"][i],
        "max_c": daily["temperature_2m_max"][i],
        "precipitation_probability_percent": daily["precipitation_probability_max"][i],
    }


def dated(iso_date: str) -> dict:
    """Date and weekday – the model gets weekdays wrong when it has to work them out itself."""
    return {"date": iso_date, "weekday": date.fromisoformat(iso_date).strftime("%A")}


async def place_name(latitude: float, longitude: float) -> str | None:
    """Town and country of a location – never the street, the visitor's exact address is private."""
    try:
        data = await nest.get("/geolocation/reverse-geocode", params={"lat": latitude, "lon": longitude, "lang": "en"})
    except Exception:
        return None
    address = data.get("address") or {}
    town = address.get("city") or address.get("town") or address.get("village") or address.get("municipality")
    return ", ".join(p for p in (town, address.get("country")) if p) or None


async def run(args: dict, context) -> dict:
    city = (args.get("city") or "").strip()
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        if city:
            location = await find_city(client, city)
            if location is None:
                return {"error": f"No place called '{city}' found."}
        elif context.latitude is not None and context.longitude is not None:
            location = {"latitude": context.latitude, "longitude": context.longitude, "place": None}
        else:
            # The frontend can then offer a button to share the browser location
            context.actions.add("request_location")
            return {"location_missing": True}

        response = await client.get(
            FORECAST_URL,
            params={
                "latitude": location["latitude"],
                "longitude": location["longitude"],
                "current": "temperature_2m,apparent_temperature,weather_code,wind_speed_10m,relative_humidity_2m",
                "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max",
                # Today and the next three days – so "and the next days?" can be answered as well
                "forecast_days": FORECAST_DAYS,
                "timezone": "auto",
            },
        )
        response.raise_for_status()
        data = response.json()

    current, daily = data["current"], data["daily"]
    place = location["place"] or await place_name(location["latitude"], location["longitude"])
    return {
        "place": place,
        "current": {
            "temperature_c": current["temperature_2m"],
            "feels_like_c": current["apparent_temperature"],
            "conditions": WEATHER_CODES.get(current["weather_code"], "unknown"),
            "wind_kmh": current["wind_speed_10m"],
            "humidity_percent": current["relative_humidity_2m"],
        },
        "today": day(daily, 0),
        "next_days": [day(daily, i) | dated(daily["time"][i]) for i in range(1, len(daily.get("time", [])))],
        "weather_gimmick_link": "/gimmicks/weather",
    }
