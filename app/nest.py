"""HTTP client for the NestJS backend (nest-aznw-api), which owns the website's data."""

import httpx

from app.config import INTERNAL_API_KEY, NEST_API_URL

TIMEOUT = httpx.Timeout(15.0)
# Locally the requests go through Cloudflare, whose bot rules may block generic Python user agents
HEADERS = {"User-Agent": "python-aznw-api (AI chat of benjamin-milcic.dev)"}


async def get(path: str, params: dict | None = None, internal: bool = False):
    """GET on the NestJS API, returns the parsed JSON.

    internal=True sends the shared secret for endpoints that only services may call.
    """
    headers = {"x-internal-api-key": INTERNAL_API_KEY} if internal else {}
    async with httpx.AsyncClient(base_url=NEST_API_URL, timeout=TIMEOUT, headers=HEADERS) as client:
        response = await client.get(path, params=params, headers=headers)
        response.raise_for_status()
        return response.json()
