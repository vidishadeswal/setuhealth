"""Local LLM client (design doc: fully-local stack) — talks to Ollama over HTTP. No
external API key: this is the one call in the pipeline that leaves the process, and it
never leaves the machine.
"""

import httpx

from backend.app.config import get_settings


async def generate(prompt: str, temperature: float = 0.0) -> str:
    settings = get_settings()
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{settings.ollama_base_url}/api/generate",
            json={
                "model": settings.ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": temperature},
            },
        )
        response.raise_for_status()
        return response.json()["response"].strip()
