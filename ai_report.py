"""Optional AI report: works with several providers through plain HTTP calls.

Supported:
    - Anthropic (Claude)
    - OpenAI, and any OpenAI-compatible service (Groq, OpenRouter, DeepSeek, Together,
      a local Ollama server, ...) by changing the base URL
    - Google Gemini

Only the summary tables built in analyzer.build_llm_prompt() are sent, never raw log rows.
Model names change over time, so they are editable in the app.
"""

import re
from datetime import datetime

import requests

PROVIDERS = {
    "Anthropic (Claude)": {
        "model": "claude-sonnet-5-5",
        "base_url": None,
    },
    "OpenAI or compatible": {
        "model": "gpt-4o-mini",
        "base_url": "https://api.openai.com/v1",
    },
    "Google Gemini": {
        "model": "gemini-3.8-flash",
        "base_url": None,
    },
}


def _check(response: requests.Response) -> dict:
    """Raise a readable error if the provider rejected the request."""
    if response.status_code >= 400:
        raise RuntimeError(f"HTTP {response.status_code}: {response.text[:300]}")
    return response.json()


def _post(url: str, headers: dict, json: dict, timeout: int) -> requests.Response:
    """POST with readable errors for slow or unreachable servers."""
    try:
        return requests.post(url, headers=headers, json=json, timeout=timeout)
    except requests.exceptions.Timeout:
        raise RuntimeError(
            f"The provider did not answer within {timeout} seconds. Try again, "
            "or switch to a faster model name in the sidebar."
        ) from None
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "Could not reach the provider. Check your internet connection, "
            "VPN or firewall, then try again."
        ) from None


def generate_report(provider: str, api_key: str, model: str, prompt: str,
                    base_url: str | None = None, timeout: int = 180) -> str:
    """Send the prompt to the chosen provider and return the text of its answer."""
    if provider == "Anthropic (Claude)":
        r = _post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={"model": model, "max_tokens": 1200,
                  "messages": [{"role": "user", "content": prompt}]},
            timeout=timeout,
        )
        return _check(r)["content"][0]["text"]

    if provider == "OpenAI or compatible":
        url = (base_url or PROVIDERS[provider]["base_url"]).rstrip("/") + "/chat/completions"
        r = _post(
            url,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": model, "messages": [{"role": "user", "content": prompt}]},
            timeout=timeout,
        )
        return _check(r)["choices"][0]["message"]["content"]

    if provider == "Google Gemini":
        r = _post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
            json={"contents": [{"parts": [{"text": prompt}]}]},
            timeout=timeout,
        )
        return _check(r)["candidates"][0]["content"]["parts"][0]["text"]

    raise ValueError(f"Unknown provider: {provider}")


# Models that cannot write a text report (speech, images, embeddings, ...)
_SKIP_WORDS = (
    "whisper", "tts", "embed", "guard", "moderation", "transcribe", "audio", "realtime",
    "image", "dall-e", "speech", "imagen", "veo", "robotics", "computer-use", "aqa",
    "live", "instruct",
)


def _get(url: str, headers: dict, timeout: int = 15) -> dict:
    """GET with readable errors."""
    try:
        r = requests.get(url, headers=headers, timeout=timeout)
    except requests.exceptions.Timeout:
        raise RuntimeError("the provider did not answer in time") from None
    except requests.exceptions.ConnectionError:
        raise RuntimeError("could not reach the provider") from None
    return _check(r)


def list_models(provider: str, api_key: str, base_url: str | None = None) -> list[str]:
    """Ask the provider which text models this key can use, newest or best first."""
    if provider == "Anthropic (Claude)":
        data = _get(
            "https://api.anthropic.com/v1/models?limit=100",
            {"x-api-key": api_key, "anthropic-version": "2023-06-01"},
        )
        return [m["id"] for m in data.get("data", [])]  # the API returns newest first

    if provider == "OpenAI or compatible":
        url = (base_url or PROVIDERS[provider]["base_url"]).rstrip("/") + "/models"
        data = _get(url, {"Authorization": f"Bearer {api_key}"})
        items = [m for m in data.get("data", []) if not any(w in m["id"].lower() for w in _SKIP_WORDS)]
        if "api.openai.com" in url:  # OpenAI's list also holds many non-chat models
            items = [m for m in items if re.match(r"^(gpt|o\d)", m["id"])]
        items.sort(key=lambda m: m.get("created", 0), reverse=True)
        return [m["id"] for m in items]

    if provider == "Google Gemini":
        data = _get(
            "https://generativelanguage.googleapis.com/v1beta/models?pageSize=200",
            {"x-goog-api-key": api_key},
        )
        names = [
            m["name"].removeprefix("models/")
            for m in data.get("models", [])
            if "generateContent" in m.get("supportedGenerationMethods", [])
        ]
        names = [n for n in names if n.startswith("gemini") and not any(w in n for w in _SKIP_WORDS)]

        def sort_key(name: str):
            v = re.search(r"gemini-(\d+(?:\.\d+)*)", name)
            version = tuple(int(x) for x in v.group(1).split(".")) if v else (0,)
            return (version, not ("preview" in name or "exp" in name), name)

        return sorted(names, key=sort_key, reverse=True)  # highest version, stable releases first

    raise ValueError(f"Unknown provider: {provider}")


def pick_default(provider: str, models: list[str]) -> str:
    """Choose a sensible, fast model from the provider's list."""
    if not models:
        return PROVIDERS[provider]["model"]
    if provider == "Anthropic (Claude)":
        return next((m for m in models if "sonnet" in m), models[0])
    if provider == "Google Gemini":
        stable_flash = [m for m in models if "flash" in m and not any(w in m for w in ("lite", "preview", "exp"))]
        any_flash = [m for m in models if "flash" in m]
        return (stable_flash or any_flash or models)[0]
    return next((m for m in models if "mini" in m), models[0])


def build_markdown(metrics: dict, ai_text: str, provider: str, model: str) -> str:
    """Wrap the AI text in a small report with a title, date and the key numbers."""
    return (
        "# Maintenance Reliability Report\n\n"
        f"Generated: {datetime.now():%d %b %Y, %H:%M}\n\n"
        "## Key numbers\n\n"
        f"- Total failures: {metrics['total_failures']}\n"
        f"- Total downtime: {metrics['total_downtime_hours']:.0f} h\n"
        f"- Total repair cost: {metrics['total_repair_cost']:,.0f}\n"
        f"- Average downtime per failure: {metrics['avg_downtime_per_failure']:.1f} h\n"
        f"- Machines covered: {metrics['machines']}\n\n"
        "## AI analysis\n\n"
        f"{ai_text}\n\n"
        "---\n\n"
        f"_Written by {provider} ({model}) from summary tables. Review before acting on it._\n"
    )


def to_html(markdown_text: str) -> str | None:
    """Convert the report to a standalone HTML page (open in a browser, then print to PDF).

    Returns None if the optional `markdown` package is not installed.
    """
    try:
        import markdown
    except ImportError:
        return None
    body = markdown.markdown(markdown_text, extensions=["tables"])
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>Maintenance Reliability Report</title>"
        "<style>body{font-family:Arial,sans-serif;max-width:800px;margin:2rem auto;"
        "padding:0 1rem;line-height:1.6;color:#222}"
        "table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:6px 10px}"
        "h1,h2{border-bottom:1px solid #ddd;padding-bottom:4px}</style></head>"
        f"<body>{body}</body></html>"
    )
