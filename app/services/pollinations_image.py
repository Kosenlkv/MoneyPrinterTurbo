# -*- coding: utf-8 -*-
"""Pollinations.ai image generation — free text-to-image without an API key.

Generates one image per script keyword through the Pollinations image
endpoint (``GET https://image.pollinations.ai/prompt/{prompt}``) and
returns raw image bytes. Anonymous use is rate-limited (about one request
per 15 seconds); a free account token raises the limits and removes the
logo. This is the zero-cost fallback material source: quality is below
Nano Banana, but no key, signup or billing is required.

Configuration (config.toml, [app] section) — all optional:
    pollinations_api_key      free token from https://auth.pollinations.ai
                              (raises rate limits, removes the logo); shared
                              with the Pollinations LLM provider entry
    pollinations_image_base_url  optional override, default
                              https://image.pollinations.ai (kept separate
                              from the LLM provider's pollinations_base_url,
                              which points at text.pollinations.ai)
    pollinations_image_model  optional override, default "flux"
"""

import random
from typing import Optional
from urllib.parse import quote

import requests
from loguru import logger

from app.config import config

DEFAULT_BASE_URL = "https://image.pollinations.ai"
DEFAULT_MODEL = "flux"
REQUEST_TIMEOUT = (30, 300)
RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
# Anonymous usage is rate-limited to 1 image per 45 seconds per IP; the API
# answers 402 FREE_LIMIT with an explicit "Retry after Ns". Treat it as a
# slow lane, not a failure: wait out the window and retry.
FREE_LIMIT_STATUS = 402
FREE_LIMIT_SLEEP_SECONDS = 50
FREE_LIMIT_MAX_ATTEMPTS = 10
MAX_ATTEMPTS = 4
RETRY_SLEEP_SECONDS = 20


class PollinationsImageError(Exception):
    """Configuration or transport error that should abort the task early."""

    def __init__(self, message: str, detail: str = ""):
        super().__init__(message)
        self.message = message
        self.detail = detail


def _setting(key: str, default: str = "") -> str:
    return str(config.app.get(key, default) or default).strip()


def api_key() -> str:
    return _setting("pollinations_api_key")


def base_url() -> str:
    return (
        _setting("pollinations_image_base_url", DEFAULT_BASE_URL).rstrip("/")
        or DEFAULT_BASE_URL
    )


def model_name() -> str:
    return _setting("pollinations_image_model", DEFAULT_MODEL) or DEFAULT_MODEL


def is_enabled(settings: Optional[dict] = None) -> bool:
    """Pollinations needs no key — the source is always usable."""
    return True


def dimensions_for_aspect(aspect_hint: str) -> tuple:
    """Width/height for an aspect hint ("9:16", "16:9", "1:1")."""
    if aspect_hint == "16:9":
        return (1280, 720)
    if aspect_hint == "1:1":
        return (1024, 1024)
    return (720, 1280)


def generate_image(prompt: str, *, aspect_hint: str = "") -> Optional[bytes]:
    """Generate one image for ``prompt``; returns image bytes or ``None``.

    Follows the same per-image failure convention as the paid providers:
    unrecoverable failures return ``None`` so the caller skips the keyword
    and continues the task.
    """
    import time

    width, height = dimensions_for_aspect(aspect_hint)
    url = f"{base_url()}/prompt/{quote(prompt, safe='')}"
    params = {
        "width": width,
        "height": height,
        "model": model_name(),
        "seed": random.randint(1, 2_000_000_000),
        "nologo": "true",
    }
    key = api_key()
    if key:
        params["token"] = key

    last_detail = ""
    # A registered token lifts the free-IP pacing, so 402 retries only make
    # sense for anonymous calls; with a token a 402 is a real rejection.
    free_limit_attempts = 0 if api_key() else FREE_LIMIT_MAX_ATTEMPTS
    attempts_left = MAX_ATTEMPTS
    while attempts_left > 0 or free_limit_attempts > 0:
        attempt_kind = "request"
        try:
            response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        except requests.RequestException as e:
            last_detail = f"network error: {e}"
            attempts_left -= 1
            logger.warning(
                f"pollinations image request failed "
                f"({attempts_left} retries left): {e}"
            )
            time.sleep(RETRY_SLEEP_SECONDS)
            continue

        status = response.status_code
        content_type = (response.headers.get("content-type") or "").lower()
        if status == 200 and content_type.startswith("image/") and response.content:
            return response.content

        last_detail = f"HTTP {status}: {response.text[:300]}"
        if status == FREE_LIMIT_STATUS and "FREE_LIMIT" in response.text:
            if free_limit_attempts <= 0:
                break
            free_limit_attempts -= 1
            attempt_kind = f"free-limit wait #{FREE_LIMIT_MAX_ATTEMPTS - free_limit_attempts}"
            logger.info(
                f"pollinations free tier pacing (1 image / 45 s per IP); "
                f"retry {FREE_LIMIT_MAX_ATTEMPTS - free_limit_attempts}/"
                f"{FREE_LIMIT_MAX_ATTEMPTS} after {FREE_LIMIT_SLEEP_SECONDS}s"
            )
            time.sleep(FREE_LIMIT_SLEEP_SECONDS)
            continue

        logger.warning(
            f"pollinations image generation failed: {last_detail}"
        )
        if status not in RETRYABLE_STATUS_CODES:
            break
        attempts_left -= 1
        logger.info(f"retrying ({attempts_left} retries left, {attempt_kind})")
        time.sleep(RETRY_SLEEP_SECONDS)

    if last_detail:
        logger.error(f"pollinations image generation gave up: {last_detail}")
    return None
