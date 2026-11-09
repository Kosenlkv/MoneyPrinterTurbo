# -*- coding: utf-8 -*-
"""Gemini image generation ("Nano Banana", gemini-2.5-flash-image).

Generates one image per script keyword through the Gemini generateContent
API and returns raw image bytes. Mirrors the failure conventions of the
other paid image providers: configuration errors raise ``GeminiImageError``,
per-image generation failures return ``None`` so the caller can skip the
keyword and continue the task.

Configuration (config.toml, [app] section):
    gemini_api_key        API key from https://aistudio.google.com/apikey
    gemini_base_url       optional override, default
                          https://generativelanguage.googleapis.com
    gemini_image_model    optional override, default gemini-2.5-flash-image
"""

import base64
from typing import Optional

import requests
from loguru import logger

from app.config import config

DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com"
DEFAULT_MODEL = "gemini-2.5-flash-image"
REQUEST_TIMEOUT = (30, 300)
RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
KEY_ERROR_STATUS_CODES = frozenset({400, 401, 403})
MAX_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = (5, 15)


class GeminiImageError(Exception):
    """Configuration or transport error that should abort the task early."""

    def __init__(self, message: str, detail: str = ""):
        super().__init__(message)
        self.message = message
        self.detail = detail


def _setting(key: str, default: str = "") -> str:
    return str(config.app.get(key, default) or default).strip()


def api_key() -> str:
    # A dedicated image key wins; otherwise reuse the LLM provider's key —
    # one Google AI Studio key unlocks both text and image generation.
    return _setting("gemini_image_api_key") or _setting("gemini_api_key")


def base_url() -> str:
    return _setting("gemini_image_base_url", DEFAULT_BASE_URL).rstrip("/") or DEFAULT_BASE_URL


def model_name() -> str:
    return _setting("gemini_image_model", DEFAULT_MODEL) or DEFAULT_MODEL


def is_enabled(settings: Optional[dict] = None) -> bool:
    """Minimal configuration check used by task preflight and the WebUI."""
    settings = config.app if settings is None else settings
    return bool(
        str(settings.get("gemini_image_api_key", "") or "").strip()
        or str(settings.get("gemini_api_key", "") or "").strip()
    )


def _extract_error_message(body) -> str:
    try:
        error = body.get("error") if isinstance(body, dict) else None
        if isinstance(error, dict):
            return str(error.get("message") or "")[:500]
    except Exception:
        pass
    return ""


def _extract_image_bytes(body) -> Optional[bytes]:
    try:
        candidates = body.get("candidates") or []
        for candidate in candidates:
            content = candidate.get("content") or {}
            for part in content.get("parts") or []:
                inline = part.get("inlineData") or part.get("inline_data")
                if not isinstance(inline, dict):
                    continue
                data = inline.get("data")
                if data:
                    return base64.b64decode(data)
    except Exception as e:
        logger.warning(f"failed to parse gemini image response: {e}")
    return None


def generate_image(prompt: str, *, aspect_hint: str = "") -> Optional[bytes]:
    """Generate one image for ``prompt``; returns image bytes or ``None``.

    ``aspect_hint`` (e.g. "9:16") is folded into the prompt text: the
    gemini-2.5-flash-image model follows natural-language aspect guidance.
    """
    key = api_key()
    if not key:
        raise GeminiImageError(
            "gemini_image_api_key_missing",
            "set gemini_image_api_key or gemini_api_key in config.toml ([app] section)",
        )

    full_prompt = prompt
    if aspect_hint:
        full_prompt = f"{prompt}\nAspect ratio: {aspect_hint}."

    url = f"{base_url()}/v1beta/models/{model_name()}:generateContent"
    payload = {
        "contents": [{"parts": [{"text": full_prompt}]}],
        "generationConfig": {"responseModalities": ["IMAGE"]},
    }
    headers = {"x-goog-api-key": key, "Content-Type": "application/json"}

    last_detail = ""
    for attempt in range(MAX_ATTEMPTS):
        try:
            response = requests.post(
                url, json=payload, headers=headers, timeout=REQUEST_TIMEOUT
            )
        except requests.RequestException as e:
            last_detail = f"network error: {e}"
            logger.warning(f"gemini image request failed (attempt {attempt + 1}): {e}")
            continue

        status = response.status_code
        try:
            body = response.json()
        except ValueError:
            body = None

        if status == 200 and isinstance(body, dict):
            image_bytes = _extract_image_bytes(body)
            if image_bytes:
                return image_bytes
            last_detail = "response contained no image data"
            logger.warning(
                f"gemini image response without image (attempt {attempt + 1}): "
                f"{str(body)[:300]}"
            )
            continue

        detail = _extract_error_message(body) if body else response.text[:300]
        last_detail = f"HTTP {status}: {detail}"
        logger.warning(
            f"gemini image generation failed (attempt {attempt + 1}): {last_detail}"
        )
        if status in KEY_ERROR_STATUS_CODES:
            # 400 usually means an unknown model id or malformed request; 401/403
            # means the key is rejected. Retrying cannot fix either.
            break
        if status not in RETRYABLE_STATUS_CODES:
            break

    if last_detail:
        logger.error(f"gemini image generation gave up: {last_detail}")
    return None
