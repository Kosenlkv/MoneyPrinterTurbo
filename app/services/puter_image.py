# -*- coding: utf-8 -*-
"""Puter image generation ("Nano Banana" through the Puter AI gateway).

Generates one image per script keyword through Puter's driver API
(``POST https://api.puter.com/drivers/call``, interface
``puter-image-generation``, method ``generate``) and returns raw image
bytes. Puter routes the request to Google image models such as
``google/gemini-2.5-flash-image`` (Nano Banana), so generation works even
where Google's own free tier allows no image generations. Usage is billed
to the Puter account the auth token belongs to (~$0.039 per image for
Nano Banana).

Mirrors the failure conventions of the other paid image providers:
configuration errors raise ``PuterImageError``, per-image generation
failures return ``None`` so the caller can skip the keyword and continue
the task.

Configuration (config.toml, [app] section):
    puter_api_key        auth token from https://puter.com/dashboard
                         ("Create token")
    puter_base_url       optional override, default https://api.puter.com
    puter_image_model    optional override, default
                         google/gemini-2.5-flash-image
"""

import base64
import binascii
from typing import Optional

import requests
from loguru import logger

from app.config import config

DEFAULT_BASE_URL = "https://api.puter.com"
DEFAULT_MODEL = "google/gemini-2.5-flash-image"
REQUEST_TIMEOUT = (30, 300)
RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
KEY_ERROR_STATUS_CODES = frozenset({400, 401, 403})
MAX_ATTEMPTS = 3


class PuterImageError(Exception):
    """Configuration or transport error that should abort the task early."""

    def __init__(self, message: str, detail: str = ""):
        super().__init__(message)
        self.message = message
        self.detail = detail


def _setting(key: str, default: str = "") -> str:
    return str(config.app.get(key, default) or default).strip()


def api_key() -> str:
    return _setting("puter_api_key")


def base_url() -> str:
    return _setting("puter_base_url", DEFAULT_BASE_URL).rstrip("/") or DEFAULT_BASE_URL


def model_name() -> str:
    return _setting("puter_image_model", DEFAULT_MODEL) or DEFAULT_MODEL


def is_enabled(settings: Optional[dict] = None) -> bool:
    """Minimal configuration check used by task preflight and the WebUI."""
    settings = config.app if settings is None else settings
    return bool(str(settings.get("puter_api_key", "") or "").strip())


def _extract_error_message(body) -> str:
    try:
        if isinstance(body, dict):
            error = body.get("error")
            if isinstance(error, dict):
                return str(error.get("message") or "")[:500]
            if isinstance(error, str):
                return error[:500]
            for key in ("message", "msg"):
                if body.get(key):
                    return str(body[key])[:500]
    except Exception:
        pass
    return ""


def _decode_base64_image(data: str) -> Optional[bytes]:
    """Decode a base64 string or ``data:image/...;base64,...`` URI."""
    if not isinstance(data, str) or not data:
        return None
    if data.startswith("data:"):
        _, _, data = data.partition(",")
    try:
        return base64.b64decode(data)
    except (ValueError, binascii.Error):
        return None


def _extract_image_bytes(body) -> Optional[bytes]:
    """Pull image bytes out of a Puter driver-call JSON envelope.

    Observed shapes: ``{"success": true, "result": <data-uri>}``,
    ``{"result": {"image": <b64>}}`` and OpenAI-style
    ``{"result": {"data": [{"b64_json": ...}]}}``. Raw ``image/*``
    responses never reach this parser — they are handled by the caller.
    """
    try:
        if not isinstance(body, dict):
            return None
        result = body.get("result", body)
        if isinstance(result, str):
            return _decode_base64_image(result)
        if isinstance(result, dict):
            for key in ("image", "b64_json", "base64", "data"):
                value = result.get(key)
                if isinstance(value, str):
                    decoded = _decode_base64_image(value)
                    if decoded:
                        return decoded
            for item in result.get("data") or []:
                if isinstance(item, dict):
                    decoded = _decode_base64_image(str(item.get("b64_json") or ""))
                    if decoded:
                        return decoded
    except Exception as e:
        logger.warning(f"failed to parse puter image response: {e}")
    return None


def generate_image(prompt: str, *, aspect_hint: str = "") -> Optional[bytes]:
    """Generate one image for ``prompt``; returns image bytes or ``None``.

    ``aspect_hint`` (e.g. "9:16") is folded into the prompt text: the
    Gemini image models behind Puter follow natural-language aspect
    guidance.
    """
    key = api_key()
    if not key:
        raise PuterImageError(
            "puter_api_key_missing",
            "set puter_api_key in config.toml ([app] section); "
            "create a token at https://puter.com/dashboard",
        )

    full_prompt = prompt
    if aspect_hint:
        full_prompt = f"{prompt}\nAspect ratio: {aspect_hint}."

    url = f"{base_url()}/drivers/call"
    payload = {
        "interface": "puter-image-generation",
        "driver": "ai-image",
        "method": "generate",
        "args": {"prompt": full_prompt, "model": model_name()},
        "auth_token": key,
    }
    headers = {
        "Authorization": f"Bearer {key}",
        # Puter's SDK sends exactly this content type for driver calls.
        "Content-Type": "text/plain;actually=json",
    }

    last_detail = ""
    for attempt in range(MAX_ATTEMPTS):
        try:
            response = requests.post(
                url, json=payload, headers=headers, timeout=REQUEST_TIMEOUT
            )
        except requests.RequestException as e:
            last_detail = f"network error: {e}"
            logger.warning(f"puter image request failed (attempt {attempt + 1}): {e}")
            continue

        status = response.status_code
        content_type = (response.headers.get("content-type") or "").lower()

        # A raw image body is a success without a JSON envelope.
        if status == 200 and content_type.startswith("image/") and response.content:
            return response.content

        try:
            body = response.json()
        except ValueError:
            body = None

        if status == 200 and isinstance(body, dict):
            if body.get("success") is False:
                detail = _extract_error_message(body) or "success=false"
                last_detail = f"HTTP 200: {detail}"
                logger.warning(
                    f"puter image generation refused (attempt {attempt + 1}): "
                    f"{last_detail}"
                )
                continue
            image_bytes = _extract_image_bytes(body)
            if image_bytes:
                return image_bytes
            last_detail = "response contained no image data"
            logger.warning(
                f"puter image response without image (attempt {attempt + 1}): "
                f"{str(body)[:300]}"
            )
            continue

        detail = _extract_error_message(body) if body else response.text[:300]
        last_detail = f"HTTP {status}: {detail}"
        logger.warning(
            f"puter image generation failed (attempt {attempt + 1}): {last_detail}"
        )
        if status in KEY_ERROR_STATUS_CODES:
            # 401/403 means the token is rejected; 400 usually means an
            # unknown model id. Retrying cannot fix either.
            break
        if status not in RETRYABLE_STATUS_CODES:
            break

    if last_detail:
        logger.error(f"puter image generation gave up: {last_detail}")
    return None
