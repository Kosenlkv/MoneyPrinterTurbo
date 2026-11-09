# -*- coding: utf-8 -*-
"""YouTube niche scout: find rising videos/small channels via yt-dlp (no API key).

Strategy: YouTube search with a server-side upload-date filter (Innertube sp
code) for candidates, then a full yt-dlp extract per candidate to get exact
view_count, upload_date and channel_follower_count. Filters: age <= days,
subs <= max_subscribers, views >= min_views. Score = views_per_day *
view_sub_ratio.
"""

import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from loguru import logger

_YDL_BASE_OPTS = {
    "quiet": True,
    "no_warnings": True,
    "ignoreerrors": True,
    "extract_flat": False,
    "skip_download": True,
    "socket_timeout": 15,
}

# Server-side "upload date" search filters (Innertube sp codes).
_SP_BY_DAYS = ((7, "EgQIAxAB"), (31, "EgQIBBAB"), (365, "EgQIBRAB"))


def _sp_for_days(days: int) -> Optional[str]:
    for limit, code in _SP_BY_DAYS:
        if days <= limit:
            return code
    return None


def _ydl():
    import yt_dlp

    return yt_dlp


def _age_days(upload_date: Optional[str], now_ts: float) -> Optional[int]:
    """upload_date in yt-dlp is 'YYYYMMDD' (UTC-ish)."""
    if not upload_date:
        return None
    try:
        dt = datetime.strptime(upload_date, "%Y%m%d").replace(tzinfo=timezone.utc)
        return max(0, int((datetime.fromtimestamp(now_ts, timezone.utc) - dt).total_seconds() // 86400))
    except Exception:
        return None


def _search_candidates(query: str, days: int) -> List[str]:
    """Return candidate video urls via YouTube search with a server-side
    upload-date filter (sp code). Returns the first result page (~20 items)."""
    ytdlp = _ydl()
    opts = dict(_YDL_BASE_OPTS)
    opts.update({"extract_flat": True, "noplaylist": True})
    url = f"https://www.youtube.com/results?search_query={urllib.parse.quote_plus(query)}"
    sp = _sp_for_days(days)
    if sp:
        url += f"&sp={sp}"
    with ytdlp.YoutubeDL(opts) as ydl:
        try:
            res = ydl.extract_info(url, download=False)
        except Exception as e:
            logger.warning(f"niche_finder search failed for '{query}': {e}")
            return []
    out: List[str] = []
    for entry in (res or {}).get("entries") or []:
        if not entry:
            continue
        vid = entry.get("id") or entry.get("url")
        if vid:
            out.append(vid if str(vid).startswith("http") else f"https://www.youtube.com/watch?v={vid}")
    return out


def _fetch_video_details(url: str, now_ts: float) -> Optional[Dict[str, Any]]:
    """Full extract of one video; returns normalized dict or None."""
    ytdlp = _ydl()
    opts = dict(_YDL_BASE_OPTS)
    opts.update({"noplaylist": True})
    try:
        with ytdlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as e:
        logger.debug(f"niche_finder extract failed for {url}: {e}")
        return None
    if not info or info.get("_type") == "playlist":
        return None
    views = info.get("view_count")
    if views is None:
        return None
    subs = info.get("channel_follower_count")
    duration = info.get("duration") or 0
    # ignore live streams / premieres and stories-like items
    if info.get("live_status") in ("is_live", "is_upcoming"):
        return None
    age = _age_days(info.get("upload_date"), now_ts)
    return {
        "video_id": info.get("id"),
        "title": info.get("title") or "",
        "url": info.get("webpage_url") or url,
        "channel": info.get("channel") or info.get("uploader") or "",
        "channel_url": info.get("channel_url") or info.get("uploader_url") or "",
        "subscribers": subs,
        "views": int(views),
        "age_days": age,
        "upload_date": info.get("upload_date"),
        "duration_sec": int(duration),
        "thumbnail": info.get("thumbnail"),
    }


def _enrich(v: Dict[str, Any]) -> Dict[str, Any]:
    age = v.get("age_days")
    views = v.get("views") or 0
    subs = v.get("subscribers")
    v["views_per_day"] = round(views / max(1, age if age is not None else 1), 1)
    v["view_sub_ratio"] = round(views / max(1, subs), 2) if subs else None
    base = v["views_per_day"] * (v["view_sub_ratio"] if v["view_sub_ratio"] else 0.5)
    v["score"] = round(base, 2)
    return v


def find_rising_videos(
    queries: List[str],
    days: int = 31,
    max_subscribers: int = 100000,
    min_views: int = 1000,
    per_query: int = 8,
    progress_callback: Optional[Callable[[str, int, int, str], None]] = None,
) -> Dict[str, Any]:
    """Scan YouTube for rising videos from small channels.

    progress_callback(stage, done, total, label); stage in {"search", "details"}.
    """
    t0 = time.time()
    now_ts = t0

    # Pass 1: collect candidates via date-filtered search pages
    seen: set = set()
    candidates: List[str] = []
    total_q = len(queries)
    for i, q in enumerate(queries, 1):
        if progress_callback:
            progress_callback("search", i - 1, total_q, q)
        for url in _search_candidates(q, days)[:per_query]:
            if url not in seen:
                seen.add(url)
                candidates.append(url)
        if progress_callback:
            progress_callback("search", i, total_q, q)

    # Pass 2: full details, in parallel
    scanned = 0
    videos: List[Dict[str, Any]] = []
    total = len(candidates)
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(_fetch_video_details, url, now_ts): url for url in candidates}
        for fut in as_completed(futures):
            scanned += 1
            info = None
            try:
                info = fut.result()
            except Exception:
                pass
            if info:
                age = info["age_days"]
                subs = info["subscribers"]
                if (
                    info["views"] >= min_views
                    and age is not None
                    and age <= days
                    and subs is not None
                    and subs <= max_subscribers
                ):
                    videos.append(_enrich(info))
            if progress_callback:
                progress_callback("details", scanned, total, futures[fut])

    videos.sort(key=lambda x: x["score"], reverse=True)
    return {"videos": videos, "scanned": scanned, "elapsed_sec": round(time.time() - t0, 1)}


def suggest_queries(topic: str, language: str = "ru", count: int = 5) -> List[str]:
    """Generate search query variations for a topic via LLM; fallback = raw topic."""
    from app.services import llm

    prompt = (
        f"Ты — исследователь YouTube-ниш. Для темы «{topic}» предложи {count} коротких "
        f"поисковых запросов на языке «{language}», по которым искать набирающие популярность "
        f"видео (подниши, форматы, вопросы аудитории). Ответь только списком, по одному запросу "
        f"на строку, без нумерации и пояснений."
    )
    try:
        text = llm._generate_response(prompt=prompt)
        lines = [ln.strip().lstrip("-•0123456789. ") for ln in (text or "").splitlines()]
        lines = [ln for ln in lines if 2 <= len(ln) <= 120 and not ln.startswith("Error:")]
        if lines:
            return lines[:count]
    except Exception as e:
        logger.warning(f"suggest_queries LLM failed: {e}")
    return [topic]
