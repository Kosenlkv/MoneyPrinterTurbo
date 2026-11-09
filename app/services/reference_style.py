# -*- coding: utf-8 -*-
"""Reference video style analysis.

Analyzes an uploaded competitor video and builds a reusable style profile:
pacing (shot lengths from ffmpeg scene detection), speech (faster-whisper
transcript, words per second, hook), and look (dominant color palette,
brightness/saturation from sampled frames). An optional LLM pass condenses
the numbers into a short style guide that is injected into the script
generation prompt so new videos copy the reference's style and mood.

The profile is persisted under ``storage/reference/profile.json`` so the
WebUI can reload it across reruns without re-analyzing the video.
"""

import json
import os
import re
import subprocess
import tempfile
from typing import Any, Callable, Dict, List, Optional

from loguru import logger

from app.config import config
from app.utils import utils

ProgressCallback = Optional[Callable[[str, int, int], None]]

_PROFILE_DIR = "reference"
_PROFILE_FILE = "profile.json"
_SCENE_THRESHOLD = 0.3
_MAX_FRAMES = 5
_MAX_TRANSCRIPT_CHARS = 3000
_MAX_SUMMARY_TRANSCRIPT_CHARS = 1500


def _profile_path() -> str:
    directory = utils.storage_dir(_PROFILE_DIR, create=True)
    return os.path.join(directory, _PROFILE_FILE)


def save_uploaded_reference(file_name: str, data: bytes) -> str:
    """Persist an uploaded reference video under storage/reference/."""
    if not data:
        raise ValueError("empty reference upload")
    directory = utils.storage_dir(_PROFILE_DIR, create=True)
    _, ext = os.path.splitext(file_name or "")
    ext = ext.lower() if ext.lower() in {".mp4", ".mov", ".webm", ".mkv"} else ".mp4"
    path = os.path.join(directory, f"reference{ext}")
    with open(path, "wb") as output:
        output.write(data)
    # A new reference invalidates the previous analysis.
    profile = _profile_path()
    if os.path.isfile(profile):
        try:
            os.remove(profile)
        except OSError:
            pass
    return path


_YT_URL_RE = re.compile(
    r"^(https?://)?(www\.|m\.)?(youtube\.com/(watch|shorts)|youtu\.be/)", re.I
)
_MAX_REFERENCE_DURATION_SEC = 1200  # 20 min — analysis cost grows with length


def is_youtube_url(value: str) -> bool:
    return bool(_YT_URL_RE.match((value or "").strip()))


def _ytdlp_ffmpeg_dir() -> str:
    """Directory containing an ``ffmpeg.exe`` yt-dlp can find.

    The bundled imageio-ffmpeg binary carries a versioned name that yt-dlp's
    executable probe does not recognize, so expose it under the canonical
    name via a hardlink (no admin rights needed, same volume).
    """
    binary = utils.get_ffmpeg_binary()
    directory = os.path.dirname(binary)
    canonical = os.path.join(directory, "ffmpeg.exe")
    if not os.path.isfile(canonical):
        try:
            os.link(binary, canonical)
        except OSError:
            import shutil

            shutil.copy2(binary, canonical)
    return directory


def download_youtube_reference(url: str, progress: ProgressCallback = None) -> str:
    """Download a YouTube video as the style reference; returns the file path.

    Uses yt-dlp (already a project dependency for the niche scout). Picks a
    ≤720p mp4 to keep downloads and scene detection fast. A new download
    invalidates the previous analysis, same as a new upload.
    """
    import yt_dlp

    url = (url or "").strip()
    if not is_youtube_url(url):
        raise ValueError(f"not a YouTube video URL: {url!r}")
    directory = utils.storage_dir(_PROFILE_DIR, create=True)
    target = os.path.join(directory, "reference.mp4")

    def _hook(d):
        if not progress or d.get("status") != "downloading":
            return
        try:
            downloaded = d.get("downloaded_bytes") or 0
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            progress("download", int(downloaded // 1024), max(int(total // 1024), 1))
        except Exception:
            pass

    # Clean previous downloads so the post-download glob is unambiguous.
    for stale in os.listdir(directory):
        if stale.startswith("reference."):
            try:
                os.remove(os.path.join(directory, stale))
            except OSError:
                pass

    opts = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        # Any ≤720p variant wins; fall back to whatever the video has.
        # Separate video+audio tracks merge into mp4 via ffmpeg.
        "format": "best[height<=720]/bestvideo[height<=720]+bestaudio/best",
        "merge_output_format": "mp4",
        "outtmpl": os.path.join(directory, "reference.%(ext)s"),
        "overwrites": True,
        "match_filter": yt_dlp.utils.match_filter_func(
            f"duration <= {_MAX_REFERENCE_DURATION_SEC} & !is_live & !is_upcoming"
        ),
        "progress_hooks": [_hook],
        # The project bundles ffmpeg (imageio-ffmpeg), it is not on PATH.
        "ffmpeg_location": _ytdlp_ffmpeg_dir(),
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])
    if not os.path.isfile(target):
        candidates = [
            os.path.join(directory, name)
            for name in os.listdir(directory)
            if name.startswith("reference.")
            and os.path.splitext(name)[1].lower() in {".mp4", ".mov", ".webm", ".mkv"}
        ]
        if not candidates:
            raise FileNotFoundError("yt-dlp finished but produced no reference file")
        target = candidates[0]
    # A new reference invalidates the previous analysis.
    profile = _profile_path()
    if os.path.isfile(profile):
        try:
            os.remove(profile)
        except OSError:
            pass
    return target


def load_profile() -> Optional[Dict[str, Any]]:
    path = _profile_path()
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            profile = json.load(handle)
        return profile if isinstance(profile, dict) and profile.get("video_path") else None
    except Exception as e:
        logger.warning(f"failed to load reference style profile: {e}")
        return None


def save_profile(profile: Dict[str, Any]) -> None:
    try:
        with open(_profile_path(), "w", encoding="utf-8") as handle:
            json.dump(profile, handle, ensure_ascii=False, indent=2)
    except OSError as e:
        logger.warning(f"failed to persist reference style profile: {e}")


def clear_profile() -> None:
    profile = _profile_path()
    if os.path.isfile(profile):
        try:
            os.remove(profile)
        except OSError:
            pass


def _run_ffmpeg(args: List[str], timeout: int = 600) -> subprocess.CompletedProcess:
    binary = utils.get_ffmpeg_binary()
    return subprocess.run(
        [binary, *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )


def _probe_duration(video_path: str) -> float:
    try:
        from moviepy.video.io.VideoFileClip import VideoFileClip

        with VideoFileClip(video_path) as clip:
            return float(clip.duration or 0.0)
    except Exception as e:
        logger.warning(f"reference probe failed: {e}")
        return 0.0


def _detect_shots(video_path: str, duration: float) -> List[float]:
    """Return shot lengths (seconds) via ffmpeg scene-cut detection."""
    result = _run_ffmpeg(
        [
            "-i", video_path,
            "-vf", f"select='gt(scene,{_SCENE_THRESHOLD})',showinfo",
            "-f", "null", "-",
        ],
        timeout=600,
    )
    cuts: List[float] = []
    for line in result.stderr.decode("utf-8", "ignore").splitlines():
        match = re.search(r"pts_time:([\d.]+)", line)
        if match:
            try:
                cuts.append(float(match.group(1)))
            except ValueError:
                continue
    cuts = sorted({round(c, 3) for c in cuts if c > 0.05})
    boundaries = [0.0, *cuts]
    if duration > 0:
        boundaries.append(duration)
    shots = [b - a for a, b in zip(boundaries, boundaries[1:]) if b - a > 0.1]
    return shots


def _extract_audio(video_path: str, output_path: str) -> bool:
    result = _run_ffmpeg(
        ["-y", "-i", video_path, "-vn", "-ac", "1", "-ar", "16000", output_path],
        timeout=300,
    )
    return result.returncode == 0 and os.path.isfile(output_path)


def _transcribe(video_path: str) -> str:
    from app.services import subtitle
    from app.utils.subtitle_reader import file_to_subtitles

    # Only the hook and speech rate are needed, so transcribe with the small
    # whisper model: large-v3 on CPU runs several times slower than realtime.
    # The small model is unloaded afterwards so real subtitle generation
    # still gets the configured (usually large) model.
    swapped_model = False
    previous_size = subtitle.model_size
    if subtitle.model is None and previous_size != "small":
        subtitle.model_size = "small"
        swapped_model = True

    with tempfile.TemporaryDirectory(prefix="reference-audio-") as temp_dir:
        audio_file = os.path.join(temp_dir, "reference.wav")
        subtitle_file = os.path.join(temp_dir, "reference.srt")
        if not _extract_audio(video_path, audio_file):
            return ""
        try:
            subtitle.create(audio_file, subtitle_file, log_details=False)
        except Exception as e:
            logger.warning(f"reference transcription failed: {e}")
            return ""
        finally:
            if swapped_model:
                subtitle.model_size = previous_size
                subtitle.model = None
        if not os.path.isfile(subtitle_file):
            return ""
        items = file_to_subtitles(subtitle_file)
        return " ".join(item[2].strip() for item in items if item[2].strip())


def _frame_stats(video_path: str, duration: float) -> Dict[str, Any]:
    """Sample frames; return palette hex colors, brightness and saturation."""
    stats: Dict[str, Any] = {"palette": [], "brightness": None, "saturation": None}
    if duration <= 0:
        return stats
    try:
        from moviepy.video.io.VideoFileClip import VideoFileClip
        from PIL import Image

        brightness_values: List[float] = []
        saturation_values: List[float] = []
        palette_votes: Dict[tuple, int] = {}
        times = [
            duration * (i + 1) / (_MAX_FRAMES + 1) for i in range(_MAX_FRAMES)
        ]
        with VideoFileClip(video_path) as clip:
            for t in times:
                try:
                    frame = clip.get_frame(min(t, max(0.0, clip.duration - 0.05)))
                except Exception:
                    continue
                image = Image.fromarray(frame).resize((96, 96))
                gray = image.convert("L")
                brightness_values.append(sum(gray.getdata()) / (96 * 96) / 255.0)
                hsv = image.convert("HSV")
                saturation_values.append(
                    sum(px[1] for px in hsv.getdata()) / (96 * 96) / 255.0
                )
                quantized = image.quantize(colors=5).convert("RGB")
                for count, color in quantized.getcolors(96 * 96) or []:
                    # Snap to a coarse grid so near-identical shades merge.
                    key = tuple((channel // 24) * 24 for channel in color)
                    palette_votes[key] = palette_votes.get(key, 0) + count
        if brightness_values:
            stats["brightness"] = round(sum(brightness_values) / len(brightness_values), 3)
        if saturation_values:
            stats["saturation"] = round(sum(saturation_values) / len(saturation_values), 3)
        top = sorted(palette_votes.items(), key=lambda item: item[1], reverse=True)[:5]
        stats["palette"] = ["#%02x%02x%02x" % color for color, _ in top]
    except Exception as e:
        logger.warning(f"reference frame analysis failed: {e}")
    return stats


def _look_description(profile: Dict[str, Any]) -> str:
    brightness = profile.get("brightness")
    saturation = profile.get("saturation")
    parts: List[str] = []
    if brightness is not None:
        if brightness < 0.35:
            parts.append("dark, moody lighting")
        elif brightness > 0.65:
            parts.append("bright, airy lighting")
        else:
            parts.append("balanced, neutral lighting")
    if saturation is not None:
        if saturation < 0.25:
            parts.append("muted, desaturated colors")
        elif saturation > 0.55:
            parts.append("vivid, saturated colors")
        else:
            parts.append("natural color rendition")
    return ", ".join(parts)


def _llm_style_summary(profile: Dict[str, Any], language: str) -> str:
    from app.services import llm

    transcript = (profile.get("transcript") or "")[:_MAX_SUMMARY_TRANSCRIPT_CHARS]
    prompt = (
        "Ты — аналитик видеоконтента. По данным конкурентного видео составь краткий "
        "гид по стилю (5-7 пунктов): настроение, темп монтажа, структура хука, тональность "
        "подачи, работа со зрителем, визуальный стиль. Пиши кратко, по делу, без воды. "
        f"Язык ответа: {language or 'ru'}.\n\n"
        f"Длительность: {profile.get('duration_sec')} c; "
        f"средний план: {profile.get('avg_shot_sec')} c; "
        f"планов в минуту: {profile.get('shots_per_min')}; "
        f"темп речи: {profile.get('words_per_sec')} слов/с; "
        f"палитра: {', '.join(profile.get('palette') or [])}; "
        f"свет: {_look_description(profile) or 'н/д'}.\n\n"
        f"Транскрипт:\n{transcript}"
    )
    try:
        text = llm._generate_response(prompt=prompt)
        return (text or "").strip()[:2000]
    except Exception as e:
        logger.warning(f"reference style LLM summary failed: {e}")
        return ""


def analyze_reference(video_path: str, language: str = "", progress: ProgressCallback = None) -> Dict[str, Any]:
    """Analyze ``video_path`` and return (and persist) the style profile."""
    if not os.path.isfile(video_path):
        raise FileNotFoundError(video_path)

    def report(stage: str, done: int, total: int) -> None:
        if progress:
            try:
                progress(stage, done, total)
            except Exception:
                pass

    total_steps = 4
    report("probe", 0, total_steps)
    duration = _probe_duration(video_path)

    report("shots", 1, total_steps)
    shots = _detect_shots(video_path, duration)
    avg_shot = round(sum(shots) / len(shots), 2) if shots else 0.0
    shots_per_min = round(60.0 * len(shots) / duration, 1) if duration > 0 else 0.0

    report("audio", 2, total_steps)
    transcript = _transcribe(video_path)[:_MAX_TRANSCRIPT_CHARS]
    words = len(transcript.split()) if transcript else 0
    words_per_sec = round(words / duration, 2) if duration > 0 and words else 0.0
    hook = " ".join(transcript.split()[:25]) if transcript else ""

    report("frames", 3, total_steps)
    frame_stats = _frame_stats(video_path, duration)

    profile: Dict[str, Any] = {
        "video_path": video_path,
        "duration_sec": round(duration, 1),
        "shot_count": len(shots),
        "avg_shot_sec": avg_shot,
        "shots_per_min": shots_per_min,
        "transcript": transcript,
        "words_per_sec": words_per_sec,
        "hook": hook,
        "palette": frame_stats.get("palette") or [],
        "brightness": frame_stats.get("brightness"),
        "saturation": frame_stats.get("saturation"),
        "look": _look_description(frame_stats),
        "language": language or "ru",
        "style_summary": "",
    }
    profile["style_summary"] = _llm_style_summary(profile, language)
    save_profile(profile)
    report("done", total_steps, total_steps)
    return profile


def suggested_clip_duration(profile: Dict[str, Any]) -> Optional[int]:
    """Round the reference's average shot length to a sane clip duration."""
    try:
        value = float(profile.get("avg_shot_sec") or 0)
    except (TypeError, ValueError):
        return None
    if value <= 0:
        return None
    return max(1, min(30, int(round(value))))


def build_style_prompt(profile: Dict[str, Any]) -> str:
    """Render the profile as an instruction block for script generation."""
    if not profile:
        return ""
    lines = [
        "STYLE REFERENCE (analyze of a competitor video — copy its style, mood "
        "and pacing, but write ORIGINAL content on the given subject; never "
        "reuse its wording):"
    ]
    if profile.get("style_summary"):
        lines.append(f"- Style guide: {profile['style_summary']}")
    if profile.get("look"):
        lines.append(f"- Look: {profile['look']}")
    palette = profile.get("palette") or []
    if palette:
        lines.append(f"- Color palette: {', '.join(palette)}")
    if profile.get("avg_shot_sec"):
        lines.append(
            f"- Editing pace: average shot {profile['avg_shot_sec']}s, "
            f"~{profile.get('shots_per_min')} cuts per minute"
        )
    if profile.get("words_per_sec"):
        lines.append(
            f"- Speech rate: ~{profile['words_per_sec']} words per second; "
            "match this narration density"
        )
    if profile.get("hook"):
        lines.append(
            f"- Hook example (first seconds of the reference, for structure "
            f"only — do NOT copy the words): \"{profile['hook']}\""
        )
    return "\n".join(lines)
