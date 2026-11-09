# -*- coding: utf-8 -*-
"""One-shot helper: full video generation styled after the analyzed YouTube reference."""
import sys
import uuid

sys.path.insert(0, ".")

from app.models.schema import VideoParams  # noqa: E402
from app.services import reference_style, task  # noqa: E402
from app.utils import utils  # noqa: E402

profile = reference_style.load_profile() or {}
style_prompt = reference_style.build_style_prompt(profile)
print("=== STYLE PROMPT ===", flush=True)
print(style_prompt, flush=True)

params = VideoParams(video_subject="")
params.video_subject = (
    "Как перестать откладывать и начать действовать: "
    "психология успеха без мотивационной воды"
)
params.video_aspect = "9:16"
params.video_language = "ru"
params.video_source = "pollinations_image"
params.image_motion = "zoom"
params.voice_name = "gemini:Charon-Male"
params.voice_volume = 1.0
params.bgm_type = "random"
params.bgm_volume = 0.2
params.subtitle_enabled = True
params.font_name = "MicrosoftYaHeiBold.ttc"
params.font_size = 60
params.paragraph_number = 1
params.video_script_prompt = (
    "Напиши короткий сценарий озвучки (60-90 секунд). " + style_prompt
).strip()

# Reuse the already-approved script and give concrete, visual search terms:
# abstract terms ("self discipline") make the image model produce random
# portraits, so each term describes a concrete filmable scene.
import json, glob  # noqa: E402

previous = glob.glob("storage/tasks/ccece6e8*/script.json")
if previous:
    with open(previous[0], encoding="utf-8") as handle:
        params.video_script = json.load(handle)["script"]
params.video_terms = [
    "man sitting alone at a desk in a dark room at night, single lamp, writing in a notebook",
    "tired man staring at a glowing laptop screen in bed at midnight, blue light",
    "silhouette of a man standing up from a chair in a dark room, dramatic window light",
    "man lacing running shoes at dawn in a dim hallway",
    "man running up concrete stairs at sunrise, low angle, cinematic haze",
    "empty city street at night, a man walking alone under a streetlight, fog",
    "close-up of hands closing a laptop decisively, dark office",
    "man looking out a rain-covered window at the city, back view, moody",
    "sunrise light breaking through curtains onto a cluttered desk with a notebook",
    "man writing a checklist on paper, dark moody office, warm lamp",
]

# Reference pacing: apply the analyzed shot length only when it stays in
# the shorts range — this reference is a static interview (one 194s shot),
# so keep the default clip duration instead of freezing one image for 30s.
suggested = reference_style.suggested_clip_duration(profile)
if suggested and suggested <= 8:
    params.video_clip_duration = suggested
    print(f"=== clip duration from reference: {suggested}s ===", flush=True)
else:
    params.video_clip_duration = 4
    print("=== reference is static; clip duration kept at 4s ===", flush=True)

task_id = uuid.uuid4().hex
print(f"=== task_id: {task_id} ===", flush=True)
task.start(task_id, params)
print("=== DONE ===", flush=True)
final_dir = utils.task_dir(task_id)
print(f"final dir: {final_dir}", flush=True)
