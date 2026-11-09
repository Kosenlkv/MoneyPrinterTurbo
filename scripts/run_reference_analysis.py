# -*- coding: utf-8 -*-
"""One-shot helper: analyze the downloaded YouTube reference and dump the profile."""
import json
import sys

sys.path.insert(0, ".")

from app.services import reference_style  # noqa: E402

profile = reference_style.analyze_reference(
    "storage/reference/reference.mp4", language="ru"
)
print(json.dumps(profile, ensure_ascii=False, indent=2), flush=True)
