# kawakura/main/part_extractors/glasses.py
from pathlib import Path
from ..utils_gemini import extract_with_prompt

CATEGORY = "glasses"

PROMPT = r"""
Output one JSON object only (no extra text):
{
  "summary": "concise phrase for glasses/sunglasses if present, else 'no glasses'",
  "attributes": {
    "type":{"value":"none|glasses|sunglasses|reading|sports|unclear","confidence":0.0-1.0},
    "frame":{"value":"full-rim|half-rim|rimless|round|square|aviator|cat-eye|wayfarer|unclear","confidence":0.0-1.0},
    "thickness":{"value":"thin|medium|thick|unclear","confidence":0.0-1.0},
    "color":{"value":"black|silver|gold|transparent|brown|tortoise|other|unclear","confidence":0.0-1.0},
    "lens_tint":{"value":"none|light|dark|mirror|colored|unclear","confidence":0.0-1.0}
  },
  "tags": ["up to 10 short tags"]
}
"""

def extract(img_path: Path):
    tags, summary, raw = extract_with_prompt(img_path, PROMPT)
    phrase = (summary + ", " + ", ".join(tags)).strip(", ")
    return tags, summary, raw, phrase
