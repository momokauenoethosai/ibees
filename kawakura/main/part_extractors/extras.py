# kawakura/main/part_extractors/extras.py
from pathlib import Path
from ..utils_gemini import extract_with_prompt

CATEGORY = "extras"

PROMPT = r"""
Output one JSON object only:
{
  "summary": "concise phrase for extra facial features if any (e.g., freckles, moles, beauty marks, scars), else 'none'",
  "attributes": {
    "freckles":{"value":"none|few|many|dense|unclear","confidence":0.0-1.0},
    "moles":{"value":"none|few|noticeable|unclear","confidence":0.0-1.0},
    "scars":{"value":"none|small|medium|large|unclear","confidence":0.0-1.0},
    "dimples":{"value":"none|present|unclear","confidence":0.0-1.0}
  },
  "tags": ["up to 10 short tags"]
}
"""

def extract(img_path: Path):
    tags, summary, raw = extract_with_prompt(img_path, PROMPT)
    phrase = (summary + ", " + ", ".join(tags)).strip(", ")
    return tags, summary, raw, phrase
