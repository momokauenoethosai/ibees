from pathlib import Path
from ..utils_gemini import extract_with_prompt
CATEGORY = "outline"
PROMPT = r"""
Output one JSON:
{
  "summary": "concise phrase for face outline",
  "attributes": {
    "shape":{"value":"oval|round|square|heart|diamond|oblong|triangle|unclear","confidence":0.0-1.0},
    "jawline":{"value":"soft|defined|sharp|wide|narrow|unclear","confidence":0.0-1.0},
    "chin":{"value":"rounded|pointed|square|unclear","confidence":0.0-1.0}
  },
  "tags":["up to 10 tags"]
}
"""
def extract(img_path: Path):
    tags, summary, raw = extract_with_prompt(img_path, PROMPT)
    phrase = (summary + ", " + ", ".join(tags)).strip(", ")
    return tags, summary, raw, phrase
