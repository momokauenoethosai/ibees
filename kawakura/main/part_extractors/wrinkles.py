# kawakura/main/part_extractors/wrinkles.py
from pathlib import Path
from ..utils_gemini import extract_with_prompt

CATEGORY = "wrinkles"

PROMPT = r"""
Output one JSON object only:
{
  "summary": "concise phrase for wrinkles/fine lines if any, else 'none'",
  "attributes": {
    "overall":{"value":"none|few|moderate|many|unclear","confidence":0.0-1.0},
    "forehead":{"value":"none|fine|moderate|deep|unclear","confidence":0.0-1.0},
    "crow_feet":{"value":"none|fine|moderate|deep|unclear","confidence":0.0-1.0},
    "nasolabial":{"value":"none|fine|moderate|deep|unclear","confidence":0.0-1.0}
  },
  "tags": ["up to 10 short tags"]
}
"""

def extract(img_path: Path):
    tags, summary, raw = extract_with_prompt(img_path, PROMPT)
    phrase = (summary + ", " + ", ".join(tags)).strip(", ")
    return tags, summary, raw, phrase
