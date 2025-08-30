from pathlib import Path
from ..utils_gemini import extract_with_prompt
CATEGORY = "beard"
PROMPT = r"""
Output one JSON:
{
  "summary": "concise phrase for facial hair",
  "attributes": {
    "type":{"value":"none|stubble|mustache|goatee|full|chin-strap|soul-patch|unclear","confidence":0.0-1.0},
    "density":{"value":"sparse|medium|dense|patchy","confidence":0.0-1.0},
    "length":{"value":"short|medium|long","confidence":0.0-1.0}
  },
  "tags":["up to 10 tags"]
}
"""
def extract(img_path: Path):
    tags, summary, raw = extract_with_prompt(img_path, PROMPT)
    phrase = (summary + ", " + ", ".join(tags)).strip(", ")
    return tags, summary, raw, phrase
