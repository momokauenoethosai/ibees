from pathlib import Path
from ..utils_gemini import extract_with_prompt
CATEGORY = "nose"
PROMPT = r"""
Output one JSON:
{
  "summary": "concise phrase for nose",
  "attributes": {
    "bridge":{"value":"low|medium|high|unclear","confidence":0.0-1.0},
    "width":{"value":"narrow|medium|wide","confidence":0.0-1.0},
    "tip":{"value":"rounded|pointed|bulbous|upturned|downturned|unclear","confidence":0.0-1.0},
    "nostrils":{"value":"narrow|medium|wide|flared|unclear","confidence":0.0-1.0},
    "length":{"value":"short|medium|long","confidence":0.0-1.0}
  },
  "tags":["up to 10 tags"]
}
"""
def extract(img_path: Path):
    tags, summary, raw = extract_with_prompt(img_path, PROMPT)
    phrase = (summary + ", " + ", ".join(tags)).strip(", ")
    return tags, summary, raw, phrase
