from pathlib import Path
from ..utils_gemini import extract_with_prompt
CATEGORY = "ear"
PROMPT = r"""
Output one JSON:
{
  "summary": "concise phrase for ears",
  "attributes": {
    "size":{"value":"small|medium|large","confidence":0.0-1.0},
    "lobe":{"value":"attached|detached|unclear","confidence":0.0-1.0},
    "protrusion":{"value":"low|medium|high","confidence":0.0-1.0},
    "piercings":{"value":"none|single|multiple|unclear","confidence":0.0-1.0}
  },
  "tags":["up to 10 tags"]
}
"""
def extract(img_path: Path):
    tags, summary, raw = extract_with_prompt(img_path, PROMPT)
    phrase = (summary + ", " + ", ".join(tags)).strip(", ")
    return tags, summary, raw, phrase
