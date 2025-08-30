from pathlib import Path
from ..utils_gemini import extract_with_prompt
CATEGORY = "mouse"
PROMPT = r"""
Output one JSON:
{
  "summary": "concise phrase for mouse/lips",
  "attributes": {
    "fullness":{"value":"thin|medium|full","confidence":0.0-1.0},
    "shape":{"value":"heart|bow|straight|downturned|upturned|unclear","confidence":0.0-1.0},
    "width":{"value":"narrow|medium|wide","confidence":0.0-1.0},
    "corners":{"value":"neutral|up|down","confidence":0.0-1.0}
  },
  "tags":["up to 10 tags"]
}
"""
def extract(img_path: Path):
    tags, summary, raw = extract_with_prompt(img_path, PROMPT)
    phrase = (summary + ", " + ", ".join(tags)).strip(", ")
    return tags, summary, raw, phrase
