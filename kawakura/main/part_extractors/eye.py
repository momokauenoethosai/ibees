# part_extractors/eye.py
from pathlib import Path
from ..utils_gemini import extract_with_prompt
CATEGORY = "eye"

PROMPT = r"""
Output one JSON object only (no extra text):
{
  "summary": "one concise english phrase for the eyes",
  "attributes": {
    "shape": {"value":"almond|round|droopy|monolid|hooded|upturned|downturned|deep-set|wide-set|close-set|unclear","confidence":0.0-1.0},
    "size": {"value":"small|medium|large","confidence":0.0-1.0},
    "tilt": {"value":"neutral|slight-up|slight-down|strong-up|strong-down","confidence":0.0-1.0},
    "openness": {"value":"narrow|medium|wide","confidence":0.0-1.0},
    "lash": {"value":"short|medium|long|dense|sparse|curled|unclear","confidence":0.0-1.0},
    "brow_gap": {"value":"small|medium|large|unclear","confidence":0.0-1.0},
    "iris_visible": {"value":"low|medium|high","confidence":0.0-1.0}
  },
  "tags": ["up to 10 tags (e.g., almond, large, slight-up)"]
}
"""

def extract(img_path: Path):
    tags, summary, raw = extract_with_prompt(img_path, PROMPT)
    phrase = (summary + ", " + ", ".join(tags)).strip(", ")
    return tags, summary, raw, phrase
