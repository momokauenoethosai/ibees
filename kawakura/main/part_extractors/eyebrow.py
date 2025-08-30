from pathlib import Path
from ..utils_gemini import extract_with_prompt
CATEGORY = "eyebrow"
PROMPT = r"""
Output one JSON:
{
  "summary": "concise phrase for eyebrows",
  "attributes": {
    "thickness":{"value":"thin|medium|thick","confidence":0.0-1.0},
    "shape":{"value":"straight|slight-arch|high-arch|rounded|angled|flat|unclear","confidence":0.0-1.0},
    "length":{"value":"short|medium|long","confidence":0.0-1.0},
    "density":{"value":"sparse|medium|dense","confidence":0.0-1.0},
    "tail":{"value":"short|medium|long|unclear","confidence":0.0-1.0}
  },
  "tags":["up to 10 tags"]
}
"""
def extract(img_path: Path):
    tags, summary, raw = extract_with_prompt(img_path, PROMPT)
    phrase = (summary + ", " + ", ".join(tags)).strip(", ")
    return tags, summary, raw, phrase
