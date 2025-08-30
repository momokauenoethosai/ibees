from pathlib import Path
from ..utils_gemini import extract_with_prompt
CATEGORY = "acc"
PROMPT = r"""
Output one JSON:
{
  "summary": "concise phrase for accessories (if visible)",
  "attributes": {
    "type":{"value":"glasses|sunglasses|earring|piercing|necklace|hat|hairpin|headband|none|unclear","confidence":0.0-1.0},
    "material":{"value":"metal|plastic|cloth|leather|wood|unclear","confidence":0.0-1.0},
    "shape":{"value":"round|oval|square|rectangle|hoop|chain|star|heart|unclear","confidence":0.0-1.0},
    "color":{"value":"black|white|grey|silver|gold|red|blue|green|yellow|brown|pink|purple|beige|unclear","confidence":0.0-1.0}
  },
  "tags":["up to 10 tags"]
}
"""
def extract(img_path: Path):
    tags, summary, raw = extract_with_prompt(img_path, PROMPT)
    phrase = (summary + ", " + ", ".join(tags)).strip(", ")
    return tags, summary, raw, phrase
