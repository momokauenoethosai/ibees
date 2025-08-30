# part_extractors/hair.py
from pathlib import Path
from ..utils_gemini import extract_with_prompt

PROMPT = r"""
出力は JSON 1オブジェクトのみ。余計な文字やコードブロックは禁止。
{
  "summary": "one concise english phrase for the hairstyle",
  "attributes": {
    "length": {"value":"very-short|short|medium|medium-long|long|very-long","confidence":0.0-1.0},
    "parting":{"value":"center-part|left-part|right-part|no-part|unclear","confidence":0.0-1.0},
    "bangs":{"value":"bangs-none|bangs-front|bangs-side|micro-bangs|see-through|unclear","confidence":0.0-1.0},
    "curl_pattern":{"value":"straight|soft-wave|wavy|curly|coily|mixed","confidence":0.0-1.0},
    "volume":{"value":"flat|medium-volume|high-volume","confidence":0.0-1.0},
    "layering":{"value":"one-length|layered|shag|wolf|step|unclear","confidence":0.0-1.0},
    "ends":{"value":"blunt|tapered|point-cut|feathered|unclear","confidence":0.0-1.0},
    "color_primary":{"value":"black|dark-brown|brown|blonde|red|grey|other","confidence":0.0-1.0}
  },
  "tags": ["up to 10 short english tags (kebab-case)"],
  "notes": "1-2 sentences in english",
  "confidence_overall": 0.0-1.0
}
"""

CATEGORY = "hair"

def extract(img_path: Path):
    """returns (tags:list[str], summary:str, raw_json:dict, search_phrase:str)"""
    tags, summary, raw = extract_with_prompt(img_path, PROMPT)
    # 検索用フレーズを合成（summary + tags）
    phrase = summary
    if tags:
        phrase = (summary + ", " + ", ".join(tags)).strip(", ")
    return tags, summary, raw, phrase