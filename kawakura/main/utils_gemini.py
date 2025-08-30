# utils_gemini.py
from __future__ import annotations
import json, re
from pathlib import Path
from typing import List, Tuple

from vertexai import init
from vertexai.generative_models import GenerativeModel, Part, Content
from .common_config import PROJECT_ID, LOCATION_GEMINI, MODEL_GEMINI

GENCFG = {
    "temperature": 0.2,
    "max_output_tokens": 1024,
    "response_mime_type": "application/json",
}

def _mime_from_suffix(suffix: str) -> str:
    s = suffix.lower().lstrip(".")
    return {
        "png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "webp": "image/webp"
    }.get(s, "application/octet-stream")

def _safe_resp_text(resp) -> str:
    try:
        for c in getattr(resp, "candidates", []) or []:
            content = getattr(c, "content", None)
            if not content: continue
            for p in getattr(content, "parts", []) or []:
                t = getattr(p, "text", None)
                if t: return str(t)
    except Exception:
        pass
    try:
        d = resp.to_dict()
        parts = d.get("candidates", [{}])[0].get("content", {}).get("parts", [])
        for p in parts:
            if "text" in p: return p["text"]
    except Exception:
        pass
    return str(resp)

def _fallback_summary(txt: str) -> str:
    t = (txt or "").strip().replace("\n", " ")
    m = re.search(r'"([^"]{10,120})"', t) or re.search(r'([A-Za-z][^\.]{10,140})', t)
    return (m.group(1).strip() if m else "appearance description")

def _uniq_lower(xs: List[str]) -> List[str]:
    seen, out = set(), []
    for x in xs:
        s = str(x).strip().lower()
        if s and s not in seen:
            seen.add(s); out.append(s)
    return out

def extract_with_prompt(img_path: Path, prompt: str):
    """画像 + プロンプト → JSON抽出（summary/tags中心）。JSONでない場合はフォールバック。"""
    init(project=PROJECT_ID, location=LOCATION_GEMINI)
    model = GenerativeModel(MODEL_GEMINI)
    image_part = Part.from_data(mime_type=_mime_from_suffix(img_path.suffix), data=img_path.read_bytes())
    user = Content(role="user", parts=[Part.from_text(prompt), image_part])
    resp = model.generate_content([user], generation_config=GENCFG)
    txt = _safe_resp_text(resp).strip()

    try:
        data = json.loads(txt)
        summary = (data.get("summary") or "").strip()
        tags = data.get("tags") or []
        return _uniq_lower(tags)[:10], summary, data
    except Exception:
        return [], _fallback_summary(txt), {"raw_text": txt}
