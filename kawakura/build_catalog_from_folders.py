#!/usr/bin/env python3
# build_catalog_with_gemini.py
# 画像フォルダを走査し、Geminiでタグ(JSON)抽出→parts_catalog.csv を自動生成（GCS経由）

import re
import csv
import time
import json
from pathlib import Path
from typing import List

from vertexai import init
from vertexai.generative_models import GenerativeModel, Part, Content
from google.cloud import storage

# ===== 設定（あなたの環境用に固定）=====
PROJECT_ID = "vision-rag"
# ★ Geminiは us-central1 が安定（GCS/BigQuery は東京でもOK）
LOCATION   = "us-central1"             # ★ Gemini 2.0 系は us-central1
MODEL_NAME = "gemini-2.0-flash"

# パーツ画像のルート（直下にカテゴリフォルダが並ぶ想定：hair/, eye/, ...）
ROOT    = Path("assets_png")
OUT_CSV = Path("parts_catalog.csv")

# 一時アップロード先（既存のGCSバケットを利用）
TMP_BUCKET = "parts-embeddings-vision-rag"
TMP_PREFIX = "tmp_tagging"

# 画像拡張子・レート制御
IMG_EXT    = {".png", ".jpg", ".jpeg", ".webp"}
RATE_SLEEP = 0.4  # 軽いレート制御（必要に応じて調整）
GENCFG = {"temperature": 0.2, "max_output_tokens": 512, "response_mime_type": "application/json"}

# ファイル名から数値ID抽出
NUM_RE = re.compile(r"(\d+)")

def extract_part_num(stem: str, fallback: int) -> int:
    m = NUM_RE.search(stem)
    return int(m.group(1)) if m else fallback

def mime_from_suffix(suffix: str) -> str:
    s = suffix.lower().lstrip(".")
    return {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "webp": "image/webp",
    }.get(s, "application/octet-stream")

# カテゴリ別の指示（必要に応じて語彙を拡張/固定）
CATEGORY_INSTRUCTIONS = {
    "hair": """以下のキーで英語の短いタグを返してください:
{
  "tags": ["long|short|medium", "left-part|right-part|center-part|no-part", "straight|soft-wave|curly|wavy", "bangs-none|bangs-front|bangs-side", "color:black|dark-brown|brown|blonde|red|grey"]
}
存在しない属性は入れず、重複しないように。""",
    "eye": """目の形とサイズ、傾きなどを英語タグで:
{"tags": ["shape:almond|round|monolid|double", "size:small|medium|large", "tilt:slight-up|slight-down|neutral"]}""",
    "eyebrow": """眉の形/太さ/角度:
{"tags": ["shape:straight|arched|rounded", "thickness:thin|medium|thick", "angle:low|medium|high"]}""",
    "nose": """鼻のタイプ（ざっくり）:
{"tags": ["bridge:low|medium|high", "tip:rounded|pointed", "size:small|medium|large"]}""",
    "mouth": """口（唇の厚み/表情）:
{"tags": ["thickness:thin|medium|full", "smile:yes|no", "open:closed|slight-open"]}""",
    "outline": """顔の輪郭形状:
{"tags": ["shape:oval|round|square|heart|diamond"]}""",
    "beard": """髭の種類:
{"tags": ["stubble|goatee|mustache|full-beard|chin", "length:short|medium|long"]}""",
    "acc": """アクセサリ（眼鏡/イヤリング/帽子など）:
{"tags": ["type:glasses|earring|hat|piercing|necklace", "material:metal|plastic|cloth", "shape:round|square|hoop|chain", "color:black|silver|gold"]}"""
}

BASE_PROMPT = """あなたは似顔絵パーツの属性抽出アシスタントです。
画像から対象カテゴリに関する「短い英語タグの配列」だけをJSONで返してください。
出力は必ず JSON 1オブジェクトのみ・余計な文章は書かない・スキーマは {"tags": [...]}。
タグは5個以内、曖昧な場合は推定しないで省略。"""

def build_prompt(category: str) -> str:
    extra = CATEGORY_INSTRUCTIONS.get(category, '{"tags": []}')
    return BASE_PROMPT + "\nカテゴリ: " + category + "\n" + extra

# ---- GCS helpers ----
_storage_client = None

def ensure_storage() -> storage.Client:
    global _storage_client
    if _storage_client is None:
        _storage_client = storage.Client(project=PROJECT_ID)
    return _storage_client

def upload_to_gcs(fp: Path) -> str:
    client = ensure_storage()
    bucket = client.bucket(TMP_BUCKET)
    key = f"{TMP_PREFIX}/{fp.name}"
    blob = bucket.blob(key)
    blob.upload_from_filename(str(fp), content_type=mime_from_suffix(fp.suffix))
    return f"gs://{TMP_BUCKET}/{key}"

def _safe_text_from_resp(resp) -> str:
    """resp.text に直接触らず、安全にテキストを取り出す"""
    # 1) 新しめのSDKは resp.candidates[0].content.parts[*].text に入る
    try:
        for cand in getattr(resp, "candidates", []) or []:
            content = getattr(cand, "content", None)
            if not content:
                continue
            for part in getattr(content, "parts", []) or []:
                txt = getattr(part, "text", None)
                if txt:
                    return str(txt)
    except Exception:
        pass
    # 2) それでも無理なら to_dict から拾う
    try:
        d = resp.to_dict()
        cands = d.get("candidates", [])
        if cands:
            parts = cands[0].get("content", {}).get("parts", [])
            for p in parts:
                if "text" in p:
                    return p["text"]
    except Exception:
        pass
    # 3) 最終フォールバック
    return str(resp)

def _mime_from_suffix(suffix: str) -> str:
    s = suffix.lower().lstrip(".")
    return {
        "png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "webp": "image/webp"
    }.get(s, "application/octet-stream")
# ---- Gemini infer（from_uri を使用）----
def extract_tags_for_image(model: GenerativeModel, category: str, fp: Path) -> list[str]:
    prompt = build_prompt(category)

    # ★ GCSは使わず、ローカル画像をバイトで渡す（あなたのSDKは Part.from_data をサポート）
    mime = _mime_from_suffix(fp.suffix)
    image_bytes = fp.read_bytes()
    image_part = Part.from_data(mime_type=mime, data=image_bytes)

    # ★ テキストも Part 化し、Content 1件にまとめて渡す
    user = Content(role="user", parts=[Part.from_text(prompt), image_part])

    resp = model.generate_content([user], generation_config=GENCFG)

    text = _safe_text_from_resp(resp)  # ← resp.text を直接触らない
    # 期待は {"tags":[...]} のJSON。失敗時はフォールバックで分割。
    try:
        data = json.loads(text or "{}")
        tags = data.get("tags", [])
    except Exception:
        txt = (text or "").strip().lower().replace("\n", " ")
        tags = [x.strip() for x in re.split(r"[,\|\;/]", txt) if x.strip()]

    # 正規化・重複除去・上限5
    norm = []
    for t in tags:
        s = str(t).strip().lower()
        if s and s not in norm:
            norm.append(s)
    return norm[:5]

def main():
    init(project=PROJECT_ID, location=LOCATION)
    model = GenerativeModel(MODEL_NAME)

    rows = []
    cat_dirs = [p for p in ROOT.iterdir() if p.is_dir()]
    if not cat_dirs:
        print(f"[WARN] カテゴリフォルダが見つかりません: {ROOT.resolve()}")
        return

    for cat_dir in sorted(cat_dirs):
        category = cat_dir.name.lower()
        files = sorted([p for p in cat_dir.iterdir() if p.suffix.lower() in IMG_EXT])
        if not files:
            continue

        print(f"[{category}] {len(files)} files")
        fallback = 1
        for fp in files:
            stem = fp.stem
            part_num = extract_part_num(stem, fallback)
            if not NUM_RE.search(stem):
                fallback += 1
            part_id = f"{category}_{part_num}"

            try:
                tags = extract_tags_for_image(model, category, fp)
                tag_str = ",".join(tags)
                print(f"  - {fp.name} -> tags: {tag_str}")
            except Exception as e:
                print(f"  ! Gemini失敗: {fp.name}: {e}")
                tag_str = ""

            rows.append([part_id, category, part_num, tag_str])
            time.sleep(RATE_SLEEP)  # レート制御（必要に応じて調整）

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["part_id", "category", "part_num", "tags"])
        w.writerows(rows)

    print(f"\nWrote {OUT_CSV.resolve()} ({len(rows)} rows)")

if __name__ == "__main__":
    main()
