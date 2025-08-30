#!/usr/bin/env python3
# build_image_embeddings.py
# 画像を Vertex AI Multimodal Embeddings で 1408次元ベクトル化 → JSONL 出力 → GCS へアップロード

import json
import re
import sys
from pathlib import Path
from typing import List

import numpy as np

from vertexai import init
from vertexai.preview.vision_models import MultiModalEmbeddingModel, Image
from google.cloud import storage

# ========= 設定（環境に合わせて変更可） =========
PROJECT_ID     = "vision-rag"
LOCATION       = "us-central1"                     # ★ Multimodal Embeddings は us-central1 が安定
MODEL_ID       = "multimodalembedding@001"

ROOT_DIR       = Path("assets_png")              # 例: ./parts_root/acc/acc_010.png
OUTPUT_JSONL   = Path("parts_vectors_img.jsonl")
GCS_BUCKET_URI = "gs://parts-embeddings-vision-rag"
GCS_OBJECT     = OUTPUT_JSONL.name                 # アップロード先オブジェクト名

# 対応拡張子
IMG_EXT = {".png", ".jpg", ".jpeg", ".webp"}

# ========= ここから下は通常変更不要 =========

_num_re = re.compile(r"(\d+)")

def extract_part_num(stem: str, fallback: int) -> int:
    """ファイル名から数字部分を抽出（無ければ連番フォールバック）"""
    m = _num_re.search(stem)
    return int(m.group(1)) if m else fallback

def l2_normalize(v: List[float]) -> List[float]:
    arr = np.asarray(v, dtype="float32")
    n = float(np.linalg.norm(arr))
    return (arr / n).astype("float32").tolist() if n > 0 else arr.tolist()

def iter_images(root: Path):
    """root直下の各フォルダ=category として画像を列挙"""
    if not root.exists():
        raise FileNotFoundError(f"画像ルートが見つかりません: {root.resolve()}")
    for cat_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        category = cat_dir.name.lower()
        files = sorted(p for p in cat_dir.iterdir() if p.suffix.lower() in IMG_EXT)
        if not files:
            continue
        yield category, files

def embed_image(model: MultiModalEmbeddingModel, img_path: Path) -> List[float]:
    """単一画像 → 1408次元ベクトル（L2正規化）"""
    img = Image(image_bytes=img_path.read_bytes())
    emb = model.get_embeddings(image=img)
    vec = list(map(float, emb.image_embedding))  # 1408次元
    return l2_normalize(vec)

def upload_to_gcs(local_path: Path, gcs_uri: str, object_name: str):
    """JSONL を GCS にアップロード"""
    bucket_name = gcs_uri.replace("gs://", "").strip("/")

    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(object_name)
    blob.upload_from_filename(str(local_path))
    print(f"[GCS] Uploaded -> gs://{bucket_name}/{object_name}")

def main():
    # Vertex 初期化
    init(project=PROJECT_ID, location=LOCATION)
    model = MultiModalEmbeddingModel.from_pretrained(MODEL_ID)

    total = 0
    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_JSONL.open("w", encoding="utf-8") as f:
        for category, files in iter_images(ROOT_DIR):
            print(f"[{category}] {len(files)} files")
            fallback = 1
            for fp in files:
                stem = fp.stem
                part_num = extract_part_num(stem, fallback)
                if not _num_re.search(stem):
                    fallback += 1
                part_id = f"{category}_{part_num}"

                try:
                    vec = embed_image(model, fp)  # 1408次元ベクトル
                except Exception as e:
                    print(f"  ! embed失敗: {fp.name}: {e}")
                    continue

                rec = {
                    "part_id": part_id,          # 例: "acc_10"
                    "category": category,        # 例: "acc"
                    "part_num": int(part_num),   # 数値ID（検索結果で返す用）
                    "vector": vec                # VECTOR<FLOAT32>[1408]
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                total += 1
                if total % 50 == 0:
                    print(f"  ... {total} embeddings")

    print(f"\n[OK] Wrote {OUTPUT_JSONL.resolve()} ({total} rows)")

    # GCSへアップロード
    try:
        upload_to_gcs(OUTPUT_JSONL, GCS_BUCKET_URI, GCS_OBJECT)
    except Exception as e:
        print(f"[WARN] GCSアップロード失敗: {e}")
        print("→ 後から手動で:  gsutil cp", str(OUTPUT_JSONL), f"{GCS_BUCKET_URI}/{GCS_OBJECT}")

    # BigQuery作成用のSQLを表示（コピーして実行）
    print("\n--- BigQuery: 作成/ロード用 SQL（コンソールで実行）---")
    print(f"""\
CREATE SCHEMA IF NOT EXISTS `vision-rag.parts`
OPTIONS(location="asia-northeast1");

CREATE TABLE IF NOT EXISTS `vision-rag.parts.catalog_img`
(
  part_id   STRING NOT NULL,
  category  STRING NOT NULL,
  part_num  INT64  NOT NULL,
  vector    VECTOR<FLOAT32>[1408] NOT NULL
);

-- 既存データ置換したい場合（任意）
-- TRUNCATE TABLE `vision-rag.parts.catalog_img`;

LOAD DATA INTO `vision-rag.parts.catalog_img`
FROM FILES (
  format = 'JSON',
  uris = ['{GCS_BUCKET_URI}/{GCS_OBJECT}']
);
""")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] 中断しました")
        sys.exit(130)
