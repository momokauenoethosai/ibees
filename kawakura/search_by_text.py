#!/usr/bin/env python3
# search_by_text.py
from __future__ import annotations
import numpy as np

from vertexai import init
from vertexai.preview.vision_models import MultiModalEmbeddingModel
from google.cloud import bigquery

PROJECT_ID = "vision-rag"
EMBED_LOCATION = "us-central1"
MODEL_ID = "multimodalembedding@001"
BQ_LOCATION = "US"
BQ_TABLE = "vision-rag.parts.catalog_img"

TEXT_QUERY = "三つ編みのような髪型"
TARGET_CATEGORY = "hair"

def l2_normalize(v):
    arr = np.asarray(v, dtype="float64")
    n = float(np.linalg.norm(arr))
    return (arr / n).tolist() if n > 0 else arr.tolist()

def get_text_embedding(model: MultiModalEmbeddingModel, text: str):
    """
    SDK差異に対応して、text引数名を順にトライ。
    返り値は list[float]（長さ1408）
    """
    # パターン1: text=...
    try:
        emb = model.get_embeddings(text=text)
        vec = emb.text_embedding
        return [float(x) for x in vec]
    except TypeError:
        pass

    # パターン2: contextual_text=...
    try:
        emb = model.get_embeddings(contextual_text=text)
        vec = emb.text_embedding
        return [float(x) for x in vec]
    except TypeError:
        pass

    # パターン3: input_text=...
    try:
        emb = model.get_embeddings(input_text=text)
        vec = emb.text_embedding
        return [float(x) for x in vec]
    except TypeError:
        pass

    # それでもダメな場合はヘルプを出して失敗
    import inspect
    raise RuntimeError(
        "このSDKでは text 引数名が見つかりませんでした。"
        f"get_embeddings シグネチャ: {inspect.signature(model.get_embeddings)}"
    )

def main():
    # 1) テキスト埋め込み
    init(project=PROJECT_ID, location=EMBED_LOCATION)
    model = MultiModalEmbeddingModel.from_pretrained(MODEL_ID)
    q = l2_normalize(get_text_embedding(model, TEXT_QUERY))

    # 2) BigQuery 類似検索（保存ベクトルは正規化済み → 内積=コサイン類似度）
    client = bigquery.Client(project=PROJECT_ID, location=BQ_LOCATION)
    sql = f"""
    SELECT
      t.part_id, t.part_num, t.category,
      (
        SELECT SUM(tv * qv2)
        FROM UNNEST(t.vector) AS tv WITH OFFSET pos
        JOIN UNNEST(@q)       AS qv2 WITH OFFSET pos2
          ON pos = pos2
      ) AS score
    FROM `{BQ_TABLE}` AS t
    WHERE category = @cat
    ORDER BY score DESC
    LIMIT 10
    """
    job = client.query(
        sql,
        job_config=bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ArrayQueryParameter("q", "FLOAT64", q),
                bigquery.ScalarQueryParameter("cat", "STRING", TARGET_CATEGORY),
            ]
        ),
    )
    print(f'Query: "{TEXT_QUERY}" (category={TARGET_CATEGORY}) → top results')
    for row in job:
        print(f"- {row.part_id} (num={row.part_num})  score={row.score:.4f}")

if __name__ == "__main__":
    main()
