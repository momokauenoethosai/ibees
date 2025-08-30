# utils_embed_bq.py
from __future__ import annotations
from typing import List, Dict, Any
import numpy as np

from vertexai import init
from vertexai.preview.vision_models import MultiModalEmbeddingModel
from google.cloud import bigquery

from .common_config import (
    PROJECT_ID, MODEL_EMBED, LOCATION_GEMINI,
    BQ_PROJECT, BQ_DATASET, BQ_TABLE
)

def l2_normalize(v: List[float]) -> List[float]:
    a = np.asarray(v, dtype="float64")
    n = float(np.linalg.norm(a))
    return (a / n).tolist() if n > 0 else a.tolist()

def get_text_embedding(text: str) -> List[float]:
    init(project=PROJECT_ID, location=LOCATION_GEMINI)
    model = MultiModalEmbeddingModel.from_pretrained(MODEL_EMBED)
    # SDK差異に対応して引数名を試行
    for kw in ("text", "contextual_text", "input_text"):
        try:
            emb = model.get_embeddings(**{kw: text})
            vec = [float(x) for x in emb.text_embedding]
            return l2_normalize(vec)
        except TypeError:
            continue
    raise RuntimeError("get_embeddings のテキスト引数名が見つかりません")

def make_bq_client() -> bigquery.Client:
    meta = bigquery.Client(project=BQ_PROJECT)
    ds = meta.get_dataset(f"{BQ_PROJECT}.{BQ_DATASET}")
    return bigquery.Client(project=BQ_PROJECT, location=ds.location)

def search_by_text_in_category(query_phrase: str, category: str, top_k: int = 10, min_score: float = 0.0) -> List[Dict[str, Any]]:
    q = get_text_embedding(query_phrase)
    client = make_bq_client()
    if category == "mouth":
        category = "mouse"
        
    sql = f"""
    SELECT
      t.part_id, t.part_num, t.category,
      (
        SELECT SUM(tv * qv2)
        FROM UNNEST(t.vector) AS tv WITH OFFSET pos
        JOIN UNNEST(@q)       AS qv2 WITH OFFSET pos2 ON pos = pos2
      ) AS score
    FROM `{BQ_TABLE}` AS t
    WHERE category = @cat
    ORDER BY score DESC
    LIMIT @k
    """
    job = client.query(
        sql,
        job_config=bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ArrayQueryParameter("q", "FLOAT64", q),
                bigquery.ScalarQueryParameter("cat", "STRING", category),
                bigquery.ScalarQueryParameter("k", "INT64", top_k),
            ]
        ),
    )
    out = []
    for r in job:
        if r.score is None or r.score < min_score:
            continue
        out.append({"part_id": r.part_id, "part_num": r.part_num, "category": r.category, "score": float(r.score)})
    return out
