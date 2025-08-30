# common_config.py
PROJECT_ID = "vision-rag"

# Vertex / Embeddings
LOCATION_GEMINI = "us-central1"
MODEL_GEMINI    = "gemini-2.0-flash"   # 精度重視なら "gemini-2.0-pro"
MODEL_EMBED     = "multimodalembedding@001"

# BigQuery
BQ_PROJECT = "vision-rag"
BQ_DATASET = "parts"   # 既存の dataset
BQ_TABLE   = f"{BQ_PROJECT}.{BQ_DATASET}.catalog_img"  # ARRAY<FLOAT64> 保存

# 検索
TOP_K      = 10
MIN_SCORE  = 0.0
