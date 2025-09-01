# Vision RAG Face Composition System

**AI駆動の顔パーツ分析・合成システム with Gemini反復調整**

人の画像から特徴を抽出し、髪・目・鼻などのパーツごとに類似するパーツ画像を検索・合成。さらにGeminiによる反復調整で元画像に近い自然な顔合成を実現。  
バックエンドに **Vertex AI + BigQuery + Gemini API** を使用し、リアルタイムUI で完全自動化を実現。

---

## 📌 フェーズ1: 実行だけする人向け

### 必要環境
- Python 3.11+
- Google Cloud 認証済み（権限は管理者が準備済み想定）

### 1. セットアップ
```bash
git clone <このリポジトリURL>
cd portrait_selection_ai_v1

python3 -m venv .venv
source .venv/bin/activate

pip install -U pip
pip install -r requirements.txt
```

### 2. 認証（初回のみ）
```bash
gcloud auth application-default login
gcloud config set project vision-rag
```

### 3. Webアプリケーション起動
```bash
cd webapp
python app.py
```
**アクセス**: http://127.0.0.1:5000

## 🎯 主要機能

### 🤖 **AI分析・合成**
1. **画像アップロード**: ドラッグ&ドロップまたはファイル選択
2. **自動パーツ分析**: Gemini Visionによる9パーツ分析（髪、目、眉、鼻、口、耳、輪郭、等）
3. **ベクトル検索**: BigQueryで700+パーツから最適選定
4. **自動合成**: 選定パーツの即座合成・表示

### 🔄 **Gemini反復調整**
1. **🔄 反復調整ボタン**: 元画像との類似度向上
2. **リアルタイム進捗**: 各反復の画像・調整指示をライブ表示
3. **自動収束**: Geminiが満足するまで継続調整
4. **履歴活用**: 過去の調整効果を学習して効率的改善

### 3. 実行
```bash
python webapp/app.py
```
ブラウザで http://127.0.0.1:5000 を開きます。  
PNG 画像をアップロードすると、類似パーツが横並びで表示され、結果 JSON は outputs/ に保存されます。

---

## 📌 フェーズ2: 開発・修正する人向け

### ディレクトリ構成
````bash
portrait_selection_ai_v1/
├─ kawakura/               # コア処理
│  ├─ main/
│  │   ├─ run_all_parts.py     # 特徴抽出 & 統合JSON
│  │   ├─ utils_embed_bq.py    # Embedding検索 / BigQuery
│  │   └─ common_config.py     # プロジェクト・モデル設定
│  ├─ assets_png/              # パーツ画像
├─ webapp/                 # Flask Web UI
│  ├─ app.py               # サーバ
│  └─ templates/index.html # UI
├─ outputs/                # 実行結果 JSON
├─ requirements.txt
````

### BigQuery 設定

#### データセット作成
```bash
bq --location=asia-northeast1 mk --dataset vision-rag:parts
```

#### テーブル作成
```sql
CREATE OR REPLACE TABLE `vision-rag.parts.catalog_img` (
  part_id   STRING,
  category  STRING,
  part_num  INT64,
  vector    ARRAY<FLOAT64>
);
```

#### データロード
```bash
bq --location=asia-northeast1 load \
  --source_format=NEWLINE_DELIMITED_JSON \
  --schema=part_id:STRING,category:STRING,part_num:INTEGER,vector:FLOAT64 \
  vision-rag:parts.catalog_img \
  gs://parts-embeddings-vision-rag/parts_vectors_img.jsonl
```

### 修正ポイント
- 特徴抽出モデル: kawakura/main/run_all_parts.py  
- 検索ロジック: kawakura/main/utils_embed_bq.py  
- UI 表示: webapp/templates/index.html  
- カテゴリ名ずれ対策: CATEGORY_DB_ALIAS で mouth → mouse など対応

### デプロイ (Cloud Run)
```bash
gcloud builds submit \
  --tag asia-northeast1-docker.pkg.dev/vision-rag/containers/portrait-ui:latest

gcloud run deploy portrait-ui \
  --image asia-northeast1-docker.pkg.dev/vision-rag/containers/portrait-ui:latest \
  --region asia-northeast1 \
  --allow-unauthenticated
```

### 出力例（JSON）
```json
{
  "input_image": "made_pictures/1.png",
  "meta": { "top_k": 1, "min_score": 0.0 },
  "parts": {
    "hair": {
      "extracted": { "summary": "Blunt bob with bangs" },
      "search": {
        "top_hits": [
          {"part_id": "hair_153", "score": 0.1157}
        ]
      },
      "selected": {
        "part_id_full": "hair_153",
        "part_num": 153,
        "score": 0.1157
      }
    },
    "eye": { ... }
  },
  "compact": {
    "hair": { "part_num": 153, "score": 0.1157 },
    "eye": { "part_num": 87, "score": 0.1038 }
  }
}
```

### 今後の改善TODO
- mouth (→mouse) のカテゴリマッピング調整
- UI: スコア色分け・スコア閾値調整
- Embeddings の更新自動化（定期ジョブ）
- Cloud Run 公開時のアクセス制御（認証あり/なし切替）