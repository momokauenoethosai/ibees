# Vision RAG Face Composition System - 技術仕様書

## 概要

本システムは、顔画像から各パーツ（髪型、目、眉毛、鼻、口、耳、輪郭等）を自動分析し、最適なパーツ画像を選定して顔合成を行うAIシステムです。Google Cloud Platform（GCP）上のVertex AI、BigQuery、Gemini APIを活用したRAG（Retrieval-Augmented Generation）アーキテクチャを採用しています。

## システム構成

### 1. 主要コンポーネント

- **画像分析エンジン** (`kawakura/`): 顔パーツの特徴抽出・選定
- **顔合成エンジン** (`face_composer/`): パーツ配置・合成処理
- **Webアプリケーション** (`webapp/`): ユーザーインターフェース
- **パーツアセット** (`kawakura/assets_png/`): 700+のパーツ画像

### 2. データフロー

```
入力画像 → パーツ分析 → ベクトル検索 → パーツ選定 → 顔合成 → Gemini修正 → 最終出力
```

## 画像ベクトル化・検索仕様

### 1. ベクトル化プロセス (`build_image_embeddings.py`)

#### 使用モデル
- **モデル**: `multimodalembedding@001` (Vertex AI Multimodal Embeddings)
- **次元数**: 1,408次元
- **正規化**: L2正規化適用
- **リージョン**: `us-central1`

#### 処理手順
1. パーツ画像（PNG/JPG/WEBP）をカテゴリ別フォルダから読み込み
2. Vertex AI Multimodal Embedding APIでベクトル化
3. L2正規化を適用し数値安定性を確保
4. JSONLファイル出力（1行1パーツ）
5. Google Cloud Storage（GCS）へアップロード

#### 出力データ構造
```json
{
  "part_id": "hair_262",
  "category": "hair", 
  "part_num": 262,
  "vector": [1408次元のfloat配列]
}
```

### 2. BigQuery設定

#### データセット・テーブル構造
```sql
CREATE TABLE `vision-rag.parts.catalog_img` (
  part_id   STRING NOT NULL,           -- 例: "hair_262"
  category  STRING NOT NULL,           -- 例: "hair"
  part_num  INT64  NOT NULL,          -- 例: 262  
  vector    VECTOR<FLOAT32>[1408] NOT NULL  -- ベクトル
);
```

#### GCSからのデータロード
- **バケット**: `gs://parts-embeddings-vision-rag`
- **ファイル**: `parts_vectors_img.jsonl`
- **リージョン**: `asia-northeast1`

### 3. ベクトル検索 (`utils_embed_bq.py`)

#### 検索クエリ処理
1. 入力テキストをVertex AI Multimodal Embeddingsでベクトル化
2. BigQueryのベクトル内積計算で類似度スコア算出
3. カテゴリ別フィルタリング適用
4. TOP-K結果を類似度順でソート

#### SQL検索クエリ
```sql
SELECT
  t.part_id, t.part_num, t.category,
  (
    SELECT SUM(tv * qv2)
    FROM UNNEST(t.vector) AS tv WITH OFFSET pos
    JOIN UNNEST(@q)       AS qv2 WITH OFFSET pos2 ON pos = pos2
  ) AS score
FROM `vision-rag.parts.catalog_img` AS t
WHERE category = @cat
ORDER BY score DESC
LIMIT @k
```

## 顔パーツ分析仕様

### 1. パーツエクストラクタ (`part_extractors/`)

#### 対応パーツカテゴリ
- **hair**: 髪型（長さ、分け目、前髪、カール、ボリューム等）
- **eye**: 目（形状、サイズ、開閉度、まつ毛等）
- **eyebrow**: 眉毛（形状、太さ、アーチ等）
- **nose**: 鼻（形状、サイズ、鼻筋等）
- **mouth**: 口（唇の形状、厚さ、色等）
- **ear**: 耳（形状、サイズ、耳たぶ等）
- **outline**: 輪郭（顔型、骨格等）
- **acc**: アクセサリー
- **beard**: ひげ・顔の毛
- **glasses**: メガネ・サングラス
- **extras**: 特徴点（ほくろ、そばかす等）
- **wrinkles**: しわ・線

### 2. Gemini Vision分析 (`utils_gemini.py`)

#### 使用モデル
- **モデル**: `gemini-2.0-flash`
- **設定**: Temperature 0.2, Max tokens 1024
- **出力形式**: JSON（構造化データ）

#### 分析プロンプト例（髪型）
```json
{
  "summary": "Blunt bob with bangs",
  "attributes": {
    "length": {"value": "short", "confidence": 0.95},
    "parting": {"value": "no-part", "confidence": 0.9},
    "bangs": {"value": "bangs-front", "confidence": 0.95},
    "curl_pattern": {"value": "straight", "confidence": 0.95}
  },
  "tags": ["blunt-bob", "straight-hair", "front-bangs", "short-hair"],
  "confidence_overall": 0.85
}
```

### 3. パーツ選定アルゴリズム (`run_all_parts.py`)

#### 選定プロセス
1. 各パーツをGemini Visionで分析
2. summary + tagsを検索フレーズに合成
3. ベクトル化してBigQuery検索実行
4. カテゴリ別閾値フィルタリング
5. ネガティブ判定（"なし"表現）でスキップ
6. トップ1選定（デフォルト）

#### カテゴリ別最小スコア
```python
CATEGORY_MIN_SCORE = {
    "hair": 0.0, "eye": 0.0, "eyebrow": 0.0, 
    "nose": 0.0, "mouth": 0.0, "ear": 0.0,
    "outline": 0.0, "acc": 0.06, "beard": 0.08,
    "glasses": 0.00, "extras": 0.00, "wrinkles": 0.00
}
```

## 顔合成エンジン仕様

### 1. 座標配置システム (`part_placement_config.py`)

#### 基準座標系（400x400キャンバス）
```python
base_positions = {
    'hair': (200, 200, 1.0),      # 中心、フルサイズ
    'eye': {
        'left': (225, 215, 0.2),   # 左目
        'right': (175, 215, 0.2),  # 右目  
        'single': (200, 215, 0.2)
    },
    'eyebrow': {
        'left': (225, 185, 0.2),   # 左眉
        'right': (175, 185, 0.2),  # 右眉
        'single': (200, 185, 0.2)
    },
    'nose': (200, 230, 0.2),      # 鼻中央
    'mouth': (200, 255, 0.25),    # 口中央
    'ear': {
        'left': (250, 220, 0.28),  # 左耳
        'right': (150, 220, 0.28)  # 右耳
    },
    'outline': (200, 200, 1.0),   # 輪郭、フルサイズ
    'acc': (200, 180, 0.3),
    'beard': (200, 300, 0.4),
    'glasses': (200, 215, 0.5)
}
```

#### 座標計算ルール
- **(x, y)**: パーツ画像の中心座標
- **scale**: スケール倍率（1.0 = 100%）
- **左右対称パーツ**: eye, eyebrow, earは左右独立配置
- **キャンバス適応**: 異なるサイズへの自動スケーリング対応

### 2. 合成処理 (`face_composer.py`)

#### レイヤー描画順序
1. **outline** (輪郭) - 最背面
2. **hair** (髪)
3. **face_shape** (顔の形)
4. **eyebrow** (眉毛)  
5. **eye** (目)
6. **nose** (鼻)
7. **mouth** (口)
8. **ear** (耳)
9. **beard** (ひげ)
10. **glasses** (メガネ)
11. **acc** (アクセサリー) - 最前面

#### 画像処理
- **透明度処理**: RGBA → RGB変換（白背景）
- **スケーリング**: PIL.Image.resize with LANCZOS
- **左右反転**: 右側パーツの自動ミラーリング
- **アルファ合成**: 透明度を考慮した重ね合わせ

## Gemini座標修正システム

### 1. 修正プロセス (`gemini_refinement.py`)

#### 使用API
- **モデル**: `gemini-1.5-flash`
- **APIキー**: `AIzaSyAt-wzZ3WLU1fc6fnzHvDhPsTZJNKnHszU`
- **入力**: 合成画像 + 現在座標 + パーツ情報

#### 修正方針
1. **顔の比例とバランス**: パーツ間の自然な位置関係
2. **左右対称性**: eye, eyebrow, earの左右バランス
3. **解剖学的正確性**: 顔の構造に従った配置  
4. **視覚的違和感**: 不自然に見える部分の修正

#### 出力形式
修正された座標を`part_placement_config.py`の`base_positions`形式で出力：
```python
{
    'hair': (x, y, scale),
    'eye': {
        'left': (x, y, scale),
        'right': (x, y, scale),
        'single': (x, y, scale)
    },
    # ... 他のパーツ
}
```

### 2. Web API統合 (`webapp/app.py`)

#### `/refine`エンドポイント
- **Method**: POST
- **Input**: 合成画像URL + パーツ情報
- **Output**: 修正座標JSON + 修正結果ファイル
- **処理時間**: 5-15秒（Gemini API応答時間依存）

## Webアプリケーション仕様

### 1. アーキテクチャ (`webapp/`)

#### フレームワーク
- **Backend**: Flask (Python)
- **Frontend**: Vanilla JavaScript + HTML5
- **通信**: Server-Sent Events (SSE) + REST API

#### 主要エンドポイント
- **GET /**: メインページ表示
- **POST /analyze**: 画像分析開始
- **GET /stream/<stream_id>**: リアルタイム進捗取得
- **POST /compose**: パーツ合成実行
- **POST /refine**: Gemini座標修正
- **GET /outputs/<filename>**: 結果ファイル配信

### 2. ユーザーフロー

1. **画像アップロード**: ドラッグ&ドロップまたはファイル選択
2. **リアルタイム分析**: 各パーツの分析進捗をSSEで表示
3. **自動合成**: 分析完了後、自動的にパーツ合成実行
4. **結果表示**: 選定パーツ一覧 + 合成結果画像
5. **Gemini修正**: ボタンクリックで座標修正実行
6. **修正結果**: 改善された座標情報を表示

### 3. UI/UX機能

#### リアルタイム進捗表示
- パーツ別進捗インジケータ
- 現在処理中パーツのハイライト
- 完了・スキップ・エラー状態の視覚化

#### 結果表示
- 各パーツの画像サムネイル
- パーツ番号・信頼度スコア表示
- 合成結果の拡大表示
- JSON生データへのリンク

## 開発・運用仕様

### 1. 開発環境

#### 言語・ライブラリ
- **Python**: 3.11+
- **主要ライブラリ**: 
  - `google-cloud-bigquery`: BigQuery操作
  - `google-cloud-storage`: GCS操作  
  - `vertexai`: Vertex AI SDK
  - `google-generativeai`: Gemini API
  - `flask`: Webフレームワーク
  - `PIL`: 画像処理
  - `numpy`: 数値計算

#### Google Cloud設定
- **プロジェクト**: `vision-rag`
- **認証**: サービスアカウントキーまたはApplication Default Credentials
- **権限**: BigQuery Data Editor, Storage Object Admin

### 2. パフォーマンス指標

#### 処理時間
- **パーツ分析**: 30-60秒（9パーツ×Gemini API）
- **ベクトル検索**: 1-3秒（BigQuery）
- **画像合成**: 2-5秒（PIL処理）
- **Gemini修正**: 5-15秒（画像解析）

#### スループット
- **同時処理**: 5-10リクエスト（Vertex AI制限依存）
- **画像サイズ**: 最大5MB推奨
- **対応形式**: PNG, JPEG, WebP

### 3. エラーハンドリング

#### 主要エラーケース
- Vertex AI API制限・障害
- BigQuery接続エラー
- 画像形式・サイズエラー
- パーツ画像ファイル不足
- Gemini API応答エラー

#### 復旧戦略
- 指数バックオフによる自動リトライ
- フォールバック処理（デフォルト座標）
- 詳細エラーログ出力
- ユーザー向けエラーメッセージ

## セキュリティ・コンプライアンス

### 1. APIキー管理
- 環境変数による機密情報管理
- GCP IAMによるアクセス制御
- 最小権限原則の適用

### 2. データ処理
- アップロード画像の一時保存（/tmp）
- 個人情報の永続化回避
- GDPR準拠のデータ処理方針

## まとめ

本システムは、最新のMultimodal AI技術（Vertex AI、Gemini）とベクトル検索（BigQuery Vector Search）を組み合わせた、高度な顔パーツ分析・合成システムです。700以上のパーツアセットから最適な組み合わせを自動選定し、AI支援による座標修正で自然な顔合成を実現します。