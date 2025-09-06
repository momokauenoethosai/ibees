# Face Analysis & Composition System

AI駆動の顔パーツ分析・合成システム

人物の画像から顔の特徴を分析し、髪・目・鼻・口などのパーツごとに類似するパーツ画像を検索・合成。さらにGeminiによる調整で自然な顔合成を実現。

## 🚀 クイックスタート

### 1. セットアップ
```bash
git clone <このリポジトリURL>
cd ibees

# Python仮想環境作成
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 依存関係インストール
pip install -r webapp/requirements.txt
```

### 2. 認証設定
```bash
# サービスアカウント認証（推奨）
# webapp/credentials/service-account.json にサービスアカウントキーを配置

# または、個人認証
gcloud auth application-default login
gcloud config set project <プロジェクトID>
```

### 3. アプリケーション起動
```bash
cd webapp
python app.py
```
ブラウザで http://127.0.0.1:8080 にアクセス

## 🎯 主要機能

### 1. 画像分析モード（認証必要）
- 画像をアップロード
- Gemini Visionによる顔パーツ分析（髪、目、眉、鼻、口、耳、輪郭など）
- BigQueryベクトル検索で700+パーツから最適なものを選定
- 選定したパーツを自動合成して表示

### 2. サンプルモード（認証不要）
- 事前に分析済みのサンプルデータを使用
- 認証なしで顔合成の機能を体験可能

## 📁 プロジェクト構成

```
ibees/
├── webapp/                    # Webアプリケーション
│   ├── app.py                # Flaskサーバー
│   ├── sample_manager.py     # サンプルデータ管理
│   ├── templates/            # HTMLテンプレート
│   ├── static/              # 静的ファイル
│   └── credentials/         # 認証情報
├── kawakura/                # コア処理
│   ├── main/
│   │   ├── run_all_parts.py # 顔パーツ分析
│   │   └── utils_*.py       # ユーティリティ
│   ├── face_parts_fitter.py # パーツ合成
│   └── assets_png/          # パーツ画像データ
├── face_composer/           # 顔合成エンジン
├── outputs/                 # 分析結果JSON
└── tools/                   # 開発・デバッグツール
```

## 🔧 技術スタック

- **バックエンド**: Flask (Python)
- **AI/ML**: 
  - Google Vertex AI (Gemini Vision)
  - BigQuery (ベクトル検索)
  - MediaPipe (顔ランドマーク検出)
- **フロントエンド**: HTML/CSS/JavaScript

## 📝 デプロイ

### Google Cloud Runへのデプロイ
```bash
# Dockerイメージビルド
gcloud builds submit --tag gcr.io/<プロジェクトID>/face-analysis-app

# Cloud Runへデプロイ
gcloud run deploy face-analysis-app \
  --image gcr.io/<プロジェクトID>/face-analysis-app \
  --platform managed \
  --region asia-northeast1 \
  --allow-unauthenticated
```

## 🔐 認証について

### サービスアカウント認証（デプロイ用）
1. GCPでサービスアカウントを作成
2. 必要な権限を付与（Vertex AI User、BigQuery Data Viewer等）
3. JSONキーをダウンロード
4. `webapp/credentials/service-account.json`に配置

### 開発時の認証
```bash
gcloud auth application-default login
```

## 📄 ライセンス

このプロジェクトは非公開プロジェクトです。