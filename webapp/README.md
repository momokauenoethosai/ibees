# Face Analysis & Composition Web UI

## 概要
このWebアプリケーションは、顔画像を分析し、データベースから適切な顔パーツを検索して合成するシステムです。

## 必要な環境
- Python 3.8+
- Google Cloud SDK
- Google Cloud認証

## セットアップ

### 1. 依存関係のインストール
```bash
pip install -r ../kawakura/requirements.txt
```

### 2. Google Cloud認証
アプリケーションはGoogle Cloud（Vertex AI、BigQuery）を使用するため、認証が必要です。

```bash
gcloud auth application-default login
```

## 起動方法

### 方法1: 起動スクリプトを使用（推奨）
```bash
cd /Users/maikerujakuson/data/2024/SpecTech/エートスAI/アイビーズ様/ibees
./webapp/start.sh
```

このスクリプトは自動的に：
- Google Cloud認証をチェック
- 必要に応じて認証を実行
- Flask アプリケーションを起動

### 方法2: 手動で起動
```bash
# 1. Google Cloud認証（初回のみ）
gcloud auth application-default login

# 2. Flask アプリケーションを起動
python webapp/app.py
```

## 使い方

1. ブラウザで `http://localhost:8080` にアクセス
2. **Step 1: 画像アップロード & 分析**
   - 画像をドラッグ&ドロップまたはクリックして選択
   - オプションでプロンプトを入力
   - 「画像を分析」ボタンをクリック
3. **Step 2: 顔パーツ合成**
   - 分析が完了したら「顔を合成」ボタンをクリック
   - 合成された画像をダウンロード

## トラブルシューティング

### 認証エラー
```
Error: 403 Permission denied
```
→ `gcloud auth application-default login` を実行してください

### BigQueryエラー
```
Error: Table not found
```
→ BigQueryテーブル `vision-rag.parts.catalog_img` が存在することを確認してください

### Vertex AIエラー
```
Error: Model not found
```
→ Vertex AI でテキスト埋め込みモデルが有効になっていることを確認してください

## アーキテクチャ

```
入力画像
  ↓
run_all_parts.py (特徴抽出・検索)
  ├── Gemini AI による顔パーツの特徴抽出
  ├── Vertex AI による埋め込みベクトル生成
  └── BigQuery でのベクトル類似検索
  ↓
JSONファイル (検索結果)
  ↓
face_parts_fitter.py (顔合成)
  ├── MediaPipe による顔ランドマーク検出
  └── 顔パーツの配置・合成
  ↓
合成画像