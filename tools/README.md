# Vision RAG Tools

このディレクトリには、顔合成システムのテスト・開発ツールが含まれています。

## 🧰 主要ツール

### 1. **iterative_face_refiner.py** 
**顔類似度ベースの反復調整システム（メインツール）**

```bash
# 基本使用法
python tools/iterative_face_refiner.py outputs/run_20250830_164634.json

# 反復回数指定
python tools/iterative_face_refiner.py outputs/run_20250830_164634.json 3
```

**機能:**
- 元の顔写真と合成画像を比較分析
- Geminiによる段階的パーツ位置・サイズ調整
- 調整履歴の追跡と活用
- 類似度スコアによる収束判定

**出力:**
- `outputs/similarity_iter_*.png` - 各反復の合成結果
- `outputs/comparison_*.png` - 元画像vs合成画像の比較

### 2. **test_with_grid_overlay.py**
**グリッド付き合成テスト（座標確認用）**

```bash
# 基本使用法  
python tools/test_with_grid_overlay.py outputs/run_20250830_164634.json

# 出力ファイル名指定
python tools/test_with_grid_overlay.py outputs/run_20250830_164634.json grid_result.png
```

**機能:**
- JSONの分析結果から合成画像を生成
- 座標確認用のグリッドオーバーレイ
- パーツ配置の視覚的検証

### 3. **create_debug_grid.py**
**デバッグ用グリッド画像作成**

```bash
python tools/create_debug_grid.py
```

**機能:**
- 座標系理解用のグリッド画像生成
- 開発・デバッグ時の座標確認

### 4. **test_face_composition.py** 
**顔合成機能の基本テスト**

```bash
python tools/test_face_composition.py
```

**機能:**
- FaceComposerの基本動作確認
- パーツ配置テスト

## 📁 ワークフロー

### 開発・テスト手順

1. **初期分析**: Webアプリで画像分析 → JSON生成
2. **座標確認**: `test_with_grid_overlay.py` でパーツ配置確認
3. **反復調整**: `iterative_face_refiner.py` で元画像類似度向上
4. **結果検証**: 生成された比較画像で効果確認

### ファイル命名規則

- `run_*.json` - 分析結果JSON
- `composed_*.png` - 基本合成画像
- `similarity_iter_*.png` - 反復調整結果
- `comparison_*.png` - 元画像vs合成画像比較
- `grid_*.png` - グリッド付き検証画像

## ⚙️ 設定

### API設定
- **Gemini API**: `gemini-2.5-pro` (高精度モデル)
- **Canvas Size**: 400x400px
- **最大反復回数**: 5回（デフォルト）

### 調整パラメータ
```python
ADJUSTMENT_STEPS = {
    'position': {
        'up/down/left/right': 5px移動,
        'up_slight/down_slight/left_slight/right_slight': 3px移動
    },
    'scale': {
        'bigger/smaller': 0.05倍変更,
        'bigger_slight/smaller_slight': 0.03倍変更
    }
}
```

## 🎯 最適な使用法

1. **高精度調整**: `iterative_face_refiner.py` - 元画像との類似度向上
2. **座標デバッグ**: `test_with_grid_overlay.py` - パーツ配置確認  
3. **基本動作確認**: `test_face_composition.py` - システム動作検証