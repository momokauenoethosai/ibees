# Vision RAG - 顔合成システム

人物画像からパーツを抽出し、選択されたパーツで顔を合成するシステム

## 📁 プロジェクト構成

```
vision_rag/
├── face_composer/              # 顔合成エンジン
│   ├── face_composer.py        # メイン合成エンジン
│   ├── landmark_detector.py    # 顔ランドマーク検出（未使用）
│   ├── part_normalizer.py      # パーツ正規化（未使用）
│   └── part_placement_config.py # パーツ配置設定⭐
├── tools/                      # テスト・デバッグツール
│   ├── test_face_composition.py # 独立合成テスト
│   ├── create_debug_grid.py     # グリッド表示
│   └── test_with_grid_overlay.py # グリッド付き合成
├── webapp/                     # WebUI
│   ├── app.py                  # Flask アプリ
│   └── templates/index.html    # WebUI
├── kawakura/                   # パーツ検索システム
│   ├── assets_png/             # パーツ画像
│   └── main/                   # AI分析
├── made_pictures/              # テスト用画像
└── outputs/                    # 分析結果JSON
```

## ⚙️ 配置設定（最重要）

### キャンバス設定
- **サイズ**: 300×300px
- **中心座標**: (150, 150) = 相対座標(0, 0)

### パーツ配置設定
`face_composer/part_placement_config.py` の `PART_PLACEMENT_CONFIGS` で設定：

| パーツ | 基準位置 | サイズ | 左右間隔 | 備考 |
|--------|----------|--------|----------|------|
| 髪 | (0, 0) | 1.1 | - | 中心配置 |
| 輪郭 | (0, 0) | 1.0 | - | 中心配置 |
| 眉毛 | (0, -15) | 0.2 | 25px | 左右対称 |
| 目 | (0, 15) | 0.2 | 25px | 左右対称 |
| 鼻 | (0, 20) | 0.2 | - | 中心より少し下 |
| 口 | (0, 130) | 0.3 | - | 中心より下 |
| 耳 | (0, 40) | 0.28 | 50px | 左右対称、小さめ |

## 🛠️ 使用方法

### 1. WebUIで合成
```bash
cd vision_rag
PORT=5001 python webapp/app.py
# http://127.0.0.1:5001 にアクセス
```

### 2. コマンドラインで高速テスト
```bash
cd vision_rag/tools
python test_face_composition.py ../outputs/run_xxx.json result.png
```

### 3. グリッド表示でデバッグ
```bash
cd vision_rag/tools  
python create_debug_grid.py ../outputs/run_xxx.json
```

## 🔧 配置調整方法

`face_composer/part_placement_config.py` を編集：

```python
# 例: 鼻を上に移動したい場合
'nose': PartPlacementConfig(
    base_x=0, 
    base_y=10,  # 20 → 10 に変更で上に移動
    initial_scale=0.2
),

# 例: 目の間隔を広げたい場合  
'eye': PartPlacementConfig(
    base_x=0, base_y=15,
    initial_scale=0.2,
    symmetrical_spacing=30  # 25 → 30 に変更で間隔拡大
),
```

## 📊 デバッグ座標系

- **キャンバス**: 300×300px
- **中心**: (150, 150) = 相対座標(0, 0)
- **X軸**: 左がマイナス、右がプラス
- **Y軸**: 上がマイナス、下がプラス

## 🎨 最新の配置調整

✅ キャンバスサイズ: 600×600 → 300×300  
✅ 鼻位置: 50px上に移動 (base_y: 70→20)  
✅ 目・眉: 間隔10px拡大 (spacing: 15→25)  
✅ 耳サイズ: 縮小 (scale: 0.4→0.28)  