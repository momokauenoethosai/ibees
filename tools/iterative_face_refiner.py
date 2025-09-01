#!/usr/bin/env python3
"""
顔類似度ベースの反復調整システム
元の顔写真と合成画像を比較して、パーツ位置関係を元画像に近づける
"""

import json
import sys
import time
from pathlib import Path
from PIL import Image
import google.generativeai as genai

# パッケージパスを追加
sys.path.append(str(Path(__file__).parent.parent))
from face_composer.face_composer import FaceComposer

# Gemini設定
GEMINI_API_KEY = "AIzaSyAt-wzZ3WLU1fc6fnzHvDhPsTZJNKnHszU"
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.5-pro')

# 相対調整ステップ
ADJUSTMENT_STEPS = {
    'position': {
        'up': (0, -5), 'down': (0, 5), 'left': (-5, 0), 'right': (5, 0),
        'up_slight': (0, -3), 'down_slight': (0, 3), 'left_slight': (-3, 0), 'right_slight': (3, 0)
    },
    'scale': {
        'bigger': 0.05, 'smaller': -0.05, 'bigger_slight': 0.03, 'smaller_slight': -0.03
    }
}

def create_face_comparison_prompt_with_history(parts_list: list, adjustment_history: list, iteration: int) -> str:
    """顔比較用のGeminiプロンプト（座標系説明付き）"""
    parts_str = ", ".join(parts_list)
    
    # 調整履歴のテキスト化
    history_text = ""
    if adjustment_history:
        history_text = f"\n## 📊 過去の調整履歴（反復{iteration}回目）\n"
        for i, hist in enumerate(adjustment_history, 1):
            similarity_before = hist.get('similarity_before', 0.0)
            similarity_after = hist.get('similarity_after', 0.0)
            adjustments = hist.get('adjustments', {})
            
            history_text += f"**反復{i}**:\n"
            if adjustments:
                for part, adj in adjustments.items():
                    pos = adj.get('position', '')
                    scale = adj.get('scale', '')
                    reason = adj.get('reason', '')
                    history_text += f"  - {part}: {pos} {scale} ({reason})\n"
                history_text += f"  → 類似度: {similarity_before:.2f} → {similarity_after:.2f}\n"
            else:
                history_text += "  - 調整なし\n"
            history_text += "\n"
    
    return f"""
2つの顔画像を比較分析してください：
- **左側**: 元の実際の顔写真（目標とする顔）
- **右側**: 現在の合成画像（調整対象）

## 🎯 分析目標
右側の合成画像を、左側の元画像のパーツ位置関係に近づけてください。
{history_text}

## 📐 重要：座標系の理解
**画像座標系では：**
- **up移動**: パーツが画面上部に移動（y値が減少）
- **down移動**: パーツが画面下部に移動（y値が増加）
- **left移動**: パーツが画面左に移動（x値が減少）
- **right移動**: パーツが画面右に移動（x値が増加）

## 🔍 重要：パーツ間の相対的関係性に注目

**分析の基本方針**:
個別パーツの絶対的位置ではなく、**パーツ同士の相対的な関係性とバランス**を元画像に近づけることが目標です。

**重点比較項目**:
1. **目と眉の間隔感**: 元画像では目と眉が近い/遠い → 合成画像でも同様の間隔に
2. **鼻と口の距離感**: 元画像の鼻-口間隔 → 合成画像でも同じ比率に
3. **顔全体に対するパーツ比率**: 元画像で口が顔幅の○％ → 合成画像でも同比率に
4. **パーツの密集度**: 元画像で目・鼻・口が密集/散らばり → 同様のバランスに
5. **表情の印象**: 元画像の表情の強さ・特徴 → 同等の印象に

## 📋 対象パーツ（必ずこの名前を使用）
{parts_str}

## ⚙️ 調整指示オプション
**位置調整**:
- up/down/left/right: 5px移動
- up_slight/down_slight/left_slight/right_slight: 3px移動

**サイズ調整**:
- bigger/smaller: 0.05倍変更
- bigger_slight/smaller_slight: 0.03倍変更

## 🎯 相対的バランス調整の具体例

**パーツ間関係性の比較**:
- 元画像で目と眉の間隔が狭い → 合成画像でも近づける → `"eyebrow": {{"position": "down_slight"}}`
- 元画像で鼻と口の距離が短い → 合成画像でも詰める → `"mouth": {{"position": "up"}}`
- 元画像で目が顔の中心寄り → 合成画像でも密集させる → `"eye": {{"position": "right"}}` (右目を中央寄りに)

**顔全体の印象バランス**:
- 元画像で口が顔幅の大きな割合 → 同じ印象に → `"mouth": {{"scale": "bigger"}}`  
- 元画像で目と口のサイズバランスが1:1.5 → 同比率に → `"eye": {{"scale": "bigger_slight"}}`

**表情の相対関係**:
- 元画像の笑顔で口と目の距離が近い → 親しみやすさの印象一致 → 両方調整

**重要**: 元画像（左側）を基準とし、合成画像（右側）をそれに近づける調整を指示してください。

## 💡 履歴活用の指針
過去の調整履歴を参考に：
1. **類似度が改善**した調整は同じ方向に継続
2. **類似度が悪化**した調整は逆方向に修正
3. **効果のなかった調整**は別のパーツに注目
4. **座標系を正確に理解**: up=画面上（y減少）、down=画面下（y増加）

## 出力形式
元画像により近づけるための調整指示をJSONで出力：

```json
{{
  "comparison_analysis": {{
    "similarity_score": 0.7,
    "main_differences": [
      "元画像は目と眉の間隔が狭く親近感がある印象だが、合成画像は間隔が広い",
      "元画像は口が顔全体に対して大きく明るい印象だが、合成画像は小さくて控えめ"
    ]
  }},
  "adjustments": {{
    "eyebrow": {{"position": "down_slight", "reason": "元画像のような目と眉の近い親近感ある関係性に"}},
    "mouth": {{"scale": "bigger", "reason": "元画像と同様に顔全体に対する口の存在感を強く"}}
  }},
  "satisfied": false,
  "notes": "パーツ間の相対的関係性を元画像の印象に近づける"
}}
```

## ⚠️ 最重要：相対的関係性重視の分析方針

**❌ 避けるべき分析**:
- "目が高い位置にある" (個別位置の絶対評価)
- "鼻が大きすぎる" (単体サイズの評価)

**✅ 目指すべき分析**:
- "元画像では目と眉が近く親近感があるが、合成画像では離れている"
- "元画像の鼻と口の距離バランスに対し、合成画像は口が相対的に遠い"
- "元画像の表情では口が顔全体の印象を左右する大きさだが、合成画像では控えめ"

## 重要ルール
1. **相対関係性優先**: パーツ同士の関係性・バランスを元画像と一致させる
2. **全体印象重視**: 個別パーツより顔全体の印象の類似を目指す
3. **表情の再現**: 元画像の表情・雰囲気を合成画像でも表現
4. **パーツ名正確性**: [{parts_str}] から正確に選択
5. **段階的調整**: 一度に最大3パーツまで
"""

def get_original_image_path(json_path: str) -> Path:
    """JSONファイルから元画像のパスを取得"""
    with open(json_path, 'r', encoding='utf-8') as f:
        analysis_result = json.load(f)
    
    input_image = analysis_result.get('input_image', '')
    
    # パスの正規化
    if input_image.startswith('/Users/'):
        # 絶対パス
        original_path = Path(input_image)
    else:
        # 相対パス
        original_path = Path(input_image)
    
    # ファイルが存在しない場合は、uploadsディレクトリから検索
    if not original_path.exists():
        filename = Path(input_image).name
        uploads_path = Path("uploads") / filename
        if uploads_path.exists():
            return uploads_path
        
        # made_picturesからも検索
        made_pictures_path = Path("made_pictures") / filename
        if made_pictures_path.exists():
            return made_pictures_path
    
    return original_path if original_path.exists() else None

def create_comparison_image(original_image: Image.Image, composed_image: Image.Image) -> Image.Image:
    """元画像と合成画像を左右に並べた比較画像を作成"""
    
    # 両画像を同じサイズにリサイズ
    target_size = (400, 400)
    original_resized = original_image.resize(target_size, Image.LANCZOS)
    composed_resized = composed_image.resize(target_size, Image.LANCZOS)
    
    # 左右に並べた比較画像を作成
    comparison = Image.new('RGB', (800, 400), 'white')
    comparison.paste(original_resized, (0, 0))
    comparison.paste(composed_resized, (400, 0))
    
    # 境界線を追加
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(comparison)
    
    # 中央の境界線
    draw.line([(400, 0), (400, 400)], fill=(200, 200, 200), width=2)
    
    # ラベル追加
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    draw.text((10, 10), "元画像（目標）", fill=(0, 0, 0), font=font)
    draw.text((410, 10), "合成画像（調整対象）", fill=(0, 0, 0), font=font)
    
    return comparison

def load_parts_from_json(json_path: str) -> dict:
    """JSONファイルからパーツ情報を読み込み"""
    
    def find_part_image_path(category: str, part_num: int) -> Path:
        assets_root = Path("kawakura/assets_png")
        category_mapping = {
            'mouth': 'mouse', 'hair': 'hair', 'eye': 'eye', 'eyebrow': 'eyebrow',
            'nose': 'nose', 'ear': 'ear', 'outline': 'outline', 'acc': 'acc',
            'beard': 'beard', 'glasses': 'glasses', 'extras': 'extras'
        }
        
        folder_name = category_mapping.get(category, category)
        category_folder = assets_root / folder_name
        file_prefix = folder_name
        
        candidates = [
            f"{file_prefix}_{part_num:03d}.png",
            f"{file_prefix}_{part_num:02d}.png", 
            f"{file_prefix}_{part_num}.png"
        ]
        
        for candidate in candidates:
            part_path = category_folder / candidate
            if part_path.exists():
                return part_path
        return None
    
    with open(json_path, 'r', encoding='utf-8') as f:
        analysis_result = json.load(f)
    
    parts_dict = {}
    parts = analysis_result.get('parts', {})
    
    for category, part_info in parts.items():
        selected = part_info.get('selected', {})
        part_num = selected.get('part_num')
        score = selected.get('score', 0.0)
        
        if part_num:
            part_image_path = find_part_image_path(category, part_num)
            if part_image_path:
                parts_dict[category] = {
                    'part_id': f"{category}_{part_num:03d}",
                    'image_path': part_image_path,
                    'part_num': part_num,
                    'score': score
                }
    
    return parts_dict

def apply_relative_adjustments(current_positions: dict, adjustments: dict) -> dict:
    """相対調整を座標に適用"""
    new_positions = json.loads(json.dumps(current_positions))
    
    for category, adj_info in adjustments.items():
        if category not in new_positions:
            continue
            
        pos_adj = adj_info.get('position')
        scale_adj = adj_info.get('scale')
        reason = adj_info.get('reason', '')
        current_pos = new_positions[category]
        
        if isinstance(current_pos, dict):
            # 左右対称パーツ
            for side in ['left', 'right']:
                if side in current_pos and len(current_pos[side]) >= 3:
                    x, y, scale = current_pos[side]
                    
                    if pos_adj and pos_adj in ADJUSTMENT_STEPS['position']:
                        dx, dy = ADJUSTMENT_STEPS['position'][pos_adj]
                        x, y = x + dx, y + dy
                    
                    if scale_adj and scale_adj in ADJUSTMENT_STEPS['scale']:
                        scale_delta = ADJUSTMENT_STEPS['scale'][scale_adj]
                        scale = max(0.1, min(1.0, scale + scale_delta))
                    
                    new_positions[category][side] = (x, y, scale)
            
            print(f"  [ADJUST] {category}: {adj_info} - {reason}")
        else:
            # 単一パーツ
            if len(current_pos) >= 3:
                x, y, scale = current_pos
                
                if pos_adj and pos_adj in ADJUSTMENT_STEPS['position']:
                    dx, dy = ADJUSTMENT_STEPS['position'][pos_adj]
                    x, y = x + dx, y + dy
                
                if scale_adj and scale_adj in ADJUSTMENT_STEPS['scale']:
                    scale_delta = ADJUSTMENT_STEPS['scale'][scale_adj]
                    scale = max(0.1, min(1.0, scale + scale_delta))
                
                new_positions[category] = (x, y, scale)
                
            print(f"  [ADJUST] {category}: {adj_info} → ({x}, {y}, {scale:.2f}) - {reason}")
    
    return new_positions

def face_similarity_refinement_test(json_path: str, max_iterations: int = 5):
    """顔類似度ベースの反復調整テスト"""
    
    print(f"👥 顔類似度調整テスト開始")
    print(f"📄 JSONファイル: {json_path}")
    
    # 1. 元画像とパーツ情報を読み込み
    try:
        original_image_path = get_original_image_path(json_path)
        if not original_image_path or not original_image_path.exists():
            print(f"❌ 元画像が見つかりません: {original_image_path}")
            return
        
        original_image = Image.open(original_image_path)
        print(f"📸 元画像: {original_image_path}")
        
        parts_dict = load_parts_from_json(json_path)
        print(f"✅ {len(parts_dict)}個のパーツ読み込み: {list(parts_dict.keys())}")
        
    except Exception as e:
        print(f"❌ 初期化エラー: {e}")
        return
    
    # 2. 初期座標
    current_positions = {
        'hair': (200, 200, 1.0),
        'eye': {'left': (225, 215, 0.2), 'right': (175, 215, 0.2)},
        'eyebrow': {'left': (225, 185, 0.2), 'right': (175, 185, 0.2)},
        'nose': (200, 230, 0.2),
        'mouth': (200, 255, 0.25),
        'ear': {'left': (250, 220, 0.28), 'right': (150, 220, 0.28)},
        'outline': (200, 200, 1.0),
        'acc': (200, 180, 0.3),
        'beard': (200, 300, 0.4),
        'glasses': (200, 215, 0.5)
    }
    
    iteration_images = []
    adjustment_history = []  # 調整履歴を追跡
    composer = FaceComposer(canvas_size=(400, 400))
    start_time = time.time()
    
    # 3. 反復ループ
    for iteration in range(max_iterations):
        print(f"\n--- 👥 反復 {iteration + 1}/{max_iterations} ---")
        
        # 3.1 現在座標で合成
        print("🎨 合成中...")
        try:
            composed_image = composer.compose_face_with_custom_positions(
                base_image_path=None,
                parts_dict=parts_dict,
                custom_positions=current_positions
            )
            
            if not composed_image:
                print(f"❌ 反復{iteration + 1}: 合成失敗")
                break
            
            # 画像保存
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            iteration_filename = f"similarity_iter_{iteration + 1}_{timestamp}.png"
            iteration_path = Path("outputs") / iteration_filename
            
            # RGB変換して保存
            if composed_image.mode == 'RGBA':
                background = Image.new('RGB', composed_image.size, (255, 255, 255))
                background.paste(composed_image, mask=composed_image.split()[-1])
                composed_image = background
            
            composed_image.save(iteration_path)
            iteration_images.append(iteration_path)
            print(f"💾 反復画像: {iteration_path}")
            
        except Exception as e:
            print(f"❌ 合成エラー: {e}")
            break
        
        # 3.2 比較画像を作成
        print("📊 比較画像作成中...")
        try:
            comparison_image = create_comparison_image(original_image, composed_image)
            
            comparison_filename = f"comparison_{iteration + 1}_{timestamp}.png"
            comparison_path = Path("outputs") / comparison_filename
            comparison_image.save(comparison_path)
            print(f"📋 比較画像: {comparison_path}")
            
        except Exception as e:
            print(f"❌ 比較画像作成エラー: {e}")
            break
        
        # 3.3 Gemini顔比較分析
        print("🤖 Gemini顔比較分析中...")
        try:
            prompt = create_face_comparison_prompt_with_history(
                list(parts_dict.keys()), 
                adjustment_history, 
                iteration + 1
            )
            response = model.generate_content([prompt, comparison_image])
            
            if not response.text:
                print(f"❌ 反復{iteration + 1}: Gemini応答なし")
                break
            
            print(f"📋 Gemini応答: {response.text[:150]}...")
            
            # JSON解析
            response_text = response.text.strip()
            
            # ```json ``` ブロックを検索
            code_block = None
            if '```json' in response_text:
                start_idx = response_text.find('```json') + 7
                end_idx = response_text.find('```', start_idx)
                if end_idx != -1:
                    code_block = response_text[start_idx:end_idx].strip()
            
            if not code_block:
                # JSONブロックが見つからない場合は全体を解析
                code_block = response_text
            
            try:
                adjustment_result = json.loads(code_block)
            except json.JSONDecodeError as e:
                print(f"❌ JSON解析失敗: {e}")
                print(f"生レスポンス: {response_text}")
                break
            
            # 結果確認
            comparison_analysis = adjustment_result.get('comparison_analysis', {})
            similarity_score = comparison_analysis.get('similarity_score', 0.0)
            differences = comparison_analysis.get('main_differences', [])
            
            satisfied = adjustment_result.get('satisfied', False)
            adjustments = adjustment_result.get('adjustments', {})
            notes = adjustment_result.get('notes', '')
            
            print(f"🎯 類似度スコア: {similarity_score:.2f}")
            print(f"📝 主な違い: {differences}")
            print(f"💬 Geminiコメント: {notes}")
            print(f"😊 満足度: {satisfied}")
            print(f"🔧 調整指示: {adjustments}")
            
            # 履歴に記録
            history_entry = {
                'iteration': iteration + 1,
                'similarity_before': 0.0 if iteration == 0 else adjustment_history[-1].get('similarity_after', 0.0),
                'similarity_after': similarity_score,
                'adjustments': adjustments,
                'notes': notes,
                'main_differences': differences
            }
            adjustment_history.append(history_entry)
            
            # 満足または調整なしなら終了
            if satisfied or not adjustments:
                print(f"✅ 反復{iteration + 1}: 目標達成！類似度 {similarity_score:.2f}")
                break
            
            # 相対調整を適用
            print("⚙️ 元画像に近づける調整を適用...")
            print("📈 調整履歴を次回Geminiに提供します")
            current_positions = apply_relative_adjustments(current_positions, adjustments)
            
        except Exception as e:
            print(f"❌ Gemini処理エラー: {e}")
            import traceback
            traceback.print_exc()
            break
    
    # 4. 結果レポート
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\n🏁 顔類似度調整テスト完了")
    print(f"⏱️ 総処理時間: {total_time:.1f}秒")
    print(f"🖼️ 生成画像数: {len(iteration_images)}枚")
    print(f"📁 保存先: outputs/similarity_iter_*.png, outputs/comparison_*.png")
    
    return current_positions, iteration_images

def main():
    if len(sys.argv) < 2:
        print("使用法:")
        print("  python face_similarity_adjuster.py <json_path> [max_iterations]")
        print("\n例:")
        print("  python face_similarity_adjuster.py outputs/run_20250830_164634.json")
        print("  python face_similarity_adjuster.py outputs/run_20250830_164634.json 3")
        
        # 利用可能なファイル表示
        json_files = list(Path("outputs").glob("run_*.json"))
        if json_files:
            print(f"\n📁 利用可能なJSONファイル:")
            for f in sorted(json_files)[-3:]:
                print(f"  {f}")
        return
    
    json_path = sys.argv[1]
    max_iterations = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    
    face_similarity_refinement_test(json_path, max_iterations)

if __name__ == "__main__":
    main()