#!/usr/bin/env python3
"""
全履歴画像送信による反復調整システム
反復N回目では、元画像 + 反復1~N-1の全画像 + 最新画像をGeminiに送信
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
model = genai.GenerativeModel('gemini-2.0-flash-exp')

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

def create_progressive_analysis_prompt(parts_list: list, iteration: int, adjustment_history: list, total_images: int) -> str:
    """全履歴分析用プロンプト"""
    parts_str = ", ".join(parts_list)
    
    # 画像の順序説明
    image_description = f"""
## 📷 画像の順序（計{total_images}枚）
**画像1**: 元の実際の顔写真（最終目標）
"""
    
    if total_images > 2:
        for i in range(2, total_images):
            image_description += f"**画像{i}**: 反復{i-1}の似顔絵（変化過程）\n"
    
    image_description += f"**画像{total_images}**: 最新の似顔絵（調整対象）"
    
    # 調整履歴のテキスト化
    history_text = ""
    if adjustment_history:
        history_text = f"\n## 📈 変化履歴（{len(adjustment_history)}回の調整）\n"
        for i, hist in enumerate(adjustment_history, 1):
            similarity_before = hist.get('similarity_before', 0.0)
            similarity_after = hist.get('similarity_after', 0.0)
            adjustments = hist.get('adjustments', {})
            
            history_text += f"**反復{i}**: "
            if adjustments:
                adj_summary = ", ".join([f"{part}({adj.get('position', '')}{adj.get('scale', '')})" 
                                       for part, adj in adjustments.items()])
                history_text += f"{adj_summary}\n"
                history_text += f"  → 類似度: {similarity_before:.2f} → {similarity_after:.2f} (変化{similarity_after-similarity_before:+.2f})\n"
            else:
                history_text += "調整なし\n"
            history_text += "\n"
    
    return f"""
🎯 全履歴進化分析 - 反復{iteration}回目

{image_description}

## 🔍 分析目標
**進化の流れを把握し、画像{total_images}（最新）を画像1（元写真）により近づける**

全ての画像を見ることで：
1. **変化の方向性**: どの調整が改善/悪化をもたらしたか
2. **収束パターン**: 目標に近づいているか、迷走しているか  
3. **次の最適手**: 過去の成功/失敗を踏まえた最良の次手

{history_text}

## 🚨 重要：人間の感覚による厳格評価

### 解剖学的異常の強制検出
画像{total_images}で以下が確認されたら **最優先修正**:
1. **目の間隔異常**: 40px未満（狭すぎ）または 70px超（離れすぎ）
2. **口サイズ異常**: 顔幅の15%未満（小さすぎ）または 40%超（大きすぎ）  
3. **眉-目距離異常**: 10px未満（近すぎ）または 40px超（離れすぎ）
4. **全体バランス崩壊**: 人間が見て明らかに不自然

### 📊 評価基準
- **human_perception_score**: 人間が見た自然さ（0.0-1.0）
- **similarity_score**: 元画像との類似度（0.0-1.0）
- **progression_score**: 過去からの改善度（-1.0 to +1.0）

## 📋 対象パーツ: {parts_str}

## ⚙️ 調整戦略

### 🎯 履歴活用による判断
- **成功パターン継続**: 類似度が向上した調整は同方向で継続
- **失敗パターン回避**: 類似度が悪化した調整は逆方向に修正
- **迷走パターン脱出**: 振動している場合は別のパーツに注目

### 🚨 強制修正モード
人間感覚スコア < 0.4 の場合：
- 解剖学的異常を最優先修正
- 類似度より自然さを重視
- より大きな調整（up/down/bigger/smaller）を採用

## 出力形式
```json
{{
  "progression_analysis": {{
    "improvement_trend": "improving|stagnating|declining",
    "successful_adjustments": ["outline scale bigger", "mouth position up"],
    "failed_adjustments": ["eye position left"],
    "next_strategy": "continue_successful|try_different|force_correction"
  }},
  "debug_analysis": {{
    "human_perception_score": 0.3,
    "anomalies_detected": [
      "目の間隔が異常に広い（約75px）",
      "口が顔全体に対して不釣り合いに大きい"
    ]
  }},
  "comparison_analysis": {{
    "similarity_score": 0.4,
    "relationship_differences": [
      "画像1と比較して全体的なパーツバランスが不自然"
    ]
  }},
  "adjustments": {{
    "eye": {{"position": "right", "reason": "異常に離れた目を自然な間隔に強制修正"}},
    "mouth": {{"scale": "smaller", "reason": "異常に大きい口を人間の感覚に合わせて修正"}}
  }},
  "satisfied": false,
  "notes": "人間の感覚で不自然な部分を解剖学的基準で強制修正"
}}
```

**最重要**: 全ての履歴画像の変化を見て、人間が自然と感じる方向に導いてください。
    """

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

def get_original_image_path(json_path: str) -> Path:
    """JSONファイルから元画像のパスを取得"""
    with open(json_path, 'r', encoding='utf-8') as f:
        analysis_result = json.load(f)
    
    input_image = analysis_result.get('input_image', '')
    
    if input_image.startswith('/Users/'):
        original_path = Path(input_image)
    else:
        original_path = Path(input_image)
    
    if not original_path.exists():
        filename = Path(input_image).name
        for search_dir in ["uploads", "made_pictures"]:
            candidate = Path(search_dir) / filename
            if candidate.exists():
                return candidate
    
    return original_path if original_path.exists() else None

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

def save_debug_session(session_id: str, iteration: int, prompt: str, input_images: list, response_text: str, parsed_result: dict = None):
    """デバッグセッションを保存"""
    debug_dir = Path("outputs/debug_sessions") / session_id
    debug_dir.mkdir(parents=True, exist_ok=True)
    
    # プロンプト保存
    prompt_file = debug_dir / f"iteration_{iteration}_prompt.txt"
    prompt_file.write_text(prompt, encoding='utf-8')
    
    # 送信画像保存
    for i, img in enumerate(input_images):
        if isinstance(img, Image.Image):
            img_file = debug_dir / f"iteration_{iteration}_input_image_{i+1}.png"
            img.save(img_file)
    
    # レスポンス保存
    response_file = debug_dir / f"iteration_{iteration}_response.txt"
    response_file.write_text(response_text, encoding='utf-8')
    
    # 解析結果保存
    if parsed_result:
        result_file = debug_dir / f"iteration_{iteration}_parsed.json"
        result_file.write_text(json.dumps(parsed_result, ensure_ascii=False, indent=2), encoding='utf-8')
    
    return debug_dir

def progressive_history_test(json_path: str, max_iterations: int = 5):
    """全履歴送信による反復調整テスト"""
    
    session_id = f"progressive_{time.strftime('%Y%m%d_%H%M%S')}"
    print(f"📈 全履歴進化分析テスト開始")
    print(f"🆔 セッションID: {session_id}")
    print(f"📄 JSONファイル: {json_path}")
    
    # 1. 元画像とパーツ情報を読み込み
    try:
        original_image_path = get_original_image_path(json_path)
        if not original_image_path or not original_image_path.exists():
            print(f"❌ 元画像が見つかりません: {original_image_path}")
            return
        
        original_image = Image.open(original_image_path).resize((400, 400), Image.LANCZOS)
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
    adjustment_history = []
    composer = FaceComposer(canvas_size=(400, 400))
    start_time = time.time()
    
    # 3. 反復ループ
    for iteration in range(max_iterations):
        print(f"\n--- 📈 反復 {iteration + 1}/{max_iterations} （全履歴分析）---")
        
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
            
            # RGB変換とリサイズ
            if composed_image.mode == 'RGBA':
                background = Image.new('RGB', composed_image.size, (255, 255, 255))
                background.paste(composed_image, mask=composed_image.split()[-1])
                composed_image = background
            
            current_image = composed_image.resize((400, 400), Image.LANCZOS)
            
            # 画像保存
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            iteration_filename = f"progressive_{iteration + 1}_{timestamp}.png"
            iteration_path = Path("outputs") / iteration_filename
            current_image.save(iteration_path)
            iteration_images.append(current_image)  # PIL Imageオブジェクトを保存
            print(f"💾 反復画像: {iteration_path}")
            
        except Exception as e:
            print(f"❌ 合成エラー: {e}")
            break
        
        # 3.2 全履歴でGemini分析
        total_images = len(iteration_images) + 1  # 元画像 + 全反復画像
        print(f"🤖📈 Gemini 全履歴分析中（{total_images}画像）...")
        
        try:
            # プロンプト作成
            prompt = create_progressive_analysis_prompt(
                list(parts_dict.keys()),
                iteration + 1, 
                adjustment_history,
                total_images
            )
            
            # 全画像を送信リストに追加
            gemini_input = [prompt, original_image] + iteration_images
            
            print(f"📷 送信: 元画像 + 反復1~{iteration+1} = 計{total_images}画像")
            print(f"📊 変化追跡: 全ての進化過程をGeminiが把握")
            
            # Gemini呼び出し
            response = model.generate_content(gemini_input)
            
            if not response.text:
                print(f"❌ 反復{iteration + 1}: Gemini応答なし")
                break
            
            response_text = response.text
            print(f"📋 Gemini応答: {len(response_text)}文字")
            print(f"📝 プレビュー: {response_text[:200]}...")
            
            # JSON解析
            try:
                response_clean = response_text.strip()
                if '```json' in response_clean:
                    start_idx = response_clean.find('```json') + 7
                    end_idx = response_clean.find('```', start_idx)
                    code_block = response_clean[start_idx:end_idx].strip()
                else:
                    code_block = response_clean
                
                adjustment_result = json.loads(code_block)
                
                # 結果確認
                progression = adjustment_result.get('progression_analysis', {})
                debug_analysis = adjustment_result.get('debug_analysis', {})
                comparison = adjustment_result.get('comparison_analysis', {})
                
                human_score = debug_analysis.get('human_perception_score', 0.0)
                similarity_score = comparison.get('similarity_score', 0.0)
                trend = progression.get('improvement_trend', 'unknown')
                anomalies = debug_analysis.get('anomalies_detected', [])
                
                satisfied = adjustment_result.get('satisfied', False)
                adjustments = adjustment_result.get('adjustments', {})
                notes = adjustment_result.get('notes', '')
                
                print(f"\n📊 分析結果:")
                print(f"  🎭 人間感覚スコア: {human_score:.2f}")
                print(f"  🎯 類似度スコア: {similarity_score:.2f}")
                print(f"  📈 改善傾向: {trend}")
                print(f"  🚨 異常検出: {anomalies}")
                print(f"  😊 満足度: {satisfied}")
                print(f"  🔧 調整指示: {adjustments}")
                print(f"  💬 コメント: {notes}")
                
                # デバッグセッション保存
                debug_dir = save_debug_session(
                    session_id, iteration + 1, prompt, 
                    [original_image] + iteration_images, 
                    response_text, adjustment_result
                )
                
                # 履歴記録
                history_entry = {
                    'iteration': iteration + 1,
                    'similarity_before': 0.0 if iteration == 0 else adjustment_history[-1].get('similarity_after', 0.0),
                    'similarity_after': similarity_score,
                    'human_perception_score': human_score,
                    'improvement_trend': trend,
                    'adjustments': adjustments,
                    'notes': notes,
                    'anomalies': anomalies
                }
                adjustment_history.append(history_entry)
                
                # 満足または調整なしなら終了
                if satisfied or not adjustments:
                    print(f"✅ 反復{iteration + 1}: 目標達成！（人間感覚: {human_score:.2f}, 類似度: {similarity_score:.2f}）")
                    break
                
                # 相対調整を適用
                print("⚙️ 履歴学習による調整を適用...")
                current_positions = apply_relative_adjustments(current_positions, adjustments)
                
            except json.JSONDecodeError as e:
                error_msg = f"JSON解析失敗: {e}"
                print(f"❌ {error_msg}")
                print(f"生レスポンス: {response_text}")
                
                # エラーもデバッグ保存
                save_debug_session(
                    session_id, iteration + 1, prompt,
                    [original_image] + iteration_images,
                    response_text
                )
                break
                
        except Exception as e:
            print(f"❌ Gemini処理エラー: {e}")
            break
    
    # 4. 結果レポート
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\n🏁 全履歴進化分析完了")
    print(f"⏱️ 総処理時間: {total_time:.1f}秒")
    print(f"🖼️ 生成画像数: {len(iteration_images)}枚")
    print(f"🐛 デバッグファイル: outputs/debug_sessions/{session_id}/")
    
    if adjustment_history:
        print(f"\n📈 進化履歴:")
        for hist in adjustment_history:
            human_score = hist.get('human_perception_score', 0.0)
            similarity_score = hist.get('similarity_after', 0.0)
            trend = hist.get('improvement_trend', 'unknown')
            print(f"  反復{hist['iteration']}: 人間{human_score:.2f}, 類似{similarity_score:.2f}, 傾向{trend}")

def main():
    if len(sys.argv) < 2:
        print("使用法:")
        print("  python progressive_history_refiner.py <json_path> [max_iterations]")
        print("\n例:")
        print("  python progressive_history_refiner.py outputs/run_20250830_170700.json")
        print("  python progressive_history_refiner.py outputs/run_20250830_170700.json 3")
        return
    
    json_path = sys.argv[1]
    max_iterations = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    
    progressive_history_test(json_path, max_iterations)

if __name__ == "__main__":
    main()