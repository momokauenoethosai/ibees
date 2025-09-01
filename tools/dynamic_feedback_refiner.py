#!/usr/bin/env python3
"""
動的座標フィードバック反復調整システム
実際の座標変化をGeminiにリアルタイムで伝え、視覚重視の調整を実現
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
model = genai.GenerativeModel('gemini-2.0-flash-lite')

# 調整ステップ
ADJUSTMENT_STEPS = {
    'position': {
        'up': (0, -5), 'down': (0, 5),
        'up_slight': (0, -3), 'down_slight': (0, 3)
    },
    'scale': {
        'bigger': 0.05, 'smaller': -0.05, 
        'bigger_slight': 0.03, 'smaller_slight': -0.03
    },
    'symmetrical': {
        'closer': (-3, +3),  'wider': (+3, -3),
        'closer_big': (-5, +5), 'wider_big': (+5, -5)
    }
}

def calculate_current_measurements(positions: dict) -> dict:
    """現在の座標から実際の測定値を計算"""
    measurements = {}
    
    # 目の間隔
    if 'eye' in positions:
        left_x = positions['eye']['left'][0]
        right_x = positions['eye']['right'][0] 
        eye_distance = abs(left_x - right_x)
        measurements['eye_distance'] = eye_distance
    
    # 眉と目の距離
    if 'eye' in positions and 'eyebrow' in positions:
        eye_y = positions['eye']['left'][1]  # 左目のy座標
        eyebrow_y = positions['eyebrow']['left'][1]  # 左眉のy座標
        eyebrow_eye_gap = abs(eye_y - eyebrow_y)
        measurements['eyebrow_eye_gap'] = eyebrow_eye_gap
    
    # 鼻と口の距離
    if 'nose' in positions and 'mouth' in positions:
        nose_y = positions['nose'][1]
        mouth_y = positions['mouth'][1]
        nose_mouth_gap = abs(mouth_y - nose_y)
        measurements['nose_mouth_gap'] = nose_mouth_gap
    
    # 口のサイズ
    if 'mouth' in positions:
        mouth_scale = positions['mouth'][2]
        measurements['mouth_scale'] = mouth_scale
    
    return measurements

def create_dynamic_feedback_prompt(parts_list: list, iteration: int, current_positions: dict, adjustment_history: list) -> str:
    """実際の座標変化を含む動的フィードバックプロンプト"""
    parts_str = ", ".join(parts_list)
    
    # 現在の実際の測定値
    measurements = calculate_current_measurements(current_positions)
    
    # 測定値のテキスト化
    current_stats = f"""
## 📏 **現在の実際の測定値**（画像と一致）
- **目の間隔**: {measurements.get('eye_distance', 'N/A')}px
- **眉と目の距離**: {measurements.get('eyebrow_eye_gap', 'N/A')}px  
- **鼻と口の距離**: {measurements.get('nose_mouth_gap', 'N/A')}px
- **口のscale**: {measurements.get('mouth_scale', 'N/A'):.2f}
"""
    
    # 変化履歴のテキスト化
    change_history = ""
    if adjustment_history:
        change_history = f"\n## 📊 実際の座標変化履歴\n"
        
        # 初期値
        initial_measurements = {
            'eye_distance': 50,
            'eyebrow_eye_gap': 30,
            'nose_mouth_gap': 25,
            'mouth_scale': 0.25
        }
        
        prev_measurements = initial_measurements
        
        for i, hist in enumerate(adjustment_history, 1):
            adjustments = hist.get('adjustments', {})
            human_score = hist.get('human_perception_score', 0.0)
            
            change_history += f"**反復{i}**: "
            
            # 調整内容
            if adjustments:
                adj_summary = []
                for part, adj in adjustments.items():
                    if 'symmetrical' in adj:
                        adj_summary.append(f"{part}_間隔({adj['symmetrical']})")
                    elif 'position' in adj:
                        adj_summary.append(f"{part}_位置({adj['position']})")
                    elif 'scale' in adj:
                        adj_summary.append(f"{part}_サイズ({adj['scale']})")
                
                change_history += ", ".join(adj_summary)
            else:
                change_history += "調整なし"
            
            change_history += f" → 人間感覚: {human_score:.2f}\n"
        
        # 最新の変化を強調
        if len(adjustment_history) > 0:
            current_eye_distance = measurements.get('eye_distance', 50)
            initial_eye_distance = 50
            total_change = current_eye_distance - initial_eye_distance
            
            change_history += f"\n🔍 **重要な変化**:\n"
            change_history += f"- 目の間隔: {initial_eye_distance}px → {current_eye_distance}px (変化{total_change:+d}px)\n"
            if abs(total_change) > 20:
                change_history += f"⚠️ 目の間隔が大幅に変化しています！画像で確認してください。\n"
    
    return f"""
🔄 動的フィードバック反復調整 - 反復{iteration}

{current_stats}
{change_history}

## 🎯 重要：視覚情報を最優先

### ⚠️ **視覚 vs テキストの優先度**
1. **画像を最優先**: 提供された画像の実際の見た目を重視
2. **数値は参考程度**: 上記の測定値は参考、実際の画像と異なれば画像を信じる
3. **変化の確認**: 過去の画像と比較して、調整の効果を確認

### 🚨 **過剰調整の防止**
- 目の間隔が30px未満になったら `symmetrical` 調整を停止
- 同じ調整を3回繰り返したら別のアプローチ
- 人間感覚スコアが悪化したら前回に戻る
- 眉と目の距離が10px未満になったら即座に調整を停止してください。近すぎます。
- パーツ同士が重なったらおかしいです。例：目と眉が重なる。

## 📐 理想的な測定値
- **目の間隔**: 35-50px（自然）
- **眉と目の距離**: 18-28px（親近感）
- **鼻と口の距離**: 20-30px（バランス）
- **口のscale**: 0.18-0.30（調和）

## ⚠️ **異常レベル（強制修正必要）**
- 目の間隔80px以上 → 即座に `symmetrical: closer_big`
- 口のscale 0.45以上 → 即座に `scale: smaller`
- 眉と目の距離60px以上 → 即座に `position: down`

## ポイント
- 目と眉の感覚/目と鼻の感覚が大事です
- 目や眉の左右の感覚も大事です。
- 口や目や鼻のサイズも重要です。

## 📋 対象パーツ: {parts_str}

## ⚙️ 調整指示（安全な範囲内）

### 🎯 対称パーツ間隔調整
- **closer/wider**: 3px変更（安全）
- **closer_big/wider_big**: 5px変更（要注意）

### 📍 位置・サイズ調整
- **slight系**: 3px or 0.03倍（推奨）
- **通常系**: 5px or 0.05倍（慎重に）

## 出力形式
```json
{{
  "visual_analysis": {{
    "actual_eye_distance_from_image": "35px（画像から直接測定）",
    "previous_change_effect": "50px→35px に改善、適切な方向",
    "stop_adjustment_needed": false
  }},
  "debug_analysis": {{
    "human_perception_score": 0.8,
    "anomalies_detected": ["なし"]
  }},
  "adjustments": {{
    "eyebrow": {{"position": "down_slight", "reason": "眉と目をあと少し近づけて完成度向上"}}
  }},
  "satisfied": false,
  "notes": "画像を見ると改善されているが、もう少し調整で完璧に"
}}
```

**最重要**: 
1. **画像を最優先で見て判断**
2. **数値は参考程度**
3. **過剰調整を絶対に避ける**
4. **変化が見えない場合は調整停止**
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

def apply_dynamic_adjustments(current_positions: dict, adjustments: dict) -> dict:
    """動的フィードバック対応の調整システム"""
    new_positions = json.loads(json.dumps(current_positions))
    
    for category, adj_info in adjustments.items():
        if category not in new_positions:
            continue
            
        pos_adj = adj_info.get('position')
        scale_adj = adj_info.get('scale')
        symmetrical_adj = adj_info.get('symmetrical')
        reason = adj_info.get('reason', '')
        current_pos = new_positions[category]
        
        if isinstance(current_pos, dict) and ('left' in current_pos and 'right' in current_pos):
            # 左右対称パーツの処理
            left_x, left_y, left_scale = current_pos['left']
            right_x, right_y, right_scale = current_pos['right']
            old_distance = abs(left_x - right_x)
            
            # 過剰調整防止チェック
            if symmetrical_adj in ['closer', 'closer_big'] and old_distance <= 25:
                print(f"  [SAFETY] {category}: 間隔{old_distance}px、これ以上狭めると異常になるため調整スキップ")
                continue
            elif symmetrical_adj in ['wider', 'wider_big'] and old_distance >= 70:
                print(f"  [SAFETY] {category}: 間隔{old_distance}px、これ以上広げると異常になるため調整スキップ")
                continue
            
            # 対称調整（間隔変更）
            if symmetrical_adj and symmetrical_adj in ADJUSTMENT_STEPS['symmetrical']:
                left_dx, right_dx = ADJUSTMENT_STEPS['symmetrical'][symmetrical_adj]
                new_left_x = left_x + left_dx
                new_right_x = right_x + right_dx
                
                new_positions[category]['left'] = (new_left_x, left_y, left_scale)
                new_positions[category]['right'] = (new_right_x, right_y, right_scale)
                
                new_distance = abs(new_left_x - new_right_x)
                print(f"  [SYMMETRICAL] {category}: {symmetrical_adj} - 間隔{old_distance}px→{new_distance}px")
            
            # 通常の位置調整（両方同時移動）
            elif pos_adj and pos_adj in ADJUSTMENT_STEPS['position']:
                dx, dy = ADJUSTMENT_STEPS['position'][pos_adj]
                new_positions[category]['left'] = (left_x + dx, left_y + dy, left_scale)
                new_positions[category]['right'] = (right_x + dx, right_y + dy, right_scale)
                print(f"  [POSITION] {category}: {pos_adj} - 両方を({dx}, {dy})移動")
            
            # スケール調整
            if scale_adj and scale_adj in ADJUSTMENT_STEPS['scale']:
                scale_delta = ADJUSTMENT_STEPS['scale'][scale_adj]
                new_left_scale = max(0.1, min(1.0, left_scale + scale_delta))
                new_right_scale = max(0.1, min(1.0, right_scale + scale_delta))
                
                left_coords = new_positions[category]['left']
                right_coords = new_positions[category]['right']
                new_positions[category]['left'] = (left_coords[0], left_coords[1], new_left_scale)
                new_positions[category]['right'] = (right_coords[0], right_coords[1], new_right_scale)
                print(f"  [SCALE] {category}: {scale_adj} - scale {left_scale:.2f}→{new_left_scale:.2f}")
            
        else:
            # 単一パーツの処理
            if len(current_pos) >= 3:
                x, y, scale = current_pos
                
                # 位置調整
                if pos_adj and pos_adj in ADJUSTMENT_STEPS['position']:
                    dx, dy = ADJUSTMENT_STEPS['position'][pos_adj]
                    x, y = x + dx, y + dy
                
                # スケール調整（安全範囲チェック）
                if scale_adj and scale_adj in ADJUSTMENT_STEPS['scale']:
                    scale_delta = ADJUSTMENT_STEPS['scale'][scale_adj]
                    new_scale = scale + scale_delta
                    
                    # 安全範囲チェック
                    if category == 'mouth' and new_scale > 0.4:
                        print(f"  [SAFETY] {category}: scale{new_scale:.2f}は異常に大きいため0.4に制限")
                        new_scale = 0.4
                    elif new_scale < 0.1:
                        print(f"  [SAFETY] {category}: scale{new_scale:.2f}は異常に小さいため0.1に制限")
                        new_scale = 0.1
                    
                    scale = new_scale
                
                new_positions[category] = (x, y, scale)
                print(f"  [SINGLE] {category}: {adj_info} → ({x}, {y}, {scale:.2f})")
        
        print(f"    理由: {reason}")
    
    return new_positions

def dynamic_feedback_test(json_path: str, max_iterations: int = 5):
    """動的フィードバック反復調整テスト"""
    
    session_id = f"dynamic_{time.strftime('%Y%m%d_%H%M%S')}"
    print(f"🔄 動的フィードバック反復調整テスト開始")
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
    
    # 初期測定値を表示
    initial_measurements = calculate_current_measurements(current_positions)
    print(f"📐 初期測定値: 目間隔{initial_measurements.get('eye_distance')}px, 眉目間隔{initial_measurements.get('eyebrow_eye_gap')}px")
    
    iteration_images = []
    adjustment_history = []
    composer = FaceComposer(canvas_size=(400, 400))
    start_time = time.time()
    
    # 3. 反復ループ
    for iteration in range(max_iterations):
        print(f"\n--- 🔄 反復 {iteration + 1}/{max_iterations} （動的フィードバック）---")
        
        # 現在の測定値を表示
        current_measurements = calculate_current_measurements(current_positions)
        print(f"📏 現在測定値: 目間隔{current_measurements.get('eye_distance')}px, 眉目間隔{current_measurements.get('eyebrow_eye_gap')}px")
        
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
            iteration_filename = f"dynamic_{iteration + 1}_{timestamp}.png"
            iteration_path = Path("outputs") / iteration_filename
            current_image.save(iteration_path)
            iteration_images.append(current_image)
            print(f"💾 反復画像: {iteration_path}")
            
        except Exception as e:
            print(f"❌ 合成エラー: {e}")
            break
        
        # 3.2 動的フィードバックでGemini分析
        print(f"🤖🔄 Gemini動的分析中（{len(iteration_images)+1}画像）...")
        try:
            prompt = create_dynamic_feedback_prompt(
                list(parts_dict.keys()), 
                iteration + 1, 
                current_positions,  # 現在の実際の座標を渡す
                adjustment_history
            )
            
            # 全履歴画像を送信
            gemini_input = [prompt, original_image] + iteration_images
            
            print(f"📷 送信: 元画像 + 反復1~{iteration+1} = 計{len(gemini_input)-1}画像")
            print(f"📊 実座標フィードバック: 目間隔{current_measurements.get('eye_distance')}px等をGeminiに通知")
            
            response = model.generate_content(gemini_input)
            
            if not response.text:
                print(f"❌ 反復{iteration + 1}: Gemini応答なし")
                break
            
            response_text = response.text
            print(f"📋 Gemini応答: {len(response_text)}文字")
            
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
                visual_analysis = adjustment_result.get('visual_analysis', {})
                debug_analysis = adjustment_result.get('debug_analysis', {})
                
                actual_eye_distance = visual_analysis.get('actual_eye_distance_from_image', 'N/A')
                change_effect = visual_analysis.get('previous_change_effect', 'N/A')
                stop_needed = visual_analysis.get('stop_adjustment_needed', False)
                
                human_score = debug_analysis.get('human_perception_score', 0.0)
                anomalies = debug_analysis.get('anomalies_detected', [])
                
                satisfied = adjustment_result.get('satisfied', False)
                adjustments = adjustment_result.get('adjustments', {})
                notes = adjustment_result.get('notes', '')
                
                print(f"\n📊 動的フィードバック結果:")
                print(f"  👁️ Geminiが認識した目間隔: {actual_eye_distance}")
                print(f"  📈 変化効果の認識: {change_effect}")
                print(f"  🛑 調整停止判定: {stop_needed}")
                print(f"  🎭 人間感覚スコア: {human_score:.2f}")
                print(f"  🚨 検出異常: {anomalies}")
                print(f"  😊 満足度: {satisfied}")
                print(f"  🔧 調整指示: {adjustments}")
                print(f"  💬 コメント: {notes}")
                
                # 履歴記録
                history_entry = {
                    'iteration': iteration + 1,
                    'similarity_before': 0.0 if iteration == 0 else adjustment_history[-1].get('similarity_after', 0.0),
                    'similarity_after': human_score,
                    'human_perception_score': human_score,
                    'actual_measurements': current_measurements,
                    'gemini_perceived_distance': actual_eye_distance,
                    'adjustments': adjustments,
                    'notes': notes,
                    'stop_needed': stop_needed
                }
                adjustment_history.append(history_entry)
                
                # 停止条件チェック
                if satisfied or not adjustments or stop_needed:
                    if stop_needed:
                        print(f"🛑 反復{iteration + 1}: Geminiが調整停止を判定")
                    else:
                        print(f"✅ 反復{iteration + 1}: 目標達成！（人間感覚: {human_score:.2f}）")
                    break
                
                # 動的調整を適用
                print("⚙️ 動的フィードバック調整適用中...")
                current_positions = apply_dynamic_adjustments(current_positions, adjustments)
                
            except json.JSONDecodeError as e:
                print(f"❌ JSON解析失敗: {e}")
                print(f"生レスポンス: {response_text}")
                break
                
        except Exception as e:
            print(f"❌ Gemini処理エラー: {e}")
            break
    
    # 4. 結果レポート
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\n🏁 動的フィードバックテスト完了")
    print(f"⏱️ 総処理時間: {total_time:.1f}秒")
    print(f"🖼️ 生成画像数: {len(iteration_images)}枚")
    
    # 測定値変化の追跡
    if adjustment_history:
        print(f"\n📏 測定値変化追跡:")
        print(f"  初期 → 最終")
        
        final_measurements = calculate_current_measurements(current_positions)
        for key in ['eye_distance', 'eyebrow_eye_gap', 'nose_mouth_gap', 'mouth_scale']:
            initial_val = initial_measurements.get(key, 'N/A')
            final_val = final_measurements.get(key, 'N/A')
            if isinstance(initial_val, (int, float)) and isinstance(final_val, (int, float)):
                change = final_val - initial_val
                print(f"  {key}: {initial_val} → {final_val} ({change:+.1f})")
            else:
                print(f"  {key}: {initial_val} → {final_val}")

def main():
    if len(sys.argv) < 2:
        print("使用法:")
        print("  python dynamic_feedback_refiner.py <json_path> [max_iterations]")
        print("\n例:")
        print("  python dynamic_feedback_refiner.py outputs/run_20250830_170700.json")
        print("  python dynamic_feedback_refiner.py outputs/run_20250830_170700.json 3")
        return
    
    json_path = sys.argv[1]
    max_iterations = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    
    dynamic_feedback_test(json_path, max_iterations)

if __name__ == "__main__":
    main()