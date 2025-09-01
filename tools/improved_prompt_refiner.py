#!/usr/bin/env python3
"""
改良プロンプトによる反復調整システム
座標解釈エラーを修正し、左右対称パーツに特化した指示を追加
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

# 改良された調整ステップ
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
        # 左右対称パーツ専用
        'closer': (-3, +3),  # 左目を右に、右目を左に
        'wider': (+3, -3),   # 左目を左に、右目を右に
        'closer_big': (-5, +5),  # 大きく間隔調整
        'wider_big': (+5, -5)
    }
}

def create_improved_prompt(parts_list: list, iteration: int, adjustment_history: list) -> str:
    """座標解釈エラーを修正した改良プロンプト"""
    parts_str = ", ".join(parts_list)
    
    # 調整履歴のテキスト化
    history_text = ""
    if adjustment_history:
        history_text = f"\n## 📊 調整履歴（{len(adjustment_history)}回実施）\n"
        for i, hist in enumerate(adjustment_history, 1):
            similarity_change = hist.get('similarity_after', 0.0) - hist.get('similarity_before', 0.0)
            adjustments = hist.get('adjustments', {})
            human_score = hist.get('human_perception_score', 0.0)
            
            history_text += f"**反復{i}**: "
            if adjustments:
                adj_summary = ", ".join([f"{part}({adj.get('position', '')}{adj.get('scale', '')}{adj.get('symmetrical', '')})" 
                                       for part, adj in adjustments.items()])
                history_text += f"{adj_summary}\n"
                history_text += f"  → 類似度: {hist.get('similarity_before', 0.0):.2f}→{hist.get('similarity_after', 0.0):.2f} / 人間感覚: {human_score:.2f}\n"
            else:
                history_text += "調整なし\n"
    
    return f"""
🔧 改良版座標調整分析 - 反復{iteration}

## 📷 画像分析
提供された画像を分析し、人間が自然と感じる顔バランスに調整してください。

{history_text}

## ⚠️ 重要：座標解釈の正確な理解

### 🎯 左右対称パーツ（eye, eyebrow, ear）の特殊指示
**間隔調整（最重要）**:
- `"eye": {{"symmetrical": "closer"}}` → 左目を右に、右目を左に移動（間隔を狭める）
- `"eye": {{"symmetrical": "wider"}}` → 左目を左に、右目を右に移動（間隔を広げる）
- `"eye": {{"symmetrical": "closer_big"}}` → 間隔を大幅に狭める（5px）
- `"eye": {{"symmetrical": "wider_big"}}` → 間隔を大幅に広げる（5px）

**個別移動**:
- `"eye": {{"position": "up"}}` → 左右両方の目を上に移動
- `"eye": {{"position": "down"}}` → 左右両方の目を下に移動

### 📏 単一パーツの座標系
- **up移動**: 画面上部に移動（y値減少）
- **down移動**: 画面下部に移動（y値増加）

## 🚨 解剖学的異常の強制検出

### 現在の座標値（参考）
- eye_left: (225, 215), eye_right: (175, 215) → **間隔50px**
- eyebrow_left: (225, 185), eyebrow_right: (175, 185) → **眉と目の距離30px**

### 📏 厳格な自然さ判定基準

#### 🎯 **satisfied=true** の厳しい条件（ALL必須）
1. **目の間隔**: 32-60px（黄金比）
2. **眉と目の距離**: 10-40px（親近感のある自然さ）
3. **鼻と口の距離**: 10-40px（中顔面の調和）
4. **口のサイズ**: scale 0.18-0.35（顔全体との調和）
5. **human_perception_score**: 0.95以上（非常に自然）
6. **元画像との印象**: 顔の雰囲気・表情が十分類似

#### 🚫 **satisfied=false** の条件（1つでも該当で継続）
- 目の間隔が50px以上または40px未満
- 眉と目の距離が28px以上または15px未満  
- 鼻と口の距離が35px以上または18px未満
- 口のscaleが0.32以上または0.15未満
- human_perception_score が 0.8未満
- パーツ配置・比率に違和感あり

#### ⚠️ **異常レベル（強制修正必要）**
- 目の間隔80px以上 → 即座に `symmetrical: closer_big`
- 口のscale 0.45以上 → 即座に `scale: smaller`
- 眉と目の距離60px以上 → 即座に `position: down`

## 📋 対象パーツ: {parts_str}

## ⚙️ 調整指示オプション

### 位置調整
- **単一パーツ**: up, down, up_slight, down_slight
- **対称パーツの間隔**: closer, wider, closer_big, wider_big

### サイズ調整  
- bigger, smaller, bigger_slight, smaller_slight

## 🎯 調整例（正確な解釈）

**❌ 間違った指示**:
- `"eye": {{"position": "right"}}` → 両目が右に移動（間隔変わらず）

**✅ 正しい指示**:
- `"eye": {{"symmetrical": "closer"}}` → 左目が右に、右目が左に（間隔が狭まる）
- `"eye": {{"symmetrical": "wider"}}` → 左目が左に、右目が右に（間隔が広がる）

## 出力形式
```json
{{
  "debug_analysis": {{
    "human_perception_score": 0.6,
    "current_measurements": {{
      "eye_distance": "50px（理想42-48px、基準外のため要調整）",
      "eyebrow_eye_gap": "30px（理想18-25px、基準外のため要調整）",
      "nose_mouth_gap": "25px（理想20-30px、許容範囲）",
      "mouth_scale": "0.25（理想0.18-0.28、許容範囲）"
    }},
    "fails_satisfaction_criteria": [
      "目の間隔50pxが理想範囲42-48pxを超過",
      "眉と目の距離30pxが理想範囲18-25pxを超過",
      "human_perception_score 0.6が最低基準0.85未満"
    ]
  }},
  "adjustments": {{
    "eye": {{"symmetrical": "closer", "reason": "50px→45px程度に狭めて理想範囲に"}},
    "eyebrow": {{"position": "down", "reason": "30px→23px程度に縮めて親近感向上"}}
  }},
  "satisfied": false,
  "notes": "厳格基準により継続調整が必要：目間隔と眉位置が理想範囲外"
}}
```

**重要**: satisfied=trueは非常に厳しい基準です。少しでも理想範囲を外れたら false で継続してください。

**最重要**: 
1. 目の間隔調整は必ず `symmetrical` を使用
2. 座標値と人間の感覚を両方考慮
3. 異常値（目間隔70px+, 口scale0.4+）は強制修正
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

def apply_improved_adjustments(current_positions: dict, adjustments: dict) -> dict:
    """改良された調整システム（左右対称パーツ対応）"""
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
            
            # 対称調整（間隔変更）
            if symmetrical_adj and symmetrical_adj in ADJUSTMENT_STEPS['symmetrical']:
                left_dx, right_dx = ADJUSTMENT_STEPS['symmetrical'][symmetrical_adj]
                new_left_x = left_x + left_dx
                new_right_x = right_x + right_dx
                
                new_positions[category]['left'] = (new_left_x, left_y, left_scale)
                new_positions[category]['right'] = (new_right_x, right_y, right_scale)
                
                old_distance = abs(left_x - right_x)
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
                
                new_positions[category]['left'] = (new_positions[category]['left'][0], new_positions[category]['left'][1], new_left_scale)
                new_positions[category]['right'] = (new_positions[category]['right'][0], new_positions[category]['right'][1], new_right_scale)
                print(f"  [SCALE] {category}: {scale_adj} - scale {left_scale:.2f}→{new_left_scale:.2f}")
            
        else:
            # 単一パーツの処理
            if len(current_pos) >= 3:
                x, y, scale = current_pos
                
                # 位置調整
                if pos_adj and pos_adj in ADJUSTMENT_STEPS['position']:
                    dx, dy = ADJUSTMENT_STEPS['position'][pos_adj]
                    x, y = x + dx, y + dy
                
                # スケール調整
                if scale_adj and scale_adj in ADJUSTMENT_STEPS['scale']:
                    scale_delta = ADJUSTMENT_STEPS['scale'][scale_adj]
                    scale = max(0.1, min(1.0, scale + scale_delta))
                
                new_positions[category] = (x, y, scale)
                print(f"  [SINGLE] {category}: {adj_info} → ({x}, {y}, {scale:.2f})")
        
        print(f"    理由: {reason}")
    
    return new_positions

def improved_refinement_test(json_path: str, max_iterations: int = 5):
    """改良プロンプトによる反復調整テスト"""
    
    session_id = f"improved_{time.strftime('%Y%m%d_%H%M%S')}"
    print(f"🚀 改良プロンプト反復調整テスト開始")
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
    
    # 2. 初期座標（問題のある設定で開始）
    current_positions = {
        'hair': (200, 200, 1.0),
        'eye': {'left': (225, 215, 0.2), 'right': (175, 215, 0.2)},  # 50px間隔
        'eyebrow': {'left': (225, 185, 0.2), 'right': (175, 185, 0.2)},
        'nose': (200, 230, 0.2),
        'mouth': (200, 255, 0.25),  # やや大きめ
        'ear': {'left': (250, 220, 0.28), 'right': (150, 220, 0.28)},
        'outline': (200, 200, 1.0),
        'acc': (200, 180, 0.3),
        'beard': (200, 300, 0.4),
        'glasses': (200, 215, 0.5)
    }
    
    # 初期状態の座標分析
    eye_distance = abs(current_positions['eye']['left'][0] - current_positions['eye']['right'][0])
    mouth_scale = current_positions['mouth'][2]
    print(f"📐 初期状態: 目の間隔{eye_distance}px, 口scale{mouth_scale:.2f}")
    
    iteration_images = []
    adjustment_history = []
    composer = FaceComposer(canvas_size=(400, 400))
    start_time = time.time()
    
    # 3. 反復ループ
    for iteration in range(max_iterations):
        print(f"\n--- 🚀 反復 {iteration + 1}/{max_iterations} （改良版）---")
        
        # 現在の座標状態を表示
        if 'eye' in current_positions:
            eye_left_x = current_positions['eye']['left'][0]
            eye_right_x = current_positions['eye']['right'][0]
            current_distance = abs(eye_left_x - eye_right_x)
            print(f"📏 現在の目間隔: {current_distance}px （左{eye_left_x}, 右{eye_right_x}）")
        
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
            iteration_filename = f"improved_{iteration + 1}_{timestamp}.png"
            iteration_path = Path("outputs") / iteration_filename
            current_image.save(iteration_path)
            iteration_images.append(current_image)
            print(f"💾 反復画像: {iteration_path}")
            
        except Exception as e:
            print(f"❌ 合成エラー: {e}")
            break
        
        # 3.2 改良プロンプトでGemini分析
        print(f"🤖🚀 Gemini改良分析中（{len(iteration_images)+1}画像）...")
        try:
            prompt = create_improved_prompt(list(parts_dict.keys()), iteration + 1, adjustment_history)
            
            # 全履歴画像を送信
            gemini_input = [prompt, original_image] + iteration_images
            
            print(f"📷 送信: 元画像 + 反復1~{iteration+1} = 計{len(gemini_input)-1}画像")
            
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
                debug_analysis = adjustment_result.get('debug_analysis', {})
                human_score = debug_analysis.get('human_perception_score', 0.0)
                anomalies = debug_analysis.get('anomalies_detected', [])
                
                satisfied = adjustment_result.get('satisfied', False)
                adjustments = adjustment_result.get('adjustments', {})
                notes = adjustment_result.get('notes', '')
                
                print(f"\n📊 改良分析結果:")
                print(f"  🎭 人間感覚スコア: {human_score:.2f}")
                print(f"  🚨 検出異常: {anomalies}")
                print(f"  😊 満足度: {satisfied}")
                print(f"  🔧 調整指示: {adjustments}")
                print(f"  💬 コメント: {notes}")
                
                # 履歴記録
                history_entry = {
                    'iteration': iteration + 1,
                    'similarity_before': 0.0 if iteration == 0 else adjustment_history[-1].get('similarity_after', 0.0),
                    'similarity_after': human_score,  # 人間感覚スコアを類似度として使用
                    'human_perception_score': human_score,
                    'adjustments': adjustments,
                    'notes': notes,
                    'anomalies': anomalies
                }
                adjustment_history.append(history_entry)
                
                # 満足または調整なしなら終了
                if satisfied or not adjustments:
                    print(f"✅ 反復{iteration + 1}: 目標達成！（人間感覚: {human_score:.2f}）")
                    break
                
                # 改良された相対調整を適用
                print("⚙️ 改良調整システム適用中...")
                current_positions = apply_improved_adjustments(current_positions, adjustments)
                
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
    
    print(f"\n🏁 改良プロンプトテスト完了")
    print(f"⏱️ 総処理時間: {total_time:.1f}秒")
    print(f"🖼️ 生成画像数: {len(iteration_images)}枚")
    print(f"📁 保存先: outputs/improved_*.png")
    
    # 最終座標確認
    if 'eye' in current_positions:
        final_eye_distance = abs(current_positions['eye']['left'][0] - current_positions['eye']['right'][0])
        final_mouth_scale = current_positions['mouth'][2]
        print(f"\n📐 最終状態:")
        print(f"  目の間隔: {eye_distance}px → {final_eye_distance}px")
        print(f"  口のscale: {mouth_scale:.2f} → {final_mouth_scale:.2f}")
    
    if adjustment_history:
        print(f"\n📈 人間感覚スコア変化:")
        for hist in adjustment_history:
            score = hist.get('human_perception_score', 0.0)
            print(f"  反復{hist['iteration']}: {score:.2f}")

def main():
    if len(sys.argv) < 2:
        print("使用法:")
        print("  python improved_prompt_refiner.py <json_path> [max_iterations]")
        print("\n例:")
        print("  python improved_prompt_refiner.py outputs/run_20250830_170700.json")
        print("  python improved_prompt_refiner.py outputs/run_20250830_170700.json 3")
        return
    
    json_path = sys.argv[1]
    max_iterations = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    
    improved_refinement_test(json_path, max_iterations)

if __name__ == "__main__":
    main()