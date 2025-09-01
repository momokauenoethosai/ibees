#!/usr/bin/env python3
"""
Gemini入出力デバッグシステム
プロンプト、送信画像、応答結果をすべて保存・確認する
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

def save_debug_session(
    session_id: str,
    iteration: int,
    prompt: str,
    images: list,
    response_text: str,
    adjustment_result: dict = None,
    error: str = None
):
    """デバッグセッションを詳細保存"""
    
    debug_dir = Path("outputs/debug_sessions") / session_id
    debug_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. プロンプトを保存
    prompt_file = debug_dir / f"iteration_{iteration}_prompt.txt"
    prompt_file.write_text(prompt, encoding='utf-8')
    
    # 2. 送信画像を保存
    for i, image in enumerate(images):
        if isinstance(image, Image.Image):
            image_file = debug_dir / f"iteration_{iteration}_input_image_{i+1}.png"
            image.save(image_file)
    
    # 3. 生レスポンスを保存
    response_file = debug_dir / f"iteration_{iteration}_response.txt"
    response_file.write_text(response_text, encoding='utf-8')
    
    # 4. 解析結果を保存
    if adjustment_result:
        result_file = debug_dir / f"iteration_{iteration}_parsed_result.json"
        result_file.write_text(
            json.dumps(adjustment_result, ensure_ascii=False, indent=2),
            encoding='utf-8'
        )
    
    # 5. エラー情報を保存
    if error:
        error_file = debug_dir / f"iteration_{iteration}_error.txt"
        error_file.write_text(f"エラー: {error}\n\n生レスポンス:\n{response_text}", encoding='utf-8')
    
    # 6. セッションサマリーを更新
    summary_file = debug_dir / "session_summary.json"
    
    summary_data = {}
    if summary_file.exists():
        try:
            summary_data = json.loads(summary_file.read_text(encoding='utf-8'))
        except:
            summary_data = {}
    
    summary_data[f"iteration_{iteration}"] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "prompt_length": len(prompt),
        "images_sent": len(images),
        "response_length": len(response_text),
        "parsed_successfully": adjustment_result is not None,
        "similarity_score": adjustment_result.get('comparison_analysis', {}).get('similarity_score', 0.0) if adjustment_result else 0.0,
        "adjustments_count": len(adjustment_result.get('adjustments', {})) if adjustment_result else 0,
        "satisfied": adjustment_result.get('satisfied', False) if adjustment_result else False,
        "error": error
    }
    
    summary_file.write_text(
        json.dumps(summary_data, ensure_ascii=False, indent=2),
        encoding='utf-8'
    )
    
    print(f"🐛 デバッグ保存: {debug_dir}")
    return debug_dir

def create_triple_image_prompt_with_debug(parts_list: list, iteration: int, adjustment_history: list) -> str:
    """デバッグ情報付きの3画像分析プロンプト"""
    parts_str = ", ".join(parts_list)
    
    # 調整履歴のテキスト化
    history_text = ""
    if adjustment_history:
        history_text = f"\n## 📊 調整履歴（{len(adjustment_history)}回実施）\n"
        for i, hist in enumerate(adjustment_history, 1):
            similarity_change = hist.get('similarity_after', 0.0) - hist.get('similarity_before', 0.0)
            adjustments = hist.get('adjustments', {})
            
            history_text += f"**反復{i}**: "
            if adjustments:
                adj_summary = ", ".join([f"{part}({adj.get('position', '')}{adj.get('scale', '')})" 
                                       for part, adj in adjustments.items()])
                history_text += f"{adj_summary} → 類似度変化 {similarity_change:+.2f}\n"
            else:
                history_text += "調整なし\n"
    
    return f"""
🐛 デバッグモード: 反復{iteration} - 詳細分析

3つの顔画像を厳密に比較分析してください：

## 📷 画像の説明
**画像1**: 元の実際の顔写真（最終目標）
**画像2**: 前回反復の似顔絵（参考・比較用）  
**画像3**: 最新の似顔絵（調整対象）

## 🎯 デバッグ重点項目

### ⚠️ 人間が認識する不自然さの検出
以下の点を特に厳しく評価してください：

1. **目の間隔異常**: 
   - 異常に離れている（顔幅の50%以上）
   - 異常に近い（顔幅の20%未満）

2. **パーツサイズ異常**:
   - 口が顔に対して異常に大きい（顔幅の40%以上）
   - 目が異常に小さい/大きい（バランス崩壊）

3. **位置関係異常**:
   - 眉と目が異常に離れている（親しみやすさ皆無）
   - 鼻と口が異常に離れている（間延び）

### 📐 数値的ガイドライン
画像3で以下が確認されたら **強制修正** してください：
- 目の左右間隔 > 80px → `"eye": {{"position": "right"}}` で狭める
- 口サイズ > 顔幅35% → `"mouth": {{"scale": "smaller"}}` で縮小
- 眉-目距離 > 35px → `"eyebrow": {{"position": "down"}}` で近づける
{history_text}

## 📋 対象パーツ: {parts_str}

## ⚙️ 調整指示オプション
**位置**: up, down, left, right (5px) / up_slight, down_slight, left_slight, right_slight (3px)
**サイズ**: bigger, smaller (0.05倍) / bigger_slight, smaller_slight (0.03倍)

## 🎯 **重要: 人間の感覚に合わせた厳格評価**

画像3を見て、人間が「明らかにおかしい」と感じる部分があれば、類似度スコアに関係なく **強制修正** してください。

**判定基準**:
- 類似度0.7でも明らかな異常があれば → satisfied: false
- 類似度0.5でも自然なバランスなら → satisfied: true

## 出力形式
```json
{{
  "debug_analysis": {{
    "human_perception_score": 0.3,
    "anomalies_detected": [
      "目の間隔が異常に広い（約80px）",
      "口が顔全体に対して異常に大きい"
    ]
  }},
  "comparison_analysis": {{
    "similarity_score": 0.4,
    "relationship_differences": [
      "画像1と比較して画像3は目が異常に離れており不自然",
      "画像1の自然な口サイズに対し画像3は明らかに大きすぎる"
    ]
  }},
  "adjustments": {{
    "eye": {{"position": "right", "reason": "異常に離れた目の間隔を人間の感覚に合わせて修正"}},
    "mouth": {{"scale": "smaller", "reason": "異常に大きい口を自然なサイズに修正"}}
  }},
  "satisfied": false,
  "notes": "人間の感覚では不自然な部分を優先的に修正"
}}
```

**最重要**: 類似度スコアより **人間が見て自然かどうか** を優先してください。
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

def debug_gemini_io_test(json_path: str):
    """Gemini入出力のデバッグテスト"""
    
    session_id = f"debug_{time.strftime('%Y%m%d_%H%M%S')}"
    print(f"🐛 Gemini I/O デバッグセッション開始")
    print(f"📋 セッションID: {session_id}")
    print(f"📄 JSONファイル: {json_path}")
    
    # 1. 元画像とパーツ情報を読み込み
    try:
        original_image_path = get_original_image_path(json_path)
        if not original_image_path or not original_image_path.exists():
            print(f"❌ 元画像が見つかりません: {original_image_path}")
            return
        
        original_image = Image.open(original_image_path).resize((400, 400), Image.LANCZOS)
        print(f"📸 元画像: {original_image_path} → リサイズ(400x400)")
        
        parts_dict = load_parts_from_json(json_path)
        print(f"✅ {len(parts_dict)}個のパーツ読み込み: {list(parts_dict.keys())}")
        
    except Exception as e:
        print(f"❌ 初期化エラー: {e}")
        return
    
    # 2. 初期座標で合成
    initial_positions = {
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
    
    print("\n🎨 初期合成実行中...")
    composer = FaceComposer(canvas_size=(400, 400))
    
    try:
        composed_image = composer.compose_face_with_custom_positions(
            base_image_path=None,
            parts_dict=parts_dict,
            custom_positions=initial_positions
        )
        
        if not composed_image:
            print("❌ 合成失敗")
            return
        
        # RGB変換とリサイズ
        if composed_image.mode == 'RGBA':
            background = Image.new('RGB', composed_image.size, (255, 255, 255))
            background.paste(composed_image, mask=composed_image.split()[-1])
            composed_image = background
        
        current_image = composed_image.resize((400, 400), Image.LANCZOS)
        
        # 合成画像を保存
        composed_path = Path("outputs") / f"debug_composed_{session_id}.png"
        current_image.save(composed_path)
        print(f"💾 合成画像: {composed_path}")
        
    except Exception as e:
        print(f"❌ 合成エラー: {e}")
        return
    
    # 3. デバッグ用プロンプト作成
    print("\n📝 デバッグプロンプト作成中...")
    prompt = create_triple_image_prompt_with_debug(list(parts_dict.keys()), 1, [])
    
    print(f"📏 プロンプト長: {len(prompt)}文字")
    print(f"📷 送信画像: 元画像(400x400) + 合成画像(400x400)")
    
    # 4. Gemini分析実行
    print("\n🤖 Gemini分析実行中...")
    
    try:
        # 送信データ
        gemini_input = [prompt, original_image, current_image]
        
        start_time = time.time()
        response = model.generate_content(gemini_input)
        end_time = time.time()
        
        response_text = response.text if response.text else "応答なし"
        analysis_time = end_time - start_time
        
        print(f"✅ Gemini応答取得: {analysis_time:.1f}秒")
        print(f"📏 応答長: {len(response_text)}文字")
        print(f"📋 応答プレビュー:\n{response_text[:300]}...")
        
        # 5. JSON解析試行
        adjustment_result = None
        error = None
        
        try:
            # JSON解析
            response_text_clean = response_text.strip()
            if '```json' in response_text_clean:
                start_idx = response_text_clean.find('```json') + 7
                end_idx = response_text_clean.find('```', start_idx)
                code_block = response_text_clean[start_idx:end_idx].strip()
            else:
                code_block = response_text_clean
            
            adjustment_result = json.loads(code_block)
            
            print(f"\n✅ JSON解析成功:")
            print(f"  類似度スコア: {adjustment_result.get('comparison_analysis', {}).get('similarity_score', 'N/A')}")
            print(f"  人間感覚スコア: {adjustment_result.get('debug_analysis', {}).get('human_perception_score', 'N/A')}")
            print(f"  調整数: {len(adjustment_result.get('adjustments', {}))}")
            print(f"  満足度: {adjustment_result.get('satisfied', False)}")
            
        except Exception as e:
            error = f"JSON解析エラー: {e}"
            print(f"❌ {error}")
        
        # 6. デバッグデータを保存
        debug_dir = save_debug_session(
            session_id=session_id,
            iteration=1,
            prompt=prompt,
            images=[original_image, current_image],
            response_text=response_text,
            adjustment_result=adjustment_result,
            error=error
        )
        
        print(f"\n🔍 デバッグファイル確認:")
        print(f"  プロンプト: {debug_dir}/iteration_1_prompt.txt")
        print(f"  入力画像: {debug_dir}/iteration_1_input_image_*.png")
        print(f"  生レスポンス: {debug_dir}/iteration_1_response.txt")
        if adjustment_result:
            print(f"  解析結果: {debug_dir}/iteration_1_parsed_result.json")
        print(f"  セッション概要: {debug_dir}/session_summary.json")
        
    except Exception as e:
        error = f"Gemini呼び出しエラー: {e}"
        print(f"❌ {error}")
        
        # エラーもデバッグ保存
        save_debug_session(
            session_id=session_id,
            iteration=1,
            prompt=prompt,
            images=[original_image, current_image],
            response_text="",
            error=error
        )

def main():
    if len(sys.argv) < 2:
        print("使用法:")
        print("  python debug_gemini_io.py <json_path>")
        print("\n例:")
        print("  python debug_gemini_io.py outputs/run_20250830_170700.json")
        
        json_files = list(Path("outputs").glob("run_*.json"))
        if json_files:
            print(f"\n📁 利用可能なJSONファイル:")
            for f in sorted(json_files)[-3:]:
                print(f"  {f}")
        return
    
    json_path = sys.argv[1]
    debug_gemini_io_test(json_path)

if __name__ == "__main__":
    main()