#!/usr/bin/env python3
"""
Gemini API統合モジュール
合成結果の座標とサイズを分析・修正するためのGemini連携機能
"""

import os
import base64
import json
import io
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from PIL import Image
import google.generativeai as genai
from .coordinate_converter import CoordinateConverter


class GeminiCoordinateRefiner:
    """Gemini APIを使用した座標修正クラス"""
    
    def __init__(self, api_key: str = "AIzaSyAt-wzZ3WLU1fc6fnzHvDhPsTZJNKnHszU", canvas_size: Tuple[int, int] = (400, 400)):
        """
        初期化
        
        Args:
            api_key: Gemini APIキー
            canvas_size: キャンバスサイズ
        """
        self.api_key = api_key
        self.canvas_size = canvas_size
        self.converter = CoordinateConverter(canvas_size)
        
        genai.configure(api_key=api_key)
        # 利用可能なモデルをテストして選択
        try:
            self.model = genai.GenerativeModel('gemini-1.5-flash')
        except Exception:
            self.model = genai.GenerativeModel('gemini-pro-vision')
        
    def _image_to_base64(self, image_path: Path) -> str:
        """画像をBase64エンコード"""
        with open(image_path, 'rb') as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def _prepare_part_info(self, parts_dict: Dict[str, Dict], current_positions: Dict[str, Any]) -> str:
        """パーツ情報を整理してプロンプト用のテキストを作成"""
        part_info_text = "現在の顔パーツ配置情報:\n\n"
        
        for category, part_data in parts_dict.items():
            part_num = part_data.get('part_num', 'N/A')
            score = part_data.get('score', 0.0)
            
            # 現在の座標情報を取得
            current_pos = current_positions.get(category, {})
            if isinstance(current_pos, dict):
                # 左右対称パーツの場合
                if 'left' in current_pos:
                    left_pos = current_pos['left']
                    right_pos = current_pos['right']
                    part_info_text += f"- {category} #{part_num} (信頼度: {score:.3f})\n"
                    part_info_text += f"  左: x={left_pos[0]}, y={left_pos[1]}, scale={left_pos[2]}\n"
                    part_info_text += f"  右: x={right_pos[0]}, y={right_pos[1]}, scale={right_pos[2]}\n"
                else:
                    # single パーツ
                    single_pos = current_pos.get('single', (200, 200, 0.3))
                    part_info_text += f"- {category} #{part_num} (信頼度: {score:.3f})\n"
                    part_info_text += f"  座標: x={single_pos[0]}, y={single_pos[1]}, scale={single_pos[2]}\n"
            else:
                # タプル形式の場合
                part_info_text += f"- {category} #{part_num} (信頼度: {score:.3f})\n"
                part_info_text += f"  座標: x={current_pos[0]}, y={current_pos[1]}, scale={current_pos[2]}\n"
            part_info_text += "\n"
        
        return part_info_text
    
    def _create_math_coordinate_prompt(self, parts_dict: Dict[str, Dict], current_positions: Dict[str, Any]) -> str:
        """数学座標系用の詳細プロンプトを生成"""
        
        # 画像座標系 → 数学座標系に変換
        math_positions = self.converter.convert_positions_to_math(current_positions)
        
        # パーツ情報をテキスト化（数学座標系）
        part_info_text = "現在の顔パーツ配置（数学座標系 - 中心原点、y軸上向き）:\n\n"
        
        for category, part_data in parts_dict.items():
            part_num = part_data.get('part_num', 'N/A')
            score = part_data.get('score', 0.0)
            
            math_pos = math_positions.get(category, {})
            
            part_info_text += f"🎭 **{category.upper()}** #{part_num} (信頼度: {score:.3f})\n"
            
            if isinstance(math_pos, dict):
                # 左右対称パーツ
                if 'left' in math_pos and 'right' in math_pos:
                    left_coords = math_pos['left']
                    right_coords = math_pos['right']
                    part_info_text += f"   左: x={left_coords[0]:+3d}, y={left_coords[1]:+3d}, scale={left_coords[2]:.2f}\n"
                    part_info_text += f"   右: x={right_coords[0]:+3d}, y={right_coords[1]:+3d}, scale={right_coords[2]:.2f}\n"
                    
                    # 対称性チェック
                    symmetry = abs(abs(left_coords[0]) - abs(right_coords[0]))
                    part_info_text += f"   📐 対称性偏差: {symmetry}px\n"
            else:
                # 単一パーツ
                if len(math_pos) >= 3:
                    part_info_text += f"   座標: x={math_pos[0]:+3d}, y={math_pos[1]:+3d}, scale={math_pos[2]:.2f}\n"
                    
                    # 中心からの距離
                    distance = (math_pos[0]**2 + math_pos[1]**2)**0.5
                    part_info_text += f"   📍 中心からの距離: {distance:.1f}px\n"
            
            part_info_text += "\n"
        
        canvas_center = self.canvas_size[0] // 2, self.canvas_size[1] // 2
        grid_info = self.converter.create_math_coordinate_grid()
        
        prompt = f"""
あなたは顔パーツ配置の専門家です。提供されたグリッド付き合成画像を分析して、数学座標系で顔パーツの座標とサイズを最適化してください。

{grid_info}

{part_info_text}

## 調整制約・ガイドライン

### 📏 **座標移動制約（数学座標系）**
- **微調整範囲**: 各パーツは現在位置から ±15px以内
- **大幅調整**: 明らかに不自然な場合のみ ±25px以内  
- **左右対称**: eye, eyebrow, earは x座標の絶対値が等しくなるよう調整

### 🔧 **スケール調整制約**
- **髪・輪郭**: 0.3 ～ 0.8 (現在から±0.2以内)
- **目・眉・鼻・口**: 0.1 ～ 0.4 (現在から±0.1以内)
- **耳**: 0.15 ～ 0.35 (現在から±0.1以内)

### 👥 **解剖学的制約（数学座標系）**
- **目の間隔**: 40-80px (左右のx座標差)
- **眉と目のy距離**: 15-35px (眉のy > 目のy)
- **鼻と口のy距離**: 20-50px (鼻のy > 口のy)
- **左右対称性**: 左右パーツのx座標は符号のみ異なる

## 出力形式
数学座標系（中心原点、y軸上向き）で修正された座標を出力してください：

```python
{{
    'hair': (x, y, scale),
    'eye': {{
        'left': (x, y, scale),
        'right': (x, y, scale)
    }},
    'eyebrow': {{
        'left': (x, y, scale),
        'right': (x, y, scale)
    }},
    'nose': (x, y, scale),
    'mouth': (x, y, scale),
    'ear': {{
        'left': (x, y, scale),
        'right': (x, y, scale)
    }},
    'outline': (x, y, scale)
}}
```

**重要**: 
- 数学座標系（中心0,0、y軸上向き）で出力してください
- グリッド画像を参照して正確な位置を判断してください
- 制約範囲内での微調整を優先してください
- 存在しないパーツは出力に含めないでください
        """
        return prompt

    def _create_refinement_prompt(self, part_info: str, canvas_size: Tuple[int, int]) -> str:
        """座標修正用のプロンプトを生成"""
        canvas_center = (canvas_size[0] // 2, canvas_size[1] // 2)
        
        prompt = f"""
あなたは顔パーツ合成システムの専門家です。提供された合成画像を分析して、顔パーツの配置座標とサイズを最適化してください。

## 画像情報
- キャンバスサイズ: {canvas_size[0]} × {canvas_size[1]}
- キャンバス中心: ({canvas_center[0]}, {canvas_center[1]})

## 現在のパーツ配置
{part_info}

## 座標システムの説明
- 座標系: (x, y) - 左上が(0,0)、右下が({canvas_size[0]}, {canvas_size[1]})
- 各パーツは (x, y, scale) で定義されます
- x, y: パーツ画像の中心座標
- scale: パーツのスケール倍率 (1.0 = 100%)

## 修正指針
画像を詳しく分析して、以下の点を考慮して座標とサイズを調整してください：

1. **顔の比例とバランス**: パーツ間の自然な位置関係
2. **左右対称性**: eye, eyebrow, earの左右バランス
3. **解剖学的正確性**: 顔の構造に従った配置
4. **視覚的違和感**: 不自然に見える部分の修正

## 出力形式
以下のPython辞書形式で、修正された座標を出力してください：

```python
{{
    'hair': (x, y, scale),
    'eye': {{
        'left': (x, y, scale),
        'right': (x, y, scale),
        'single': (x, y, scale)
    }},
    'eyebrow': {{
        'left': (x, y, scale),
        'right': (x, y, scale),
        'single': (x, y, scale)
    }},
    'nose': (x, y, scale),
    'mouth': (x, y, scale),
    'ear': {{
        'left': (x, y, scale),
        'right': (x, y, scale)
    }},
    'outline': (x, y, scale),
    'acc': (x, y, scale),
    'beard': (x, y, scale),
    'glasses': (x, y, scale)
}}
```

**重要**: 存在しないパーツは出力に含めないでください。Pythonコードブロック内で辞書のみを出力し、説明や追加テキストは含めないでください。
        """
        return prompt
    
    def refine_coordinates(
        self,
        composed_image_path: Path,
        parts_dict: Dict[str, Dict],
        current_positions: Dict[str, Any],
        canvas_size: Tuple[int, int] = (400, 400)
    ) -> Optional[Dict[str, Any]]:
        """
        Gemini APIを使用して座標を修正
        
        Args:
            composed_image_path: 合成画像のパス
            parts_dict: パーツ情報辞書
            current_positions: 現在の座標設定
            canvas_size: キャンバスサイズ
            
        Returns:
            修正された座標辞書、エラー時はNone
        """
        try:
            # 画像を読み込み
            image = Image.open(composed_image_path)
            
            # 数学座標系用プロンプトを生成
            prompt = self._create_math_coordinate_prompt(parts_dict, current_positions)
            
            print(f"[DEBUG] Gemini APIに送信中...")
            print(f"[DEBUG] 画像: {composed_image_path}")
            
            # Gemini APIで分析
            response = self.model.generate_content([prompt, image])
            
            if not response.text:
                print(f"[ERROR] Geminiからの応答が空です")
                return None
            
            print(f"[DEBUG] Gemini応答: {response.text[:200]}...")
            
            # レスポンスからPythonコードブロックを抽出
            response_text = response.text.strip()
            
            # ```python ``` ブロックを検索
            start_markers = ['```python', '```']
            end_marker = '```'
            
            code_block = None
            for start_marker in start_markers:
                if start_marker in response_text:
                    start_idx = response_text.find(start_marker) + len(start_marker)
                    end_idx = response_text.find(end_marker, start_idx)
                    if end_idx != -1:
                        code_block = response_text[start_idx:end_idx].strip()
                        break
            
            if not code_block:
                # コードブロックが見つからない場合は全体をコードとして解釈
                code_block = response_text
            
            # Pythonコードを評価して辞書を取得
            try:
                math_refined_positions = eval(code_block)
                if isinstance(math_refined_positions, dict):
                    print(f"[DEBUG] Gemini座標修正成功（数学座標系）: {len(math_refined_positions)}個のパーツ")
                    
                    # 数学座標系 → 画像座標系に変換
                    image_refined_positions = self.converter.convert_positions_to_image(math_refined_positions)
                    
                    print(f"[DEBUG] 座標系変換完了（画像座標系）: {len(image_refined_positions)}個のパーツ")
                    print("[DEBUG] 変換例:")
                    for category, coords in list(image_refined_positions.items())[:3]:
                        print(f"  {category}: {coords}")
                    
                    return image_refined_positions
                else:
                    print(f"[ERROR] 無効な形式: {type(math_refined_positions)}")
                    return None
            except Exception as eval_error:
                print(f"[ERROR] コード評価エラー: {eval_error}")
                print(f"[DEBUG] コードブロック: {code_block}")
                return None
            
        except Exception as e:
            print(f"[ERROR] Gemini API エラー: {e}")
            return None
    
    def test_api_connection(self) -> bool:
        """API接続テスト"""
        try:
            # シンプルなテスト
            test_response = self.model.generate_content("Hello, please respond with 'OK'")
            return test_response.text and "OK" in test_response.text.upper()
        except Exception as e:
            print(f"[ERROR] API接続テスト失敗: {e}")
            return False


if __name__ == "__main__":
    # テスト用コード
    refiner = GeminiCoordinateRefiner()
    
    # API接続テスト
    print("API接続テスト...")
    if refiner.test_api_connection():
        print("✅ API接続成功")
    else:
        print("❌ API接続失敗")