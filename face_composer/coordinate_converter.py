#!/usr/bin/env python3
"""
座標系変換モジュール
画像座標系（左上原点、y軸下向き）↔ 数学座標系（中心原点、y軸上向き）の変換
"""

from typing import Dict, Any, Tuple, Union


class CoordinateConverter:
    """座標系変換クラス"""
    
    def __init__(self, canvas_size: Tuple[int, int] = (400, 400)):
        """
        初期化
        
        Args:
            canvas_size: キャンバスサイズ (width, height)
        """
        self.canvas_size = canvas_size
        self.canvas_center = (canvas_size[0] // 2, canvas_size[1] // 2)
    
    def image_to_math(self, x: int, y: int) -> Tuple[int, int]:
        """
        画像座標系 → 数学座標系に変換
        
        画像座標系: 左上(0,0), x右向き, y下向き
        数学座標系: 中心(0,0), x右向き, y上向き
        
        Args:
            x, y: 画像座標系での座標
            
        Returns:
            (math_x, math_y): 数学座標系での座標
        """
        center_x, center_y = self.canvas_center
        
        math_x = x - center_x  # 中心を原点に
        math_y = center_y - y  # y軸を反転（上向きに）
        
        return math_x, math_y
    
    def math_to_image(self, math_x: int, math_y: int) -> Tuple[int, int]:
        """
        数学座標系 → 画像座標系に変換
        
        Args:
            math_x, math_y: 数学座標系での座標
            
        Returns:
            (x, y): 画像座標系での座標
        """
        center_x, center_y = self.canvas_center
        
        x = math_x + center_x  # 原点を中心に戻す
        y = center_y - math_y  # y軸を反転（下向きに）
        
        return x, y
    
    def convert_positions_to_math(self, image_positions: Dict[str, Any]) -> Dict[str, Any]:
        """
        パーツ座標辞書を画像座標系 → 数学座標系に変換
        
        Args:
            image_positions: 画像座標系でのパーツ座標辞書
            
        Returns:
            数学座標系でのパーツ座標辞書
        """
        math_positions = {}
        
        for category, pos_data in image_positions.items():
            if isinstance(pos_data, dict):
                # 左右対称パーツ
                math_pos = {}
                for side, coords in pos_data.items():
                    if isinstance(coords, (tuple, list)) and len(coords) >= 3:
                        x, y, scale = coords[0], coords[1], coords[2]
                        math_x, math_y = self.image_to_math(x, y)
                        math_pos[side] = (math_x, math_y, scale)
                math_positions[category] = math_pos
            elif isinstance(pos_data, (tuple, list)) and len(pos_data) >= 3:
                # 単一パーツ
                x, y, scale = pos_data[0], pos_data[1], pos_data[2]
                math_x, math_y = self.image_to_math(x, y)
                math_positions[category] = (math_x, math_y, scale)
        
        return math_positions
    
    def convert_positions_to_image(self, math_positions: Dict[str, Any]) -> Dict[str, Any]:
        """
        パーツ座標辞書を数学座標系 → 画像座標系に変換
        
        Args:
            math_positions: 数学座標系でのパーツ座標辞書
            
        Returns:
            画像座標系でのパーツ座標辞書
        """
        image_positions = {}
        
        for category, pos_data in math_positions.items():
            if isinstance(pos_data, dict):
                # 左右対称パーツ
                image_pos = {}
                for side, coords in pos_data.items():
                    if isinstance(coords, (tuple, list)) and len(coords) >= 3:
                        math_x, math_y, scale = coords[0], coords[1], coords[2]
                        x, y = self.math_to_image(math_x, math_y)
                        image_pos[side] = (x, y, scale)
                image_positions[category] = image_pos
            elif isinstance(pos_data, (tuple, list)) and len(pos_data) >= 3:
                # 単一パーツ
                math_x, math_y, scale = pos_data[0], pos_data[1], pos_data[2]
                x, y = self.math_to_image(math_x, math_y)
                image_positions[category] = (x, y, scale)
        
        return image_positions
    
    def create_math_coordinate_grid(self) -> str:
        """数学座標系用のグリッド説明テキストを生成"""
        center_x, center_y = self.canvas_center
        half_width = center_x
        half_height = center_y
        
        grid_text = f"""
## 座標系情報（数学座標系）

### 座標系定義
- **原点**: 画面中心 (0, 0)
- **X軸**: 右向きが正 (-{half_width} ～ +{half_width})
- **Y軸**: 上向きが正 (-{half_height} ～ +{half_height})
- **グリッド**: 25px間隔（薄線）, 50px間隔（太線・数値）

### 座標範囲
- **左端**: x = -{half_width}
- **右端**: x = +{half_width}  
- **下端**: y = -{half_height}
- **上端**: y = +{half_height}

### パーツ配置の目安
- **髪**: y = +50～-50 (顔上部～中央)
- **眉**: y = +15～+35 (目より上)
- **目**: y = +15 (中央やや上)
- **鼻**: y = +30～-30 (中央)
- **口**: y = -55～-25 (中央やや下)
- **耳**: x = ±50～±100 (左右)
        """
        return grid_text


def test_coordinate_conversion():
    """座標変換のテスト"""
    converter = CoordinateConverter((400, 400))
    
    print("=== 座標系変換テスト ===")
    print(f"キャンバス: {converter.canvas_size}, 中心: {converter.canvas_center}")
    
    # テスト座標
    test_coords = [
        (200, 200),  # 中心
        (225, 215),  # 右目
        (175, 185),  # 左眉
        (100, 100),  # 左上
        (300, 300),  # 右下
    ]
    
    print("\n画像座標 → 数学座標 → 画像座標:")
    for img_x, img_y in test_coords:
        math_x, math_y = converter.image_to_math(img_x, img_y)
        back_x, back_y = converter.math_to_image(math_x, math_y)
        print(f"  ({img_x:3d}, {img_y:3d}) → ({math_x:+4d}, {math_y:+4d}) → ({back_x:3d}, {back_y:3d})")


if __name__ == "__main__":
    test_coordinate_conversion()