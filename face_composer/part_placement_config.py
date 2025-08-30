#!/usr/bin/env python3
"""
パーツ配置座標計算モジュール
キャンバスサイズに応じて座標とスケールを動的に計算
"""

from typing import Tuple, Dict, Any
import math

class PartPlacementCalculator:
    """パーツ配置座標計算クラス"""
    
    # 基準キャンバスサイズ（座標定義の基準）
    BASE_CANVAS_SIZE = (400, 400)
    BASE_CENTER = (200, 200)
    
    def __init__(self, canvas_size: Tuple[int, int] = (400, 400)):
        """
        初期化
        
        Args:
            canvas_size: 実際のキャンバスサイズ
        """
        self.canvas_size = canvas_size
        self.canvas_center = (canvas_size[0] // 2, canvas_size[1] // 2)
        
        print(f"[DEBUG] キャンバス: {canvas_size}, 中心: {self.canvas_center}")
        print(f"[DEBUG] パーツ座標は固定値を使用（スケーリングなし）")
    
    def calculate_part_position(self, category: str, part_num: int, 
                              is_left_part: bool = False, is_right_part: bool = False) -> Tuple[int, int, float]:
        """
        パーツの配置座標とスケールを計算
        
        Args:
            category: パーツカテゴリ
            part_num: パーツ番号
            is_left_part: 左側パーツかどうか
            is_right_part: 右側パーツかどうか
            
        Returns:
            (final_x, final_y, scale) タプル
        """
        # 基準座標を取得（400x400ベース）
        base_x, base_y, base_scale = self._get_base_coordinates(
            category, part_num, is_left_part, is_right_part
        )
        
        # 中心座標の差分を計算
        center_offset_x = self.canvas_center[0] - self.BASE_CENTER[0]
        center_offset_y = self.canvas_center[1] - self.BASE_CENTER[1]
        
        # オフセットを適用（中心からの相対位置を保持）
        final_x = base_x + center_offset_x
        final_y = base_y + center_offset_y
        
        return final_x, final_y, base_scale
    
    def _get_base_coordinates(self, category: str, part_num: int, 
                            is_left_part: bool = False, is_right_part: bool = False) -> Tuple[int, int, float]:
        """
        基準座標を取得（400x400ベース）
        
        Returns:
            (x, y, scale) タプル
        """
        
        # 基準座標定義（400x400キャンバス用）
        base_positions = {
            'hair': (200, 200, 1.0),
            'eye': {
                'left': (225, 215, 0.2),
                'right': (175, 215, 0.2),
                'single': (200, 215, 0.2)
            },
            'eyebrow': {
                'left': (225, 185, 0.2),
                'right': (175, 185, 0.2),
                'single': (200, 185, 0.2)
            },
            'nose': (200, 230, 0.2),
            'mouth': (200, 255, 0.25),
            'ear': {
                'left': (250, 220, 0.28),
                'right': (150, 220, 0.28)
            },
            'outline': (200, 200, 1.0),
            'acc': (200, 180, 0.3),
            'beard': (200, 300, 0.4),
            'glasses': (200, 215, 0.5)
        }
        
        # カテゴリ別座標取得
        if category in base_positions:
            pos_data = base_positions[category]
            
            if isinstance(pos_data, dict):
                # 左右対称パーツ
                if is_left_part and 'left' in pos_data:
                    return pos_data['left']
                elif is_right_part and 'right' in pos_data:
                    return pos_data['right']
                elif 'single' in pos_data:
                    return pos_data['single']
                else:
                    # フォールバック
                    return list(pos_data.values())[0]
            else:
                # 単一パーツ
                return pos_data
        
        # デフォルト座標
        return (200, 200, 0.3)
    
    def is_symmetrical_category(self, category: str) -> bool:
        """カテゴリが左右対称パーツかどうか"""
        symmetrical_categories = {'eye', 'eyebrow', 'ear'}
        return category in symmetrical_categories
    
    def get_canvas_info(self) -> Dict[str, Any]:
        """キャンバス情報を返す"""
        return {
            'canvas_size': self.canvas_size,
            'canvas_center': self.canvas_center,
            'scale_factor': self.scale_factor,
            'base_canvas_size': self.BASE_CANVAS_SIZE,
            'base_center': self.BASE_CENTER
        }

# 互換性のための定数（廃止予定）
CANVAS_CENTER = (200, 200)  # 基準値のみ、実際の計算では使用しない#!/usr/bin/env python3