#!/usr/bin/env python3
"""
デバッグ用グリッド表示モジュール
キャンバスの座標系、中心点、パーツ配置位置を視覚的に表示する
"""

from __future__ import annotations
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys

# パッケージパスを追加
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from face_composer.part_placement_config import PartPlacementCalculator, CANVAS_CENTER

class DebugGridRenderer:
    """デバッグ用グリッド描画クラス"""
    
    def __init__(self, canvas_size: Tuple[int, int] = (600, 600)):
        """
        初期化
        
        Args:
            canvas_size: キャンバスサイズ (width, height)
        """
        self.canvas_size = canvas_size
        self.canvas_center = CANVAS_CENTER
        self.placement_calculator = PartPlacementCalculator()
        
        # グリッド設定
        self.grid_spacing = 50  # グリッド間隔
        self.axis_color = (255, 0, 0, 180)      # 座標軸の色（赤）
        self.grid_color = (200, 200, 200, 100)  # グリッド線の色（薄いグレー）
        self.center_color = (0, 255, 0, 255)    # 中心点の色（緑）
        self.text_color = (0, 0, 0, 255)        # テキストの色（黒）
        
        # フォント設定
        try:
            self.font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 12)
            self.font_small = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 10)
        except:
            self.font = ImageFont.load_default()
            self.font_small = ImageFont.load_default()
    
    def create_grid_overlay(self) -> Image.Image:
        """
        グリッドオーバーレイ画像を作成
        
        Returns:
            グリッド付きの透明画像
        """
        # 透明キャンバス作成
        overlay = Image.new('RGBA', self.canvas_size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        # 1. グリッド線を描画
        self._draw_grid_lines(draw)
        
        # 2. 座標軸を描画
        self._draw_coordinate_axes(draw)
        
        # 3. 中心点を描画
        self._draw_center_point(draw)
        
        # 4. 座標ラベルを描画
        self._draw_coordinate_labels(draw)
        
        return overlay
    
    def _draw_grid_lines(self, draw: ImageDraw.Draw):
        """グリッド線を描画"""
        width, height = self.canvas_size
        
        # 縦線
        for x in range(0, width + 1, self.grid_spacing):
            draw.line([(x, 0), (x, height)], fill=self.grid_color, width=1)
        
        # 横線
        for y in range(0, height + 1, self.grid_spacing):
            draw.line([(0, y), (width, y)], fill=self.grid_color, width=1)
    
    def _draw_coordinate_axes(self, draw: ImageDraw.Draw):
        """座標軸（X軸、Y軸）を描画"""
        center_x, center_y = self.canvas_center
        width, height = self.canvas_size
        
        # X軸（水平線）
        draw.line([(0, center_y), (width, center_y)], fill=self.axis_color, width=2)
        
        # Y軸（垂直線）
        draw.line([(center_x, 0), (center_x, height)], fill=self.axis_color, width=2)
    
    def _draw_center_point(self, draw: ImageDraw.Draw):
        """中心点を描画"""
        center_x, center_y = self.canvas_center
        radius = 5
        
        # 中心点の円
        draw.ellipse([
            center_x - radius, center_y - radius,
            center_x + radius, center_y + radius
        ], fill=self.center_color, outline=(0, 0, 0, 255), width=1)
        
        # 中心点ラベル
        draw.text((center_x + 8, center_y + 8), "(0,0)", fill=self.text_color, font=self.font)
    
    def _draw_coordinate_labels(self, draw: ImageDraw.Draw):
        """座標ラベルを描画"""
        center_x, center_y = self.canvas_center
        
        # X軸のメモリ
        for x in range(0, self.canvas_size[0] + 1, self.grid_spacing):
            if x == center_x:
                continue  # 中心点はスキップ
            
            coord_x = x - center_x  # 相対座標
            draw.text((x - 10, center_y + 10), str(coord_x), fill=self.text_color, font=self.font_small)
        
        # Y軸のメモリ
        for y in range(0, self.canvas_size[1] + 1, self.grid_spacing):
            if y == center_y:
                continue  # 中心点はスキップ
            
            coord_y = y - center_y  # 相対座標
            draw.text((center_x + 10, y - 6), str(coord_y), fill=self.text_color, font=self.font_small)
    
    def add_part_position_markers(self, overlay: Image.Image, selected_parts: Dict[str, Dict]) -> Image.Image:
        """
        パーツの配置位置にマーカーを追加
        
        Args:
            overlay: グリッドオーバーレイ画像
            selected_parts: 選択されたパーツ情報
            
        Returns:
            マーカー追加済みのオーバーレイ画像
        """
        draw = ImageDraw.Draw(overlay)
        
        # パーツカテゴリの色分け
        category_colors = {
            'hair': (255, 100, 100, 200),      # 赤
            'eye': (100, 100, 255, 200),       # 青  
            'eyebrow': (100, 255, 100, 200),   # 緑
            'nose': (255, 255, 100, 200),      # 黄
            'mouth': (255, 100, 255, 200),     # マゼンタ
            'mouse': (255, 100, 255, 200),     # マゼンタ（互換）
            'ear': (100, 255, 255, 200),       # シアン
            'outline': (150, 150, 150, 200),   # グレー
            'beard': (200, 150, 100, 200),     # ブラウン
            'glasses': (50, 50, 50, 200),      # ダークグレー
            'acc': (255, 200, 0, 200),         # オレンジ
        }
        
        for category, part_info in selected_parts.items():
            try:
                part_num = part_info.get('part_num') or self._extract_part_num_from_id(part_info.get('part_id', ''))
                color = category_colors.get(category, (128, 128, 128, 200))
                
                # 対称パーツかどうかチェック
                is_symmetrical = self.placement_calculator.is_symmetrical_category(category)
                
                if is_symmetrical:
                    # 左右両方のマーカーを表示
                    for is_left, is_right in [(True, False), (False, True)]:
                        final_x, final_y, scale = self.placement_calculator.calculate_part_position(
                            category, part_num, 
                            is_left_part=is_left, 
                            is_right_part=is_right
                        )
                        
                        side = "L" if is_left else "R"
                        self._draw_position_marker(draw, final_x, final_y, f"{category}_{part_num:03d}_{side}", color)
                else:
                    # 単一パーツのマーカー
                    final_x, final_y, scale = self.placement_calculator.calculate_part_position(category, part_num)
                    self._draw_position_marker(draw, final_x, final_y, f"{category}_{part_num:03d}", color)
                    
            except Exception as e:
                print(f"マーカー描画エラー ({category}): {e}")
        
        return overlay
    
    def _draw_position_marker(self, draw: ImageDraw.Draw, x: float, y: float, label: str, color: Tuple[int, int, int, int]):
        """位置マーカーを描画"""
        radius = 4
        x, y = int(x), int(y)
        
        # マーカー円
        draw.ellipse([
            x - radius, y - radius,
            x + radius, y + radius
        ], fill=color, outline=(0, 0, 0, 255), width=1)
        
        # 座標ラベル
        center_x, center_y = self.canvas_center
        rel_x = x - center_x
        rel_y = y - center_y
        
        # ラベル位置を調整（マーカーと重ならないように）
        label_x = x + 8
        label_y = y - 15
        
        draw.text((label_x, label_y), f"{label}", fill=self.text_color, font=self.font_small)
        draw.text((label_x, label_y + 12), f"({rel_x},{rel_y})", fill=self.text_color, font=self.font_small)
    
    def _extract_part_num_from_id(self, part_id: str) -> int:
        """part_idから番号を抽出"""
        try:
            if '_' in part_id:
                return int(part_id.split('_')[-1])
            return 0
        except:
            return 0
    
    def create_debug_composition(self, 
                                base_image_path: Optional[Path], 
                                selected_parts: Dict[str, Dict],
                                show_grid: bool = True,
                                show_markers: bool = True) -> Image.Image:
        """
        デバッグ用の合成画像を作成
        
        Args:
            base_image_path: ベース画像（Noneの場合は透明キャンバス）
            selected_parts: 選択されたパーツ
            show_grid: グリッド表示
            show_markers: パーツ位置マーカー表示
            
        Returns:
            デバッグ情報付きの合成画像
        """
        # ベースキャンバス作成
        if base_image_path and base_image_path.exists():
            base = Image.open(base_image_path).convert('RGBA')
            # キャンバスサイズにリサイズ
            base = base.resize(self.canvas_size, Image.Resampling.LANCZOS)
        else:
            # 白背景キャンバス
            base = Image.new('RGBA', self.canvas_size, (255, 255, 255, 255))
        
        # グリッドオーバーレイ作成
        overlay = Image.new('RGBA', self.canvas_size, (0, 0, 0, 0))
        
        if show_grid:
            grid_overlay = self.create_grid_overlay()
            overlay = Image.alpha_composite(overlay, grid_overlay)
        
        if show_markers:
            marker_overlay = self.add_part_position_markers(
                Image.new('RGBA', self.canvas_size, (0, 0, 0, 0)), 
                selected_parts
            )
            overlay = Image.alpha_composite(overlay, marker_overlay)
        
        # 最終合成
        result = Image.alpha_composite(base, overlay)
        
        return result


def create_debug_visualization(json_path: str, output_path: str = None):
    """
    分析結果からデバッグ用視覚化画像を作成
    
    Args:
        json_path: 分析結果JSONのパス
        output_path: 出力画像パス
    """
    import json
    
    print(f"=== デバッグ視覚化作成 ===")
    print(f"分析結果JSON: {json_path}")
    
    # JSONファイル読み込み
    json_file = Path(json_path)
    if not json_file.exists():
        print(f"JSONファイルが見つかりません: {json_path}")
        return
    
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            analysis_result = json.load(f)
    except Exception as e:
        print(f"JSON読み込みエラー: {e}")
        return
    
    # パーツ情報を変換
    parts_dict = {}
    parts = analysis_result.get('parts', {})
    for category, part_info in parts.items():
        selected = part_info.get('selected', {})
        part_num = selected.get('part_num')
        if part_num:
            parts_dict[category] = {
                'part_id': f"{category}_{part_num:03d}",
                'part_num': part_num,
                'score': selected.get('score', 0.0)
            }
    
    if not parts_dict:
        print("選択されたパーツがありません")
        return
    
    # デバッグ視覚化作成
    renderer = DebugGridRenderer()
    debug_image = renderer.create_debug_composition(
        base_image_path=None,  # 透明背景
        selected_parts=parts_dict,
        show_grid=True,
        show_markers=True
    )
    
    # 結果保存
    if not output_path:
        output_path = f"debug_grid_{json_file.stem}.png"
    
    result_path = Path(output_path)
    debug_image.save(result_path)
    print(f"✅ デバッグ視覚化完了: {result_path}")
    
    # パーツ配置座標を表示
    print("\n=== パーツ配置座標 ===")
    calculator = PartPlacementCalculator()
    
    for category, part_info in parts_dict.items():
        part_num = part_info['part_num']
        is_symmetrical = calculator.is_symmetrical_category(category)
        
        if is_symmetrical:
            # 左右対称パーツ
            for is_left, is_right in [(True, False), (False, True)]:
                final_x, final_y, scale = calculator.calculate_part_position(
                    category, part_num, 
                    is_left_part=is_left, 
                    is_right_part=is_right
                )
                side = "左" if is_left else "右"
                rel_x = final_x - CANVAS_CENTER[0]
                rel_y = final_y - CANVAS_CENTER[1]
                print(f"{category}_{part_num:03d}({side}): キャンバス座標({final_x}, {final_y}) / 相対座標({rel_x:+d}, {rel_y:+d}) / スケール{scale}")
        else:
            # 単一パーツ
            final_x, final_y, scale = calculator.calculate_part_position(category, part_num)
            rel_x = final_x - CANVAS_CENTER[0]
            rel_y = final_y - CANVAS_CENTER[1]
            print(f"{category}_{part_num:03d}: キャンバス座標({final_x}, {final_y}) / 相対座標({rel_x:+d}, {rel_y:+d}) / スケール{scale}")


def main():
    """メイン関数"""
    if len(sys.argv) < 2:
        print("使用法:")
        print("  python debug_grid.py <json_path> [output_path]")
        print("\n例:")
        print("  python debug_grid.py outputs/run_1_20250829_182259.json")
        print("  python debug_grid.py outputs/run_1_20250829_182259.json debug_grid_custom.png")
        return
    
    json_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    create_debug_visualization(json_path, output_path)


if __name__ == "__main__":
    main()