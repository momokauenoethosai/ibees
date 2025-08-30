#!/usr/bin/env python3
"""
デバッグ用グリッド表示スクリプト
キャンバスの座標系、中心点、パーツ配置位置を視覚的に表示する
"""

import json
import sys
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from typing import Dict, Tuple

# 設定値（参考アプリから取得）
CANVAS_SIZE = (600, 600)
CANVAS_CENTER = (300, 300)

# カテゴリ別基準位置とスケール
CATEGORY_CONFIGS = {
    'outline': {'base_x': 0, 'base_y': 0, 'scale': 1.0},
    'hair': {'base_x': 0, 'base_y': 0, 'scale': 1.1},
    'eyebrow': {'base_x': 0, 'base_y': -15, 'scale': 0.2, 'spacing': 15},
    'eye': {'base_x': 0, 'base_y': 15, 'scale': 0.2, 'spacing': 15},
    'ear': {'base_x': 0, 'base_y': 40, 'scale': 0.4, 'spacing': 50},
    'nose': {'base_x': 0, 'base_y': 0, 'scale': 0.2},
    'mouth': {'base_x': 0, 'base_y': 130, 'scale': 0.3},
    'mouse': {'base_x': 0, 'base_y': 130, 'scale': 0.3},
    'beard': {'base_x': 0, 'base_y': 0, 'scale': 1.2},
    'glasses': {'base_x': 0, 'base_y': 20, 'scale': 0.5},
    'acc': {'base_x': 0, 'base_y': 0, 'scale': 1.8},
}

def create_grid_canvas() -> Image.Image:
    """グリッド付きキャンバスを作成"""
    # 白背景キャンバス
    canvas = Image.new('RGB', CANVAS_SIZE, (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    
    # フォント設定
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 12)
        font_small = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 10)
    except:
        font = ImageFont.load_default()
        font_small = ImageFont.load_default()
    
    # グリッド線（50px間隔）
    grid_color = (220, 220, 220)
    for x in range(0, CANVAS_SIZE[0] + 1, 50):
        draw.line([(x, 0), (x, CANVAS_SIZE[1])], fill=grid_color, width=1)
    for y in range(0, CANVAS_SIZE[1] + 1, 50):
        draw.line([(0, y), (CANVAS_SIZE[0], y)], fill=grid_color, width=1)
    
    # 座標軸（赤色、太線）
    center_x, center_y = CANVAS_CENTER
    draw.line([(0, center_y), (CANVAS_SIZE[0], center_y)], fill=(255, 0, 0), width=3)  # X軸
    draw.line([(center_x, 0), (center_x, CANVAS_SIZE[1])], fill=(255, 0, 0), width=3)  # Y軸
    
    # 中心点（緑色の円）
    radius = 6
    draw.ellipse([
        center_x - radius, center_y - radius,
        center_x + radius, center_y + radius
    ], fill=(0, 255, 0), outline=(0, 0, 0), width=2)
    
    # 中心点ラベル
    draw.text((center_x + 10, center_y + 10), "(0,0)", fill=(0, 0, 0), font=font)
    
    # 座標メモリ
    for x in range(0, CANVAS_SIZE[0] + 1, 50):
        if x == center_x:
            continue
        coord_x = x - center_x
        draw.text((x - 8, center_y + 15), str(coord_x), fill=(100, 100, 100), font=font_small)
    
    for y in range(0, CANVAS_SIZE[1] + 1, 50):
        if y == center_y:
            continue
        coord_y = y - center_y
        draw.text((center_x + 15, y - 6), str(coord_y), fill=(100, 100, 100), font=font_small)
    
    return canvas

def calculate_part_position(category: str, part_num: int, is_left: bool = False, is_right: bool = False) -> Tuple[int, int, float]:
    """パーツの配置位置を計算"""
    config = CATEGORY_CONFIGS.get(category, {'base_x': 0, 'base_y': 0, 'scale': 1.0})
    
    center_x, center_y = CANVAS_CENTER
    base_x = config['base_x']
    base_y = config['base_y']
    scale = config['scale']
    
    # 左右対称パーツの処理
    symmetrical_x = 0
    if 'spacing' in config:
        if is_left:
            symmetrical_x = config['spacing']  # 左パーツは右側に
        elif is_right:
            symmetrical_x = -config['spacing']  # 右パーツは左側に
    
    final_x = center_x + base_x + symmetrical_x
    final_y = center_y + base_y
    
    return final_x, final_y, scale

def add_part_markers(canvas: Image.Image, selected_parts: Dict[str, Dict]) -> Image.Image:
    """パーツ位置マーカーを追加"""
    draw = ImageDraw.Draw(canvas)
    
    try:
        font_small = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 10)
    except:
        font_small = ImageFont.load_default()
    
    # カテゴリ別色分け
    colors = {
        'hair': (255, 100, 100),     # 赤
        'eye': (100, 100, 255),      # 青
        'eyebrow': (100, 255, 100),  # 緑
        'nose': (255, 255, 100),     # 黄
        'mouth': (255, 100, 255),    # マゼンタ
        'mouse': (255, 100, 255),    # マゼンタ
        'ear': (100, 255, 255),      # シアン
        'outline': (150, 150, 150),  # グレー
        'beard': (200, 150, 100),    # ブラウン
        'glasses': (50, 50, 50),     # ダークグレー
        'acc': (255, 200, 0),        # オレンジ
    }
    
    for category, part_info in selected_parts.items():
        part_num = part_info.get('part_num', 0)
        color = colors.get(category, (128, 128, 128))
        
        # 対称パーツかチェック
        config = CATEGORY_CONFIGS.get(category, {})
        is_symmetrical = 'spacing' in config
        
        if is_symmetrical:
            # 左右両方のマーカー
            for is_left, is_right in [(True, False), (False, True)]:
                x, y, scale = calculate_part_position(category, part_num, is_left, is_right)
                side = "L" if is_left else "R"
                
                # マーカー描画
                draw.ellipse([x-5, y-5, x+5, y+5], fill=color, outline=(0, 0, 0), width=1)
                
                # ラベル
                center_x, center_y = CANVAS_CENTER
                rel_x = x - center_x
                rel_y = y - center_y
                draw.text((x + 8, y - 15), f"{category}_{side}", fill=(0, 0, 0), font=font_small)
                draw.text((x + 8, y + 5), f"({rel_x:+d},{rel_y:+d})", fill=(0, 0, 0), font=font_small)
        else:
            # 単一パーツ
            x, y, scale = calculate_part_position(category, part_num)
            
            # マーカー描画
            draw.ellipse([x-5, y-5, x+5, y+5], fill=color, outline=(0, 0, 0), width=1)
            
            # ラベル
            center_x, center_y = CANVAS_CENTER
            rel_x = x - center_x
            rel_y = y - center_y
            draw.text((x + 8, y - 15), f"{category}", fill=(0, 0, 0), font=font_small)
            draw.text((x + 8, y + 5), f"({rel_x:+d},{rel_y:+d})", fill=(0, 0, 0), font=font_small)
    
    return canvas

def create_debug_visualization(json_path: str, output_path: str = None):
    """デバッグ視覚化を作成"""
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
    
    # グリッドキャンバス作成
    canvas = create_grid_canvas()
    
    # パーツマーカー追加
    canvas_with_markers = add_part_markers(canvas, parts_dict)
    
    # 結果保存
    if not output_path:
        output_path = f"debug_grid_{json_file.stem}.png"
    
    result_path = Path(output_path)
    canvas_with_markers.save(result_path)
    print(f"✅ デバッグ視覚化完了: {result_path}")
    
    # パーツ配置座標を表示
    print(f"\n=== キャンバス情報 ===")
    print(f"サイズ: {CANVAS_SIZE[0]}x{CANVAS_SIZE[1]}px")
    print(f"中心: ({CANVAS_CENTER[0]}, {CANVAS_CENTER[1]}) = 相対座標(0, 0)")
    
    print(f"\n=== パーツ配置座標 ===")
    for category, part_info in parts_dict.items():
        part_num = part_info['part_num']
        config = CATEGORY_CONFIGS.get(category, {})
        is_symmetrical = 'spacing' in config
        
        if is_symmetrical:
            # 左右対称パーツ
            for is_left, is_right in [(True, False), (False, True)]:
                x, y, scale = calculate_part_position(category, part_num, is_left, is_right)
                side = "左" if is_left else "右"
                rel_x = x - CANVAS_CENTER[0]
                rel_y = y - CANVAS_CENTER[1]
                print(f"  {category}_{part_num:03d}({side}): ({x}, {y}) = ({rel_x:+d}, {rel_y:+d}) scale={scale}")
        else:
            # 単一パーツ
            x, y, scale = calculate_part_position(category, part_num)
            rel_x = x - CANVAS_CENTER[0]
            rel_y = y - CANVAS_CENTER[1]
            print(f"  {category}_{part_num:03d}: ({x}, {y}) = ({rel_x:+d}, {rel_y:+d}) scale={scale}")

def main():
    """メイン関数"""
    if len(sys.argv) < 2:
        print("使用法:")
        print("  python create_debug_grid.py <json_path> [output_path]")
        print("\n例:")
        print("  python create_debug_grid.py outputs/run_1_20250829_182259.json")
        print("  python create_debug_grid.py outputs/run_1_20250829_182259.json debug_custom.png")
        return
    
    json_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    create_debug_visualization(json_path, output_path)

if __name__ == "__main__":
    main()