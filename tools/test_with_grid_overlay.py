#!/usr/bin/env python3
"""
グリッド付き合成結果を作成して、座標計算の正確性を視覚的に確認する
"""

import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
# パッケージパスを追加
import sys
sys.path.append(str(Path(__file__).parent.parent))
from face_composer.face_composer import FaceComposer

# ========== 統一設定 ==========
CANVAS_SIZE = (300, 300)  # ここを変更するだけで全体のサイズが変わる
CANVAS_CENTER = (CANVAS_SIZE[0] // 2, CANVAS_SIZE[1] // 2)  # 自動計算

# 他のサイズ例：
# CANVAS_SIZE = (600, 600)  # 高解像度
# CANVAS_SIZE = (800, 800)  # 超高解像度
# =============================

def create_grid_overlay(canvas_size=CANVAS_SIZE, canvas_center=CANVAS_CENTER):
    """グリッドオーバーレイを作成"""
    overlay = Image.new('RGBA', canvas_size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    try:
        font_small = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 10)
    except:
        font_small = ImageFont.load_default()
    
    # グリッド線（50px間隔、薄い色）
    grid_color = (200, 200, 200, 80)
    for x in range(0, canvas_size[0] + 1, 50):
        draw.line([(x, 0), (x, canvas_size[1])], fill=grid_color, width=1)
    for y in range(0, canvas_size[1] + 1, 50):
        draw.line([(0, y), (canvas_size[0], y)], fill=grid_color, width=1)
    
    # 座標軸（赤色、半透明）
    center_x, center_y = canvas_center
    axis_color = (255, 0, 0, 120)
    draw.line([(0, center_y), (canvas_size[0], center_y)], fill=axis_color, width=2)  # X軸
    draw.line([(center_x, 0), (center_x, canvas_size[1])], fill=axis_color, width=2)  # Y軸
    
    # 中心点（緑色）
    center_color = (0, 255, 0, 200)
    radius = 4
    draw.ellipse([
        center_x - radius, center_y - radius,
        center_x + radius, center_y + radius
    ], fill=center_color, outline=(0, 0, 0, 255), width=1)
    
    # 中心点ラベル
    draw.text((center_x + 8, center_y + 8), "(0,0)", fill=(0, 0, 0, 255), font=font_small)
    
    return overlay

def test_composition_with_grid(json_path: str, output_path: str = None):
    """グリッド付きで合成テスト"""
    print(f"=== グリッド付き合成テスト ===")
    print(f"キャンバスサイズ: {CANVAS_SIZE}, 中心: {CANVAS_CENTER}")
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
    
    # 入力画像パスを修正
    input_image_path = analysis_result.get('input_image')
    if not input_image_path:
        print("入力画像パスが見つかりません")
        return
    
    original_path = Path(input_image_path)
    if not original_path.exists():
        image_name = original_path.name
        local_image_path = Path("../made_pictures") / image_name
        
        if local_image_path.exists():
            base_image_path = local_image_path
            print(f"ローカル画像を使用: {base_image_path}")
        else:
            print(f"入力画像が見つかりません")
            return
    else:
        base_image_path = original_path
    
    # パーツ情報を変換
    def find_part_image_path(category: str, part_num: int) -> Path:
        assets_root = Path("../kawakura/assets_png")
        category_mapping = {
            'mouth': 'mouse', 'hair': 'hair', 'eye': 'eye', 'eyebrow': 'eyebrow',
            'nose': 'nose', 'ear': 'ear', 'outline': 'outline', 'acc': 'acc',
            'beard': 'beard', 'glasses': 'glasses', 'extras': 'extras'
        }
        
        folder_name = category_mapping.get(category, category)
        category_folder = assets_root / folder_name
        
        # ファイル名はフォルダ名に合わせる（mouthの場合はmouse）
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
    
    if not parts_dict:
        print("選択されたパーツがありません")
        return
    
    print("選択されたパーツ:")
    for category, part_info in parts_dict.items():
        print(f"  {category}: {part_info['part_id']} (score: {part_info['score']:.3f})")
    
    # 顔合成実行
    print("\n顔合成を実行中...")
    try:
        # 統一されたキャンバスサイズを使用
        composer = FaceComposer(canvas_size=CANVAS_SIZE)
        print(f"[DEBUG] FaceComposer作成: キャンバス{CANVAS_SIZE}")
        composed_image = composer.compose_face(base_image_path, parts_dict)
        
        if composed_image:
            # グリッドオーバーレイを作成（統一されたサイズ）
            grid_overlay = create_grid_overlay()
            
            # デバッグ情報を追加
            print(f"[DEBUG] composed_image: サイズ{composed_image.size}, モード{composed_image.mode}")
            print(f"[DEBUG] grid_overlay: サイズ{grid_overlay.size}, モード{grid_overlay.mode}")
            
            # 合成結果にグリッドを重ねる
            if composed_image.mode != 'RGBA':
                composed_image = composed_image.convert('RGBA')
            
            final_result = Image.alpha_composite(composed_image, grid_overlay)
            
            # 結果保存
            if not output_path:
                output_path = f"grid_overlay_{json_file.stem}.png"
            
            result_path = Path(output_path)
            final_result.save(result_path)
            print(f"✅ グリッド付き合成完了: {result_path}")
        else:
            print("❌ 顔合成に失敗しました")
            
    except Exception as e:
        print(f"❌ 合成エラー: {e}")
        import traceback
        traceback.print_exc()

def main():
    import sys
    
    if len(sys.argv) < 2:
        print("使用法:")
        print("  python test_with_grid_overlay.py <json_path> [output_path]")
        print(f"  現在のキャンバスサイズ: {CANVAS_SIZE}")
        print("\n例:")
        print("  python test_with_grid_overlay.py outputs/run_1_20250829_182259.json")
        return
    
    json_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    test_composition_with_grid(json_path, output_path)

if __name__ == "__main__":
    main()