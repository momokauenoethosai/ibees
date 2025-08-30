#!/usr/bin/env python3
"""
顔合成機能の独立テストスクリプト
outputsフォルダの既存結果を使用して顔合成をテストする
"""

import json
import sys
from pathlib import Path

# パッケージパスを追加
sys.path.append(str(Path(__file__).parent.parent))
from face_composer.face_composer import FaceComposer

def load_analysis_result(json_path: Path) -> dict:
    """分析結果JSONを読み込み"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"JSON読み込みエラー: {e}")
        return {}

def find_part_image_path(category: str, part_num: int) -> Path:
    """パーツ画像のパスを検索"""
    assets_root = Path("../kawakura/assets_png")
    
    # カテゴリフォルダのマッピング（mouth → mouse など）
    category_mapping = {
        'mouth': 'mouse',
        'hair': 'hair',
        'eye': 'eye',
        'eyebrow': 'eyebrow',
        'nose': 'nose',
        'ear': 'ear',
        'outline': 'outline',
        'acc': 'acc',
        'beard': 'beard',
        'glasses': 'glasses',
        'extras': 'extras'
    }
    
    folder_name = category_mapping.get(category, category)
    category_folder = assets_root / folder_name
    
    if not category_folder.exists():
        print(f"カテゴリフォルダが見つかりません: {category_folder}")
        return None
    
    # パーツファイル名の候補
    candidates = [
        f"{category}_{part_num:03d}.png",
        f"{category}_{part_num:02d}.png", 
        f"{category}_{part_num}.png"
    ]
    
    for candidate in candidates:
        part_path = category_folder / candidate
        if part_path.exists():
            return part_path
    
    print(f"パーツ画像が見つかりません: {category}_{part_num}")
    return None

def convert_analysis_to_parts_dict(analysis_result: dict) -> dict:
    """分析結果を合成用のパーツ辞書に変換"""
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
                    'score': score
                }
    
    return parts_dict

def test_face_composition(json_path: str, output_path: str = None):
    """顔合成テスト実行"""
    print(f"=== 顔合成テスト開始 ===")
    print(f"分析結果JSON: {json_path}")
    
    # JSONファイル読み込み
    json_file = Path(json_path)
    if not json_file.exists():
        print(f"JSONファイルが見つかりません: {json_path}")
        return
    
    analysis_result = load_analysis_result(json_file)
    if not analysis_result:
        print("分析結果の読み込みに失敗しました")
        return
    
    # 入力画像パスを取得・修正
    input_image_path = analysis_result.get('input_image')
    if not input_image_path:
        print("入力画像パスが見つかりません")
        return
    
    # パスを現在のプロジェクトに合わせて修正
    original_path = Path(input_image_path)
    if not original_path.exists():
        # made_pictures フォルダで画像ファイル名を検索
        image_name = original_path.name
        local_image_path = Path("../made_pictures") / image_name
        
        if local_image_path.exists():
            base_image_path = local_image_path
            print(f"ローカル画像を使用: {base_image_path}")
        else:
            print(f"入力画像が見つかりません: {input_image_path}")
            print(f"ローカルパスも見つかりません: {local_image_path}")
            return
    else:
        base_image_path = original_path
    
    print(f"ベース画像: {base_image_path}")
    
    # パーツ辞書に変換
    parts_dict = convert_analysis_to_parts_dict(analysis_result)
    if not parts_dict:
        print("選択されたパーツがありません")
        return
    
    print("選択されたパーツ:")
    for category, part_info in parts_dict.items():
        print(f"  {category}: {part_info['part_id']} (score: {part_info['score']:.3f})")
    
    # 顔合成実行
    print("\n顔合成を実行中...")
    try:
        composer = FaceComposer(canvas_size=(400, 400))
        composed_image = composer.compose_face(base_image_path, parts_dict)
        
        if composed_image:
            # 結果保存
            if not output_path:
                output_path = f"test_composition_{json_file.stem}.png"
            
            result_path = Path(output_path)
            
            # RGBA → RGB変換（白背景）
            if composed_image.mode == 'RGBA':
                from PIL import Image
                background = Image.new('RGB', composed_image.size, (255, 255, 255))
                background.paste(composed_image, mask=composed_image.split()[-1])
                composed_image = background
            
            composed_image.save(result_path)
            print(f"✅ 合成完了: {result_path}")
        else:
            print("❌ 顔合成に失敗しました")
            
    except Exception as e:
        print(f"❌ 顔合成エラー: {e}")
        import traceback
        traceback.print_exc()

def list_available_results():
    """利用可能な分析結果一覧を表示"""
    outputs_dir = Path("outputs")
    if not outputs_dir.exists():
        print("outputsフォルダが見つかりません")
        return
    
    json_files = list(outputs_dir.glob("run_*.json"))
    if not json_files:
        print("分析結果JSONファイルが見つかりません")
        return
    
    print("利用可能な分析結果:")
    for i, json_file in enumerate(json_files, 1):
        print(f"  {i}. {json_file.name}")
    
    return json_files

def main():
    if len(sys.argv) < 2:
        print("使用法:")
        print("  python test_face_composition.py <json_path> [output_path]")
        print("  python test_face_composition.py list  # 利用可能結果一覧")
        print("\n例:")
        print("  python test_face_composition.py outputs/run_20250829_182259.json")
        print("  python test_face_composition.py list")
        return
    
    if sys.argv[1] == "list":
        list_available_results()
        return
    
    json_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    test_face_composition(json_path, output_path)

if __name__ == "__main__":
    main()