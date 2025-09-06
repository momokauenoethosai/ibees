import json
import sys
import time
from pathlib import Path
from PIL import Image

# パッケージパスを追加
sys.path.append(str(Path(__file__).parent.parent))

from face_composer.face_composer import FaceComposer
from tools.iterative_face_refiner import load_parts_from_json
from face_composer.initial_position_generator import generate_initial_positions

def compose_face_from_json(json_path: str):
    print(f"🎨 JSONファイルから顔を合成します。JSONファイル: {json_path}")

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            analysis_result = json.load(f)
        
        # パーツ情報を読み込み
        parts_dict = load_parts_from_json(json_path)
        if not parts_dict:
            print("❌ パーツ情報が読み込めませんでした。")
            return

        # original_image_path は generate_initial_positions の中で処理されるため、ここでは不要
        # ただし、load_parts_from_json には必要なので、iterative_face_refiner から取得
        from tools.iterative_face_refiner import get_original_image_path
        original_image_path = get_original_image_path(json_path)
        if not original_image_path or not original_image_path.exists():
            print(f"❌ 元画像が見つかりません: {original_image_path}")
            return

        parts_dict = load_parts_from_json(json_path)
        if not parts_dict:
            print("❌ パーツ情報が読み込めませんでした。")
            return

        # ランドマーク検出に基づいて初期座標と注釈付き元画像を取得
        current_positions, _ = generate_initial_positions(original_image_path, parts_dict)

        # FaceComposerを初期化
        composer = FaceComposer(canvas_size=(400, 400)) # 仮のキャンバスサイズ

        # パーツを合成
        composed_image = composer.compose_face_with_custom_positions(
            base_image_path=None, # ベース画像はFaceComposer内で処理される
            parts_dict=parts_dict,
            custom_positions=current_positions
        )

        if composed_image:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_filename = f"composed_from_json_{timestamp}.png"
            output_path = Path("outputs") / output_filename
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            composed_image.save(output_path)
            print(f"✅ JSONから合成された画像が保存されました: {output_path}")
        else:
            print("❌ 画像の合成に失敗しました。")

    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使用法:")
        print("  python compose_from_json.py <json_path>")
        print("\n例:")
        print("  python compose_from_json.py outputs/run_20250830_170700.json")
        
        # 利用可能なファイル表示
        json_files = list(Path("outputs").glob("run_*.json"))
        if json_files:
            print(f"\n📁 利用可能なJSONファイル:")
            for f in sorted(json_files)[-3:]: # 最新の3つを表示
                print(f"  {f}")
    else:
        json_path = sys.argv[1]
        compose_face_from_json(json_path)
