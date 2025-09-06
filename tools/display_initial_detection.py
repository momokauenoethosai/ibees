import sys
from pathlib import Path
from PIL import Image
import time

# パッケージパスを追加
sys.path.append(str(Path(__file__).parent.parent))

from face_composer.initial_position_generator import generate_initial_positions
from tools.iterative_face_refiner import get_original_image_path, load_parts_from_json

def display_initial_detection(json_path: str):
    print(f"🔍 初期検出結果の表示を開始します。JSONファイル: {json_path}")

    try:
        original_image_path = get_original_image_path(json_path)
        if not original_image_path or not original_image_path.exists():
            print(f"❌ 元画像が見つかりません: {original_image_path}")
            return
        
        parts_dict = load_parts_from_json(json_path)
        if not parts_dict:
            print("❌ パーツ情報が読み込めませんでした。")
            return

        # generate_initial_positions は (current_positions, annotated_original_image) を返す
        _, annotated_original_image = generate_initial_positions(original_image_path, parts_dict)

        if annotated_original_image:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_filename = f"initial_detection_display_{timestamp}.png"
            output_path = Path("outputs") / output_filename
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            annotated_original_image.save(output_path)
            print(f"✅ 検出結果が描画された画像が保存されました: {output_path}")
        else:
            print("❌ ランドマークが描画された画像を取得できませんでした。")

    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使用法:")
        print("  python display_initial_detection.py <json_path>")
        print("\n例:")
        print("  python display_initial_detection.py outputs/run_20250830_164634.json")
        
        # 利用可能なファイル表示
        json_files = list(Path("outputs").glob("run_*.json"))
        if json_files:
            print(f"\n📁 利用可能なJSONファイル:")
            for f in sorted(json_files)[-3:]:
                print(f"  {f}")
    else:
        json_path = sys.argv[1]
        display_initial_detection(json_path)
