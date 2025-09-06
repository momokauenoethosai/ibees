from pathlib import Path
from typing import Dict, Tuple, Optional, List
import cv2
import numpy as np
from PIL import Image
from face_composer.landmark_detector import FaceLandmarkDetector, FaceLandmarks

def generate_initial_positions(image_path: Path, parts_dict: dict) -> Tuple[dict, Image.Image]:
    """
    顔のランドマーク検出に基づいて、各パーツの初期位置を生成します。
    ランドマーク検出に失敗した場合は、デフォルトの初期座標と元の画像を返します。
    """
    original_image_cv2 = cv2.imread(str(image_path))
    if original_image_cv2 is None:
        print(f"❌ 元画像の読み込みに失敗: {image_path}")
        # エラー時はデフォルト座標と空のPIL Imageを返す
        return {
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
        }, Image.new('RGB', (400, 400), color = 'white') # Return a blank image

    # 元画像のサイズを取得
    original_h, original_w = original_image_cv2.shape[:2]
    target_canvas_size = (400, 400) # FaceComposerのキャンバスサイズ

    # スケーリング係数を計算
    scale_factor_w = target_canvas_size[0] / original_w
    scale_factor_h = target_canvas_size[1] / original_h
    # 縦横比を維持しつつ、両方に収まるように小さい方のスケールファクターを使用
    overall_scale_factor = min(scale_factor_w, scale_factor_h)

    annotated_image_cv2 = original_image_cv2.copy()

    if not landmarks:
        print("❌ ランドマーク検出に失敗しました。デフォルトの初期座標を使用します。")
        # PIL Imageに変換して返す
        return {
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
        }, Image.fromarray(cv2.cvtColor(original_image_cv2, cv2.COLOR_BGR2RGB))
    else:
        # ランドマークを描画
        for lm in landmarks.all_landmarks:
            # ランドマーク座標もスケーリングして描画
            scaled_lm_x = int(lm[0] * overall_scale_factor)
            scaled_lm_y = int(lm[1] * overall_scale_factor)
            cv2.circle(annotated_image_cv2, (scaled_lm_x, scaled_lm_y), 1, (0, 255, 0), -1) # 緑色の点
        
        # 主要なランドマークを強調（スケーリング済み座標を使用）
        cv2.circle(annotated_image_cv2, (int(landmarks.left_eye_center[0] * overall_scale_factor), int(landmarks.left_eye_center[1] * overall_scale_factor)), 3, (255, 0, 0), -1) # 青
        cv2.circle(annotated_image_cv2, (int(landmarks.right_eye_center[0] * overall_scale_factor), int(landmarks.right_eye_center[1] * overall_scale_factor)), 3, (255, 0, 0), -1) # 青
        cv2.circle(annotated_image_cv2, (int(landmarks.nose_tip[0] * overall_scale_factor), int(landmarks.nose_tip[1] * overall_scale_factor)), 3, (0, 0, 255), -1) # 赤
        cv2.circle(annotated_image_cv2, (int(landmarks.mouth_center[0] * overall_scale_factor), int(landmarks.mouth_center[1] * overall_scale_factor)), 3, (0, 255, 255), -1) # 黄
        cv2.circle(annotated_image_cv2, (int(landmarks.face_center[0] * overall_scale_factor), int(landmarks.face_center[1] * overall_scale_factor)), 3, (255, 255, 0), -1) # シアン

        current_positions = {}
        for category in parts_dict.keys():
            placement_info = detector.get_part_placement_info(landmarks, category)
            if placement_info:
                # ランドマーク検出器からの座標とスケールをキャンバスサイズに合わせてスケーリング
                x = placement_info.get('x', 0) * overall_scale_factor
                y = placement_info.get('y', 0) * overall_scale_factor
                scale = placement_info.get('scale', 1.0) * overall_scale_factor # スケールも調整

                # 特殊なケース（左右対称パーツ）のハンドリング
                if category in ['eye', 'eyebrow', 'ear']:
                    if category == 'eye':
                        left_x = landmarks.left_eye_center[0] * overall_scale_factor
                        right_x = landmarks.right_eye_center[0] * overall_scale_factor
                        y_val = landmarks.left_eye_center[1] * overall_scale_factor
                        current_positions[category] = {
                            'left': (int(left_x), int(y_val), scale),
                            'right': (int(right_x), int(y_val), scale)
                        }
                    elif category == 'eyebrow':
                        left_x = landmarks.left_eye_center[0] * overall_scale_factor
                        right_x = landmarks.right_eye_center[0] * overall_scale_factor
                        y_val = (landmarks.left_eye_center[1] - (landmarks.face_height * 0.08)) * overall_scale_factor
                        current_positions[category] = {
                            'left': (int(left_x), int(y_val), scale),
                            'right': (int(right_x), int(y_val), scale)
                        }
                    elif category == 'ear':
                        left_x = (landmarks.face_center[0] - landmarks.face_width / 2) * overall_scale_factor
                        right_x = (landmarks.face_center[0] + landmarks.face_width / 2) * overall_scale_factor
                        y_val = landmarks.face_center[1] * overall_scale_factor
                        current_positions[category] = {
                            'left': (int(left_x), int(y_val), scale),
                            'right': (int(right_x), int(y_val), scale)
                        }
                else:
                    current_positions[category] = (int(x), int(y), scale)
        print("✅ ランドマーク検出に基づいて初期座標を設定しました。")
        # PIL Imageに変換して返す
        return current_positions, Image.fromarray(cv2.cvtColor(annotated_image_cv2, cv2.COLOR_BGR2RGB))
