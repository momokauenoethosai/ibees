#!/usr/bin/env python3
"""
顔ランドマーク検出モジュール
MediaPipeを使用して顔の基準点を検出し、パーツ配置座標を計算する
"""

from __future__ import annotations
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass

@dataclass
class FaceLandmarks:
    """顔ランドマーク情報を格納するデータクラス"""
    # 基準点の座標 (x, y)
    left_eye_center: Tuple[float, float]
    right_eye_center: Tuple[float, float] 
    nose_tip: Tuple[float, float]
    mouth_center: Tuple[float, float]
    face_center: Tuple[float, float]
    
    # 顔の境界情報
    face_width: float
    face_height: float
    
    # 全ランドマーク座標 (468点)
    all_landmarks: List[Tuple[float, float]]
    
    # 画像サイズ
    image_width: int
    image_height: int

class FaceLandmarkDetector:
    """顔ランドマーク検出クラス"""
    
    def __init__(self):
        """MediaPipe Face Meshの初期化"""
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # 重要なランドマークポイントのインデックス定義
        # MediaPipe Face Mesh の468点ランドマークから重要な点を抽出
        self.LANDMARK_INDICES = {
            'left_eye': [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
            'right_eye': [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
            'nose': [1, 2, 5, 4, 6, 19, 20, 94, 125, 141, 235, 236, 237, 238, 239, 240, 241, 242],
            'mouth': [0, 11, 12, 13, 14, 15, 16, 17, 18, 200, 199, 175, 61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318],
            'face_outline': [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162]
        }
    
    def detect_landmarks(self, image_path: Path) -> Optional[FaceLandmarks]:
        """
        画像から顔ランドマークを検出する
        
        Args:
            image_path: 画像ファイルのパス
            
        Returns:
            FaceLandmarks: 検出された顔ランドマーク情報、失敗時はNone
        """
        try:
            # 画像を読み込み
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"画像の読み込みに失敗: {image_path}")
                return None
            
            height, width = image.shape[:2]
            
            # BGR -> RGB変換
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 顔ランドマーク検出
            results = self.face_mesh.process(rgb_image)
            
            if not results.multi_face_landmarks:
                print("顔が検出されませんでした")
                return None
            
            # 最初の顔のランドマークを取得
            face_landmarks = results.multi_face_landmarks[0]
            
            # ランドマーク座標を抽出
            landmarks = []
            for lm in face_landmarks.landmark:
                x = lm.x * width
                y = lm.y * height
                landmarks.append((x, y))
            
            # 重要な基準点を計算
            left_eye_center = self._calculate_eye_center(landmarks, self.LANDMARK_INDICES['left_eye'])
            right_eye_center = self._calculate_eye_center(landmarks, self.LANDMARK_INDICES['right_eye'])
            nose_tip = landmarks[1]  # MediaPipeの鼻先端インデックス
            mouth_center = self._calculate_mouth_center(landmarks, self.LANDMARK_INDICES['mouth'])
            
            # 顔の中心点を計算
            face_center = (
                (left_eye_center[0] + right_eye_center[0]) / 2,
                (left_eye_center[1] + right_eye_center[1] + nose_tip[1] + mouth_center[1]) / 4
            )
            
            # 顔のサイズを計算
            face_width = abs(right_eye_center[0] - left_eye_center[0]) * 2.5  # 目の間隔から推定
            face_height = abs(mouth_center[1] - ((left_eye_center[1] + right_eye_center[1]) / 2)) * 2.5
            
            return FaceLandmarks(
                left_eye_center=left_eye_center,
                right_eye_center=right_eye_center,
                nose_tip=nose_tip,
                mouth_center=mouth_center,
                face_center=face_center,
                face_width=face_width,
                face_height=face_height,
                all_landmarks=landmarks,
                image_width=width,
                image_height=height
            )
            
        except Exception as e:
            print(f"ランドマーク検出エラー: {e}")
            return None
    
    def _calculate_eye_center(self, landmarks: List[Tuple[float, float]], eye_indices: List[int]) -> Tuple[float, float]:
        """目の中心座標を計算"""
        eye_points = [landmarks[i] for i in eye_indices if i < len(landmarks)]
        if not eye_points:
            return (0, 0)
        
        center_x = sum(point[0] for point in eye_points) / len(eye_points)
        center_y = sum(point[1] for point in eye_points) / len(eye_points)
        return (center_x, center_y)
    
    def _calculate_mouth_center(self, landmarks: List[Tuple[float, float]], mouth_indices: List[int]) -> Tuple[float, float]:
        """口の中心座標を計算"""
        mouth_points = [landmarks[i] for i in mouth_indices if i < len(landmarks)]
        if not mouth_points:
            return (0, 0)
        
        center_x = sum(point[0] for point in mouth_points) / len(mouth_points)
        center_y = sum(point[1] for point in mouth_points) / len(mouth_points)
        return (center_x, center_y)
    
    def get_part_placement_info(self, landmarks: FaceLandmarks, part_category: str) -> Dict[str, float]:
        """
        パーツカテゴリに応じた配置情報を計算
        
        Args:
            landmarks: 検出された顔ランドマーク
            part_category: パーツカテゴリ ('hair', 'eye', 'nose', etc.)
            
        Returns:
            配置情報 (位置、サイズ、回転角度など)
        """
        placement_info = {
            'x': 0,
            'y': 0,
            'scale': 1.0,
            'rotation': 0.0,
            'width': 100,
            'height': 100
        }
        
        if part_category == 'hair':
            # 髪の毛は顔の上部全体
            placement_info.update({
                'x': landmarks.face_center[0],
                'y': landmarks.face_center[1] - landmarks.face_height * 0.4,
                'scale': landmarks.face_width / 200,  # 基準サイズ200pxとして計算
                'width': int(landmarks.face_width * 1.2),
                'height': int(landmarks.face_height * 0.8)
            })
        
        elif part_category == 'eye':
            # 目は左右の目の中心に配置
            eye_center_x = (landmarks.left_eye_center[0] + landmarks.right_eye_center[0]) / 2
            eye_center_y = (landmarks.left_eye_center[1] + landmarks.right_eye_center[1]) / 2
            eye_distance = abs(landmarks.right_eye_center[0] - landmarks.left_eye_center[0])
            
            placement_info.update({
                'x': eye_center_x,
                'y': eye_center_y,
                'scale': eye_distance / 100,
                'width': int(eye_distance * 1.5),
                'height': int(eye_distance * 0.8)
            })
        
        elif part_category == 'nose':
            placement_info.update({
                'x': landmarks.nose_tip[0],
                'y': landmarks.nose_tip[1],
                'scale': landmarks.face_width / 300,
                'width': int(landmarks.face_width * 0.3),
                'height': int(landmarks.face_height * 0.25)
            })
        
        elif part_category == 'mouth':
            placement_info.update({
                'x': landmarks.mouth_center[0],
                'y': landmarks.mouth_center[1],
                'scale': landmarks.face_width / 250,
                'width': int(landmarks.face_width * 0.4),
                'height': int(landmarks.face_height * 0.2)
            })
        
        elif part_category in ['eyebrow']:
            # 眉毛は目の上部に配置
            eye_center_x = (landmarks.left_eye_center[0] + landmarks.right_eye_center[0]) / 2
            eye_center_y = (landmarks.left_eye_center[1] + landmarks.right_eye_center[1]) / 2
            eye_distance = abs(landmarks.right_eye_center[0] - landmarks.left_eye_center[0])
            
            placement_info.update({
                'x': eye_center_x,
                'y': eye_center_y - landmarks.face_height * 0.08,
                'scale': eye_distance / 120,
                'width': int(eye_distance * 1.6),
                'height': int(landmarks.face_height * 0.15)
            })
        
        # その他のパーツ（ear, beard, accessories等）はデフォルト値
        
        return placement_info


def test_landmark_detection(image_path: str):
    """ランドマーク検出のテスト関数"""
    detector = FaceLandmarkDetector()
    landmarks = detector.detect_landmarks(Path(image_path))
    
    if landmarks:
        print("ランドマーク検出成功!")
        print(f"左目中心: {landmarks.left_eye_center}")
        print(f"右目中心: {landmarks.right_eye_center}")
        print(f"鼻先端: {landmarks.nose_tip}")
        print(f"口中心: {landmarks.mouth_center}")
        print(f"顔中心: {landmarks.face_center}")
        print(f"顔サイズ: {landmarks.face_width:.1f} x {landmarks.face_height:.1f}")
    else:
        print("ランドマーク検出失敗")

if __name__ == "__main__":
    # テスト実行例
    import sys
    if len(sys.argv) > 1:
        test_landmark_detection(sys.argv[1])
    else:
        print("使用法: python landmark_detector.py <image_path>")