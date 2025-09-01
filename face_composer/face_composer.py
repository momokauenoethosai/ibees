#!/usr/bin/env python3
"""
顔合成エンジンモジュール
検出されたランドマークとパーツ画像を使用して顔を合成する
"""

from __future__ import annotations
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass

from .landmark_detector import FaceLandmarks, FaceLandmarkDetector
from .part_normalizer import PartNormalizer, PartScaler
from .part_placement_config import PartPlacementCalculator

@dataclass
class CompositionLayer:
    """合成レイヤー情報"""
    category: str
    part_image: Image.Image
    position: Tuple[int, int]  # (x, y)
    scale: float
    rotation: float = 0.0
    opacity: float = 1.0
    blend_mode: str = 'normal'  # 'normal', 'multiply', 'overlay', etc.

class FaceComposer:
    """顔合成エンジンクラス"""
    
    # レイヤーの描画順序（後のものが上に描画される）
    LAYER_ORDER = [
        'outline',    # 輪郭（最背面）
        'hair',       # 髪
        'face_shape', # 顔の形
        'eyebrow',    # 眉毛
        'eye',        # 目
        'nose',       # 鼻
        'mouth',      # 口
        'ear',        # 耳
        'beard',      # ひげ
        'glasses',    # メガネ
        'acc',        # アクセサリー（最前面）
    ]
    
    def __init__(self, canvas_size: Tuple[int, int] = (400, 400)):
        """
        初期化
        
        Args:
            canvas_size: 合成キャンバスサイズ (width, height)
        """
        self.canvas_size = canvas_size
        self.canvas_center = (canvas_size[0] // 2, canvas_size[1] // 2)
        
        # コンポーネント初期化（canvas_sizeを渡す）
        self.landmark_detector = FaceLandmarkDetector()
        self.part_normalizer = PartNormalizer()
        self.part_scaler = PartScaler()
        self.placement_calculator = PartPlacementCalculator(canvas_size=canvas_size)  # 修正
        self.logger = logging.getLogger(__name__)
        
        # 合成レイヤーリスト
        self.layers: List[CompositionLayer] = []
        
        # デバッグ情報
        print(f"[DEBUG] FaceComposer初期化: キャンバス{canvas_size}, 中心{self.canvas_center}")
    
    def compose_face_with_custom_positions(self,
                                           base_image_path: Optional[Path],
                                           parts_dict: Dict[str, Dict],
                                           custom_positions: Dict[str, Any]) -> Optional[Image.Image]:
        """
        カスタム座標を使用した顔合成
        
        Args:
            base_image_path: ベース顔画像のパス（オプション）
            parts_dict: 選択されたパーツ情報
            custom_positions: カスタム座標辞書（Gemini修正結果）
            
        Returns:
            合成された顔画像、失敗時はNone
        """
        try:
            # 1. 透明キャンバスを作成
            canvas = Image.new('RGBA', self.canvas_size, (0, 0, 0, 0))
            print(f"[DEBUG] カスタム座標で合成開始: {self.canvas_size}")
            
            # 2. カスタム座標で各パーツレイヤーを作成
            self._create_layers_with_custom_positions(parts_dict, custom_positions)
            
            # 3. レイヤーを順番に合成
            composed_image = self._compose_layers(canvas)
            
            # 4. 後処理
            final_image = self._apply_post_processing(composed_image)
            
            return final_image
            
        except Exception as e:
            print(f"[ERROR] Custom position composition failed: {e}")
            return None

    def _create_layers_with_custom_positions(self, parts_dict: Dict[str, Dict], custom_positions: Dict[str, Any]):
        """カスタム座標でレイヤーを作成"""
        self.layers.clear()
        
        for category, part_data in parts_dict.items():
            if category not in custom_positions:
                print(f"[WARN] No custom position for {category}, skipping")
                continue
                
            part_image_path = part_data['image_path']
            if not part_image_path.exists():
                print(f"[WARN] Part image not found: {part_image_path}")
                continue
                
            # カスタム座標を取得
            custom_pos = custom_positions[category]
            
            try:
                part_image = Image.open(part_image_path).convert('RGBA')
                
                if isinstance(custom_pos, dict):
                    # 左右対称パーツの処理
                    if 'left' in custom_pos and 'right' in custom_pos:
                        # 左側パーツ
                        left_pos = custom_pos['left']
                        if len(left_pos) >= 3:
                            x, y, scale = left_pos[0], left_pos[1], left_pos[2]
                            scaled_image = self._scale_part_image(part_image, scale)
                            layer = CompositionLayer(
                                category=f"{category}_left",
                                part_image=scaled_image,
                                position=(x, y),
                                scale=scale
                            )
                            self.layers.append(layer)
                            print(f"[DEBUG] {category}_left: pos=({x}, {y}), scale={scale}")
                        
                        # 右側パーツ（左右反転）
                        right_pos = custom_pos['right']
                        if len(right_pos) >= 3:
                            x, y, scale = right_pos[0], right_pos[1], right_pos[2]
                            flipped_image = part_image.transpose(Image.FLIP_LEFT_RIGHT)
                            scaled_image = self._scale_part_image(flipped_image, scale)
                            layer = CompositionLayer(
                                category=f"{category}_right",
                                part_image=scaled_image,
                                position=(x, y),
                                scale=scale
                            )
                            self.layers.append(layer)
                            print(f"[DEBUG] {category}_right: pos=({x}, {y}), scale={scale}")
                    elif 'single' in custom_pos:
                        # 単一パーツ
                        single_pos = custom_pos['single']
                        if len(single_pos) >= 3:
                            x, y, scale = single_pos[0], single_pos[1], single_pos[2]
                            scaled_image = self._scale_part_image(part_image, scale)
                            layer = CompositionLayer(
                                category=category,
                                part_image=scaled_image,
                                position=(x, y),
                                scale=scale
                            )
                            self.layers.append(layer)
                            print(f"[DEBUG] {category}_single: pos=({x}, {y}), scale={scale}")
                else:
                    # タプル形式の座標
                    if len(custom_pos) >= 3:
                        x, y, scale = custom_pos[0], custom_pos[1], custom_pos[2]
                        scaled_image = self._scale_part_image(part_image, scale)
                        layer = CompositionLayer(
                            category=category,
                            part_image=scaled_image,
                            position=(x, y),
                            scale=scale
                        )
                        self.layers.append(layer)
                        print(f"[DEBUG] {category}: pos=({x}, {y}), scale={scale}")
                        
            except Exception as e:
                print(f"[ERROR] Failed to process {category}: {e}")
                continue

    def _scale_part_image(self, image: Image.Image, scale: float) -> Image.Image:
        """パーツ画像をスケーリング"""
        if scale == 1.0:
            return image
        
        new_width = int(image.width * scale)
        new_height = int(image.height * scale)
        return image.resize((new_width, new_height), Image.LANCZOS)

    def compose_face(self, 
                    base_image_path: Path, 
                    selected_parts: Dict[str, Dict]) -> Optional[Image.Image]:
        """
        メイン合成関数（固定座標システム使用）
        
        Args:
            base_image_path: ベース顔画像のパス
            selected_parts: 選択されたパーツ情報
                           {'category': {'part_id': '...', 'image_path': Path, 'score': float}}
            
        Returns:
            合成された顔画像、失敗時はNone
        """
        try:
            # 1. 透明キャンバスを作成
            canvas = Image.new('RGBA', self.canvas_size, (0, 0, 0, 0))
            print(f"[DEBUG] キャンバス作成: {self.canvas_size}")
            
            # 2. 各パーツレイヤーを作成（ランドマーク検出なしの固定配置）
            self._create_composition_layers_fixed(selected_parts)
            
            # 3. レイヤーを順番に合成
            composed_image = self._compose_layers(canvas)
            
            # 4. 後処理（色調整、シャープネス等）
            final_image = self._apply_post_processing(composed_image)
            
            return final_image
            
        except Exception as e:
            self.logger.error(f"顔合成エラー: {e}")
            return None
    
    def _create_composition_layers_fixed(self, selected_parts: Dict[str, Dict]):
        """固定配置システムでレイヤーを作成"""
        self.layers.clear()
        
        for category, part_info in selected_parts.items():
            if 'image_path' not in part_info:
                continue
            
            part_path = Path(part_info['image_path'])
            if not part_path.exists():
                continue
            
            try:
                # パーツ番号を取得
                part_num = part_info.get('part_num') or self._extract_part_num_from_id(part_info.get('part_id', ''))
                
                # パーツ画像を読み込み
                part_image = Image.open(part_path).convert('RGBA')
                
                # 固定配置ロジックで位置とスケールを計算
                is_symmetrical = self.placement_calculator.is_symmetrical_category(category)
                
                if is_symmetrical:
                    # 左右対称パーツの場合は両方作成
                    for is_left in [True, False]:
                        # 右側パーツは左右反転させる
                        part_img = part_image if is_left else part_image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                        
                        layer = self._create_single_layer_fixed(
                            category, part_num, part_img, 
                            is_left_part=is_left, is_right_part=not is_left
                        )
                        if layer:
                            self.layers.append(layer)
                else:
                    # 単一パーツの場合
                    layer = self._create_single_layer_fixed(category, part_num, part_image)
                    if layer:
                        self.layers.append(layer)
                
            except Exception as e:
                self.logger.error(f"レイヤー作成エラー ({category}): {e}")
    
    def _create_single_layer_fixed(self, category: str, part_num: int, part_image: Image.Image,
                                  is_left_part: bool = False, is_right_part: bool = False) -> Optional[CompositionLayer]:
        """固定配置システムで単一レイヤーを作成"""
        try:
            # 固定配置計算システムで位置とスケールを取得（自動スケール適用）
            final_x, final_y, initial_scale = self.placement_calculator.calculate_part_position(
                category, part_num, 
                is_left_part=is_left_part, 
                is_right_part=is_right_part
            )
            
            # デバッグ出力
            side_text = ""
            if is_left_part:
                side_text = "(左)"
            elif is_right_part:
                side_text = "(右・反転済み)"
            print(f"[DEBUG] {category}_{part_num:03d}{side_text}: スケール後座標({final_x}, {final_y}), スケール{initial_scale:.3f}")
            
            # パーツ画像をスケール調整
            scaled_width = int(part_image.width * initial_scale)
            scaled_height = int(part_image.height * initial_scale)
            
            print(f"[DEBUG] {category}_{part_num:03d}{side_text}: 元サイズ({part_image.width}x{part_image.height}) -> スケール後({scaled_width}x{scaled_height})")
            
            if scaled_width > 0 and scaled_height > 0:
                scaled_part = part_image.resize((scaled_width, scaled_height), Image.Resampling.LANCZOS)
            else:
                scaled_part = part_image
            
            # 配置座標計算（中心座標から左上角座標に変換）
            position_x = int(final_x - scaled_part.width // 2)
            position_y = int(final_y - scaled_part.height // 2)
            
            print(f"[DEBUG] {category}_{part_num:03d}{side_text}: 最終配置座標({position_x}, {position_y}) (左上角)")
            print(f"[DEBUG] {category}_{part_num:03d}{side_text}: 画像中心座標({final_x}, {final_y})")
            
            return CompositionLayer(
                category=category,
                part_image=scaled_part,
                position=(position_x, position_y),
                scale=initial_scale,
                rotation=0.0,
                opacity=self._get_category_opacity(category),
                blend_mode=self._get_category_blend_mode(category)
            )
            
        except Exception as e:
            self.logger.error(f"単一レイヤー作成エラー ({category}): {e}")
            return None
    
    def _extract_part_num_from_id(self, part_id: str) -> int:
        """part_idから番号を抽出"""
        try:
            # "hair_153" -> 153
            if '_' in part_id:
                return int(part_id.split('_')[-1])
            return 0
        except:
            return 0
    
    def _compose_layers(self, base_image: Image.Image) -> Image.Image:
        """レイヤーを順番に合成"""
        result = base_image.copy()
        
        # レイヤー順序に従って合成
        for category in self.LAYER_ORDER:
            # 対称パーツ（eye_left, eye_right等）も含めて検索
            layers = [layer for layer in self.layers 
                     if layer.category == category or layer.category.startswith(f"{category}_")]
            for layer in layers:
                result = self._blend_layer(result, layer)
        
        return result
    
    def _blend_layer(self, base: Image.Image, layer: CompositionLayer) -> Image.Image:
        """レイヤーをベース画像に合成"""
        try:
            # オーバーレイ用のキャンバス作成
            overlay = Image.new('RGBA', self.canvas_size, (0, 0, 0, 0))
            
            # パーツ画像をオーバーレイに配置
            part_image = layer.part_image.copy()
            
            # 回転処理
            if layer.rotation != 0:
                part_image = part_image.rotate(layer.rotation, expand=True, fillcolor=(0, 0, 0, 0))
            
            # 透明度調整
            if layer.opacity < 1.0:
                alpha = part_image.split()[-1]
                alpha = ImageEnhance.Brightness(alpha).enhance(layer.opacity)
                part_image.putalpha(alpha)
            
            # パーツの中心座標を左上角座標に変換
            center_x, center_y = layer.position
            left = center_x - part_image.width // 2
            top = center_y - part_image.height // 2
            
            print(f"[DEBUG] {layer.category}: 中心({center_x}, {center_y}) → 左上({left}, {top}), サイズ({part_image.width}, {part_image.height})")
            
            # オーバーレイに配置
            overlay.paste(part_image, (left, top), part_image)
            
            # ブレンドモードに応じて合成
            if layer.blend_mode == 'normal':
                result = Image.alpha_composite(base, overlay)
            else:
                # 他のブレンドモードは後で実装
                result = Image.alpha_composite(base, overlay)
            
            return result
            
        except Exception as e:
            self.logger.error(f"レイヤー合成エラー ({layer.category}): {e}")
            return base
    
    def _get_category_opacity(self, category: str) -> float:
        """カテゴリ別の標準透明度"""
        opacity_map = {
            'hair': 0.95,
            'eye': 1.0,
            'eyebrow': 0.9,
            'nose': 0.95,
            'mouth': 0.95,
            'beard': 0.85,
            'glasses': 0.95,
            'acc': 1.0,
        }
        return opacity_map.get(category, 1.0)
    
    def _get_category_blend_mode(self, category: str) -> str:
        """カテゴリ別のブレンドモード"""
        blend_map = {
            'hair': 'normal',
            'eye': 'normal',
            'eyebrow': 'normal', 
            'nose': 'normal',
            'mouth': 'normal',
            'beard': 'normal',
            'glasses': 'normal',
            'acc': 'normal',
        }
        return blend_map.get(category, 'normal')
    
    def _apply_post_processing(self, image: Image.Image) -> Image.Image:
        """後処理（色調整、シャープネスなど）"""
        try:
            # 軽微なシャープネス向上
            enhanced = image.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=3))
            
            return enhanced
            
        except Exception as e:
            self.logger.error(f"後処理エラー: {e}")
            return image
    
    def create_composition_preview(self, base_image_path: Path, selected_parts: Dict[str, Dict], 
                                  preview_size: Tuple[int, int] = (512, 512)) -> Optional[Image.Image]:
        """
        プレビュー用の小さな合成画像を作成
        
        Args:
            base_image_path: ベース顔画像
            selected_parts: 選択パーツ
            preview_size: プレビューサイズ
            
        Returns:
            プレビュー画像
        """
        original_canvas_size = self.canvas_size
        original_center = self.canvas_center
        
        # 一時的にプレビューサイズに変更
        self.canvas_size = preview_size
        self.canvas_center = (preview_size[0] // 2, preview_size[1] // 2)
        self.placement_calculator = PartPlacementCalculator(canvas_size=preview_size)
        
        try:
            result = self.compose_face(base_image_path, selected_parts)
            return result
        finally:
            # 元の設定に戻す
            self.canvas_size = original_canvas_size
            self.canvas_center = original_center
            self.placement_calculator = PartPlacementCalculator(canvas_size=original_canvas_size)


def test_face_composition():
    """顔合成のテスト"""
    # カスタムサイズでテスト
    test_canvas_size = (600, 600)
    composer = FaceComposer(canvas_size=test_canvas_size)
    
    # テスト用のパス（実際のファイルに置き換える）
    base_image = Path("/Users/uenomomoka/Desktop/Projects/vision_rag/made_pictures/1.png")
    
    selected_parts = {
        'eye': {
            'part_id': 'eye_010',
            'image_path': Path("/Users/uenomomoka/Desktop/Projects/vision_rag/kawakura/assets_png/eye/eye_010.png"),
            'score': 0.85,
            'part_num': 10
        }
    }
    
    if base_image.exists():
        result = composer.compose_face(base_image, selected_parts)
        if result:
            output_path = Path("/tmp/test_composition.png")
            result.save(output_path)
            print(f"合成完了: {output_path}")
        else:
            print("合成失敗")
    else:
        print(f"ベース画像が見つかりません: {base_image}")


if __name__ == "__main__":
    test_face_composition()