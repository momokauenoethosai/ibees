#!/usr/bin/env python3
"""
パーツ画像正規化モジュール
パーツ画像のサイズ統一、透明度処理、配置基準の標準化を行う
"""

from __future__ import annotations
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import logging

class PartNormalizer:
    """パーツ画像正規化クラス"""
    
    def __init__(self, standard_size: Tuple[int, int] = (512, 512)):
        """
        初期化
        
        Args:
            standard_size: 標準化サイズ (width, height)
        """
        self.standard_size = standard_size
        self.logger = logging.getLogger(__name__)
    
    def normalize_part_image(self, part_path: Path, part_category: str) -> Optional[Image.Image]:
        """
        パーツ画像を正規化する
        
        Args:
            part_path: パーツ画像のパス
            part_category: パーツカテゴリ
            
        Returns:
            正規化されたPIL Image、失敗時はNone
        """
        try:
            # 画像を開く
            part_image = Image.open(part_path)
            
            # RGBA形式に変換（透明度対応）
            if part_image.mode != 'RGBA':
                part_image = part_image.convert('RGBA')
            
            # カテゴリ別の前処理
            part_image = self._preprocess_by_category(part_image, part_category)
            
            # 標準サイズにリサイズ（アスペクト比維持）
            part_image = self._resize_keeping_aspect(part_image)
            
            # 中央配置で標準キャンバスに配置
            normalized_image = self._place_on_standard_canvas(part_image)
            
            return normalized_image
            
        except Exception as e:
            self.logger.error(f"パーツ正規化エラー {part_path}: {e}")
            return None
    
    def _preprocess_by_category(self, image: Image.Image, category: str) -> Image.Image:
        """カテゴリ別の前処理"""
        if category == 'hair':
            # 髪の毛：輪郭を少しぼかして自然に
            return image.filter(ImageFilter.SMOOTH_MORE)
        
        elif category in ['eye', 'eyebrow']:
            # 目・眉毛：エッジを少し強調
            return image.filter(ImageFilter.EDGE_ENHANCE_MORE)
        
        elif category == 'mouth':
            # 口：色合いを調整
            return image
        
        elif category in ['beard', 'acc', 'glasses']:
            # アクセサリー系：そのまま
            return image
        
        return image
    
    def _resize_keeping_aspect(self, image: Image.Image) -> Image.Image:
        """アスペクト比を維持してリサイズ"""
        original_width, original_height = image.size
        target_width, target_height = self.standard_size
        
        # アスペクト比計算
        aspect_ratio = original_width / original_height
        target_aspect = target_width / target_height
        
        if aspect_ratio > target_aspect:
            # 幅基準でリサイズ
            new_width = target_width
            new_height = int(target_width / aspect_ratio)
        else:
            # 高さ基準でリサイズ
            new_height = target_height
            new_width = int(target_height * aspect_ratio)
        
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    def _place_on_standard_canvas(self, image: Image.Image) -> Image.Image:
        """標準キャンバスに中央配置"""
        canvas_width, canvas_height = self.standard_size
        img_width, img_height = image.size
        
        # 透明キャンバス作成
        canvas = Image.new('RGBA', self.standard_size, (0, 0, 0, 0))
        
        # 中央配置座標計算
        x = (canvas_width - img_width) // 2
        y = (canvas_height - img_height) // 2
        
        # 画像を配置
        canvas.paste(image, (x, y), image)
        
        return canvas
    
    def get_part_bounds(self, image: Image.Image) -> Dict[str, int]:
        """
        パーツ画像の実際の境界を取得
        
        Args:
            image: RGBA形式のパーツ画像
            
        Returns:
            境界情報 {'left', 'top', 'right', 'bottom', 'width', 'height'}
        """
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        # アルファチャンネルからマスクを作成
        alpha = image.split()[-1]
        bbox = alpha.getbbox()
        
        if bbox is None:
            # 完全透明画像の場合
            return {
                'left': 0, 'top': 0, 'right': 0, 'bottom': 0,
                'width': 0, 'height': 0
            }
        
        left, top, right, bottom = bbox
        return {
            'left': left,
            'top': top, 
            'right': right,
            'bottom': bottom,
            'width': right - left,
            'height': bottom - top
        }
    
    def create_mask_from_part(self, image: Image.Image, feather: int = 2) -> Image.Image:
        """
        パーツ画像からソフトマスクを作成
        
        Args:
            image: RGBA形式のパーツ画像
            feather: マスクのぼかし強度
            
        Returns:
            マスク画像
        """
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        # アルファチャンネルを取得
        mask = image.split()[-1]
        
        # ぼかし処理
        if feather > 0:
            mask = mask.filter(ImageFilter.GaussianBlur(radius=feather))
        
        return mask
    
    def batch_normalize_parts(self, parts_folder: Path, output_folder: Path) -> Dict[str, int]:
        """
        フォルダ内のパーツを一括正規化
        
        Args:
            parts_folder: パーツ画像フォルダ
            output_folder: 出力フォルダ
            
        Returns:
            処理結果統計
        """
        output_folder.mkdir(parents=True, exist_ok=True)
        
        stats = {'success': 0, 'failed': 0, 'total': 0}
        
        # カテゴリフォルダを走査
        for category_folder in parts_folder.iterdir():
            if not category_folder.is_dir():
                continue
                
            category = category_folder.name
            category_output = output_folder / category
            category_output.mkdir(exist_ok=True)
            
            # カテゴリ内のパーツファイルを処理
            for part_file in category_folder.glob('*.png'):
                stats['total'] += 1
                
                normalized = self.normalize_part_image(part_file, category)
                if normalized:
                    output_path = category_output / part_file.name
                    normalized.save(output_path, 'PNG')
                    stats['success'] += 1
                else:
                    stats['failed'] += 1
                    self.logger.warning(f"正規化失敗: {part_file}")
        
        return stats


class PartScaler:
    """パーツスケール調整クラス"""
    
    @staticmethod
    def scale_part_for_face(part_image: Image.Image, 
                           placement_info: Dict[str, Any], 
                           smooth_edges: bool = True) -> Image.Image:
        """
        顔のサイズに合わせてパーツをスケール調整
        
        Args:
            part_image: 正規化済みパーツ画像
            placement_info: 配置情報（landmark_detector.pyから取得）
            smooth_edges: エッジスムージング有効化
            
        Returns:
            スケール調整済みパーツ画像
        """
        target_width = int(placement_info.get('width', 100))
        target_height = int(placement_info.get('height', 100))
        scale_factor = placement_info.get('scale', 1.0)
        
        # スケール適用
        final_width = max(1, int(target_width * scale_factor))
        final_height = max(1, int(target_height * scale_factor))
        
        # リサイズ
        scaled_part = part_image.resize(
            (final_width, final_height), 
            Image.Resampling.LANCZOS
        )
        
        # エッジスムージング
        if smooth_edges:
            scaled_part = scaled_part.filter(ImageFilter.SMOOTH)
        
        return scaled_part


def test_part_normalization():
    """パーツ正規化のテスト"""
    normalizer = PartNormalizer()
    
    # テスト用のパーツ画像パス
    test_part = Path("/Users/uenomomoka/Desktop/Projects/vision_rag/kawakura/assets_png/eye/eye_010.png")
    
    if test_part.exists():
        normalized = normalizer.normalize_part_image(test_part, "eye")
        if normalized:
            # 結果保存
            output_path = Path("/tmp/test_normalized_eye.png")
            normalized.save(output_path)
            print(f"正規化完了: {output_path}")
            
            # 境界情報表示
            bounds = normalizer.get_part_bounds(normalized)
            print(f"境界情報: {bounds}")
        else:
            print("正規化失敗")
    else:
        print(f"テスト画像が見つかりません: {test_part}")


if __name__ == "__main__":
    test_part_normalization()