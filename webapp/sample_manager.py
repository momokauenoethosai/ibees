import json
import os
from pathlib import Path
from typing import List, Dict, Any

class SampleManager:
    """サンプルデータを管理するクラス"""
    
    def __init__(self, samples_dir="webapp/static/samples"):
        self.samples_dir = Path(samples_dir)
        self.samples_dir.mkdir(parents=True, exist_ok=True)
        self.samples_config_file = self.samples_dir / "samples.json"
        self._init_default_samples()
    
    def _init_default_samples(self):
        """デフォルトのサンプルデータを初期化"""
        if not self.samples_config_file.exists():
            default_samples = [
                {
                    "id": "sample1",
                    "title": "サンプル1: ビジネス",
                    "description": "フォーマルな雰囲気",
                    "thumbnail": "/static/samples/sample1_thumb.jpg",
                    "image": "/static/samples/sample1.jpg",
                    "json_data": {
                        "input_image": "/static/samples/sample1.jpg",
                        "compact": {
                            "hair": {"part_num": 15, "score": 0.95},
                            "eye": {"part_num": 12, "score": 0.92},
                            "eyebrow": {"part_num": 45, "score": 0.90},
                            "nose": {"part_num": 23, "score": 0.89},
                            "mouth": {"part_num": 34, "score": 0.88},
                            "ear": {"part_num": 41, "score": 0.87},
                            "outline": {"part_num": 32, "score": 0.86}
                        }
                    }
                },
                {
                    "id": "sample2",
                    "title": "サンプル2: カジュアル",
                    "description": "親しみやすい印象",
                    "thumbnail": "/static/samples/sample2_thumb.jpg",
                    "image": "/static/samples/sample2.jpg",
                    "json_data": {
                        "input_image": "/static/samples/sample2.jpg",
                        "compact": {
                            "hair": {"part_num": 29, "score": 0.94},
                            "eye": {"part_num": 38, "score": 0.91},
                            "eyebrow": {"part_num": 93, "score": 0.89},
                            "nose": {"part_num": 66, "score": 0.90},
                            "mouth": {"part_num": 80, "score": 0.88},
                            "ear": {"part_num": 17, "score": 0.85},
                            "outline": {"part_num": 19, "score": 0.87}
                        }
                    }
                },
                {
                    "id": "sample3",
                    "title": "サンプル3: クリエイティブ",
                    "description": "個性的なスタイル",
                    "thumbnail": "/static/samples/sample3_thumb.jpg",
                    "image": "/static/samples/sample3.jpg",
                    "json_data": {
                        "input_image": "/static/samples/sample3.jpg",
                        "compact": {
                            "hair": {"part_num": 42, "score": 0.93},
                            "eye": {"part_num": 55, "score": 0.90},
                            "eyebrow": {"part_num": 67, "score": 0.88},
                            "nose": {"part_num": 81, "score": 0.91},
                            "mouth": {"part_num": 56, "score": 0.87},
                            "ear": {"part_num": 8, "score": 0.86},
                            "outline": {"part_num": 51, "score": 0.85}
                        }
                    }
                }
            ]
            
            with open(self.samples_config_file, 'w', encoding='utf-8') as f:
                json.dump(default_samples, f, ensure_ascii=False, indent=2)
    
    def get_samples(self) -> List[Dict[str, Any]]:
        """サンプルデータの一覧を取得"""
        if self.samples_config_file.exists():
            with open(self.samples_config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []
    
    def get_sample_by_id(self, sample_id: str) -> Dict[str, Any] | None:
        """IDでサンプルデータを取得"""
        samples = self.get_samples()
        for sample in samples:
            if sample['id'] == sample_id:
                return sample
        return None
    
    def add_sample(self, sample_data: Dict[str, Any]):
        """新しいサンプルを追加"""
        samples = self.get_samples()
        
        # IDの重複チェック
        if any(s['id'] == sample_data['id'] for s in samples):
            raise ValueError(f"Sample ID '{sample_data['id']}' already exists")
        
        samples.append(sample_data)
        
        with open(self.samples_config_file, 'w', encoding='utf-8') as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)
    
    def create_sample_from_analysis(self, analysis_result: Dict[str, Any], 
                                  title: str, description: str = "") -> str:
        """分析結果からサンプルを作成"""
        import uuid
        
        sample_id = f"custom_{uuid.uuid4().hex[:8]}"
        
        sample_data = {
            "id": sample_id,
            "title": title,
            "description": description,
            "thumbnail": analysis_result.get('input_image', ''),
            "image": analysis_result.get('input_image', ''),
            "json_data": {
                "input_image": analysis_result.get('input_image', ''),
                "compact": analysis_result.get('compact', {})
            }
        }
        
        self.add_sample(sample_data)
        return sample_id