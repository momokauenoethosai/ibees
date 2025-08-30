#!/usr/bin/env python3
"""
JSONファイルのパーツ情報を使って顔合成をリクエストするスクリプト
"""

import json
import requests
from pathlib import Path

def compose_face_from_json(json_path: str, server_url: str = "http://localhost:5000"):
    """
    JSONファイルの情報を使って顔合成をリクエスト
    
    Args:
        json_path: 分析結果のJSONファイルパス
        server_url: FlaskサーバーのURL
    """
    
    # JSONファイルを読み込み
    json_file = Path(json_path)
    if not json_file.exists():
        print(f"JSONファイルが見つかりません: {json_path}")
        return
    
    with open(json_file, 'r', encoding='utf-8') as f:
        analysis_result = json.load(f)
    
    # 入力画像パスを取得
    input_image = analysis_result.get('input_image')
    if not input_image:
        print("入力画像パスが見つかりません")
        return
    
    # 画像をアップロード（既にサーバーにある場合はスキップ）
    image_path = Path(input_image)
    if not image_path.exists():
        print(f"入力画像が見つかりません: {input_image}")
        return
    
    # 画像をサーバーにアップロード
    print(f"画像をアップロード中: {image_path.name}")
    with open(image_path, 'rb') as f:
        files = {'image': (image_path.name, f, 'image/png')}
        upload_response = requests.post(f"{server_url}/analyze", files=files)
    
    if not upload_response.json().get('ok'):
        print(f"アップロードエラー: {upload_response.text}")
        return
    
    # パーツ情報を変換
    parts = []
    compact_data = analysis_result.get('compact', {})
    
    for category, part_info in compact_data.items():
        part_num = part_info.get('part_num')
        score = part_info.get('score', 0.0)
        
        if part_num:
            parts.append({
                'category': category,
                'part_num': part_num,
                'score': score
            })
    
    if not parts:
        print("選択されたパーツがありません")
        return
    
    print(f"選択されたパーツ: {len(parts)}個")
    for part in parts:
        print(f"  - {part['category']}: {part['part_num']} (score: {part['score']:.3f})")
    
    # 合成リクエストを送信
    compose_data = {
        'input_image_url': f"/uploads/{image_path.name}",
        'parts': parts
    }
    
    print("\n顔合成を実行中...")
    compose_response = requests.post(
        f"{server_url}/compose",
        json=compose_data,
        headers={'Content-Type': 'application/json'}
    )
    
    if compose_response.status_code == 200:
        result = compose_response.json()
        if result.get('ok'):
            composed_url = result.get('composed_image_url')
            print(f"✅ 合成完了: {server_url}{composed_url}")
            
            # 結果画像をダウンロード
            download_response = requests.get(f"{server_url}{composed_url}")
            if download_response.status_code == 200:
                output_path = Path(f"composed_result_{result['timestamp']}.png")
                output_path.write_bytes(download_response.content)
                print(f"✅ 画像保存: {output_path}")
            else:
                print("画像ダウンロードに失敗")
        else:
            print(f"❌ 合成エラー: {result.get('error')}")
    else:
        print(f"❌ リクエストエラー: {compose_response.text}")

if __name__ == "__main__":
    # JSONファイルのパス
    json_path = "/Users/uenomomoka/Desktop/Projects/vision_rag/outputs/run_2_20250829_183346.json"
    
    # 顔合成実行
    compose_face_from_json(json_path)