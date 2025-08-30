#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import os
import time
from pathlib import Path
from typing import List, Optional

from flask import Flask, render_template, request, jsonify, send_from_directory, abort, Response
import threading
import queue
import json
import uuid

# ルート解決
PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# kawakura のメイン処理を呼ぶ
from kawakura.main.run_all_parts import run_once

# face_composer のモジュールをインポート
from face_composer.face_composer import FaceComposer
from face_composer.landmark_detector import FaceLandmarkDetector
from PIL import Image

# Cloud Run では書き込みは /tmp のみ
UPLOADS_DIR = Path(os.environ.get("UPLOADS_DIR", "/tmp/uploads"))
OUTPUTS_DIR = Path(os.environ.get("OUTPUTS_DIR", "/tmp/outputs"))
ASSETS_ROOT = PROJECT_ROOT / "kawakura" / "assets_png"  # リポジトリ同梱

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# mouth は mouse フォルダという命名に対応
CATEGORY_TO_FOLDER = {
    "hair": "hair",
    "eye": "eye",
    "eyebrow": "eyebrow",
    "nose": "nose",
    "mouth": "mouse",
    "ear": "ear",
    "outline": "outline",
    "acc": "acc",
    "beard": "beard",
    "glasses": "glasses",
    "extras": "extras",
    "wrinkles": "wrinkles",
}

def candidate_filenames(category: str, part_num: int) -> List[str]:
    s = str(part_num)
    return [f"{category}_{s.zfill(3)}.png", f"{category}_{s.zfill(2)}.png", f"{category}_{s}.png"]

def find_part_image(category: str, part_num: int) -> Optional[Path]:
    folder = CATEGORY_TO_FOLDER.get(category, category)
    dir_path = ASSETS_ROOT / folder
    if not dir_path.exists():
        return None
    for name in candidate_filenames(category, part_num):
        p = dir_path / name
        if p.exists():
            return p
    # 後方一致の保険
    for p in dir_path.glob("*.png"):
        st = p.stem
        if st.endswith(str(part_num)) or st.endswith(str(part_num).zfill(3)):
            return p
    return None

def to_url_path(p: Path) -> str:
    p = p.resolve()
    if UPLOADS_DIR in p.parents or p == UPLOADS_DIR:
        return f"/uploads/{p.name}"
    if ASSETS_ROOT in p.parents or p == ASSETS_ROOT:
        rel = p.relative_to(ASSETS_ROOT)
        return f"/assets/{rel.as_posix()}"
    return ""

app = Flask(__name__, template_folder="templates")

# SSE用のクライアント管理
active_streams = {}

@app.get("/")
def index():
    return render_template("index.html")

@app.get("/healthz")
def healthz():
    return "ok", 200

@app.get("/stream/<stream_id>")
def stream_progress(stream_id):
    """Server-Sent Events (SSE) エンドポイント"""
    def event_stream():
        # キューを作成してactive_streamsに登録
        q = queue.Queue()
        active_streams[stream_id] = q
        
        try:
            while True:
                try:
                    # キューから進捗情報を取得（タイムアウト30秒）
                    data = q.get(timeout=30)
                    yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                    
                    # 処理完了の場合は終了
                    if data.get("status") == "finished":
                        break
                        
                except queue.Empty:
                    # タイムアウト時はハートビート送信
                    yield "data: {\"status\": \"heartbeat\"}\n\n"
                    
        except GeneratorExit:
            pass
        finally:
            # クリーンアップ
            if stream_id in active_streams:
                del active_streams[stream_id]
    
    return Response(event_stream(), mimetype="text/event-stream")

@app.get("/assets/<path:subpath>")
def serve_assets(subpath: str):
    safe_root = ASSETS_ROOT
    target = (safe_root / subpath).resolve()
    if safe_root not in target.parents and target != safe_root:
        abort(403)
    return send_from_directory(safe_root, subpath)

@app.get("/uploads/<path:filename>")
def serve_uploads(filename: str):
    return send_from_directory(UPLOADS_DIR, filename)

@app.post("/analyze")
def analyze():
    """
    1) 画像を /tmp に保存
    2) run_once() 実行（SSE進捗付き）
    3) 各カテゴリの選択パーツの PNG を解決して返す
    """
    if "image" not in request.files:
        return jsonify(ok=False, error="image is required"), 400

    file = request.files["image"]
    if not file.filename:
        return jsonify(ok=False, error="invalid filename"), 400

    ts = time.strftime("%Y%m%d_%H%M%S")
    ext = Path(file.filename).suffix
    safe_name = f"upload_{ts}{ext}"
    up_path = UPLOADS_DIR / safe_name
    file.save(str(up_path))

    # SSEストリーム用のIDを生成
    stream_id = str(uuid.uuid4())

    # 進捗コールバック関数を定義
    def progress_callback(data):
        if stream_id in active_streams:
            try:
                active_streams[stream_id].put_nowait(data)
            except queue.Full:
                pass  # キューが満杯の場合は無視

    # バックグラウンドで解析を実行
    def run_analysis():
        try:
            # 開始通知
            progress_callback({
                "status": "started",
                "message": "分析を開始しました",
                "timestamp": ts
            })

            # 解析実行
            result = run_once(up_path, progress_callback=progress_callback)

            # JSON保存
            json_path = OUTPUTS_DIR / f"run_{ts}.json"
            json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

            # 結果をパーツ画像URLに変換
            parts = result.get("parts", {})
            resolved = []
            parts_dict = {}  # 合成用のパーツ辞書
            
            for category, info in parts.items():
                sel = (info or {}).get("selected") or {}
                num = sel.get("part_num")
                score = sel.get("score", 0.0)
                if not isinstance(num, int):
                    continue
                img_path = find_part_image(category, num)
                if not img_path:
                    continue
                resolved.append({
                    "category": category,
                    "part_num": num,
                    "score": score,
                    "image_url": to_url_path(img_path),
                })
                
                # 合成用のパーツ情報を準備
                parts_dict[category] = {
                    'part_id': f"{category}_{num:03d}",
                    'image_path': img_path,
                    'part_num': num,
                    'score': score
                }

            # パーツ合成実行
            composed_image_url = None
            if parts_dict:
                try:
                    progress_callback({
                        "status": "composing",
                        "message": "パーツ合成を実行中...",
                        "timestamp": ts
                    })
                    
                    # 顔合成実行（400x400のキャンバス）
                    composer = FaceComposer(canvas_size=(400, 400))
                    composed_image = composer.compose_face(up_path, parts_dict)
                    
                    if composed_image:
                        # 結果画像を保存
                        composed_filename = f"composed_{ts}.png"
                        composed_path = OUTPUTS_DIR / composed_filename
                        
                        # RGBに変換して保存（透明度を白背景で埋める）
                        if composed_image.mode == 'RGBA':
                            background = Image.new('RGB', composed_image.size, (255, 255, 255))
                            background.paste(composed_image, mask=composed_image.split()[-1])
                            composed_image = background
                        
                        composed_image.save(composed_path, 'PNG')
                        composed_image_url = f"/outputs/{composed_filename}"
                        
                except Exception as e:
                    print(f"合成エラー: {e}")
                    # 合成エラーは致命的ではないので続行

            # 完了通知
            progress_callback({
                "status": "finished",
                "input_image_url": to_url_path(up_path),
                "parts": resolved,
                "composed_image_url": composed_image_url,
                "raw_json_url": f"/outputs/{json_path.name}",
                "timestamp": ts
            })

        except Exception as e:
            # エラー通知
            progress_callback({
                "status": "error",
                "error": str(e),
                "timestamp": ts
            })

    # バックグラウンドスレッドで実行
    thread = threading.Thread(target=run_analysis)
    thread.daemon = True
    thread.start()

    # ストリームIDを返す
    return jsonify(ok=True, stream_id=stream_id)

@app.post("/compose")
def compose_face():
    """
    顔合成エンドポイント
    選択されたパーツを使って顔を合成する
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify(ok=False, error="JSON data is required"), 400
        
        input_image_url = data.get('input_image_url')
        selected_parts = data.get('parts', [])
        
        if not input_image_url:
            return jsonify(ok=False, error="input_image_url is required"), 400
        
        # input_image_urlから実際のファイルパスを取得
        if input_image_url.startswith('/uploads/'):
            filename = input_image_url.replace('/uploads/', '')
            base_image_path = UPLOADS_DIR / filename
        else:
            return jsonify(ok=False, error="Invalid input_image_url"), 400
        
        if not base_image_path.exists():
            return jsonify(ok=False, error="Input image not found"), 400
        
        # パーツ情報を変換
        parts_dict = {}
        for part in selected_parts:
            category = part.get('category')
            part_num = part.get('part_num')
            
            if category and part_num:
                # パーツ画像のパスを構築
                part_image_path = find_part_image(category, part_num)
                if part_image_path:
                    parts_dict[category] = {
                        'part_id': f"{category}_{part_num:03d}",
                        'image_path': part_image_path,
                        'score': part.get('score', 0.0)
                    }
        
        if not parts_dict:
            return jsonify(ok=False, error="No valid parts selected"), 400
        
        # 顔合成実行（400x400のキャンバス、中心200,200）
        composer = FaceComposer(canvas_size=(400, 400))
        composed_image = composer.compose_face(base_image_path, parts_dict)
        
        if not composed_image:
            return jsonify(ok=False, error="Face composition failed"), 500
        
        # 結果画像を保存
        ts = time.strftime("%Y%m%d_%H%M%S")
        result_filename = f"composed_{ts}.png"
        result_path = OUTPUTS_DIR / result_filename
        
        # RGBに変換して保存（透明度を白背景で埋める）
        if composed_image.mode == 'RGBA':
            background = Image.new('RGB', composed_image.size, (255, 255, 255))
            background.paste(composed_image, mask=composed_image.split()[-1])
            composed_image = background
        
        composed_image.save(result_path, 'PNG')
        
        return jsonify({
            "ok": True,
            "composed_image_url": f"/outputs/{result_filename}",
            "timestamp": ts
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify(ok=False, error=str(e)), 500

@app.get("/outputs/<path:filename>")
def serve_outputs(filename: str):
    return send_from_directory(OUTPUTS_DIR, filename)

if __name__ == "__main__":
    # Cloud Run では gunicorn を使うが、ローカル確認用に起動可能にしておく
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)
