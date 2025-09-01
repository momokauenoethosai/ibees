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
from face_composer.gemini_refinement import GeminiCoordinateRefiner
from PIL import Image
import google.generativeai as genai

# 開発環境ではプロジェクトのoutputsフォルダを使用、Cloud Runでは /tmp を使用
UPLOADS_DIR = Path(os.environ.get("UPLOADS_DIR", PROJECT_ROOT / "uploads"))
OUTPUTS_DIR = Path(os.environ.get("OUTPUTS_DIR", PROJECT_ROOT / "outputs"))
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

@app.post("/refine")
def refine_coordinates():
    """
    Gemini APIを使用して座標を修正する（outputsフォルダの既存ファイルを使用）
    """
    try:
        data = request.get_json()
        print(f"[DEBUG] Received refine request: {data}")
        
        if not data:
            return jsonify(ok=False, error="JSON data is required"), 400
        
        # 既存のJSONファイルとPNGファイルのパスを取得
        raw_json_url = data.get('raw_json_url')
        composed_image_url = data.get('composed_image_url')
        
        print(f"[DEBUG] raw_json_url: {raw_json_url}")
        print(f"[DEBUG] composed_image_url: {composed_image_url}")
        
        if not raw_json_url or not composed_image_url:
            return jsonify(ok=False, error="raw_json_url and composed_image_url are required"), 400
        
        # JSONファイルパスを取得
        if raw_json_url.startswith('/outputs/'):
            json_filename = raw_json_url.replace('/outputs/', '')
            json_path = OUTPUTS_DIR / json_filename
        else:
            return jsonify(ok=False, error="Invalid raw_json_url"), 400
        
        # 合成画像パスを取得
        if composed_image_url.startswith('/outputs/'):
            image_filename = composed_image_url.replace('/outputs/', '')
            composed_image_path = OUTPUTS_DIR / image_filename
        else:
            return jsonify(ok=False, error="Invalid composed_image_url"), 400
        
        # ファイル存在確認
        if not json_path.exists():
            return jsonify(ok=False, error=f"JSON file not found: {json_path}"), 400
        
        if not composed_image_path.exists():
            return jsonify(ok=False, error=f"Composed image not found: {composed_image_path}"), 400
        
        # JSONファイルから分析結果を読み込み
        try:
            analysis_result = json.loads(json_path.read_text(encoding='utf-8'))
        except Exception as e:
            return jsonify(ok=False, error=f"Failed to parse JSON: {e}"), 400
        
        # パーツ情報を抽出
        parts = analysis_result.get('parts', {})
        parts_dict = {}
        
        for category, part_info in parts.items():
            selected = part_info.get('selected', {})
            part_num = selected.get('part_num')
            score = selected.get('score', 0.0)
            
            if part_num:
                part_image_path = find_part_image(category, part_num)
                if part_image_path:
                    parts_dict[category] = {
                        'part_id': f"{category}_{part_num:03d}",
                        'image_path': part_image_path,
                        'part_num': part_num,
                        'score': score
                    }
        
        if not parts_dict:
            return jsonify(ok=False, error="No valid parts found in JSON"), 400
        
        print(f"[DEBUG] Found {len(parts_dict)} parts: {list(parts_dict.keys())}")
        
        # 現在の座標設定
        base_positions = {
            'hair': (200, 200, 1.0),
            'eye': {
                'left': (225, 215, 0.2),
                'right': (175, 215, 0.2),
                'single': (200, 215, 0.2)
            },
            'eyebrow': {
                'left': (225, 185, 0.2),
                'right': (175, 185, 0.2),
                'single': (200, 185, 0.2)
            },
            'nose': (200, 230, 0.2),
            'mouth': (200, 255, 0.25),
            'ear': {
                'left': (250, 220, 0.28),
                'right': (150, 220, 0.28)
            },
            'outline': (200, 200, 1.0),
            'acc': (200, 180, 0.3),
            'beard': (200, 300, 0.4),
            'glasses': (200, 215, 0.5)
        }
        
        # Gemini APIで座標修正
        print(f"[DEBUG] Calling Gemini API for coordinate refinement...")
        refiner = GeminiCoordinateRefiner()
        refined_positions = refiner.refine_coordinates(
            composed_image_path=composed_image_path,
            parts_dict=parts_dict,
            current_positions=base_positions,
            canvas_size=(400, 400)
        )
        
        if not refined_positions:
            return jsonify(ok=False, error="Failed to refine coordinates with Gemini"), 500
        
        # 修正結果をJSONとして保存
        ts = time.strftime("%Y%m%d_%H%M%S")
        refined_filename = f"refined_{ts}.json"
        refined_json_path = OUTPUTS_DIR / refined_filename
        
        refinement_result = {
            "original_json_file": raw_json_url,
            "original_composed_image": composed_image_url,
            "refined_positions": refined_positions,
            "parts_dict": {k: {'part_num': v['part_num'], 'score': v['score']} for k, v in parts_dict.items()},
            "timestamp": ts
        }
        
        refined_json_path.write_text(
            json.dumps(refinement_result, ensure_ascii=False, indent=2), 
            encoding="utf-8"
        )
        
        print(f"[DEBUG] Refinement completed successfully!")
        
        return jsonify({
            "ok": True,
            "refined_positions": refined_positions,
            "refinement_json_url": f"/outputs/{refined_filename}",
            "message": "Coordinates refined successfully using existing output files.",
            "timestamp": ts
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify(ok=False, error=str(e)), 500

@app.post("/recompose")
def recompose_with_refined_coordinates():
    """
    Geminiで修正された座標を使って再合成する
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify(ok=False, error="JSON data is required"), 400
        
        refinement_json_url = data.get('refinement_json_url')
        if not refinement_json_url:
            return jsonify(ok=False, error="refinement_json_url is required"), 400
        
        print(f"[DEBUG] Recomposing with refined coordinates: {refinement_json_url}")
        
        # refinementファイルを読み込み
        if refinement_json_url.startswith('/outputs/'):
            refinement_filename = refinement_json_url.replace('/outputs/', '')
            refinement_path = OUTPUTS_DIR / refinement_filename
        else:
            return jsonify(ok=False, error="Invalid refinement_json_url"), 400
        
        if not refinement_path.exists():
            return jsonify(ok=False, error="Refinement JSON not found"), 400
        
        try:
            refinement_data = json.loads(refinement_path.read_text(encoding='utf-8'))
        except Exception as e:
            return jsonify(ok=False, error=f"Failed to parse refinement JSON: {e}"), 400
        
        # 修正された座標とパーツ情報を取得
        refined_positions = refinement_data.get('refined_positions', {})
        parts_dict = refinement_data.get('parts_dict', {})
        
        if not refined_positions or not parts_dict:
            return jsonify(ok=False, error="Invalid refinement data"), 400
        
        # パーツ画像パスを解決
        full_parts_dict = {}
        for category, part_info in parts_dict.items():
            part_num = part_info.get('part_num')
            score = part_info.get('score', 0.0)
            
            if part_num:
                part_image_path = find_part_image(category, part_num)
                if part_image_path:
                    full_parts_dict[category] = {
                        'part_id': f"{category}_{part_num:03d}",
                        'image_path': part_image_path,
                        'part_num': part_num,
                        'score': score
                    }
        
        # カスタム座標で顔合成実行
        print(f"[DEBUG] Recomposing with {len(full_parts_dict)} parts using refined coordinates")
        
        # FaceComposerに修正された座標を適用
        composer = FaceComposer(canvas_size=(400, 400))
        
        # 座標を一時的に適用（後で詳細実装）
        composed_image = composer.compose_face_with_custom_positions(
            base_image_path=None,
            parts_dict=full_parts_dict,
            custom_positions=refined_positions
        )
        
        if not composed_image:
            return jsonify(ok=False, error="Recomposition failed"), 500
        
        # 結果画像を保存
        ts = time.strftime("%Y%m%d_%H%M%S")
        recomposed_filename = f"recomposed_{ts}.png"
        recomposed_path = OUTPUTS_DIR / recomposed_filename
        
        # RGBに変換して保存
        if composed_image.mode == 'RGBA':
            background = Image.new('RGB', composed_image.size, (255, 255, 255))
            background.paste(composed_image, mask=composed_image.split()[-1])
            composed_image = background
        
        composed_image.save(recomposed_path, 'PNG')
        
        return jsonify({
            "ok": True,
            "recomposed_image_url": f"/outputs/{recomposed_filename}",
            "refined_positions": refined_positions,
            "timestamp": ts,
            "message": "Successfully recomposed with refined coordinates"
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify(ok=False, error=str(e)), 500

@app.post("/iterative_refine")
def iterative_refine():
    """
    反復的相対調整をSSEで実行
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify(ok=False, error="JSON data is required"), 400
        
        raw_json_url = data.get('raw_json_url')
        if not raw_json_url:
            return jsonify(ok=False, error="raw_json_url is required"), 400
        
        # SSEストリーム用のIDを生成
        stream_id = str(uuid.uuid4())
        
        def progress_callback(data):
            if stream_id in active_streams:
                try:
                    active_streams[stream_id].put_nowait(data)
                except queue.Full:
                    pass
        
        def run_iterative_refinement():
            try:
                # JSONファイル読み込み
                if raw_json_url.startswith('/outputs/'):
                    json_filename = raw_json_url.replace('/outputs/', '')
                    json_path = OUTPUTS_DIR / json_filename
                else:
                    raise ValueError("Invalid raw_json_url")
                
                if not json_path.exists():
                    raise FileNotFoundError(f"JSON file not found: {json_path}")
                
                analysis_result = json.loads(json_path.read_text(encoding='utf-8'))
                
                # パーツ情報を抽出
                parts = analysis_result.get('parts', {})
                parts_dict = {}
                
                for category, part_info in parts.items():
                    selected = part_info.get('selected', {})
                    part_num = selected.get('part_num')
                    score = selected.get('score', 0.0)
                    
                    if part_num:
                        part_image_path = find_part_image(category, part_num)
                        if part_image_path:
                            parts_dict[category] = {
                                'part_id': f"{category}_{part_num:03d}",
                                'image_path': part_image_path,
                                'part_num': part_num,
                                'score': score
                            }
                
                if not parts_dict:
                    raise ValueError("No valid parts found")
                
                progress_callback({
                    "status": "started",
                    "message": f"反復調整を開始（{len(parts_dict)}個のパーツ）",
                    "parts": list(parts_dict.keys())
                })
                
                # 初期座標
                current_positions = {
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
                }
                
                # Gemini設定
                genai.configure(api_key="AIzaSyAt-wzZ3WLU1fc6fnzHvDhPsTZJNKnHszU")
                gemini_model = genai.GenerativeModel('gemini-1.5-flash')
                
                # 相対調整ステップ定義
                ADJUSTMENT_STEPS = {
                    'position': {
                        'up': (0, -5), 'down': (0, 5), 'left': (-5, 0), 'right': (5, 0),
                        'up_slight': (0, -3), 'down_slight': (0, 3), 'left_slight': (-3, 0), 'right_slight': (3, 0)
                    },
                    'scale': {
                        'bigger': 0.05, 'smaller': -0.05, 'bigger_slight': 0.03, 'smaller_slight': -0.03
                    }
                }
                
                def apply_relative_adjustments(positions, adjustments):
                    new_positions = json.loads(json.dumps(positions))
                    
                    for category, adj_info in adjustments.items():
                        if category not in new_positions:
                            continue
                            
                        pos_adj = adj_info.get('position')
                        scale_adj = adj_info.get('scale')
                        current_pos = new_positions[category]
                        
                        if isinstance(current_pos, dict):
                            for side in ['left', 'right']:
                                if side in current_pos:
                                    x, y, scale = current_pos[side]
                                    
                                    if pos_adj and pos_adj in ADJUSTMENT_STEPS['position']:
                                        dx, dy = ADJUSTMENT_STEPS['position'][pos_adj]
                                        x, y = x + dx, y + dy
                                    
                                    if scale_adj and scale_adj in ADJUSTMENT_STEPS['scale']:
                                        scale_delta = ADJUSTMENT_STEPS['scale'][scale_adj]
                                        scale = max(0.1, min(1.0, scale + scale_delta))
                                    
                                    new_positions[category][side] = (x, y, scale)
                        else:
                            x, y, scale = current_pos
                            
                            if pos_adj and pos_adj in ADJUSTMENT_STEPS['position']:
                                dx, dy = ADJUSTMENT_STEPS['position'][pos_adj]
                                x, y = x + dx, y + dy
                            
                            if scale_adj and scale_adj in ADJUSTMENT_STEPS['scale']:
                                scale_delta = ADJUSTMENT_STEPS['scale'][scale_adj]
                                scale = max(0.1, min(1.0, scale + scale_delta))
                            
                            new_positions[category] = (x, y, scale)
                    
                    return new_positions
                
                # 反復調整ループ
                iteration_images = []
                max_iterations = 5
                
                for iteration in range(max_iterations):
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    
                    progress_callback({
                        "status": "composing",
                        "iteration": iteration + 1,
                        "max_iterations": max_iterations,
                        "message": f"反復 {iteration + 1}/{max_iterations}: 合成中..."
                    })
                    
                    # 1. 現在の座標で合成
                    composer = FaceComposer(canvas_size=(400, 400))
                    composed_image = composer.compose_face_with_custom_positions(
                        base_image_path=None,
                        parts_dict=parts_dict,
                        custom_positions=current_positions
                    )
                    
                    if not composed_image:
                        raise Exception(f"反復{iteration + 1}: 合成失敗")
                    
                    # 画像保存
                    iteration_filename = f"iteration_{iteration + 1}_{ts}.png"
                    iteration_path = OUTPUTS_DIR / iteration_filename
                    
                    if composed_image.mode == 'RGBA':
                        background = Image.new('RGB', composed_image.size, (255, 255, 255))
                        background.paste(composed_image, mask=composed_image.split()[-1])
                        composed_image = background
                    
                    composed_image.save(iteration_path)
                    iteration_url = f"/outputs/{iteration_filename}"
                    iteration_images.append(iteration_url)
                    
                    progress_callback({
                        "status": "iteration_image",
                        "iteration": iteration + 1,
                        "image_url": iteration_url,
                        "message": f"反復 {iteration + 1}: 画像生成完了"
                    })
                    
                    # 2. Gemini分析
                    progress_callback({
                        "status": "analyzing",
                        "iteration": iteration + 1,
                        "message": f"反復 {iteration + 1}: Gemini分析中..."
                    })
                    
                    parts_list = ", ".join(list(parts_dict.keys()))
                    prompt = f"""
この顔合成画像を分析し、不自然な配置があれば以下のパーツの相対的調整指示を出してください。

## 対象パーツ（これらの名前のみ使用）
{parts_list}

## 調整指示オプション
**位置**: up, down, left, right (5px) / up_slight, down_slight, left_slight, right_slight (3px)
**サイズ**: bigger, smaller (0.05倍) / bigger_slight, smaller_slight (0.03倍)

## 出力形式（JSON のみ）
```json
{{
  "adjustments": {{
    "hair": {{"position": "up_slight", "scale": "smaller"}},
    "eye": {{"position": "down"}}
  }},
  "satisfied": false,
  "notes": "調整理由"
}}
```

**重要**: パーツ名は [{parts_list}] から正確に選択。満足なら satisfied: true。
                    """
                    
                    response = gemini_model.generate_content([prompt, composed_image])
                    
                    if not response.text:
                        raise Exception(f"反復{iteration + 1}: Gemini応答なし")
                    
                    # JSON解析
                    response_text = response.text.strip()
                    if '```json' in response_text:
                        start_idx = response_text.find('```json') + 7
                        end_idx = response_text.find('```', start_idx)
                        code_block = response_text[start_idx:end_idx].strip()
                    else:
                        code_block = response_text
                    
                    try:
                        adjustment_result = json.loads(code_block)
                    except json.JSONDecodeError:
                        raise Exception(f"反復{iteration + 1}: JSON解析失敗")
                    
                    satisfied = adjustment_result.get('satisfied', False)
                    adjustments = adjustment_result.get('adjustments', {})
                    notes = adjustment_result.get('notes', '')
                    
                    progress_callback({
                        "status": "adjustment_result",
                        "iteration": iteration + 1,
                        "satisfied": satisfied,
                        "adjustments": adjustments,
                        "notes": notes,
                        "message": f"反復 {iteration + 1}: {notes}"
                    })
                    
                    # 満足なら終了
                    if satisfied or not adjustments:
                        progress_callback({
                            "status": "finished",
                            "iteration": iteration + 1,
                            "final_image_url": iteration_url,
                            "final_positions": current_positions,
                            "iteration_images": iteration_images,
                            "message": "反復調整完了！" + (f" ({notes})" if notes else "")
                        })
                        return
                    
                    # 相対調整を適用
                    current_positions = apply_relative_adjustments(current_positions, adjustments)
                
                # 最大回数に達した場合
                progress_callback({
                    "status": "finished",
                    "iteration": max_iterations,
                    "final_image_url": iteration_images[-1] if iteration_images else None,
                    "final_positions": current_positions,
                    "iteration_images": iteration_images,
                    "message": f"最大反復回数（{max_iterations}回）に達しました"
                })
                
            except Exception as e:
                progress_callback({
                    "status": "error",
                    "error": str(e)
                })
        
        # バックグラウンドで実行
        thread = threading.Thread(target=run_iterative_refinement)
        thread.daemon = True
        thread.start()
        
        return jsonify(ok=True, stream_id=stream_id)
        
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
