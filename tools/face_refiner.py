#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
顔類似度ベースの反復調整システム（堅牢版・Geminiキー固定）
- EXIF向き補正 & 最小サイズ拡大
- ランドマーク検出: face_recognition → MediaPipe FaceMesh の順で自動フォールバック
- 特徴量は存在するもののみで類似度を計算（重み自動再正規化）
- 1手ずつ採用するヒルクライム最適化
- Geminiは任意のヒント用（APIキーはコード内に固定）
"""

import os
import re
import json
import sys
import time
from pathlib import Path
from typing import Dict, Tuple, Any, List, Optional

from PIL import Image, ImageDraw, ImageFont, ImageOps

# --- 依存（任意：存在すれば使う） ---
try:
    import numpy as np
except Exception:
    np = None

try:
    import face_recognition  # 68点ランドマーク
    HAS_FACE_REC = True
except Exception:
    HAS_FACE_REC = False

try:
    import mediapipe as mp  # FaceMesh: 468点ランドマーク
    HAS_MP = True
except Exception:
    HAS_MP = False

try:
    import google.generativeai as genai
    HAS_GEMINI = True
except Exception:
    HAS_GEMINI = False

# FaceComposer（既存）
sys.path.append(str(Path(__file__).parent.parent))
from face_composer.face_composer import FaceComposer  # noqa

# ----------------------------
# 設定
# ----------------------------
ADJUSTMENT_STEPS = {
    'position': {
        'up': (0, -5), 'down': (0, 5), 'left': (-5, 0), 'right': (5, 0),
        'up_slight': (0, -3), 'down_slight': (0, 3), 'left_slight': (-3, 0), 'right_slight': (3, 0)
    },
    'scale': {
        'bigger': 0.05, 'smaller': -0.05, 'bigger_slight': 0.03, 'smaller_slight': -0.03
    }
}
SCALE_MIN, SCALE_MAX = 0.1, 2.0
CANDIDATE_MOVES = ['up','down','left','right','up_slight','down_slight','left_slight','right_slight']
CANDIDATE_SCALES = ['bigger','smaller','bigger_slight','smaller_slight']
OUTPUT_DIR = Path("outputs"); OUTPUT_DIR.mkdir(exist_ok=True)

DEFAULT_WEIGHTS = {
    'eye_y_diff': 1.0,
    'brow_eye_gap': 2.0,
    'nose_mouth': 2.0,
    'mouth_ratio': 1.5,
    'compact': 1.0,
}

# ★ Gemini APIキーを固定で使用（ユーザー要望どおり）
GEMINI_API_KEY = "AIzaSyAt-wzZ3WLU1fc6fnzHvDhPsTZJNKnHszU"
def to_pil_rgb(img) -> Image.Image:
    """
    入力が:
      - PIL.Image: RGBA→RGB へ正規化
      - NumPy(OpenCV): BGR/BGRA→RGB/RGBA に変換して PIL.Image へ
    """
    if isinstance(img, Image.Image):
        # 透過を含む場合の予期せぬ退色も防ぐため一度 RGBA→RGB に統一
        if img.mode == "RGBA":
            return img.convert("RGBA").convert("RGB")
        return img.convert("RGB")

    # NumPy配列（OpenCV想定）
    arr = np.asarray(img)
    if arr.ndim == 2:  # グレースケール
        return Image.fromarray(arr)
    if arr.ndim == 3:
        if arr.shape[2] == 3:      # BGR → RGB
            return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))
        if arr.shape[2] == 4:      # BGRA → RGBA → RGB
            rgba = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGBA)
            return Image.fromarray(rgba).convert("RGB")
    # フォールバック
    return Image.fromarray(arr)
# ----------------------------
# 画像ユーティリティ
# ----------------------------
def exif_fix_and_minify(img: Image.Image, min_side: int = 512) -> Image.Image:
    """EXIF向き補正 + 最小辺が小さければ等倍拡大"""
    img = ImageOps.exif_transpose(img)
    w, h = img.size
    if min(w, h) < min_side:
        scale = min_side / min(w, h)
        img = img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
    return img.convert("RGB")

def create_comparison_image(original_image: Image.Image, composed_image: Image.Image) -> Image.Image:
    target_size = (400, 400)
    o = original_image.resize(target_size, Image.LANCZOS)
    c = composed_image.resize(target_size, Image.LANCZOS)
    comp = Image.new('RGB', (800, 400), 'white')
    comp.paste(o, (0, 0)); comp.paste(c, (400, 0))
    draw = ImageDraw.Draw(comp)
    draw.line([(400, 0), (400, 400)], fill=(200, 200, 200), width=2)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
    draw.text((10, 10), "元画像（目標）", fill=(0, 0, 0), font=font)
    draw.text((410, 10), "合成画像（調整対象）", fill=(0, 0, 0), font=font)
    return comp

# ----------------------------
# 入力（JSON/パーツ）
# ----------------------------
def get_original_image_path(json_path: str) -> Optional[Path]:
    with open(json_path, 'r', encoding='utf-8') as f:
        analysis_result = json.load(f)
    input_image = analysis_result.get('input_image', '')
    candidates = [Path(input_image),
                  Path("uploads")/Path(input_image).name,
                  Path("made_pictures")/Path(input_image).name]
    for p in candidates:
        if p.exists():
            return p
    return None

def load_parts_from_json(json_path: str) -> Dict[str, Dict[str, Any]]:
    def find_part_image_path(category: str, part_num: int) -> Optional[Path]:
        assets_root = Path("kawakura/assets_png")
        category_mapping = {
            'mouth': 'mouth',
            'hair': 'hair', 'eye': 'eye', 'eyebrow': 'eyebrow',
            'nose': 'nose', 'ear': 'ear', 'outline': 'outline',
            'acc': 'acc', 'beard': 'beard', 'glasses': 'glasses', 'extras': 'extras'
        }
        folder = assets_root / category_mapping.get(category, category)
        pref = folder.name
        for name in (f"{pref}_{part_num:03d}.png", f"{pref}_{part_num:02d}.png", f"{pref}_{part_num}.png"):
            p = folder / name
            if p.exists():
                return p
        return None

    with open(json_path, 'r', encoding='utf-8') as f:
        analysis = json.load(f)
    parts = analysis.get('parts', {})
    parts_dict: Dict[str, Dict[str, Any]] = {}
    for cat, info in parts.items():
        sel = info.get('selected', {})
        num = sel.get('part_num')
        score = sel.get('score', 0.0)
        if num:
            p = find_part_image_path(cat, num)
            if p:
                parts_dict[cat] = {'part_id': f"{cat}_{num:03d}", 'image_path': p, 'part_num': num, 'score': score}
    return parts_dict

# ----------------------------
# ランドマーク検出（face_recognition → MediaPipe フォールバック）
# ----------------------------
def _center_of(points: List[Tuple[float, float]]) -> Tuple[float, float]:
    if not points: return (0.0, 0.0)
    sx = sum(p[0] for p in points); sy = sum(p[1] for p in points); n = len(points)
    return (sx / n, sy / n)

def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    import math; return math.hypot(a[0] - b[0], a[1] - b[1])

def detect_landmarks_face_rec(img: Image.Image) -> Optional[Dict[str, Any]]:
    if not (HAS_FACE_REC and np is not None): return None
    arr = np.array(img)
    boxes = face_recognition.face_locations(arr, model="hog")
    if not boxes:
        return None
    lmk_list = face_recognition.face_landmarks(arr, model="large")
    if not lmk_list: return None
    lmk = lmk_list[0]
    left_eye_c  = _center_of(lmk.get('left_eye', []))
    right_eye_c = _center_of(lmk.get('right_eye', []))
    left_brow_c = _center_of(lmk.get('left_eyebrow', []))
    right_brow_c= _center_of(lmk.get('right_eyebrow', []))
    nose_tip_pts= lmk.get('nose_tip', []) or lmk.get('nose_bridge', [])
    nose_tip    = nose_tip_pts[-1] if nose_tip_pts else (0.0, 0.0)
    top_lip = lmk.get('top_lip', []); bottom_lip = lmk.get('bottom_lip', [])
    mouth_all = top_lip + bottom_lip
    mouth_c = _center_of(mouth_all)
    mouth_w = 0.0
    if mouth_all:
        xs = [p[0] for p in mouth_all]; ys = [p[1] for p in mouth_all]
        mouth_w = ((_dist((min(xs), min(ys)), (max(xs), max(ys)))))

    top, right, bottom, left = boxes[0]
    face_w = max(right-left, 1); face_h = max(bottom-top, 1)
    face_diag = (face_w**2 + face_h**2) ** 0.5
    return {
        'left_eye_center': left_eye_c, 'right_eye_center': right_eye_c,
        'left_brow_center': left_brow_c, 'right_brow_center': right_brow_c,
        'nose_tip': nose_tip, 'mouth_center': mouth_c, 'mouth_width': float(mouth_w),
        'face_width': float(face_w), 'face_height': float(face_h), 'face_diag': float(face_diag)
    }

# MediaPipe FaceMesh の主要インデックス
MP_LEFT_EYE   = [33, 133, 160, 159, 158, 157, 173]
MP_RIGHT_EYE  = [263, 362, 385, 386, 387, 388, 466]
MP_LEFT_BROW  = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
MP_RIGHT_BROW = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]
MP_NOSE_TIP   = [1, 4]
MP_MOUTH_OUT  = [61, 291, 0, 17, 267, 269, 270, 409, 291]

def detect_landmarks_mp(img: Image.Image) -> Optional[Dict[str, Any]]:
    if not HAS_MP: return None
    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as fm:
        rgb = np.array(img) if np is not None else None
        if rgb is None: return None
        res = fm.process(rgb)
        if not res.multi_face_landmarks: return None
        lm = res.multi_face_landmarks[0]
        h, w = rgb.shape[:2]

        def pts(idx_list):
            return [(lm.landmark[i].x * w, lm.landmark[i].y * h) for i in idx_list if 0 <= i < len(lm.landmark)]

        left_eye_c  = _center_of(pts(MP_LEFT_EYE))
        right_eye_c = _center_of(pts(MP_RIGHT_EYE))
        left_brow_c = _center_of(pts(MP_LEFT_BROW))
        right_brow_c= _center_of(pts(MP_RIGHT_BROW))
        nose_cands  = pts(MP_NOSE_TIP)
        nose_tip    = nose_cands[-1] if nose_cands else (0.0, 0.0)
        mouth_pts   = pts(MP_MOUTH_OUT)
        mouth_c     = _center_of(mouth_pts)
        mouth_w     = 0.0
        if mouth_pts:
            xs = [p[0] for p in mouth_pts]; ys = [p[1] for p in mouth_pts]
            mouth_w = ((_dist((min(xs), min(ys)), (max(xs), max(ys)))))

        xs_all = [p.x*w for p in lm.landmark]; ys_all = [p.y*h for p in lm.landmark]
        left, right = min(xs_all), max(xs_all); top, bottom = min(ys_all), max(ys_all)
        face_w = max(int(right-left), 1); face_h = max(int(bottom-top), 1)
        face_diag = (face_w**2 + face_h**2) ** 0.5

        return {
            'left_eye_center': left_eye_c, 'right_eye_center': right_eye_c,
            'left_brow_center': left_brow_c, 'right_brow_center': right_brow_c,
            'nose_tip': nose_tip, 'mouth_center': mouth_c, 'mouth_width': float(mouth_w),
            'face_width': float(face_w), 'face_height': float(face_h), 'face_diag': float(face_diag)
        }

def detect_landmarks(img: Image.Image) -> Optional[Dict[str, Any]]:
    """堅牢検出（EXIF補正+拡大 → FR → MP の順で試す）"""
    img = exif_fix_and_minify(img)
    lmk = detect_landmarks_face_rec(img)
    if lmk is not None:
        return lmk
    return detect_landmarks_mp(img)

# ----------------------------
# 特徴量 & 類似度（存在する要素のみで計算）
# ----------------------------
def compute_face_features(lmk: Dict[str, Any]) -> Optional[Dict[str, float]]:
    if lmk is None: return None
    left_eye  = lmk['left_eye_center'];  right_eye = lmk['right_eye_center']
    lb = lmk['left_brow_center'];        rb = lmk['right_brow_center']
    nose = lmk['nose_tip'];              mouth_c = lmk['mouth_center']
    mouth_w = lmk.get('mouth_width', 0.0)
    face_w = max(lmk['face_width'], 1e-6); face_h = max(lmk['face_height'], 1e-6)
    ipd = _dist(left_eye, right_eye) + 1e-6
    face_diag = max(lmk['face_diag'], 1e-6)

    def valid(pt): return (pt != (0.0, 0.0))
    feats = {}

    if valid(left_eye) and valid(right_eye):
        feats['eye_y_diff'] = abs(left_eye[1] - right_eye[1]) / face_h

    gaps = []
    if valid(lb) and valid(left_eye):
        gaps.append((lb[1] - left_eye[1]) / ipd)
    if valid(rb) and valid(right_eye):
        gaps.append((rb[1] - right_eye[1]) / ipd)
    if gaps:
        feats['brow_eye_gap'] = float(sum(gaps) / len(gaps))

    if valid(nose) and valid(mouth_c):
        feats['nose_mouth'] = abs(nose[1] - mouth_c[1]) / face_h

    if mouth_w > 0.0:
        feats['mouth_ratio'] = (mouth_w / face_w)

    core_pts = [p for p in (left_eye, right_eye, nose, mouth_c) if valid(p)]
    if len(core_pts) >= 3:
        cx = sum(p[0] for p in core_pts) / len(core_pts)
        cy = sum(p[1] for p in core_pts) / len(core_pts)
        centroid = (cx, cy)
        feats['compact'] = sum(_dist(centroid, p) for p in core_pts) / (len(core_pts) * face_diag)

    return {k: float(v) for k, v in feats.items()}

def similarity_score(f_orig: Dict[str, float], f_comp: Dict[str, float],
                     weights: Dict[str, float] = None) -> float:
    if f_orig is None or f_comp is None: return 0.0
    if weights is None: weights = DEFAULT_WEIGHTS
    keys = [k for k in f_orig.keys() if k in f_comp]
    if not keys: return 0.0
    w_used = {k: weights.get(k, 1.0) for k in keys}
    w_sum = sum(w_used.values()) or 1.0
    w_used = {k: w / w_sum for k, w in w_used.items()}
    sse = 0.0
    for k in keys:
        d = (f_orig[k] - f_comp[k])
        sse += w_used[k] * (d * d)
    return 1.0 / (1.0 + sse)

def detect_and_features(img: Image.Image) -> Optional[Dict[str, float]]:
    lmk = detect_landmarks(img)
    if lmk is None: return None
    return compute_face_features(lmk)

# ----------------------------
# 調整適用（サイド指定対応）
# ----------------------------
def _parse_side_key(category_key: str) -> Tuple[str, Optional[str]]:
    if '.' in category_key:
        cat, side = category_key.split('.', 1)
        if side in ('left','right'): return cat, side
    return category_key, None

def apply_relative_adjustments(current_positions: Dict[str, Any],
                               adjustments: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    new_positions = json.loads(json.dumps(current_positions))
    for key, adj in adjustments.items():
        category, side = _parse_side_key(key)
        if category not in new_positions: continue
        pos_adj = adj.get('position'); scale_adj = adj.get('scale')

        if isinstance(new_positions[category], dict):
            targets = [side] if side in ('left','right') else ['left','right']
            for sd in targets:
                if sd not in new_positions[category]: continue
                x, y, sc = new_positions[category][sd]
                if pos_adj in ADJUSTMENT_STEPS['position']:
                    dx, dy = ADJUSTMENT_STEPS['position'][pos_adj]; x += dx; y += dy
                if scale_adj in ADJUSTMENT_STEPS['scale']:
                    sc = max(SCALE_MIN, min(SCALE_MAX, sc + ADJUSTMENT_STEPS['scale'][scale_adj]))
                new_positions[category][sd] = (x, y, sc)
        else:
            if len(new_positions[category]) >= 3:
                x, y, sc = new_positions[category]
                if pos_adj in ADJUSTMENT_STEPS['position']:
                    dx, dy = ADJUSTMENT_STEPS['position'][pos_adj]; x += dx; y += dy
                if scale_adj in ADJUSTMENT_STEPS['scale']:
                    sc = max(SCALE_MIN, min(SCALE_MAX, sc + ADJUSTMENT_STEPS['scale'][scale_adj]))
                new_positions[category] = (x, y, sc)
    return new_positions

# ----------------------------
# Gemini（任意：キー固定でセットアップ）
# ----------------------------
def setup_gemini_fixed():
    if not HAS_GEMINI: return None
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        return genai.GenerativeModel('gemini-2.5-pro')
    except Exception:
        return None

def parse_json_from_text(text: str) -> Optional[dict]:
    if not text: return None
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    blk = m.group(1) if m else text
    try:
        return json.loads(blk)
    except json.JSONDecodeError:
        return None

def gemini_hint(model, parts: List[str], history: List[dict], comp_img: Image.Image) -> dict:
    if model is None: return {}
    prompt = f"""
次反復で注目すべき最大3パーツ（[{", ".join(parts)}]）と小さな調整案をJSONで。
position: up/down/left/right/_slight, scale: bigger/smaller/_slight。
形式: {{"adjustments": {{"eyebrow": {{"position": "down_slight"}}}}}}
""".strip()
    try:
        resp = model.generate_content([prompt, comp_img])
        js = parse_json_from_text((resp.text or "").strip())
        if isinstance(js, dict):
            return js.get("adjustments", {})
    except Exception:
        pass
    return {}

# ----------------------------
# ヒルクライム最適化
# ----------------------------
def hill_climb_optimize(original_image: Image.Image,
                        composer: FaceComposer,
                        parts_dict: Dict[str, Any],
                        init_positions: Dict[str, Any],
                        max_iters: int = 20,
                        use_gemini_hint: bool = False) -> Tuple[Dict[str, Any], float, List[Path]]:
    feat_orig = detect_and_features(original_image)
    if feat_orig is None:
        raise RuntimeError("元画像からランドマークが検出できませんでした。")

    best_positions = json.loads(json.dumps(init_positions))
    composed = composer.compose_face_with_custom_positions(None, parts_dict, best_positions)
    comp_img = composed.convert("RGB") if composed.mode != "RGB" else composed
    feat_comp = detect_and_features(comp_img)
    best_score = similarity_score(feat_orig, feat_comp)
    print(f"初期スコア: {best_score:.4f}")

    model = setup_gemini_fixed() if use_gemini_hint else None
    history: List[dict] = []
    saved: List[Path] = []

    step_sets = [
        {'moves': ['up','down','left','right'], 'scales': ['bigger','smaller']},
        {'moves': ['up_slight','down_slight','left_slight','right_slight'], 'scales': ['bigger_slight','smaller_slight']},
    ]

    for it in range(1, max_iters+1):
        print(f"\n--- 反復 {it}/{max_iters} ---")
        cmp = create_comparison_image(original_image, comp_img)
        p = OUTPUT_DIR / f"comparison_{it:02d}_{time.strftime('%Y%m%d_%H%M%S')}.png"
        cmp.save(p); saved.append(p); print(f"比較画像: {p}")

        hint = gemini_hint(model, list(parts_dict.keys()), history, cmp) if model else {}
        hint_parts = list(hint.keys())

        improved = False
        trial_best = (best_score, None, None)

        cand_keys: List[str] = []
        for part in parts_dict.keys():
            if isinstance(best_positions.get(part), dict):
                cand_keys += [f"{part}.left", f"{part}.right"]
            else:
                cand_keys.append(part)
        if hint_parts:
            prio = []
            for h in hint_parts:
                if isinstance(best_positions.get(h), dict):
                    prio += [f"{h}.left", f"{h}.right"]
                elif h in cand_keys:
                    prio.append(h)
            cand_keys = list(dict.fromkeys(prio + cand_keys))

        for step in step_sets:
            for key in cand_keys:
                for mv in step['moves']:
                    adj = {key: {'position': mv}}
                    trial_pos = apply_relative_adjustments(best_positions, adj)
                    comp = composer.compose_face_with_custom_positions(None, parts_dict, trial_pos)
                    comp_p = comp.convert("RGB") if comp.mode != "RGB" else comp
                    feat = detect_and_features(comp_p)
                    sc = similarity_score(feat_orig, feat)
                    if sc > trial_best[0]:
                        trial_best = (sc, key, {'position': mv}); improved = True

            for key in cand_keys:
                for sca in step['scales']:
                    adj = {key: {'scale': sca}}
                    trial_pos = apply_relative_adjustments(best_positions, adj)
                    comp = composer.compose_face_with_custom_positions(None, parts_dict, trial_pos)
                    comp_p = comp.convert("RGB") if comp.mode != "RGB" else comp
                    feat = detect_and_features(comp_p)
                    sc = similarity_score(feat_orig, feat)
                    if sc > trial_best[0]:
                        trial_best = (sc, key, {'scale': sca}); improved = True

            if improved and trial_best[1] is not None:
                best_score = trial_best[0]
                best_positions = apply_relative_adjustments(best_positions, {trial_best[1]: trial_best[2]})
                composed = composer.compose_face_with_custom_positions(None, parts_dict, best_positions)
                comp_img = composed.convert("RGB") if composed.mode != "RGB" else composed
                print(f"採用: {trial_best[1]} -> {trial_best[2]} | スコア {best_score:.4f}")
                history.append({'iter': it, 'adopted_key': trial_best[1], 'adopted_adj': trial_best[2], 'score': best_score})
                break

        if not improved:
            print("改善が見られないため終了。")
            break

        outp = OUTPUT_DIR / f"similarity_iter_{it:02d}_{time.strftime('%Y%m%d_%H%M%S')}.png"
        comp_img.save(outp); saved.append(outp); print(f"合成画像: {outp}")

    return best_positions, best_score, saved

# ----------------------------
# エントリポイント
# ----------------------------
def face_similarity_refinement(json_path: str, max_iterations: int = 10, use_gemini_hint: bool = False):
    print("👥 顔類似度調整（堅牢版・Geminiキー固定）開始")
    print(f"JSON: {json_path}")

    orig_path = get_original_image_path(json_path)
    if not orig_path or not orig_path.exists():
        print(f"❌ 元画像が見つかりません: {orig_path}"); return
    original_image = Image.open(orig_path)

    parts_dict = load_parts_from_json(json_path)
    print(f"✅ パーツ読み込み: {list(parts_dict.keys())}")

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

    composer = FaceComposer(canvas_size=(400, 400))
    start = time.time()
    try:
        best_pos, best_score, images = hill_climb_optimize(
            original_image=original_image,
            composer=composer,
            parts_dict=parts_dict,
            init_positions=current_positions,
            max_iters=max_iterations,
            use_gemini_hint=use_gemini_hint
        )
    except RuntimeError as e:
        print(f"❌ エラー: {e}")
        print("ヒント: 元画像が横顔・マスク・強い傾き・逆光・極端な低解像度だと検出に失敗します。別画像か高解像版で再試行してください。")
        return

    elapsed = time.time() - start
    print("\n🏁 完了")
    print(f"最終スコア: {best_score:.4f}")
    print(f"出力枚数: {len(images)}")
    print(f"処理時間: {elapsed:.1f}s")

    result = {'best_score': best_score, 'best_positions': best_pos, 'outputs': [str(p) for p in images]}
    out_json = OUTPUT_DIR / f"refine_result_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"結果JSON: {out_json}")

def main():
    if len(sys.argv) < 2:
        print("使用法:")
        print("  python iterative_face_refiner.py <json_path> [max_iterations] [--gemini]")
        print("例:")
        print("  python iterative_face_refiner.py outputs/run_20250830_164634.json 15 --gemini")
        return
    json_path = sys.argv[1]
    max_iters = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 10
    use_gemini = ('--gemini' in sys.argv)
    face_similarity_refinement(json_path, max_iterations=max_iters, use_gemini_hint=use_gemini)

if __name__ == "__main__":
    main()
