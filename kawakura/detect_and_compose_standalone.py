#!/usr/bin/env python3
# coding: utf-8
"""
座標検出（MediaPipe）→ 400x400キャンバスで顔パーツ合成
- 右側パーツは左右反転
- PNG透明余白を自動トリム
- BBoxをパーツ別マージンで拡張
- アスペクト維持で拡大（パーツ別ブースト＆最小サイズ）
- 中心合わせで貼り付け
入力: run_*.json（input_image と compact/parts.selected を含む）
出力: outputs/composed_<timestamp>.png, outputs/debug_overlay_<timestamp>.png
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
from PIL import Image, ImageOps, ImageChops

import mediapipe as mp
from mediapipe.python.solutions.face_mesh_connections import (
    FACEMESH_LEFT_EYEBROW, FACEMESH_RIGHT_EYEBROW,
    FACEMESH_LEFT_EYE, FACEMESH_RIGHT_EYE,
    FACEMESH_NOSE, FACEMESH_LIPS
)

# ====== 設定 ======
CANVAS_SIZE = (400, 400)  # (W, H)

ASSETS_ROOT = Path("kawakura/assets_png")
CATEGORY_FOLDER = {
    "hair": "hair",
    "outline": "outline",
    "eye": "eye",
    "eyebrow": "eyebrow",
    "nose": "nose",
    "mouth": "mouse",   # 注意: mouth はフォルダ名が mouse
    "ear": "ear",
    "acc": "acc",
}
ALL_CATEGORIES = ["outline", "hair", "eyebrow", "eye", "nose", "mouth", "ear", "acc"]
LAYER_ORDER = ["outline", "hair", "eyebrow", "eye", "nose", "mouth", "ear", "acc"]

# パーツ別：検出BBoxの拡張倍率（小さく検出されやすい部位を広げる）
BBOX_MARGIN = {
    "eye_left": 1.8, "eye_right": 1.8,
    "eyebrow_left": 2.0, "eyebrow_right": 2.0,
    "nose": 1.3,
    "mouth": 1.4,
    "ear_left": 1.6, "ear_right": 1.6,
}

# パーツ別：拡大ブースト（最終サイズの底上げ）
SCALE_BOOST = {
    "eye_left": 1.4, "eye_right": 1.4,
    "eyebrow_left": 1.6, "eyebrow_right": 1.6,
    "nose": 1.15,
    "mouth": 1.25,
    "ear_left": 1.4, "ear_right": 1.4,
}

# パーツ別：最小サイズ（w,h）
MIN_SIZE = {
    "eye_left": (70, 40), "eye_right": (70, 40),
    "eyebrow_left": (80, 30), "eyebrow_right": (80, 30),
    "nose": (60, 60),
    "mouth": (110, 45),
    "ear_left": (70, 90), "ear_right": (70, 90),
}

# ====== ユーティリティ ======
def _connections_to_index_set(conns) -> List[int]:
    idx = set()
    for a, b in conns:
        idx.add(a); idx.add(b)
    return sorted(idx)

PARTS_INDEX = {
    "eyebrow_left":  _connections_to_index_set(FACEMESH_LEFT_EYEBROW),
    "eyebrow_right": _connections_to_index_set(FACEMESH_RIGHT_EYEBROW),
    "eye_left":      _connections_to_index_set(FACEMESH_LEFT_EYE),
    "eye_right":     _connections_to_index_set(FACEMESH_RIGHT_EYE),
    "nose":          _connections_to_index_set(FACEMESH_NOSE),
    "mouth":         _connections_to_index_set(FACEMESH_LIPS),
    # 耳はFaceMeshに明示の接続がないため近傍点を簡易指定（うまく出ない画像もあります）
    "ear_left_hint":  list(range(234, 254)),
    "ear_right_hint": list(range(454, 474)),
}

def read_image_any(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(str(path))
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        # RGBA → BGR（白背景合成）
        bgr = np.full((img.shape[0], img.shape[1], 3), 255, dtype=np.uint8)
        alpha = img[:, :, 3:4].astype(np.float32) / 255.0
        bgr = (img[:, :, :3] * alpha + bgr * (1 - alpha)).astype(np.uint8)
        img = bgr
    return img

def ensure_outputs() -> Path:
    out = Path("outputs")
    out.mkdir(parents=True, exist_ok=True)
    return out

def load_selected_parts(run_json: Dict) -> Dict[str, Dict]:
    """
    parts.selected or compact から使用パーツ画像を決定
    戻り値: {category: {"image_path": Path, "part_num": int}}
    """
    selected: Dict[str, Dict] = {}
    # compact 優先、なければ parts.selected
    if "compact" in run_json:
        for cat, info in run_json["compact"].items():
            part_num = info.get("part_num")
            if not part_num:
                continue
            folder = CATEGORY_FOLDER.get(cat, cat)
            p = ASSETS_ROOT / folder / f"{folder}_{int(part_num):03d}.png"
            if p.exists():
                selected[cat] = {"image_path": p, "part_num": int(part_num)}
            else:
                print(f"[WARN] パーツ画像が見つかりません: {p}")
    if "parts" in run_json:
        for cat, info in run_json["parts"].items():
            sel = info.get("selected", {})
            part_num = sel.get("part_num")
            if not part_num:
                continue
            if cat in selected:  # compactで既に決定済みならスキップ
                continue
            folder = CATEGORY_FOLDER.get(cat, cat)
            p = ASSETS_ROOT / folder / f"{folder}_{int(part_num):03d}.png"
            if p.exists():
                selected[cat] = {"image_path": p, "part_num": int(part_num)}
            else:
                print(f"[WARN] パーツ画像が見つかりません: {p}")
    return selected

# ====== MediaPipe 検出 ======
def detect_landmark_positions(image_path: Path):
    """
    画像から各パーツの bbox/中心を作成し、(x, y, 1.0) 形式に正規化
    戻り値:
      positions: FaceComposer互換の辞書
      bboxes_scaled: 400x400基準のBBox辞書
      (H, W): 入力画像の原寸
    """
    img = read_image_any(image_path)
    H, W = img.shape[:2]

    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as fm:
        res = fm.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if not res.multi_face_landmarks:
        print("[ERROR] 顔が検出できませんでした")
        return {}, {}, (H, W)

    lm = res.multi_face_landmarks[0].landmark
    pts = np.array([[l.x * W, l.y * H] for l in lm], dtype=np.float32)

    def bbox_of(indices: List[int]) -> Tuple[int, int, int, int]:
        sub = pts[indices, :]
        x0, y0 = sub[:, 0].min(), sub[:, 1].min()
        x1, y1 = sub[:, 0].max(), sub[:, 1].max()
        return int(x0), int(y0), int(x1), int(y1)

    # 元画像座標のBBox
    bboxes = {
        "eye_left":    bbox_of(PARTS_INDEX["eye_left"]),
        "eye_right":   bbox_of(PARTS_INDEX["eye_right"]),
        "eyebrow_left":  bbox_of(PARTS_INDEX["eyebrow_left"]),
        "eyebrow_right": bbox_of(PARTS_INDEX["eyebrow_right"]),
        "nose":        bbox_of(PARTS_INDEX["nose"]),
        "mouth":       bbox_of(PARTS_INDEX["mouth"]),
        "ear_left":    bbox_of(PARTS_INDEX["ear_left_hint"]),
        "ear_right":   bbox_of(PARTS_INDEX["ear_right_hint"]),
    }

    # 400x400 に命中させるスケール
    sx, sy = CANVAS_SIZE[0] / W, CANVAS_SIZE[1] / H

    def to_canvas_bbox(x0, y0, x1, y1):
        x0n, y0n = int(x0 * sx), int(y0 * sy)
        x1n, y1n = int(x1 * sx), int(y1 * sy)
        return x0n, y0n, x1n, y1n

    centers: Dict[str, Tuple[int,int,float]] = {}
    bboxes_scaled: Dict[str, Tuple[int,int,int,int]] = {}

    for key in ["eye_left","eye_right","eyebrow_left","eyebrow_right","nose","mouth","ear_left","ear_right"]:
        x0, y0, x1, y1 = to_canvas_bbox(*bboxes[key])
        cx, cy = (x0 + x1)//2, (y0 + y1)//2
        # パーツ別マージンで拡張
        mx = BBOX_MARGIN.get(key, 1.0)
        w, h = x1 - x0, y1 - y0
        w2, h2 = max(1, int(w * mx)), max(1, int(h * mx))
        x0 = max(0, cx - w2 // 2); y0 = max(0, cy - h2 // 2)
        x1 = min(CANVAS_SIZE[0], cx + w2 // 2); y1 = min(CANVAS_SIZE[1], cy + h2 // 2)

        centers[key] = (cx, cy, 1.0)
        bboxes_scaled[key] = (x0, y0, x1, y1)

    positions = {
        "eye": {"left": centers["eye_left"], "right": centers["eye_right"]},
        "eyebrow": {"left": centers["eyebrow_left"], "right": centers["eyebrow_right"]},
        "nose": centers["nose"],
        "mouth": centers["mouth"],
        "ear": {"left": centers["ear_left"], "right": centers["ear_right"]},
    }
    return positions, bboxes_scaled, (H, W)

# ====== 合成 ======
def trim_alpha(img: Image.Image) -> Image.Image:
    """PNGの透明余白をトリム"""
    if img.mode != "RGBA":
        return img
    bg = Image.new("RGBA", img.size, (0,0,0,0))
    diff = ImageChops.difference(img, bg)
    bbox = diff.getbbox()
    return img if bbox is None else img.crop(bbox)

def resize_to_bbox(img_rgba: Image.Image, bbox: Tuple[int,int,int,int], key: str) -> Image.Image:
    """余白トリム＋ブースト＋最小サイズ＋アスペクト維持でフィット"""
    img_rgba = trim_alpha(img_rgba)

    x0, y0, x1, y1 = bbox
    w, h = max(1, x1 - x0), max(1, y1 - y0)

    # ブースト
    boost = SCALE_BOOST.get(key, 1.0)
    tw, th = int(w * boost), int(h * boost)

    # 最小サイズ
    if key in MIN_SIZE:
        mw, mh = MIN_SIZE[key]
        tw = max(tw, mw)
        th = max(th, mh)

    # アスペクト維持：横基準 → 不足なら縦基準
    pw, ph = img_rgba.width, img_rgba.height
    if pw == 0 or ph == 0:
        return img_rgba
    scale = tw / pw
    tw_a = int(pw * scale); th_a = int(ph * scale)
    if th_a < th:
        scale = th / ph
        tw_a = int(pw * scale); th_a = int(ph * scale)

    return img_rgba.resize((tw_a, th_a), Image.LANCZOS)

def alpha_paste(base: Image.Image, part: Image.Image, left_top: Tuple[int, int]):
    base.alpha_composite(part, left_top)

def compose_on_canvas(selected_parts: Dict[str, Dict],
                      positions: Dict,
                      bboxes_scaled: Dict[str, Tuple[int,int,int,int]],
                      target_parts: Optional[List[str]] = None) -> Image.Image:
    """
    target_parts: 合成対象カテゴリ（Noneなら全カテゴリ）
    """
    if target_parts is None:
        target_parts = ALL_CATEGORIES

    canvas = Image.new("RGBA", CANVAS_SIZE, (255, 255, 255, 0))

    # 1) 輪郭と髪（全面貼り）
    for cat in ["outline", "hair"]:
        if cat not in target_parts or cat not in selected_parts:
            continue
        path = selected_parts[cat]["image_path"]
        if not Path(path).exists():
            print(f"[WARN] {cat} 画像が見つかりません: {path}")
            continue
        img = Image.open(path).convert("RGBA").resize(CANVAS_SIZE, Image.LANCZOS)
        alpha_paste(canvas, img, (0, 0))

    # 2) 主要パーツ（右側は反転、中心合わせ）
    mapping = {
        "eyebrow": ("eyebrow_left","eyebrow_right"),
        "eye": ("eye_left","eye_right"),
        "nose": ("nose",),
        "mouth": ("mouth",),
        "ear": ("ear_left","ear_right"),
        "acc": ("acc",),
    }
    for cat, keys in mapping.items():
        if cat not in target_parts or cat not in selected_parts:
            continue
        src_path = selected_parts[cat]["image_path"]
        if not Path(src_path).exists():
            print(f"[WARN] {cat} 画像が見つかりません: {src_path}")
            continue
        base_part = Image.open(src_path).convert("RGBA")

        for key in keys:
            # 座標チェック
            if ("left" in key or "right" in key) and cat in positions:
                side = "left" if "left" in key else "right"
                if side not in positions[cat]:
                    print(f"[WARN] {cat}:{side} が検出されませんでした → 重なりの可能性")
                    continue
                cx, cy, _ = positions[cat][side]
            else:
                if cat not in positions:
                    print(f"[WARN] {cat} が検出されませんでした → 重なりの可能性")
                    continue
                cx, cy, _ = positions[cat]

            # BBox 決定（フォールバックあり）
            if key in bboxes_scaled:
                bbox = bboxes_scaled[key]
            else:
                bb_key = key
                if key not in bboxes_scaled and cat in ["eyebrow","eye","ear"]:
                    bb_key = f"{cat}_{'left' if 'left' in key else 'right'}"
                bbox = bboxes_scaled.get(bb_key, (cx-20, cy-10, cx+20, cy+10))

            # 右側は反転
            part_img = ImageOps.mirror(base_part) if "right" in key else base_part

            # リサイズ（キー渡し）→ 中心合わせ
            resized = resize_to_bbox(part_img, bbox, key)
            x0, y0, x1, y1 = bbox
            cx, cy = (x0 + x1)//2, (y0 + y1)//2
            left = int(cx - resized.width//2)
            top  = int(cy - resized.height//2)
            alpha_paste(canvas, resized, (left, top))

    return canvas

# ====== 実行器 ======
def run_once(run_json_path: str, target_parts: Optional[List[str]] = None):
    with open(run_json_path, "r", encoding="utf-8") as f:
        run = json.load(f)

    # 入力画像の解決（存在しない場合は uploads/ や made_pictures/ を探す）
    img_path = Path(run.get("input_image", ""))
    if not img_path.exists():
        for base in ("uploads", "made_pictures"):
            cand = Path(base) / img_path.name
            if cand.exists():
                img_path = cand
                break
    if not img_path.exists():
        raise FileNotFoundError(f"入力画像が見つかりません: {run.get('input_image')}")

    # 使用するパーツ（JSONにあるもの全部）
    selected_parts = load_selected_parts(run)
    if not selected_parts:
        print("[ERROR] 使用パーツが見つかりません（compact / parts.selected を確認）")
        return

    # 座標検出
    positions, bboxes_scaled, _ = detect_landmark_positions(img_path)
    if not positions:
        return

    # 合成
    composed = compose_on_canvas(selected_parts, positions, bboxes_scaled, target_parts)

    # デバッグ可視化（検出BBoxオーバーレイ）
    debug = cv2.imread(str(img_path))
    debug = cv2.resize(debug, CANVAS_SIZE)
    for k, (x0,y0,x1,y1) in bboxes_scaled.items():
        cv2.rectangle(debug, (x0,y0), (x1,y1), (0, 128, 255), 1)
        cv2.putText(debug, k, (x0, max(0,y0-4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,128,255), 1, cv2.LINE_AA)

    # 保存
    out_dir = ensure_outputs()
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_png = out_dir / f"composed_{ts}.png"
    composed.save(out_png)
    dbg_png = out_dir / f"debug_overlay_{ts}.png"
    cv2.imwrite(str(dbg_png), debug)

    print(f"✅ 合成完了: {out_png}")
    print(f"🧪 デバッグ: {dbg_png}")

# ====== CLI ======
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("run_json_path", help="run_*.json のパス")
    parser.add_argument(
        "--parts",
        type=str,
        default="",  # 未指定なら全部
        help="表示するパーツカテゴリ (カンマ区切り, 例: eye,eyebrow,nose). 未指定なら全部"
    )
    args = parser.parse_args()

    target_parts = None
    if args.parts.strip():
        toks = [p.strip().lower() for p in args.parts.split(",") if p.strip()]
        target_parts = [p for p in toks if p in ALL_CATEGORIES]
        unknown = [p for p in toks if p not in ALL_CATEGORIES]
        if unknown:
            print(f"[WARN] 未知のカテゴリを無視します: {unknown}")

    run_once(args.run_json_path, target_parts)
