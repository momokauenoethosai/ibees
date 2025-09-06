#!/usr/bin/env python3
# coding: utf-8
"""
JSON(analysis結果) → アセット自動解決 → 顔パーツ合成
- 入力: run_*.json（input_image, compact または parts.selected を含む）
- 使うパーツ: ear / eyebrow / eye / nose / mouth / (hair, outlineも利用)
- 右側パーツは左右反転して配置
- 目は縦ストレッチ＋下げオフセット（固定値）
- 鼻は FACEMESH_NOSE のBBoxにフィット（アルファ余白トリム、縦横倍率固定）
- 口は口角(61,291)で角度合わせ＆縦横倍率固定
- 輪郭は「髪の上端（＝頭頂）」〜「顎(#152)」に縦フィットし、横は耳の“内側幅”に収めつつ比率維持
出力: outputs/face_composed_<timestamp>.png
"""

from pathlib import Path
import math
import json
import time
import argparse

import cv2
import mediapipe as mp
from PIL import Image, ImageOps
from mediapipe.python.solutions.face_mesh_connections import (
    FACEMESH_NOSE, FACEMESH_LIPS,
    FACEMESH_LEFT_EYEBROW, FACEMESH_RIGHT_EYEBROW, FACEMESH_FACE_OVAL
)

# ==========================
# 固定パラメータ（引数では受け取らない）
# ==========================
EYE_DY = 1
EYE_STRETCH_Y = 1.4
EYE_STRETCH_X = 1.2

NOSE_MARGIN_X = 1.0
NOSE_MARGIN_Y = 1.0

MOUTH_SCALE_X = 1.2
MOUTH_SCALE_Y = 1.4
MOUTH_DX = 0
MOUTH_DY = 1

BROW_MARGIN_X = 0.0
BROW_MARGIN_Y = 0.0
BROW_TIGHTEN_Y = 1.0
BROW_DY = 1

# === 耳サイズ・位置調整（横を細く、縦を長く） ===
EAR_W_RATIO   = 0.18
EAR_H_RATIO   = 0.25
EAR_SCALE_X   = 0.90
EAR_SCALE_Y   = 1.10
EAR_W_MIN_FR  = 0.12
EAR_W_MAX_FR  = 0.30
EAR_H_MIN_FR  = 0.30
EAR_H_MAX_FR  = 0.70
EAR_PUSH_OUT_FR = 0.10
EAR_LIFT_UP_FR  = 0.05
EAR_LIFT_UP_PX  = 0
LEFT_EAR_HINTS  = list(range(234, 254))
RIGHT_EAR_HINTS = list(range(454, 474))

# ===== 髪パラメータ =====
HAIR_PUSH_OUT_FR = 0.25
HAIR_TOP_LIFT_FR = 2.0  # 顔高×この分だけ上に持ち上げ（正の値で上）
HAIR_SHIFT_X = 0


OUTLINE_CHIN_EXT_FR = 0.95
OUTLINE_WIDTH_SCALE = 0.9
mp_face_mesh = mp.solutions.face_mesh

# ==========================
# ユーティリティ
# ==========================
def _lm_xy(lm, idx, W, H):
    p = lm.landmark[idx]
    return int(p.x * W), int(p.y * H)

def read_rgba(path: Path) -> Image.Image:
    return Image.open(path).convert("RGBA")

def pil_alpha_paste_center(base: Image.Image, part: Image.Image, center_xy):
    x, y = center_xy
    base.alpha_composite(part, (int(x - part.width // 2), int(y - part.height // 2)))

def to_xy(lm, idx, W, H):
    p = lm.landmark[idx]
    return (int(p.x * W), int(p.y * H))

def angle_deg(p1, p2):
    dx, dy = (p2[0] - p1[0]), (p2[1] - p1[1])
    return -cv2.fastAtan2(dy, dx)

def scale_to_width(img: Image.Image, target_w: float, min_scale=0.1) -> Image.Image:
    if img.width <= 0:
        return img
    scale = max(min_scale, target_w / img.width)
    nw, nh = max(1, int(img.width * scale)), max(1, int(img.height * scale))
    return img.resize((nw, nh), Image.LANCZOS)

def alpha_bbox(img_rgba: Image.Image):
    return img_rgba.split()[-1].getbbox() if img_rgba.mode == "RGBA" else None

def detect_landmarks(image_path: Path):
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise FileNotFoundError(f"Cannot read: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as fm:
        res = fm.process(rgb)
    if not res.multi_face_landmarks:
        return None, bgr.shape[1], bgr.shape[0]
    return res.multi_face_landmarks[0], bgr.shape[1], bgr.shape[0]

def _collect_indices_from_connections(conns):
    s = set()
    for a, b in conns:
        s.add(a); s.add(b)
    return sorted(list(s))

def estimate_outline_center_x(lm, W, H):
    idxs = {a for a,b in FACEMESH_FACE_OVAL} | {b for a,b in FACEMESH_FACE_OVAL}
    xs = [int(lm.landmark[i].x * W) for i in idxs]
    return (min(xs) + max(xs)) // 2 if xs else W // 2

def _bbox_from_indices(lm, indices, W, H, margin_x=0.0, margin_y=0.0):
    xs, ys = [], []
    for i in indices:
        p = lm.landmark[i]
        xs.append(int(p.x * W)); ys.append(int(p.y * H))
    x0, x1 = max(0, min(xs)), min(W - 1, max(xs))
    y0, y1 = max(0, min(ys)), min(H - 1, max(ys))
    bw, bh = max(1, x1 - x0), max(1, y1 - y0)
    x0 = max(0, int(x0 - bw * margin_x)); x1 = min(W - 1, int(x1 + bw * margin_x))
    y0 = max(0, int(y0 - bh * margin_y)); y1 = min(H - 1, int(y1 + bh * margin_y))
    return (x0, y0, x1, y1)

def _resize_to_bbox_cropped(asset: Image.Image, bbox, tighten_x=1.0, tighten_y=1.0):
    l, t, r, b = alpha_bbox(asset) or (0, 0, asset.width, asset.height)
    cropped = asset.crop((l, t, r, b))
    x0, y0, x1, y1 = bbox
    bw, bh = max(1, x1 - x0), max(1, y1 - y0)
    tw = max(1, int(bw * tighten_x)); th = max(1, int(bh * tighten_y))
    return cropped.resize((tw, th), Image.LANCZOS)
def plan_outline_frame(lm, W, H,
                       push_out_fr: float,
                       top_lift_fr: float,
                       chin_ext_fr: float,
                       inner_pad_fr: float = 0.04):
    """髪・輪郭を置くための枠(上端y, 左右x)を先に決める"""
    def _xy(idx):
        p = lm.landmark[idx]
        return (int(p.x * W), int(p.y * H))

    left_cheek  = _xy(234)
    right_cheek = _xy(454)
    forehead    = _xy(10)
    chin        = _xy(152)
    face_w = max(1, abs(right_cheek[0] - left_cheek[0]))
    face_h = max(1, abs(chin[1] - forehead[1]))

    # 横（耳前アンカー±余白）
    ear_left_x  = _xy(127)[0]
    ear_right_x = _xy(356)[0]
    left_x  = max(0, int(ear_left_x  - face_w * push_out_fr))
    right_x = min(W - 1, int(ear_right_x + face_w * push_out_fr))
    # 輪郭は耳“内側”に収めたいので少し内側に寄せる
    inner_pad = int(face_w * inner_pad_fr)
    inner_left_x  = min(right_x-1, max(0, left_x  + inner_pad))
    inner_right_x = min(W-1,       max(inner_left_x+1, right_x - inner_pad))

    # 上（頭頂）＝ FACE_OVAL の最小y をさらに持ち上げ
    oval_idxs = {a for a, b in FACEMESH_FACE_OVAL} | {b for a, b in FACEMESH_FACE_OVAL}
    ys = [int(lm.landmark[i].y * H) for i in oval_idxs]
    y_top_oval = min(ys) if ys else forehead[1]
    top_of_head = int(max(0, y_top_oval - face_h * top_lift_fr))

    # 縦（顎までの距離を基準に下方向へ拡張：輪郭用）
    base_h = max(1, chin[1] - top_of_head)
    outline_h = base_h * chin_ext_fr

    return top_of_head, inner_left_x, inner_right_x, outline_h
def get_outline_center_frac_x(outline_png: Path) -> float:
    """輪郭PNGのアルファ有効範囲の中心Xを、その幅に対する相対(0..1)で返す"""
    im = Image.open(outline_png).convert("RGBA")
    ab = im.split()[-1].getbbox()  # (l,t,r,b)
    if not ab:
        return 0.5
    l, t, r, b = ab
    w = max(1, r - l)
    center_x = (l + r) / 2.0
    return float(center_x - l) / float(w)
def get_nose_center_x(lm, W, H) -> int:
    idxs = _collect_indices_from_connections(FACEMESH_NOSE)
    x0, y0, x1, y1 = _bbox_from_indices(lm, idxs, W, H)
    cx = (x0 + x1) // 2
    return cx

def recenter_frame_around_axis(axis_x: int, left_x: int, right_x: int, W: int):
    rect_w = max(1, right_x - left_x)
    new_left  = int(axis_x - rect_w / 2)
    new_right = new_left + rect_w
    # 画像内にクリップ
    if new_left < 0:
        shift = -new_left
        new_left = 0
        new_right = min(W, new_right + shift)
    if new_right > W:
        shift = new_right - W
        new_right = W
        new_left  = max(0, new_left - shift)
    # 念のため整合性
    if new_right <= new_left:
        new_right = min(W, new_left + 1)
    return new_left, new_right

# ==========================
# パーツ合成（目/鼻/口/耳/眉）
# ==========================
def place_two_eyes_from_left_asset(base_rgba: Image.Image, lm, left_eye_asset: Path,
                                   eye_dy=EYE_DY,
                                   eye_stretch_y=EYE_STRETCH_Y,
                                   eye_stretch_x=EYE_STRETCH_X):
    W, H = base_rgba.size
    def eye_geom(outer_idx, inner_idx):
        p_outer = to_xy(lm, outer_idx, W, H)
        p_inner = to_xy(lm, inner_idx, W, H)
        cx = (p_outer[0] + p_inner[0]) // 2
        cy = (p_outer[1] + p_inner[1]) // 2
        w  = math.hypot(p_inner[0]-p_outer[0], p_inner[1]-p_outer[1])
        ang = angle_deg(p_outer, p_inner)
        return (cx, cy), w, ang

    Rc, Rw, Rang = eye_geom(33, 133)
    Lc, Lw, Lang = eye_geom(362, 263)

    eye_asset = read_rgba(left_eye_asset)

    L_img = scale_to_width(eye_asset, Lw)
    L_img = L_img.resize(
        (int(L_img.width * eye_stretch_x),
         int(L_img.height * eye_stretch_y)),
        Image.LANCZOS
    )
    L_img = L_img.rotate(Lang, resample=Image.BICUBIC, expand=True)
    pil_alpha_paste_center(base_rgba, L_img, (Lc[0], Lc[1] + eye_dy))

    # 右目
    R_img = ImageOps.mirror(eye_asset)
    R_img = scale_to_width(R_img, Rw)
    R_img = R_img.resize(
        (int(R_img.width * eye_stretch_x),
         int(R_img.height * eye_stretch_y)),
        Image.LANCZOS
    )
    R_img = R_img.rotate(Rang, resample=Image.BICUBIC, expand=True)
    pil_alpha_paste_center(base_rgba, R_img, (Rc[0], Rc[1] + eye_dy))

def place_nose_fit_bbox_autocrop(base_rgba: Image.Image, lm, nose_asset_path: Path,
                                 margin_x=NOSE_MARGIN_X, margin_y=NOSE_MARGIN_Y):
    W, H = base_rgba.size
    idxs = _collect_indices_from_connections(FACEMESH_NOSE)
    x0, y0, x1, y1 = _bbox_from_indices(lm, idxs, W, H)
    bw, bh = max(1, x1 - x0), max(1, y1 - y0)

    nose_raw = read_rgba(nose_asset_path)
    L, T, R, B = alpha_bbox(nose_raw) or (0, 0, nose_raw.width, nose_raw.height)
    nose_crop = nose_raw.crop((L, T, R, B))

    target_w = max(1, int(bw * margin_x))
    target_h = max(1, int(bh * margin_y))
    nose_resized = nose_crop.resize((target_w, target_h), Image.LANCZOS)

    cx = (x0 + x1) // 2
    cy = (y0 + y1) // 2
    pil_alpha_paste_center(base_rgba, nose_resized, (cx, cy))

def place_mouth_fit(base_rgba: Image.Image, lm, mouth_asset_path: Path,
                    scale_x=MOUTH_SCALE_X, scale_y=MOUTH_SCALE_Y,
                    dx=MOUTH_DX, dy=MOUTH_DY):
    W, H = base_rgba.size
    right_corner = to_xy(lm, 61, W, H)
    left_corner  = to_xy(lm, 291, W, H)
    cx = (right_corner[0] + left_corner[0]) // 2
    cy = (right_corner[1] + left_corner[1]) // 2
    base_w = math.hypot(left_corner[0]-right_corner[0], left_corner[1]-right_corner[1])
    ang = angle_deg(right_corner, left_corner)

    mouth_raw = read_rgba(mouth_asset_path)
    L, T, R, B = alpha_bbox(mouth_raw) or (0, 0, mouth_raw.width, mouth_raw.height)
    mouth_crop = mouth_raw.crop((L, T, R, B))

    mouth_fit = scale_to_width(mouth_crop, base_w)
    mouth_fit = mouth_fit.resize((int(mouth_fit.width * scale_x),
                                  int(mouth_fit.height * scale_y)), Image.LANCZOS)
    mouth_fit = mouth_fit.rotate(ang, resample=Image.BICUBIC, expand=True)
    pil_alpha_paste_center(base_rgba, mouth_fit, (cx + dx, cy + dy))

def place_ears_from_left_asset(base_rgba: Image.Image, lm, left_ear_png: Path):
    """
    与えるアセットは『被写体の左耳』（画像右側の耳）。
      - 画像左側(=被写体の右耳)にはミラー版
      - 画像右側(=被写体の左耳)にはそのまま
    戻り値: (left_rect, right_rect)
      left_rect  = (x0,y0,x1,y1) 画像左(被写体右耳)
      right_rect = (x0,y0,x1,y1) 画像右(被写体左耳)
    """
    W, H = base_rgba.size
    ear_asset = read_rgba(left_ear_png)

    def _xy(idx):
        p = lm.landmark[idx]
        return (int(p.x * W), int(p.y * H))

    # 顔寸法
    p_left_cheek  = _xy(234)
    p_right_cheek = _xy(454)
    p_forehead    = _xy(10)
    p_chin        = _xy(152)

    face_w = max(1, abs(p_right_cheek[0] - p_left_cheek[0]))
    face_h = max(1, abs(p_chin[1] - p_forehead[1]))

    # 耳の高さ（目と口の中間）
    eye_cy   = (_xy(263)[1] + _xy(33)[1]) // 2
    mouth_cy = (_xy(61)[1] + _xy(291)[1]) // 2
    ear_cy   = (eye_cy + mouth_cy) // 2

    # オフセット
    push_out = int(face_w * EAR_PUSH_OUT_FR)
    lift_up  = int(face_h * EAR_LIFT_UP_FR) + int(EAR_LIFT_UP_PX)

    # 仕上がりサイズ
    tgt_w = int(face_w * EAR_W_RATIO)
    tgt_h = int(face_h * EAR_H_RATIO)
    tgt_w = max(int(face_w*EAR_W_MIN_FR), min(int(face_w*EAR_W_MAX_FR), tgt_w))
    tgt_h = max(int(face_h*EAR_H_MIN_FR), min(int(face_h*EAR_H_MAX_FR), tgt_h))
    tgt_w = max(1, int(tgt_w * EAR_SCALE_X))
    tgt_h = max(1, int(tgt_h * EAR_SCALE_Y))

    # 余白トリム
    ab = alpha_bbox(ear_asset)
    ear_crop = ear_asset.crop(ab) if ab else ear_asset

    # そのままW×Hに歪ませてリサイズ（縦長・細耳）
    right_ear_img = ear_crop.resize((tgt_w, tgt_h), Image.LANCZOS)            # 被写体の左耳（画像右）
    left_ear_img  = ImageOps.mirror(ear_crop).resize((tgt_w, tgt_h), Image.LANCZOS)  # 被写体の右耳（画像左）

    # アンカー（耳前）
    left_anchor_x  = _xy(127)[0]
    right_anchor_x = _xy(356)[0]

    # 左耳（画像左＝被写体右耳）
    left_cx = left_anchor_x - push_out + tgt_w // 4
    left_cy = ear_cy - lift_up
    pil_alpha_paste_center(base_rgba, left_ear_img, (left_cx, left_cy))
    left_rect = (int(left_cx - tgt_w/2), int(left_cy - tgt_h/2),
                 int(left_cx + tgt_w/2), int(left_cy + tgt_h/2))

    # 右耳（画像右＝被写体左耳）
    right_cx = right_anchor_x + push_out - tgt_w // 4
    right_cy = ear_cy - lift_up
    pil_alpha_paste_center(base_rgba, right_ear_img, (right_cx, right_cy))
    right_rect = (int(right_cx - tgt_w/2), int(right_cy - tgt_h/2),
                  int(right_cx + tgt_w/2), int(right_cy + tgt_h/2))

    return left_rect, right_rect

def place_eyebrows_from_left_asset(base_rgba: Image.Image, lm, left_brow_png: Path):
    W, H = base_rgba.size
    brow_asset = read_rgba(left_brow_png)

    left_idx  = _collect_indices_from_connections(FACEMESH_LEFT_EYEBROW)
    bbL = _bbox_from_indices(lm, left_idx, W, H,
                             margin_x=BROW_MARGIN_X, margin_y=BROW_MARGIN_Y)
    L_img = _resize_to_bbox_cropped(brow_asset, bbL,
                                    tighten_x=1.00, tighten_y=BROW_TIGHTEN_Y)
    base_rgba.alpha_composite(L_img, (bbL[0], bbL[1] + BROW_DY))

    right_idx = _collect_indices_from_connections(FACEMESH_RIGHT_EYEBROW)
    bbR = _bbox_from_indices(lm, right_idx, W, H,
                             margin_x=BROW_MARGIN_X, margin_y=BROW_MARGIN_Y)
    R_img = ImageOps.mirror(brow_asset)
    R_img = _resize_to_bbox_cropped(R_img, bbR,
                                    tighten_x=1.00, tighten_y=BROW_TIGHTEN_Y)
    base_rgba.alpha_composite(R_img, (bbR[0], bbR[1] + BROW_DY))

# ==========================
# 髪・輪郭のための関数
# ==========================
def place_hair_by_head_and_ears(base_rgba: Image.Image, lm, hair_png: Path,
                                push_out_fr: float,
                                top_lift_fr: float,
                                align_center_x: int):
    W, H = base_rgba.size

    # 1) まず“枠”を再計算（hair幅を決めるため）
    hair_top_y, inner_left_x, inner_right_x, _ = plan_outline_frame(
        lm, W, H, push_out_fr, top_lift_fr, chin_ext_fr=1.0, inner_pad_fr=0.04
    )
    rect_w = max(1, inner_right_x - inner_left_x)

    # 2) 髪を横幅 rect_w にスケール（アスペクト維持）、上端=hair_top_y、中心x=align_center_x に完全一致
    hair_raw = Image.open(hair_png).convert("RGBA")
    ab = hair_raw.split()[-1].getbbox()
    hair_crop = hair_raw.crop(ab) if ab else hair_raw

    scale = rect_w / max(1, hair_crop.width)
    new_w = max(1, int(hair_crop.width * scale))
    new_h = max(1, int(hair_crop.height * scale))
    hair_scaled = hair_crop.resize((new_w, new_h), Image.LANCZOS)

    paste_x = int(align_center_x - new_w / 2)
    paste_y = hair_top_y

    # クリップ
    paste_x = max(0, min(W - new_w, paste_x))
    paste_y = max(0, min(H - new_h, paste_y))

    base_rgba.alpha_composite(hair_scaled, (paste_x, paste_y))

    # 返却：髪上端と“耳内側枠”（この後の輪郭用）
    # （髪を軸優先で貼ったので、左右枠も軸中心に合わせ直したければここで調整して返すのも可）
    return hair_top_y, inner_left_x, inner_right_x



def place_outline_by_hair_ears(
    base_rgba, lm, outline_png: Path,
    hair_top_y: int,              # 髪を貼ったときの上端y（＝輪郭の上端）
    inner_left_x: int,            # 輪郭の左端（耳の“内側”）
    inner_right_x: int,           # 輪郭の右端（耳の“内側”）
    chin_ext_fr: float = OUTLINE_CHIN_EXT_FR
):
    W, H = base_rgba.size

    def _xy(idx):
        p = lm.landmark[idx]
        return (int(p.x * W), int(p.y * H))

    chin_y = _xy(152)[1]  # 顎先
    rect_w = max(1, inner_right_x - inner_left_x)

    # 髪上端→顎までの距離を基準に、下方向へ拡張
    base_h = max(1, chin_y - hair_top_y)
    rect_h = max(1, int(base_h * chin_ext_fr))

    # 画像外クリップ（はみ出しても良ければ外してOK）
    if hair_top_y + rect_h > H:
        rect_h = H - hair_top_y

    # 輪郭アセットを、rect_w × rect_h に“独立スケール”でフィット
    outline_raw = Image.open(outline_png).convert("RGBA")
    ab = outline_raw.split()[-1].getbbox()
    outline_crop = outline_raw.crop(ab) if ab else outline_raw
    outline_resized = outline_crop.resize((rect_w, rect_h), Image.LANCZOS)

    # 上端＝髪上端、左端＝inner_left_x で貼り付け
    base_rgba.alpha_composite(outline_resized, (inner_left_x, hair_top_y))


# ==========================
# JSON → アセット解決
# ==========================
ASSETS_ROOT = Path("kawakura/assets_png")
FOLDER_MAP = {
    "eye": "eye",
    "eyebrow": "eyebrow",
    "nose": "nose",
    "mouth": "mouse",   # 注意
    "ear": "ear",
    "hair": "hair",
    "outline": "outline",
}

def find_asset(category: str, part_num: int) -> Path | None:
    folder = FOLDER_MAP.get(category, category)
    base = ASSETS_ROOT / folder
    for name in (
        f"{folder}_{part_num:03d}.png",
        f"{folder}_{part_num:02d}.png",
        f"{folder}_{part_num}.png",
    ):
        p = base / name
        if p.exists():
            return p
    return None

def resolve_input_image(run: dict) -> Path:
    ip = Path(run.get("input_image", ""))
    if ip.exists():
        return ip
    cand = Path("uploads") / ip.name
    if cand.exists():
        return cand
    cand2 = Path("made_pictures") / ip.name
    if cand2.exists():
        return cand2
    return ip

def gather_parts_from_json(run: dict) -> dict:
    chosen = {}
    src = run.get("compact", {})
    if not src and "parts" in run:
        src = {k: {"part_num": v.get("selected", {}).get("part_num")}
               for k, v in run["parts"].items()}
    for cat in ["ear", "eyebrow", "eye", "nose", "mouth", "hair", "outline"]:
        info = src.get(cat)
        if not info:
            continue
        pn = info.get("part_num")
        if not pn and "parts" in run:
            sel = run["parts"].get(cat, {}).get("selected", {})
            pn = sel.get("part_num")
        if not pn:
            continue
        ap = find_asset(cat, int(pn))
        if ap:
            chosen[cat] = ap
        else:
            print(f"[WARN] アセットが見つかりません: {cat}_{pn:03d}")
    return chosen

# ==========================
# メイン
# ==========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_json", help="run_*.json のパス")
    ap.add_argument("--out", help="出力PNG（未指定なら自動命名）")
    ap.add_argument("--outline-chin-ext-fr", type=float, default=OUTLINE_CHIN_EXT_FR,
                help="髪上端→顎距離に対する下方向延長倍率（1.0で顎まで）")
    ap.add_argument(
        "--outline-width-scale", type=float, default=OUTLINE_WIDTH_SCALE,
        help="輪郭/髪フレームの横幅スケール。1.0=そのまま, <1で狭く, >1で広く"
    )
    args = ap.parse_args()

    with open(args.run_json, "r", encoding="utf-8") as f:
        run = json.load(f)

    face_path = resolve_input_image(run)
    if not face_path.exists():
        raise FileNotFoundError(f"入力画像が見つかりません: {face_path}")

    parts = gather_parts_from_json(run)
    if not parts:
        print("❌ 使用パーツが見つかりません")
        return

    lm, W, H = detect_landmarks(face_path)
    W, H = Image.open(face_path).size

    if lm is None:
        print("❌ 顔が検出できませんでした")
        return

    # 画像読み込み・ランドマーク検出は既存のまま…

    base = Image.new("RGBA", (W, H), (255, 255, 255, 0))  # 透明背景

    # ここで輪郭中心xを算出（＝顔外周の中心x）
    outline_center_x = estimate_outline_center_x(lm, W, H)

    # --- まず耳（必要なら） ---
    left_ear_rect = right_ear_rect = None
    if "ear" in parts:
        left_ear_rect, right_ear_rect = place_ears_from_left_asset(base, lm, parts["ear"])

    # --- 髪（輪郭中心xに合わせて水平位置を調整） ---
    hair_top_y_plan, inner_l_x_plan, inner_r_x_plan, outline_h_plan = plan_outline_frame(
        lm, W, H,
        push_out_fr=HAIR_PUSH_OUT_FR,
        top_lift_fr=HAIR_TOP_LIFT_FR,
        chin_ext_fr=args.outline_chin_ext_fr,
        inner_pad_fr=0.04
    )

    axis_x = get_nose_center_x(lm, W, H)

    # 追加：幅スケール
    rect_w_plan = inner_r_x_plan - inner_l_x_plan
    rect_w = max(1, int(rect_w_plan * args.outline_width_scale))

    # スケール後の幅で鼻中心に完全一致させる左右枠を再構成
    left_x  = max(0, int(axis_x - rect_w / 2))
    right_x = min(W, left_x + rect_w)
    left_x  = max(0, right_x - rect_w)

    hair_top_y = hair_top_y_plan

    # 髪（必要なら髪もこの枠で描くよう変更可能）
    if "hair" in parts:
        place_hair_by_head_and_ears(
            base, lm, parts["hair"],
            push_out_fr=HAIR_PUSH_OUT_FR,
            top_lift_fr=HAIR_TOP_LIFT_FR,
            align_center_x=axis_x
        )

    # 輪郭（上端・左右をこの枠で）
    if "outline" in parts:
        place_outline_by_hair_ears(
            base, lm, parts["outline"],
            hair_top_y=hair_top_y,
            inner_left_x=left_x,
            inner_right_x=right_x,
            chin_ext_fr=args.outline_chin_ext_fr
        )
    # --- 残りのパーツ ---
    if "eyebrow" in parts:
        place_eyebrows_from_left_asset(base, lm, parts["eyebrow"])

    if "eye" in parts:
        place_two_eyes_from_left_asset(base, lm, parts["eye"],
                                       eye_dy=EYE_DY, eye_stretch_y=EYE_STRETCH_Y)

    if "nose" in parts:
        place_nose_fit_bbox_autocrop(base, lm, parts["nose"],
                                     margin_x=NOSE_MARGIN_X, margin_y=NOSE_MARGIN_Y)

    if "mouth" in parts:
        place_mouth_fit(base, lm, parts["mouth"],
                        scale_x=MOUTH_SCALE_X, scale_y=MOUTH_SCALE_Y,
                        dx=MOUTH_DX, dy=MOUTH_DY)

    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = Path(args.out) if args.out else (
        out_dir / f"face_composed_{time.strftime('%Y%m%d_%H%M%S')}.png"
    )

    # ★ 元画像と横並び比較を作る
    orig_rgba = Image.open(face_path).convert("RGBA")

    # 高さを揃える（もし違う場合はリサイズ）
    if orig_rgba.height != base.height:
        scale = base.height / orig_rgba.height
        new_w = int(orig_rgba.width * scale)
        orig_rgba = orig_rgba.resize((new_w, base.height), Image.LANCZOS)

    # 横並びキャンバスを作る
    comp_w = orig_rgba.width + base.width
    comp_h = max(orig_rgba.height, base.height)
    comp = Image.new("RGBA", (comp_w, comp_h), (255, 255, 255, 0))

    # 左に元画像、右に合成済み画像を貼る
    comp.paste(orig_rgba, (0, 0))
    comp.paste(base, (orig_rgba.width, 0))

    # 保存
    comp.save(out_path)
    print(f"✅ 合成完了: {out_path}")

if __name__ == "__main__":
    main()
