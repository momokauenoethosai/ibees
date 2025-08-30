#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
入力画像1枚に対して：
- 各部位 extractor で特徴JSON抽出（summary/tags 等）
- summary+tags を検索フレーズにしてベクトル化
- BigQuery でカテゴリ絞りの類似検索を実行
- “無し”と判断した部位はスキップ
- 弱いマッチはカテゴリ別しきい値で除外
- 各部位のトップヒットを1件返す
- UI用に compact（part_num/score のみ）も併せて出力

使い方:
  cd <project-root>/portrait_selection_ai_v1
  python -m kawakura.main.run_all_parts made_pictures/1.png
  # 必要なら一時的に件数を変える（既定は 1）:
  python -m kawakura.main.run_all_parts made_pictures/1.png 1
"""
from __future__ import annotations
import sys
import json
import re
from pathlib import Path
from typing import Dict, Any, List

# 同ディレクトリ（main/）内
from .common_config import MIN_SCORE
from .utils_embed_bq import search_by_text_in_category
from .part_extractors import hair, eye, eyebrow, nose, mouth, ear, outline, acc, beard

# トップヒットは1件固定（引数で上書き可だが既定は1）
DEFAULT_TOP_K = 1

EXTRACTORS = {
    "hair": hair,
    "eye": eye,
    "eyebrow": eyebrow,
    "nose": nose,
    "mouth": mouth,
    "ear": ear,
    "outline": outline,
    "acc": acc,
    "beard": beard,
}

# “無し”判定のキーワード（英語中心。必要なら日本語も追加可）
NEG_PATTERNS = {
    "beard": [
        r"\bno facial hair\b", r"\bno beard\b", r"\bclean[- ]?shaven\b", r"\bnone\b",
        r"ひげなし", r"髭なし", r"なし",
    ],
    "acc": [
        r"\bno accessories\b", r"\bno accessory\b", r"\bnone\b",
        r"アクセ(サリー)?なし", r"なし",
    ],
    # ★ 追加分 ↓↓↓
    "glasses": [
        r"\bno glasses\b", r"\bno sunglasses\b", r"\bnone\b",
        r"めがねなし", r"メガネなし", r"サングラスなし", r"なし",
    ],
    "extras": [
        r"\bnone\b", r"なし", r"特になし",
        r"\bno freckles\b", r"\bno moles\b", r"\bno scars\b",
    ],
    "wrinkles": [
        r"\bnone\b", r"なし", r"しわなし", r"シワなし", r"しわがない", r"シワがない",
        r"\bno wrinkles\b", r"\bno fine lines\b",
    ],
}

# カテゴリ別 しきい値（弱マッチ除去用）
CATEGORY_MIN_SCORE = {
    "hair": 0.0,
    "eye": 0.0,
    "eyebrow": 0.0,
    "nose": 0.0,
    "mouth": 0.0,
    "ear": 0.0,
    "outline": 0.0,
    "acc": 0.06,
    "beard": 0.08,
    "glasses": 0.00,
    "extras": 0.00,
    "wrinkles": 0.00,
}


def looks_negative(category: str, summary: str, tags: List[str]) -> bool:
    """“無い”系の表現なら True を返す（その部位をスキップ）"""
    text = " ".join([summary or ""] + (tags or [])).lower()
    for pat in NEG_PATTERNS.get(category, []):
        if re.search(pat, text):
            return True
    return False


def _safe_extract(mod, img_path: Path) -> Dict[str, Any]:
    """各 extractor の extract() を安全に呼び、失敗しても壊れない形で返す。"""
    try:
        tags, summary, raw, phrase = mod.extract(img_path)
        if not isinstance(tags, list):
            tags = []
        summary = (summary or "").strip()
        phrase = (phrase or summary or "").strip()
        return {
            "ok": True,
            "tags": tags,
            "summary": summary,
            "raw": raw,
            "phrase": phrase if phrase else summary,
        }
    except Exception as e:
        return {
            "ok": False,
            "err": f"{type(e).__name__}: {e}",
            "tags": [],
            "summary": "",
            "raw": {"error": str(e)},
            "phrase": "",
        }


def _safe_search(category: str, phrase: str, top_k: int, min_score: float) -> Dict[str, Any]:
    """検索呼び出し（例外に強い）"""
    if not phrase:
        return {"ok": True, "hits": []}
    try:
        hits = search_by_text_in_category(phrase, category=category, top_k=top_k, min_score=min_score)
        return {"ok": True, "hits": hits[:top_k]}
    except Exception as e:
        return {"ok": False, "err": f"{type(e).__name__}: {e}", "hits": []}


def run_once(img_path: Path, top_k: int = DEFAULT_TOP_K, min_score: float = MIN_SCORE, progress_callback=None) -> Dict[str, Any]:
    """画像1枚に対する全カテゴリ処理のメイン本体。統合JSONを返す。"""
    result: Dict[str, Any] = {
        "input_image": str(img_path),
        "meta": {"top_k": top_k, "min_score": min_score},
        "parts": {},
    }
    
    total_categories = len(EXTRACTORS)
    current_index = 0

    for category, mod in EXTRACTORS.items():
        current_index += 1
        
        # 進捗コールバック送信
        if progress_callback:
            progress_callback({
                "status": "processing",
                "current_part": category,
                "progress": current_index,
                "total": total_categories,
                "percentage": int((current_index / total_categories) * 100)
            })
        
        # 1) 特徴抽出
        ext = _safe_extract(mod, img_path)
        summary = ext.get("summary", "")
        tags = ext.get("tags", [])
        phrase = ext.get("phrase", "")

        # (A) 特徴が全く無い → スキップ
        if not phrase and not tags and not summary:
            print(f"[{category}] skipped (no features)")
            if progress_callback:
                progress_callback({
                    "status": "skipped",
                    "current_part": category,
                    "reason": "no features",
                    "progress": current_index,
                    "total": total_categories
                })
            continue

        # (B) "無し"と判定（例：no facial hair / no accessories） → スキップ
        if looks_negative(category, summary, tags):
            print(f"[{category}] skipped (negative: none)")
            if progress_callback:
                progress_callback({
                    "status": "skipped",
                    "current_part": category,
                    "reason": "negative detection",
                    "progress": current_index,
                    "total": total_categories
                })
            continue

        # 2) 類似検索（カテゴリごとの最小スコアで弱マッチ除去）
        cat_min = CATEGORY_MIN_SCORE.get(category, min_score)
        srch = _safe_search(category, phrase, top_k=top_k, min_score=cat_min)
        hits = [
            h for h in srch.get("hits", [])
            if h.get("score") is not None and h["score"] >= cat_min
        ][:1]
        best = hits[0] if hits else None

        # (C) 弱マッチのみ → スキップ
        if not best:
            print(f"[{category}] skipped (weak match)")
            if progress_callback:
                progress_callback({
                    "status": "skipped",
                    "current_part": category,
                    "reason": "weak match",
                    "progress": current_index,
                    "total": total_categories
                })
            continue

        # 3) まとめ（selected は単一オブジェクト）
        part_out: Dict[str, Any] = {
            "extracted": {
                "summary": summary,
                "tags": tags,
                "query_phrase": phrase,
                "raw": ext.get("raw", {}),
            },
            "search": {"top_hits": hits},
            "selected": {
                "part_id_full": best["part_id"],
                "part_num": best["part_num"],
                "score": best["score"],
            },
        }

        result["parts"][category] = part_out
        print(f"[{category}] phrase='{phrase}' -> best={best['part_id']} ({best['score']:.4f})")
        
        # 成功時の進捗コールバック
        if progress_callback:
            progress_callback({
                "status": "completed",
                "current_part": category,
                "part_id": best["part_id"],
                "score": best["score"],
                "progress": current_index,
                "total": total_categories
            })

    # 4) UI向けの軽量 JSON（part_num/score のみ）
    compact = {}
    for cat, data in result["parts"].items():
        sel = data.get("selected")
        if sel:
            compact[cat] = {"part_num": sel["part_num"], "score": sel["score"]}
    result["compact"] = compact

    return result


def main():
    if len(sys.argv) < 2:
        print("使い方: python -m kawakura.main.run_all_parts <face_image_path> [top_k]\n"
              "※ 既定のトップヒット件数は 1 です。")
        sys.exit(1)

    img_path = Path(sys.argv[1])
    if not img_path.exists():
        print(f"画像が見つかりません: {img_path}")
        sys.exit(2)

    try:
        top_k = int(sys.argv[2]) if len(sys.argv) >= 3 else DEFAULT_TOP_K
        if top_k <= 0:
            top_k = DEFAULT_TOP_K
    except Exception:
        top_k = DEFAULT_TOP_K

    output = run_once(img_path, top_k=top_k, min_score=MIN_SCORE)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
