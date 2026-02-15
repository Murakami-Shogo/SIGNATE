#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import argparse
from datetime import datetime
from typing import Dict, Tuple, Optional, List

import pandas as pd


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_table(path_abs: str) -> pd.DataFrame:
    ext = os.path.splitext(path_abs)[1].lower()
    if ext in [".tsv", ".txt"]:
        return pd.read_csv(path_abs, sep="\t", dtype=str)
    return pd.read_csv(path_abs, dtype=str)


def _clean_text(x: str) -> str:
    if x is None:
        return ""
    s = str(x)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def build_base_json(df_base: pd.DataFrame) -> list[dict]:
    out = []
    for _, r in df_base.iterrows():
        out.append(
            {
                "id": int(str(r["id"]).strip()),
                "category": _clean_text(r.get("category", "")),
                "title": _clean_text(r.get("title", "")),
                "story": _clean_text(r.get("story", "")),
            }
        )
    out.sort(key=lambda d: d["id"])
    return out


def build_practice_json(df_prac: pd.DataFrame) -> list[dict]:
    out = []
    for _, r in df_prac.iterrows():
        out.append(
            {
                "id_a": int(str(r["id_a"]).strip()),
                "id_b": int(str(r["id_b"]).strip()),
                "title_a": _clean_text(r.get("title_a", "")),
                "title_b": _clean_text(r.get("title_b", "")),
                "story": _clean_text(r.get("story", "")),
            }
        )
    return out


def build_test_json(df_test: pd.DataFrame) -> list[dict]:
    out = []
    for _, r in df_test.iterrows():
        out.append(
            {
                "id": int(str(r["id"]).strip()),
                "story": _clean_text(r.get("story", "")),
            }
        )
    out.sort(key=lambda d: d["id"])
    return out


def json_pretty(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def normalize_pair(a: int, b: int) -> Tuple[int, int]:
    return (a, b) if a <= b else (b, a)


def read_pred_file(path_abs: str) -> Dict[int, Tuple[int, int]]:
    """
    予測ファイル（CSV想定）
    - ヘッダなし: id,pred_a,pred_b
    - ヘッダあり: id,pred_a,pred_b でも読めるようにする
    """
    # まず header=None で読む
    df = pd.read_csv(path_abs, header=None, dtype=str)

    # ヘッダっぽいなら読み直す
    if df.shape[1] >= 3:
        first = str(df.iloc[0, 0]).strip().lower()
        if first in ["id", "index"] or not first.isdigit():
            df = pd.read_csv(path_abs, header=0, dtype=str)
    else:
        raise ValueError(f"pred file must have 3 columns: {path_abs}")

    # 列名補正（ヘッダなしで読んだ場合）
    if df.shape[1] >= 3 and (set(df.columns) == {0, 1, 2}):
        df.columns = ["id", "pred_a", "pred_b"]

    need = ["id", "pred_a", "pred_b"]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"pred file missing column '{c}': {path_abs}")

    out: Dict[int, Tuple[int, int]] = {}
    for _, r in df.iterrows():
        try:
            rid = int(str(r["id"]).strip())
            a = int(str(r["pred_a"]).strip())
            b = int(str(r["pred_b"]).strip())
            out[rid] = normalize_pair(a, b)  # 重複IDは後勝ち
        except Exception:
            continue
    return out


def build_disagree_items(
    test_items: list[dict],
    pred1: Dict[int, Tuple[int, int]],
    pred2: Dict[int, Tuple[int, int]],
) -> list[dict]:
    """
    pred1/pred2 が一致しない test だけ集める。
    付加情報:
      - pred_1, pred_2
      - overlap_count: 0 or 1
      - overlap_ids: []
      - relation: "no_overlap" / "one_overlap"
      - missing: "pred1_missing" / "pred2_missing" / ""
    """
    out = []
    for t in test_items:
        rid = int(t["id"])
        p1 = pred1.get(rid)
        p2 = pred2.get(rid)

        missing = ""
        if p1 is None and p2 is None:
            missing = "both_missing"
        elif p1 is None:
            missing = "pred1_missing"
        elif p2 is None:
            missing = "pred2_missing"

        # 両方ある場合の一致判定（順序無視）
        if p1 is not None and p2 is not None and set(p1) == set(p2):
            continue  # 一致は除外

        # 付加情報
        overlap_ids: List[int] = []
        overlap_count = 0
        relation = ""
        if p1 is not None and p2 is not None:
            overlap_ids = sorted(list(set(p1) & set(p2)))
            overlap_count = len(overlap_ids)
            relation = "one_overlap" if overlap_count == 1 else "no_overlap"

        item = {
            "id": rid,
            "story": t["story"],
            "pred_1": {"pred_a": p1[0], "pred_b": p1[1]} if p1 is not None else None,
            "pred_2": {"pred_a": p2[0], "pred_b": p2[1]} if p2 is not None else None,
            "overlap_count": overlap_count,
            "overlap_ids": overlap_ids,
            "relation": relation,
            "missing": missing,
        }
        out.append(item)

    out.sort(key=lambda d: d["id"])
    return out


def build_prompt(
    base_items: list[dict],
    practice_items: list[dict],
    disagree_items_chunk: list[dict],
    pred1_name: str,
    pred2_name: str,
) -> str:
    n = len(disagree_items_chunk)

    instruction = f"""「架空作品のあらすじ」から元ネタ作品を推定するタスクを解きます。
各架空作品は「候補作品一覧（50作品）」の中の2作品を混ぜて作られています。

あなたの仕事：
- 「予測対象のあらすじ{n}件」それぞれについて、元ネタになった候補作品IDを2つ推定する

補足：
- 予測対象には、別の2つのモデルの予測結果（{pred1_name}, {pred2_name}）を参考情報として付けてある
- pred_1 と pred_2 が一致しなかったものだけを渡している
- overlap_count=1 のときは、2つの予測が1作品だけ同じIDを含む
- overlap_count=0 のときは、2つの予測が共通IDなし
- ただし、pred_1/pred_2 はどちらも間違っている可能性があるので、候補作品一覧と練習事例とあらすじから最終判断する

制約：
- 必ず候補ID（1~50）の中から2つ選ぶ
- 2つのIDは異なる
- 出力はCSV形式のみ（ヘッダなし）
- 1行 = 「予測対象id,pred_a,pred_b」
- pred_a,pred_b の順序は任意
- 説明・理由・余計な文章は禁止

入力データは、下の3つのJSONブロックとして与える：
1) 候補作品一覧（50作品）: 各要素は {{id, category, title, story}}
2) 練習事例（20件）: 各要素は {{id_a, id_b, title_a, title_b, story}}
3) 予測対象（{n}件）: 各要素は {{
     id, story,
     pred_1, pred_2, overlap_count, overlap_ids, relation, missing
   }}
   pred_1/pred_2 は {{pred_a, pred_b}} または null

出力は「予測対象（{n}件）」の件数と同じ{n}行にすること。
"""

    prompt = (
        instruction
        + "\n"
        + "=== 候補作品一覧（50作品）===\n"
        + json_pretty(base_items)
        + "\n\n"
        + "=== 練習事例（20件）===\n"
        + json_pretty(practice_items)
        + "\n\n"
        + f"=== 予測対象（{n}件）===\n"
        + json_pretty(disagree_items_chunk)
        + "\n\n"
        + f"=== 出力（{n}行）===\n"
    )
    return prompt


def chunk_list(items: list[dict], chunk_size: int) -> list[list[dict]]:
    if chunk_size <= 0:
        raise ValueError("--chunk_size must be >= 1")
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", type=str, default="data/base_stories.tsv")
    parser.add_argument(
        "--practice_path", type=str, default="data/fiction_stories_practice.tsv"
    )
    parser.add_argument(
        "--test_path", type=str, default="data/fiction_stories_test.tsv"
    )

    # 2つの予測ファイル
    parser.add_argument("--pred1_path", type=str, required=True)
    parser.add_argument("--pred2_path", type=str, required=True)
    parser.add_argument("--pred1_name", type=str, default="pred_file_1")
    parser.add_argument("--pred2_name", type=str, default="pred_file_2")

    parser.add_argument("--output_dir", type=str, default="prompts_disagree")

    # 1プロンプトあたりの件数
    parser.add_argument("--chunk_size", type=int, default=10)

    args = parser.parse_args()

    base_path_abs = os.path.join(PROJECT_ROOT, args.base_path)
    practice_path_abs = os.path.join(PROJECT_ROOT, args.practice_path)
    test_path_abs = os.path.join(PROJECT_ROOT, args.test_path)

    pred1_path_abs = (
        os.path.join(PROJECT_ROOT, args.pred1_path)
        if not os.path.isabs(args.pred1_path)
        else args.pred1_path
    )
    pred2_path_abs = (
        os.path.join(PROJECT_ROOT, args.pred2_path)
        if not os.path.isabs(args.pred2_path)
        else args.pred2_path
    )

    for p, name in [
        (base_path_abs, "base"),
        (practice_path_abs, "practice"),
        (test_path_abs, "test"),
        (pred1_path_abs, "pred1"),
        (pred2_path_abs, "pred2"),
    ]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"{name} file not found: {p}")

    df_base = read_table(base_path_abs)
    df_prac = read_table(practice_path_abs)
    df_test = read_table(test_path_abs)

    base_items = build_base_json(df_base)
    practice_items = build_practice_json(df_prac)
    test_items = build_test_json(df_test)

    pred1 = read_pred_file(pred1_path_abs)
    pred2 = read_pred_file(pred2_path_abs)

    disagree_items = build_disagree_items(test_items, pred1, pred2)

    out_dir_abs = os.path.join(PROJECT_ROOT, args.output_dir)
    ensure_dir(out_dir_abs)
    stamp = now_stamp()

    if not disagree_items:
        print("No disagreements found. Nothing to do.")
        return

    chunks = chunk_list(disagree_items, args.chunk_size)

    for ch in chunks:
        start_id = ch[0]["id"]
        end_id = ch[-1]["id"]

        prompt = build_prompt(
            base_items=base_items,
            practice_items=practice_items,
            disagree_items_chunk=ch,
            pred1_name=args.pred1_name,
            pred2_name=args.pred2_name,
        )

        out_path_abs = os.path.join(
            out_dir_abs,
            f"webui_prompt_disagree_{stamp}_{start_id}-{end_id}.txt",
        )
        with open(out_path_abs, "w", encoding="utf-8") as f:
            f.write(prompt)

        print(f"Saved: {out_path_abs}  (ids: {start_id}-{end_id}, n={len(ch)})")

    print(
        f"Done. disagree={len(disagree_items)}, chunk_size={args.chunk_size}, files={len(chunks)}"
    )


if __name__ == "__main__":
    main()
