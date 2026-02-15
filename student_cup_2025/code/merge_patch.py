#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
from datetime import datetime
from typing import Dict, Tuple, Optional

import pandas as pd


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def normalize_pair(a: int, b: int) -> Tuple[int, int]:
    return (a, b) if a <= b else (b, a)


def read_pred_csv(path_abs: str) -> Dict[int, Tuple[int, int]]:
    """
    CSV: id,pred_a,pred_b
    - ヘッダなし/あり両対応
    - 変換できない行はスキップ
    - 同じidが複数あれば後勝ち
    """
    df = pd.read_csv(path_abs, header=None, dtype=str)

    # ヘッダっぽいなら読み直す
    if df.shape[1] >= 3:
        first = str(df.iloc[0, 0]).strip().lower()
        if first in ["id", "index"] or not first.isdigit():
            df = pd.read_csv(path_abs, header=0, dtype=str)

    # ヘッダなしで読んだ場合の列名補正
    if set(df.columns) == {0, 1, 2}:
        df.columns = ["id", "pred_a", "pred_b"]

    need = ["id", "pred_a", "pred_b"]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"missing column '{c}' in {path_abs}")

    out: Dict[int, Tuple[int, int]] = {}
    for _, r in df.iterrows():
        try:
            rid = int(str(r["id"]).strip())
            a = int(str(r["pred_a"]).strip())
            b = int(str(r["pred_b"]).strip())
            if not (1 <= a <= 50 and 1 <= b <= 50) or a == b:
                continue
            out[rid] = normalize_pair(a, b)
        except Exception:
            continue
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_submit", type=str, required=True, help="元の提出ファイル（全件）"
    )
    parser.add_argument(
        "--patch_submit",
        type=str,
        required=True,
        help="差分推論ファイル（置換したいidだけ）",
    )
    parser.add_argument("--output_dir", type=str, default="submissions")
    parser.add_argument("--output_name", type=str, default="", help="空なら自動命名")

    # 置換の安全策
    parser.add_argument(
        "--only_if_present",
        action="store_true",
        help="base_submit に存在する id だけ置換する（未知idは無視）",
    )
    parser.add_argument(
        "--skip_if_same",
        action="store_true",
        help="patchがbaseと同じ予測なら置換しない（ログ的に意味がある場合）",
    )

    args = parser.parse_args()

    base_path = (
        os.path.join(PROJECT_ROOT, args.base_submit)
        if not os.path.isabs(args.base_submit)
        else args.base_submit
    )
    patch_path = (
        os.path.join(PROJECT_ROOT, args.patch_submit)
        if not os.path.isabs(args.patch_submit)
        else args.patch_submit
    )

    if not os.path.exists(base_path):
        raise FileNotFoundError(f"base_submit not found: {base_path}")
    if not os.path.exists(patch_path):
        raise FileNotFoundError(f"patch_submit not found: {patch_path}")

    base = read_pred_csv(base_path)
    patch = read_pred_csv(patch_path)

    base_ids = set(base.keys())
    patch_ids = set(patch.keys())

    # 置換対象id
    if args.only_if_present:
        target_ids = sorted(patch_ids & base_ids)
        ignored = sorted(patch_ids - base_ids)
    else:
        target_ids = sorted(patch_ids)
        ignored = []

    replaced = 0
    skipped_same = 0
    for rid in target_ids:
        if args.skip_if_same and (rid in base) and (set(base[rid]) == set(patch[rid])):
            skipped_same += 1
            continue
        base[rid] = patch[rid]
        replaced += 1

    out_dir = os.path.join(PROJECT_ROOT, args.output_dir)
    ensure_dir(out_dir)

    if args.output_name:
        out_path = os.path.join(out_dir, args.output_name)
    else:
        stamp = now_stamp()
        out_path = os.path.join(out_dir, f"submission_merged_{stamp}.csv")

    # id 昇順で保存
    with open(out_path, "w", encoding="utf-8") as f:
        for rid in sorted(base.keys()):
            a, b = base[rid]
            f.write(f"{rid},{a},{b}\n")

    print(f"Saved: {out_path}")
    print(f"base_submit : {base_path} (n={len(base_ids)})")
    print(f"patch_submit: {patch_path} (n={len(patch_ids)})")
    print(f"replaced    : {replaced}")
    if args.skip_if_same:
        print(f"skipped_same: {skipped_same}")
    if ignored:
        print(
            f"ignored_ids (not in base): {ignored[:20]}{'...' if len(ignored) > 20 else ''}"
        )


if __name__ == "__main__":
    main()
