#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import argparse
from datetime import datetime

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


def build_prompt(
    base_items: list[dict],
    practice_items: list[dict],
    test_items_chunk: list[dict],
) -> str:
    # ★このチャンクで実際に予測する件数（最後は端数になりうる）
    n = len(test_items_chunk)

    instruction = f"""「架空作品のあらすじ」から元ネタ作品を推定するタスクを解きます。
各架空作品は「候補作品一覧（50作品）」の中の2作品を混ぜて作られています。

あなたの仕事：
- 「予測対象のあらすじ{n}件」それぞれについて、元ネタになった候補作品IDを2つ推定する

制約：
- 必ず候補ID（1~50）の中から2つ選ぶ
- 2つのIDは異なる
- 出力はCSV形式のみ（ヘッダなし）
- 1行 = 「予測対象id,pred_a,pred_b」
- pred_a,pred_b の順序は任意
- 説明・理由・余計な文章は禁止

参考情報：
- 練習事例（20件）には正解（id_a, id_b）が含まれている。混ぜ方や言い換え方の傾向を学ぶのに使ってよい。

入力データは、下の3つのJSONブロックとして与える：
1) 候補作品一覧（50作品）: 各要素は {{id, category, title, story}}
2) 練習事例（20件）: 各要素は {{id_a, id_b, title_a, title_b, story}}
3) 予測対象（{n}件）: 各要素は {{id, story}}

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
        + json_pretty(test_items_chunk)
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
    parser.add_argument("--output_dir", type=str, default="prompts")

    # 1プロンプトあたりのテスト件数
    parser.add_argument("--chunk_size", type=int, default=10)

    args = parser.parse_args()

    base_path_abs = os.path.join(PROJECT_ROOT, args.base_path)
    practice_path_abs = os.path.join(PROJECT_ROOT, args.practice_path)
    test_path_abs = os.path.join(PROJECT_ROOT, args.test_path)

    if not os.path.exists(base_path_abs):
        raise FileNotFoundError(f"base file not found: {base_path_abs}")
    if not os.path.exists(practice_path_abs):
        raise FileNotFoundError(f"practice file not found: {practice_path_abs}")
    if not os.path.exists(test_path_abs):
        raise FileNotFoundError(f"test file not found: {test_path_abs}")

    df_base = read_table(base_path_abs)
    df_prac = read_table(practice_path_abs)
    df_test = read_table(test_path_abs)

    base_items = build_base_json(df_base)
    practice_items = build_practice_json(df_prac)
    test_items = build_test_json(df_test)

    chunks = chunk_list(test_items, args.chunk_size)

    out_dir_abs = os.path.join(PROJECT_ROOT, args.output_dir)
    ensure_dir(out_dir_abs)

    stamp = now_stamp()

    for ch in chunks:
        if not ch:
            continue
        start_id = ch[0]["id"]
        end_id = ch[-1]["id"]

        prompt = build_prompt(base_items, practice_items, ch)

        out_path_abs = os.path.join(
            out_dir_abs,
            f"webui_prompt_{stamp}_{start_id}-{end_id}.txt",
        )
        with open(out_path_abs, "w", encoding="utf-8") as f:
            f.write(prompt)

        print(f"Saved: {out_path_abs}  (test_ids: {start_id}-{end_id}, n={len(ch)})")

    print(f"Done. chunk_size={args.chunk_size}, files={len(chunks)}")


if __name__ == "__main__":
    main()
