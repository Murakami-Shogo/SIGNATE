#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import argparse
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


# code/embedding_*.py の場所
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# code/ の1つ上（プロジェクト直下）
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def safe_slug(s: str) -> str:
    """モデル名などをファイル名に入れても安全な形にする"""
    s = s.strip().replace("/", "__")
    s = re.sub(r"[^0-9A-Za-z_.\-]+", "_", s)
    return s[:120]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def setup_logging(log_file_abs: str) -> None:
    ensure_dir(os.path.dirname(log_file_abs))

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if logger.handlers:
        logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(log_file_abs, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)


def read_table(path_abs: str) -> pd.DataFrame:
    ext = os.path.splitext(path_abs)[1].lower()
    if ext in [".tsv", ".txt"]:
        return pd.read_csv(path_abs, sep="\t", dtype=str)
    return pd.read_csv(path_abs, dtype=str)


def get_model_max_len(tokenizer, model) -> int:
    """
    安全な最大長を返す。
    - tokenizer.model_max_length を優先（多くの場合 512 などが入っている）
    - RoBERTa/XLM-R 系は config.max_position_embeddings が 514 でも実用上は 512 が安全
    """
    tok_cap = getattr(tokenizer, "model_max_length", None)
    tok_ok = isinstance(tok_cap, int) and 0 < tok_cap < 100000

    cfg_cap = getattr(model.config, "max_position_embeddings", None)
    cfg_ok = isinstance(cfg_cap, int) and 0 < cfg_cap < 100000

    caps = []
    if tok_ok:
        caps.append(tok_cap)

    if cfg_ok:
        model_type = getattr(model.config, "model_type", "")
        if model_type in {"roberta", "xlm-roberta"}:
            caps.append(cfg_cap - 2)  # 514 -> 512 が安全
        else:
            caps.append(cfg_cap)

    if not caps:
        return 512

    return max(16, min(caps))


def estimate_token_lengths(
    texts: list[str],
    tokenizer,
    prefix: str = "",
    batch_size: int = 64,
) -> np.ndarray:
    """
    truncation=False でトークン数を測る。
    add_special_tokens=True なので [CLS]/[SEP] 相当も含めた長さ。
    """
    lengths: list[int] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        if prefix:
            batch = [prefix + t for t in batch]

        enc = tokenizer(
            batch,
            add_special_tokens=True,
            truncation=False,
            padding=False,
        )
        lengths.extend([len(ids) for ids in enc["input_ids"]])

    return np.asarray(lengths, dtype=np.int32)


def choose_max_length_max_plus_alpha(
    base_texts: list[str],
    test_texts: list[str],
    tokenizer,
    model,
    doc_prefix: str,
    query_prefix: str,
    buffer_tokens: int,
) -> int:
    """
    max_length = max_token_len_in_data + buffer_tokens
    ただしモデル上限(cap)を超えたらcapで丸める。
    """
    cap = get_model_max_len(tokenizer, model)

    lens_doc = estimate_token_lengths(base_texts, tokenizer, prefix=doc_prefix)
    lens_qry = estimate_token_lengths(test_texts, tokenizer, prefix=query_prefix)

    max_doc = int(lens_doc.max()) if len(lens_doc) else 0
    max_qry = int(lens_qry.max()) if len(lens_qry) else 0
    max_all = max(max_doc, max_qry)

    chosen_before_cap = max_all + int(buffer_tokens)
    chosen = chosen_before_cap

    if chosen > cap:
        chosen = cap
        logging.info(
            f"[auto max_length] WARNING: data_max+buffer={chosen_before_cap} > model_cap={cap}. "
            f"Use cap={cap} (some truncation may occur)."
        )

    chosen = max(chosen, 16)

    def stats(arr: np.ndarray) -> str:
        return (
            f"min={int(arr.min())}, med={int(np.median(arr))}, "
            f"p95={int(np.percentile(arr, 95))}, p99={int(np.percentile(arr, 99))}, "
            f"max={int(arr.max())}"
        )

    trunc_doc = float(np.mean(lens_doc > chosen)) if len(lens_doc) else 0.0
    trunc_qry = float(np.mean(lens_qry > chosen)) if len(lens_qry) else 0.0

    logging.info(f"[auto max_length] model_cap={cap}")
    logging.info(f"[auto max_length] doc token stats: {stats(lens_doc)}")
    logging.info(f"[auto max_length] qry token stats: {stats(lens_qry)}")
    logging.info(
        f"[auto max_length] max_doc={max_doc}, max_qry={max_qry}, max_all={max_all}"
    )
    logging.info(f"[auto max_length] buffer_tokens={buffer_tokens} -> chosen={chosen}")
    logging.info(
        f"[auto max_length] trunc_rate doc={trunc_doc:.3f}, qry={trunc_qry:.3f}"
    )

    return chosen


@torch.no_grad()
def encode_texts_mean_pooling(
    texts: list[str],
    tokenizer,
    model,
    device: torch.device,
    batch_size: int,
    max_length: int,
    prefix: str = "",
    fp16: bool = False,
) -> torch.Tensor:
    """
    AutoModel の last_hidden_state を attention mask で mean pooling → L2 normalize
    戻り値: torch.Tensor (N, D) on CPU
    """
    model.eval()
    out_embs = []

    use_amp = fp16 and (device.type == "cuda")
    autocast_ctx = torch.cuda.amp.autocast(enabled=use_amp)

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        if prefix:
            batch = [prefix + t for t in batch]

        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with autocast_ctx:
            outputs = model(**enc)
            last_hidden = outputs.last_hidden_state  # (B, L, H)
            attention_mask = enc["attention_mask"]  # (B, L)

            mask = attention_mask.unsqueeze(-1).type_as(last_hidden)  # (B, L, 1)
            summed = torch.sum(last_hidden * mask, dim=1)  # (B, H)
            counts = torch.clamp(mask.sum(dim=1), min=1e-9)  # (B, 1)
            emb = summed / counts  # (B, H)

        emb = F.normalize(emb, p=2, dim=1)
        out_embs.append(emb.detach().cpu())

    return torch.cat(out_embs, dim=0)


def make_pair_embeddings(base_emb: torch.Tensor, base_ids: np.ndarray):
    """
    base_emb: (50, D) L2-normalized (CPU)
    base_ids: (50,) int
    return:
      pair_emb: (1225, D) L2-normalized (CPU)
      pair_a_ids: (1225,) int
      pair_b_ids: (1225,) int
    """
    n = base_emb.shape[0]
    # 上三角（i<j）の全ペア
    i_idx, j_idx = torch.triu_indices(n, n, offset=1)  # (P,), (P,)
    # 合体（平均）→ 正規化
    pair_emb = (base_emb[i_idx] + base_emb[j_idx]) * 0.5
    pair_emb = F.normalize(pair_emb, p=2, dim=1)

    pair_a_ids = base_ids[i_idx.numpy()]
    pair_b_ids = base_ids[j_idx.numpy()]

    return pair_emb, pair_a_ids, pair_b_ids


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--base_path", type=str, default="data/base_stories.tsv")
    parser.add_argument(
        "--test_path", type=str, default="data/fiction_stories_test.tsv"
    )

    parser.add_argument("--output_dir", type=str, default="submissions")
    parser.add_argument("--logs_dir", type=str, default="logs")

    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)

    # - 普段は --max_length を固定値で指定（デフォルト256）
    # - 自動にしたい時だけ --auto_max_length を付ける
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--auto_max_length", action="store_true")
    parser.add_argument("--max_length_buffer", type=int, default=16)

    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--fp16", action="store_true")

    parser.add_argument("--query_prefix", type=str, default="")
    parser.add_argument("--doc_prefix", type=str, default="")

    args = parser.parse_args()

    stamp = now_stamp()
    model_slug = safe_slug(args.model_name)

    logs_dir_abs = os.path.join(PROJECT_ROOT, args.logs_dir)
    output_dir_abs = os.path.join(PROJECT_ROOT, args.output_dir)
    ensure_dir(logs_dir_abs)
    ensure_dir(output_dir_abs)

    log_file_abs = os.path.join(
        logs_dir_abs, f"embedding_pair_{stamp}_{model_slug}.log"
    )
    setup_logging(log_file_abs)

    logging.info("=== embedding_pair start ===")
    logging.info(f"PROJECT_ROOT = {PROJECT_ROOT}")
    logging.info(f"model_name   = {args.model_name}")
    logging.info(f"batch_size   = {args.batch_size}")
    logging.info(f"max_length   = {args.max_length}")
    logging.info(f"auto_max_length = {args.auto_max_length}")
    logging.info(f"max_len_buf  = {args.max_length_buffer}")
    logging.info(f"fp16         = {args.fp16}")
    logging.info(f"trust_remote_code = {args.trust_remote_code}")
    logging.info(f"query_prefix = {repr(args.query_prefix)}")
    logging.info(f"doc_prefix   = {repr(args.doc_prefix)}")
    logging.info(f"log_file     = {log_file_abs}")

    base_path_abs = os.path.join(PROJECT_ROOT, args.base_path)
    test_path_abs = os.path.join(PROJECT_ROOT, args.test_path)

    logging.info(f"base_path(abs) = {base_path_abs}")
    logging.info(f"test_path(abs) = {test_path_abs}")

    if not os.path.exists(base_path_abs):
        raise FileNotFoundError(f"base file not found: {base_path_abs}")
    if not os.path.exists(test_path_abs):
        raise FileNotFoundError(f"test file not found: {test_path_abs}")

    df_base = read_table(base_path_abs)
    df_test = read_table(test_path_abs)

    for col in ["id", "story"]:
        if col not in df_base.columns:
            raise ValueError(
                f"base file must have column '{col}' (found: {list(df_base.columns)})"
            )
    for col in ["id", "story"]:
        if col not in df_test.columns:
            raise ValueError(
                f"test file must have column '{col}' (found: {list(df_test.columns)})"
            )

    df_base["story"] = df_base["story"].fillna("").astype(str).str.strip()
    df_test["story"] = df_test["story"].fillna("").astype(str).str.strip()

    base_ids = df_base["id"].astype(int).to_numpy()
    base_texts = df_base["story"].tolist()

    test_ids = df_test["id"].astype(int).to_numpy()
    test_texts = df_test["story"].tolist()

    logging.info(f"base shape = {df_base.shape}, test shape = {df_test.shape}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"device = {device}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=args.trust_remote_code
    )
    model = AutoModel.from_pretrained(
        args.model_name, trust_remote_code=args.trust_remote_code
    )
    model.to(device)

    if args.fp16 and device.type == "cuda":
        model.half()
        logging.info("model cast to fp16 (half).")

    # max_length 自動決定（最大+α）
    if args.auto_max_length:
        args.max_length = choose_max_length_max_plus_alpha(
            base_texts=base_texts,
            test_texts=test_texts,
            tokenizer=tokenizer,
            model=model,
            doc_prefix=args.doc_prefix,
            query_prefix=args.query_prefix,
            buffer_tokens=args.max_length_buffer,
        )
        logging.info(f"auto-decided max_length = {args.max_length}")

    logging.info(f"final max_length = {args.max_length}")

    # base/test の埋め込み
    logging.info("encoding base stories ...")
    base_emb = encode_texts_mean_pooling(
        base_texts,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        prefix=args.doc_prefix,
        fp16=args.fp16,
    )  # (50, D) CPU, normalized

    logging.info("encoding test stories ...")
    test_emb = encode_texts_mean_pooling(
        test_texts,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        prefix=args.query_prefix,
        fp16=args.fp16,
    )  # (N, D) CPU, normalized

    logging.info(
        f"base_emb shape = {tuple(base_emb.shape)}, test_emb shape = {tuple(test_emb.shape)}"
    )

    # 50C2 のペア埋め込みを作る
    pair_emb, pair_a_ids, pair_b_ids = make_pair_embeddings(base_emb, base_ids)
    logging.info(f"pair_emb shape = {tuple(pair_emb.shape)} (should be 1225 x D)")

    # 類似度計算（cosine: 正規化済みなので内積）
    sim = test_emb @ pair_emb.T  # (N, 1225)
    best_pair_idx = torch.argmax(sim, dim=1).numpy()  # (N,)

    pred_a = pair_a_ids[best_pair_idx]
    pred_b = pair_b_ids[best_pair_idx]

    # 念のため a < b に正規化（評価は順不同だけど、提出が安定する）
    pred_min = np.minimum(pred_a, pred_b)
    pred_max = np.maximum(pred_a, pred_b)

    # 提出ファイル作成（ヘッダーなし: id,pred_a,pred_b）
    sub_path_abs = os.path.join(output_dir_abs, f"submission_{stamp}_{model_slug}.csv")
    sub_df = pd.DataFrame({"id": test_ids, "pred_a": pred_min, "pred_b": pred_max})
    sub_df.to_csv(sub_path_abs, index=False, header=False)

    logging.info(f"saved submission = {sub_path_abs}")
    logging.info("=== done ===")


if __name__ == "__main__":
    main()
