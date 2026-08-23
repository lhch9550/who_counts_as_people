#!/usr/bin/env python
"""
04_compute_frame_scores.py
==========================
Compute baseline-corrected frame scores for every contextual window using the
era-specific fine-tuned models (script 03).

For each window, the target token is replaced with [MASK] and passed through
its era's fine-tuned model. For each category we take the mean pseudo-log-
likelihood (PLL) of its top-N highest-scoring dictionary words, then subtract
the mean PLL of the top-N random-noun baseline words:

    Score(c) = mean_topN( log P(w | masked) for w in dict[c] )
             - mean_topN( log P(w | masked) for w in random_baseline )

The random baseline is sampled once (fixed seed) from the most frequent corpus
nouns that are not already in any category dictionary.

Inputs
------
    --windows     windows.csv        (from 01; must contain era-assignable year)
    --dict        dictionary.json    (from 02)
    --nouns       noun_counts.csv    (from 02; for the random baseline)
    --era-models  ./era_models       (from 03)

Output
------
    --output scores.csv : instance-level scores. Columns include, for the
        chosen TOP_N (default 3), per category c in {nation, civic, global}:
            top{N}_{c}_pll        absolute category PLL
            top{N}_{c}_rel_pll    baseline-corrected frame score (Score(c))
            top{N}_{c}_ratio      softmax over absolute PLLs
            top{N}_{c}_ratio_corr softmax over baseline-corrected PLLs
        plus: top{N}_random_pll, top{N}_pll_dominant, top{N}_pll_confidence,
              year, speaker, speech_idx, era.

Example
-------
    python 04_compute_frame_scores.py \
        --windows windows.csv \
        --dict dictionary.json \
        --nouns noun_counts.csv \
        --era-models ./era_models \
        --output scores.csv \
        --top-n 3
"""

import argparse
import ast
import json
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM

CATS = ["nation", "civic", "global"]
TARGET_WORD = "people"

# Must match the periodization used for fine-tuning in script 03.
ERA_BOUNDARIES = {
    "founding_era":        (1789, 1828),
    "jacksonian_civilwar": (1829, 1877),
    "gilded_progressive":  (1878, 1932),
    "newdeal_postwar":     (1933, 1968),
    "cold_war_late":       (1969, 1991),
    "post_cold_war":       (1992, 2026),
}


def assign_era(year):
    for era, (start, end) in ERA_BOUNDARIES.items():
        if start <= year <= end:
            return era
    return None


def to_single_token_ids(words, tokenizer):
    """Keep only words that map to exactly one token; return their ids."""
    ids = []
    for w in words:
        if w == TARGET_WORD:
            continue
        t = tokenizer.encode(w, add_special_tokens=False)
        if len(t) == 1:
            ids.append(t[0])
    return ids


def build_random_baseline(noun_counts_df, dict_words, k, seed, tokenizer):
    """Sample k baseline nouns from top corpus nouns not in any dictionary."""
    random.seed(seed)
    all_cat_words = {w.lower() for words in dict_words.values() for w in words}
    pool = [
        n for n in noun_counts_df.sort_values("count", ascending=False)["noun"]
              .head(500).tolist()
        if str(n).lower() not in all_cat_words and n != TARGET_WORD
    ]
    sampled = random.sample(pool, k=min(k, len(pool)))
    print(f"[baseline] random words: {sampled}")
    return sampled


def pll_probe_batch(model, tokenizer, device, batch_ids, batch_pos,
                    valid_cat_words, valid_random_words, top_n):
    """Return a list of per-row score dicts for one batch."""
    pad_id = tokenizer.pad_token_id
    mask_id = tokenizer.mask_token_id

    max_len = max(ids.size(0) for ids in batch_ids)
    padded = torch.full((len(batch_ids), max_len), pad_id, dtype=torch.long)
    attn = torch.zeros_like(padded)

    for i, ids in enumerate(batch_ids):
        cur = ids.size(0)
        masked = ids.clone()
        masked[min(batch_pos[i], cur - 1)] = mask_id
        padded[i, :cur] = masked
        attn[i, :cur] = 1

    with torch.no_grad():
        logits = model(input_ids=padded.to(device),
                       attention_mask=attn.to(device)).logits

    out_rows = []
    for i, pos in enumerate(batch_pos):
        log_probs = F.log_softmax(logits[i, min(pos, padded.size(1) - 1)], dim=-1)

        cat_sorted = {}
        for cat, tids in valid_cat_words.items():
            cat_sorted[cat] = sorted(
                [(log_probs[t].item(), t) for t in tids], reverse=True) if tids else []
        rand_sorted = sorted(
            [(log_probs[t].item(), t) for t in valid_random_words], reverse=True) \
            if valid_random_words else []

        cat_pll, cat_top_words = {}, {}
        for cat in CATS:
            scored = cat_sorted[cat]
            if not scored:
                cat_pll[cat], cat_top_words[cat] = -20.0, []
                continue
            top_k = scored[:top_n]
            cat_pll[cat] = sum(s for s, _ in top_k) / len(top_k)
            cat_top_words[cat] = [tokenizer.decode([t]) for _, t in top_k]

        random_pll = (sum(s for s, _ in rand_sorted[:top_n]) / len(rand_sorted[:top_n])
                      if rand_sorted else -20.0)

        vals = np.array([cat_pll[c] for c in CATS])
        exp = np.exp(vals - vals.max())
        ratio = {c: exp[j] / exp.sum() for j, c in enumerate(CATS)}

        rel = {c: cat_pll[c] - random_pll for c in CATS}
        rel_vals = np.array([rel[c] for c in CATS])
        exp_rel = np.exp(rel_vals - rel_vals.max())
        ratio_corr = {c: exp_rel[j] / exp_rel.sum() for j, c in enumerate(CATS)}

        dominant = max(cat_pll, key=cat_pll.get)
        s = sorted(cat_pll.values(), reverse=True)
        confidence = s[0] - s[1]

        p = f"top{top_n}_"
        row = {}
        row.update({p + c + "_pll": cat_pll[c] for c in CATS})
        row.update({p + c + "_ratio": ratio[c] for c in CATS})
        row[p + "random_pll"] = random_pll
        row.update({p + c + "_rel_pll": rel[c] for c in CATS})
        row.update({p + c + "_ratio_corr": ratio_corr[c] for c in CATS})
        row[p + "pll_dominant"] = dominant
        row[p + "pll_confidence"] = confidence
        row.update({p + c + "_top_words": cat_top_words[c] for c in CATS})
        out_rows.append(row)

    return out_rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--windows", required=True)
    ap.add_argument("--dict", required=True, help="dictionary.json from step 02.")
    ap.add_argument("--nouns", required=True, help="noun_counts.csv from step 02.")
    ap.add_argument("--era-models", default="./era_models")
    ap.add_argument("--base-model", default="ddore14/RooseBERT-scr-cased",
                    help="Fallback tokenizer / special-token source.")
    ap.add_argument("--output", default="scores.csv")
    ap.add_argument("--top-n", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--random-k", type=int, default=20,
                    help="Number of random baseline words to sample.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    cls_id, sep_id = tokenizer.cls_token_id, tokenizer.sep_token_id

    dict_words = json.load(open(args.dict))
    noun_counts_df = pd.read_csv(args.nouns)

    valid_cat_words = {cat: to_single_token_ids(dict_words[cat], tokenizer)
                       for cat in CATS}
    for cat in CATS:
        print(f"  {cat:8s}: {len(valid_cat_words[cat])} usable single-token words")

    random_words = build_random_baseline(
        noun_counts_df, dict_words, k=args.random_k, seed=args.seed, tokenizer=tokenizer)
    valid_random_words = to_single_token_ids(random_words, tokenizer)

    # ── load windows, assign era ──────────────────────────────────────────
    print(f"[load] windows: {args.windows}")
    df = pd.read_csv(args.windows)
    df["window_ids"] = df["window_ids"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    df["era"] = df["year"].apply(assign_era)
    df = df.dropna(subset=["era"]).reset_index(drop=True)
    print(df["era"].value_counts())

    # ── score each era with its own fine-tuned model ──────────────────────
    final_rows = []
    for era_name in ERA_BOUNDARIES:
        era_df = df[df["era"] == era_name]
        if len(era_df) == 0:
            continue
        model_path = os.path.join(args.era_models, era_name)
        if not os.path.exists(model_path):
            print(f"[skip] {era_name}: no fine-tuned model at {model_path}")
            continue

        print(f"=== Scoring {era_name} (n={len(era_df)}) with {model_path} ===")
        model = AutoModelForMaskedLM.from_pretrained(model_path).to(device).eval()

        data = era_df.to_dict("records")
        for i in tqdm(range(0, len(data), args.batch_size),
                      desc=f"PLL [{era_name}]", leave=False):
            chunk = data[i:i + args.batch_size]
            batch_ids, batch_pos = [], []
            for row in chunk:
                l_pos = int(row["local_pos"])
                ids = torch.tensor([cls_id] + list(row["window_ids"]) + [sep_id])
                batch_ids.append(ids)
                batch_pos.append(l_pos + 1)

            scores = pll_probe_batch(
                model, tokenizer, device, batch_ids, batch_pos,
                valid_cat_words, valid_random_words, args.top_n)

            for row, sc in zip(chunk, scores):
                res = {**sc, "era": era_name}
                for col in ["speaker", "year", "speech_idx",
                            "n_people_in_speech", "window_text"]:
                    if col in row:
                        res[col] = row[col]
                final_rows.append(res)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    df_scores = pd.DataFrame(final_rows)
    df_scores.to_csv(args.output, index=False)
    print(f"[done] {len(df_scores):,} instance scores -> {args.output}")


if __name__ == "__main__":
    main()
