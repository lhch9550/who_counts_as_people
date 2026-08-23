#!/usr/bin/env python
"""
02_build_dictionary.py
======================
Build the curated probe-word dictionary for the three frames
(Nation / Civic / Global) through a four-step procedure:

  1. Seed selection      : theory-driven seed words per category (defined below).
  2. Neighbor retrieval   : for each category, average the seed-word static
                            embeddings into a centroid and retrieve the nearest
                            vocabulary tokens by cosine similarity.
  3. Filtering            : keep only common nouns (NLTK POS tag NN/NNS) that
                            occur at least `--min-freq` times in the corpus.
  4. Disambiguation       : if a word is a candidate in multiple categories,
                            assign it to the category whose centroid is nearest
                            (highest cosine similarity).

It also computes the within/between-category pairwise mean cosine similarity
matrix used for dictionary validation.

Input
-----
The corpus CSV (same file used in 01), read for noun-frequency counts.

Outputs
-------
    --out-dict   dictionary.json      : {"nation": [...], "civic": [...], "global": [...]}
    --out-nouns  noun_counts.csv      : noun, count   (used later for the random baseline)
    --out-cosine cosine_matrix.csv    : 3x3 within/between cosine similarity table

Example
-------
    python 02_build_dictionary.py \
        --input emnlp_us.csv \
        --out-dict dictionary.json \
        --out-nouns noun_counts.csv \
        --out-cosine cosine_matrix.csv \
        --min-freq 20 --top-k 20
"""

import argparse
import json
import re
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForMaskedLM

import nltk

CATS = ["nation", "civic", "global"]

# ── Step 1: theory-driven seed words ──────────────────────────────────────
SEED_WORDS = {
    "nation": ["nationals", "state", "nation", "country"],
    "civic":  ["residents", "taxpayers", "voters", "citizens", "electors"],
    "global": ["peoples", "mankind", "humanity", "humankind", "inhabitants"],
}


def ensure_nltk():
    for pkg in ["punkt", "punkt_tab",
                "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng"]:
        nltk.download(pkg, quiet=True)


def count_nouns(texts):
    """Count corpus-wide noun frequencies (NN/NNS, alphabetic only)."""
    noun_counter = Counter()
    for i, text in enumerate(texts):
        tokens = nltk.word_tokenize(text)
        tagged = nltk.pos_tag(tokens)
        nouns = [
            w.lower() for w, tag in tagged
            if tag in ("NN", "NNS") and re.fullmatch(r"[a-zA-Z]+", w)
        ]
        noun_counter.update(nouns)
        if i % 200 == 0:
            print(f"  noun counting {i}/{len(texts)}")
    return noun_counter


def get_centroid(words, tokenizer, embedding_matrix):
    vecs = []
    for w in words:
        ids = tokenizer.encode(w, add_special_tokens=False)
        vec = embedding_matrix[ids].mean(dim=0) if len(ids) != 1 else embedding_matrix[ids[0]]
        vecs.append(vec)
    centroid = torch.stack(vecs).mean(dim=0)
    return F.normalize(centroid, p=2, dim=0)


def get_top_similar_words(centroid, normalized_embeddings, tokenizer,
                          top_k=20, exclude_words=None):
    """Return [(token, cosine_sim), ...] of the nearest vocabulary tokens."""
    sims = torch.matmul(normalized_embeddings, centroid)
    sorted_indices = torch.argsort(sims, descending=True)
    results = []
    exclude_words = exclude_words or set()
    for idx in sorted_indices.tolist():
        token = tokenizer.convert_ids_to_tokens([idx])[0]
        if token is None:
            continue
        if token.startswith("##") or token in tokenizer.all_special_tokens:
            continue
        clean = token.replace("Ġ", "").strip()
        if not re.fullmatch(r"[a-zA-Z]+", clean):
            continue
        if clean.lower() in exclude_words:
            continue
        results.append((clean, sims[idx].item()))
        if len(results) >= top_k:
            break
    return results


def get_word_vector(word, tokenizer, embedding_matrix):
    ids = tokenizer.encode(word, add_special_tokens=False)
    vec = embedding_matrix[ids].mean(dim=0) if len(ids) != 1 else embedding_matrix[ids[0]]
    return F.normalize(vec, p=2, dim=0)


def cosine_validation_matrix(dict_words, tokenizer, embedding_matrix):
    """Within/between-category pairwise mean cosine similarity (3x3)."""
    cat_vecs = {}
    for cat in CATS:
        seen, vecs = set(), []
        for w in dict_words[cat]:
            wl = w.lower()
            if wl in seen:
                continue
            seen.add(wl)
            vecs.append(get_word_vector(w, tokenizer, embedding_matrix).cpu().numpy())
        cat_vecs[cat] = np.stack(vecs)

    def mean_intra(V):
        S = V @ V.T
        iu = np.triu_indices(S.shape[0], k=1)
        return S[iu].mean()

    def mean_inter(V1, V2):
        return (V1 @ V2.T).mean()

    M = np.zeros((len(CATS), len(CATS)))
    for i, ci in enumerate(CATS):
        for j, cj in enumerate(CATS):
            if i == j:
                M[i, j] = mean_intra(cat_vecs[ci])
            elif i < j:
                val = mean_inter(cat_vecs[ci], cat_vecs[cj])
                M[i, j] = M[j, i] = val
    return pd.DataFrame(M, index=CATS, columns=CATS)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="Corpus CSV (for noun counts).")
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--model-name", default="ddore14/RooseBERT-scr-cased")
    ap.add_argument("--top-k", type=int, default=20,
                    help="Nearest vocabulary tokens retrieved per category.")
    ap.add_argument("--min-freq", type=int, default=20,
                    help="Minimum corpus noun frequency to keep a candidate.")
    ap.add_argument("--out-dict", default="dictionary.json")
    ap.add_argument("--out-nouns", default="noun_counts.csv")
    ap.add_argument("--out-cosine", default="cosine_matrix.csv")
    args = ap.parse_args()

    ensure_nltk()

    print(f"[load] model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    mlm_model = AutoModelForMaskedLM.from_pretrained(args.model_name)

    embedding_matrix = mlm_model.get_input_embeddings().weight.detach()
    normalized_embeddings = F.normalize(embedding_matrix, p=2, dim=1)

    # ── noun frequency ────────────────────────────────────────────────────
    print(f"[load] corpus: {args.input}")
    texts = pd.read_csv(args.input)[args.text_col].fillna("").tolist()
    print("[step] counting corpus nouns")
    noun_counter = count_nouns(texts)
    print(f"       unique nouns: {len(noun_counter):,}")

    # ── Step 2: centroid + neighbor retrieval ─────────────────────────────
    print("[step] retrieving nearest vocabulary tokens per category")
    centroids = {cat: get_centroid(words, tokenizer, embedding_matrix)
                 for cat, words in SEED_WORDS.items()}
    candidates = {
        cat: get_top_similar_words(centroids[cat], normalized_embeddings,
                                   tokenizer, top_k=args.top_k,
                                   exclude_words={"people"})
        for cat in CATS
    }

    # ── Step 3: frequency filter ──────────────────────────────────────────
    filtered = {cat: [] for cat in CATS}
    for cat in CATS:
        for word, sim in candidates[cat]:
            if noun_counter.get(word.lower(), 0) >= args.min_freq:
                filtered[cat].append((word, sim))
        filtered[cat].sort(key=lambda x: x[1], reverse=True)

    # ── Step 4: cross-category disambiguation (assign to nearest centroid) ─
    word_best_cat = {}   # word_lower -> (best_cat, best_sim)
    for cat in CATS:
        for word, sim in filtered[cat]:
            wl = word.lower()
            if wl not in word_best_cat or sim > word_best_cat[wl][1]:
                word_best_cat[wl] = (cat, sim)

    dict_words = {cat: [] for cat in CATS}
    seen = {cat: set() for cat in CATS}
    for cat in CATS:
        for word, sim in filtered[cat]:
            wl = word.lower()
            if word_best_cat[wl][0] != cat:
                continue          # assigned to a nearer category
            if wl in seen[cat]:
                continue          # in-category case-duplicate
            seen[cat].add(wl)
            dict_words[cat].append(word)

    for cat in CATS:
        print(f"  {cat:8s}: {len(dict_words[cat])} words -> {dict_words[cat]}")

    # ── Dictionary validation: cosine matrix ──────────────────────────────
    print("[step] computing cosine validation matrix")
    cosine_df = cosine_validation_matrix(dict_words, tokenizer, embedding_matrix)
    print(cosine_df.round(3))

    # ── Save ──────────────────────────────────────────────────────────────
    with open(args.out_dict, "w") as f:
        json.dump(dict_words, f, indent=2, ensure_ascii=False)

    (pd.DataFrame(sorted(noun_counter.items(), key=lambda x: -x[1]),
                  columns=["noun", "count"])
     .to_csv(args.out_nouns, index=False))

    cosine_df.to_csv(args.out_cosine)

    print(f"[done] dictionary -> {args.out_dict}")
    print(f"[done] noun counts -> {args.out_nouns}")
    print(f"[done] cosine matrix -> {args.out_cosine}")


if __name__ == "__main__":
    main()
