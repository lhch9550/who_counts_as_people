#!/usr/bin/env python
"""
01_extract_windows.py
=====================
Extract fixed-size contextual windows around each occurrence of the target
token ("people") in a corpus of presidential speeches.

For every occurrence of the target token in a speech, a window of
`window_size` tokens on each side is collected (truncated at the sequence
boundary). Each occurrence becomes one row in the output.

Input
-----
A CSV with (at least) the columns:
    text       : full speech text
    year       : year of the speech
    president  : speaker name

Output
------
A CSV of contextual windows with columns:
    year, speaker, window_ids, local_pos, window_text,
    n_people_in_speech, speech_idx

Note: `window_ids` is stored as a Python-list literal string; downstream
scripts read it back with ast.literal_eval.

Example
-------
    python 01_extract_windows.py \
        --input speech_us.csv \
        --output windows.csv \
        --window-size 50
"""

import argparse

import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer


def extract_people_windows(df, tokenizer, target_word="people",
                           text_col="text", year_col="year",
                           speaker_col="president", window_size=50):
    """Return a DataFrame with one row per occurrence of `target_word`."""
    target_ids = list(tokenizer.encode(target_word, add_special_tokens=False))
    tL = len(target_ids)
    rows = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting windows"):
        text = str(row[text_col]) if pd.notna(row[text_col]) else ""
        enc = list(tokenizer.encode(text.lower(), add_special_tokens=False))

        positions = [
            i for i in range(len(enc) - tL + 1)
            if enc[i:i + tL] == target_ids
        ]
        if not positions:
            continue

        for pos in positions:
            start = max(0, pos - window_size)
            end = min(len(enc), pos + tL + window_size)
            chunk_ids = enc[start:end]

            rows.append({
                "year": row[year_col],
                "speaker": row[speaker_col],
                "window_ids": chunk_ids,
                "local_pos": pos - start,
                "window_text": tokenizer.decode(chunk_ids),
                "n_people_in_speech": len(positions),
                "speech_idx": idx,
            })

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="Path to the corpus CSV.")
    ap.add_argument("--output", default="windows.csv", help="Output CSV path.")
    ap.add_argument("--model-name", default="ddore14/RooseBERT-scr-cased",
                    help="HuggingFace tokenizer to use for windowing.")
    ap.add_argument("--target-word", default="people", help="Target token.")
    ap.add_argument("--window-size", type=int, default=50,
                    help="Number of tokens on each side of the target.")
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--year-col", default="year")
    ap.add_argument("--speaker-col", default="president")
    args = ap.parse_args()

    print(f"[load] tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    print(f"[load] corpus: {args.input}")
    df = pd.read_csv(args.input)

    df_windows = extract_people_windows(
        df, tokenizer,
        target_word=args.target_word,
        text_col=args.text_col,
        year_col=args.year_col,
        speaker_col=args.speaker_col,
        window_size=args.window_size,
    )

    df_windows.to_csv(args.output, index=False)
    print(f"[done] {len(df_windows):,} windows written to {args.output}")


if __name__ == "__main__":
    main()
