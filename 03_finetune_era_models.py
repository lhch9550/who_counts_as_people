#!/usr/bin/env python
"""
03_finetune_era_models.py
=========================
Fine-tune a separate masked-language model (starting from RooseBERT) on each
historical era's subset of the contextual windows. These era-specific models
are later used to compute frame scores (script 04).

Each window's decoded text is used as an MLM training example (whole-word
masking is *not* used; standard DataCollatorForLanguageModeling with
mlm_probability=0.15). One model is saved per era under `--output-dir/<era>`.

Input
-----
windows.csv produced by 01_extract_windows.py (must contain `window_ids`,
`year`).

Output
------
    <output-dir>/<era_name>/   : a fine-tuned MLM per era (HF save_pretrained)

Era boundaries are defined below and can be edited to match the paper's
periodization.

Example
-------
    python 03_finetune_era_models.py \
        --windows windows.csv \
        --output-dir ./era_models \
        --epochs 3 --batch-size 16
"""

import argparse
import ast
import os

import pandas as pd
import torch
from transformers import (
    AutoTokenizer, AutoModelForMaskedLM,
    DataCollatorForLanguageModeling, Trainer, TrainingArguments,
)
from datasets import Dataset

# ── Era periodization (edit to match the paper) ───────────────────────────
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


def build_era_dataset(era_df, tokenizer, max_length=128):
    """Decode window ids back to text and tokenize for MLM training."""
    texts = [tokenizer.decode(ids, skip_special_tokens=True)
             for ids in era_df["window_ids"]]
    ds = Dataset.from_dict({"text": texts})

    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True,
                         max_length=max_length, padding="max_length")

    return ds.map(tokenize_fn, batched=True, remove_columns=["text"])


def finetune_era_model(era_name, era_df, base_model, tokenizer, device,
                       output_dir, epochs, batch_size, lr, max_length):
    save_path = os.path.join(output_dir, era_name)
    if os.path.exists(save_path):
        print(f"[skip] {era_name} already fine-tuned at {save_path}")
        return save_path

    print(f"=== Fine-tuning era: {era_name} (n={len(era_df)}) ===")
    model = AutoModelForMaskedLM.from_pretrained(base_model).to(device)
    train_ds = build_era_dataset(era_df, tokenizer, max_length=max_length)

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

    args = TrainingArguments(
        output_dir=f"./tmp_{era_name}",
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        save_strategy="no",
        logging_steps=50,
        learning_rate=lr,
        fp16=torch.cuda.is_available(),
        report_to=[],
    )

    Trainer(model=model, args=args,
            train_dataset=train_ds, data_collator=collator).train()

    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"[saved] {era_name} -> {save_path}")
    return save_path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--windows", required=True, help="windows.csv from step 01.")
    ap.add_argument("--base-model", default="ddore14/RooseBERT-scr-cased")
    ap.add_argument("--output-dir", default="./era_models")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max-length", type=int, default=128)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    print(f"[load] windows: {args.windows}")
    df = pd.read_csv(args.windows)
    df["window_ids"] = df["window_ids"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    df["era"] = df["year"].apply(assign_era)
    df = df.dropna(subset=["era"])
    print(df["era"].value_counts())

    os.makedirs(args.output_dir, exist_ok=True)
    for era_name, era_df in df.groupby("era"):
        finetune_era_model(
            era_name, era_df, args.base_model, tokenizer, device,
            output_dir=args.output_dir, epochs=args.epochs,
            batch_size=args.batch_size, lr=args.lr, max_length=args.max_length,
        )

    print("[done] all era models fine-tuned.")


if __name__ == "__main__":
    main()
