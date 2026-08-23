# Frame Score Pipeline

Measuring how "the people" is framed (Nation / Civic / Global) in U.S.
presidential speeches, using era-specific masked language models built on
RooseBERT. The pipeline is five scripts that chain together via CSV/JSON files.

## Overview

**Abstract.** Political elites rarely define "the people" explicitly; they
invoke it in ways that quietly bound it by ancestry, by citizenship, or by
appeals to a broader human community. This pipeline measures which of three
theory-driven frames — **Nation** (shared ancestry, bounded sovereignty),
**Civic** (rights-bearing citizenship, constitutional principles), and
**Global** (universalist obligations beyond the state) — prevails in each use
of the word, and traces how the balance shifts across U.S. presidential
rhetoric from 1789 to 2025.

**Data.** A corpus of U.S. presidential speeches (Miller Center + American
Presidency Project), one row per speech with the speech text, year, and
speaker. Each occurrence of the token *people* is turned into a fixed-size
contextual window, yielding the instance-level unit of analysis.

**Model.** [RooseBERT](https://huggingface.co/ddore14/RooseBERT-scr-cased), a
BERT-based masked language model pre-trained on political debate and speech.
Because the meaning of "the people" shifts over time, a separate copy is
fine-tuned on each historical era's windows, and every instance is scored with
its own era-specific model.

**Method.** For each windowed occurrence, the target token is replaced with
`[MASK]`. For each frame we take the mean pseudo-log-likelihood (PLL) of its
top-N probe words at the masked slot, then subtract the mean PLL of a
random-noun baseline, giving a **baseline-corrected frame score**. Instance-
level scores are aggregated and regressed on macro-level political and economic
indicators (Long-term and Modern specifications).

## Steps

| # | Script | Description | Input | Output |
|---|--------|-------------|-------|--------|
| 1 | `01_extract_windows.py` | Extract a fixed-size token window around each occurrence of "people" (one row per occurrence). | corpus CSV (`text`, `year`, `president`) | `windows.csv` |
| 2 | `02_build_dictionary.py` | Build the probe-word dictionary (seed → expand → filter → disambiguate) and its cosine validation matrix. | corpus CSV | `dictionary.json`, `noun_counts.csv`, `cosine_matrix.csv` |
| 3 | `03_finetune_era_models.py` | Fine-tune a separate MLM per historical era on that era's windows. | `windows.csv` | `./era_models/<era>/` |
| 4 | `04_compute_frame_scores.py` | Score every window with its era-specific model and compute baseline-corrected frame scores. | `windows.csv`, `dictionary.json`, `noun_counts.csv`, `./era_models` | `scores.csv` |
| 5 | `05_regression.py` | Merge macro indicators and estimate Long/Modern OLS (instance-level clustered by year + year-level robustness). | `scores.csv`, V-Dem CSV, macro CSV | regression tables |

## Example run

### 1. Contextual windows around every "people"

    python 01_extract_windows.py --input speech_us.csv --output windows.csv --window-size 50

### 2. Curated probe-word dictionary (seed -> expand -> filter -> disambiguate)

    python 02_build_dictionary.py --input speech_us.csv \
        --out-dict dictionary.json --out-nouns noun_counts.csv --out-cosine cosine_matrix.csv \
        --top-k 20 --min-freq 20

### 3. Fine-tune one MLM per historical era

    python 03_finetune_era_models.py --windows windows.csv --output-dir ./era_models --epochs 3

### 4. Baseline-corrected frame scores with era-specific models

    python 04_compute_frame_scores.py --windows windows.csv \
        --dict dictionary.json --nouns noun_counts.csv --era-models ./era_models \
        --output scores.csv --top-n 3

### 5. Long/Modern OLS (instance-level clustered by year + year-level robustness)

    python 05_regression.py --scores scores.csv \
        --vdem V-Dem-CD-v15.csv --macro macro_indicators.csv \
        --top-n 3 --out-prefix regression

## Notes

- **Era boundaries** are defined identically in `03` and `04`
  (`ERA_BOUNDARIES`). Edit both if you change the periodization.
- **Frame score** is the baseline-corrected `top{N}_{c}_rel_pll`: mean top-N
  category PLL minus mean top-N random-noun-baseline PLL.
- The random baseline is sampled once with a fixed seed (`--seed 42`) from the
  most frequent corpus nouns not present in any dictionary.
- Requires a GPU for steps 3 and 4 in practice.
- Each script has a detailed module docstring; run `python <script> --help` for
  all arguments.
