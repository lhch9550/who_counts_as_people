# Who Counts as "the people"? Measuring Contested Political Concepts from Political Speech using Masked Language Models

What do political elites mean when they invoke "the people", and how has this meaning shifted across centuries of democratic history? Political theorists have long held that "the people" is a discursively constructed and contested category. Yet empirical work remains largely qualitative or small in scale, and computational approaches have yet to address how contested concepts themselves are framed over time. We address this gap with a generalizable framework that models contested concepts as configurations of theoretically grounded frames, and apply it to American presidential rhetoric using three: Nation, Civic, and Global. Using a fine-tuned political masked language model, we mask each occurrence of "people" and score each frame by the pseudo-log-likelihood of its probe words. We apply the framework to 3,402 U.S. presidential speeches from 1789 to 2025. The resulting measurements agree with expert human coding at a level comparable to inter-annotator agreement, while tracking major junctures such as the Civil War, the World Wars, the Civil Rights era, and a recent nationalist turn. They also covary with external indicators as expected: higher democracy is associated with less national and global framing and more civic framing. Beyond "the people", the framework may generalize to other contested political concepts.

## Data, Model, and Method

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

## Pipeline Steps

|   | Script | Description | Input | Output |
|---|--------|-------------|-------|--------|
| 1 | `01_extract_windows.py` | Extract a fixed-size token window around each occurrence of "people" (one row per occurrence). | corpus CSV (`text`, `year`, `president`) | `windows.csv` |
| 2 | `02_build_dictionary.py` | Build the probe-word dictionary (seed → expand → filter → disambiguate) and its cosine validation matrix. | corpus CSV | `dictionary.json`, `noun_counts.csv`, `cosine_matrix.csv` |
| 3 | `03_finetune_era_models.py` | Fine-tune a separate MLM per historical era on that era's windows. | `windows.csv` | `./era_models/<era>/` |
| 4 | `04_compute_frame_scores.py` | Score every window with its era-specific model and compute baseline-corrected frame scores. | `windows.csv`, `dictionary.json`, `noun_counts.csv`, `./era_models` | `scores.csv` |
| 5 | `05_regression.py` | Merge macro indicators and estimate Long/Modern OLS (instance-level clustered by year + year-level robustness). | `scores.csv`, V-Dem CSV, macro CSV | regression tables |

## Usage

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

### Notes

- **Era boundaries** are defined identically in `03` and `04`
  (`ERA_BOUNDARIES`). Edit both if you change the periodization.
- Requires a GPU for steps 3 and 4 in practice.
- Each script has a detailed module docstring; run `python <script> --help` for
  all arguments.

## License

The code in this repository is released under the MIT License (see LICENSE).

Note that external models and datasets are not covered by this license and
remain subject to their own terms — including RooseBERT (distributed for
non-commercial academic research) and the external indicators used in the
regression step (e.g., V-Dem, WDI). Please consult and comply with each
resource's original license before use.
