# Frame Score Pipeline

Measuring how "the people" is framed (Nation / Civic / Global) in U.S.
presidential speeches, using era-specific masked language models built on
RooseBERT.

The pipeline is five scripts that chain together via CSV/JSON files.

## Steps

| # | Script | Input | Output |
|---|--------|-------|--------|
| 1 | `01_extract_windows.py` | corpus CSV (`text`, `year`, `president`) | `windows.csv` |
| 2 | `02_build_dictionary.py` | corpus CSV | `dictionary.json`, `noun_counts.csv`, `cosine_matrix.csv` |
| 3 | `03_finetune_era_models.py` | `windows.csv` | `./era_models/<era>/` |
| 4 | `04_compute_frame_scores.py` | `windows.csv`, `dictionary.json`, `noun_counts.csv`, `./era_models` | `scores.csv` |
| 5 | `05_regression.py` | `scores.csv`, V-Dem CSV, macro CSV | regression tables |

## Example run

```bash
# 1. contextual windows around every "people"
python 01_extract_windows.py --input emnlp_us.csv --output windows.csv --window-size 50

# 2. curated probe-word dictionary (seed -> expand -> filter -> disambiguate)
python 02_build_dictionary.py --input emnlp_us.csv \
    --out-dict dictionary.json --out-nouns noun_counts.csv --out-cosine cosine_matrix.csv \
    --top-k 20 --min-freq 20

# 3. fine-tune one MLM per historical era
python 03_finetune_era_models.py --windows windows.csv --output-dir ./era_models --epochs 3

# 4. baseline-corrected frame scores with era-specific models
python 04_compute_frame_scores.py --windows windows.csv \
    --dict dictionary.json --nouns noun_counts.csv --era-models ./era_models \
    --output scores.csv --top-n 3

# 5. Long/Modern OLS (instance-level clustered by year + year-level robustness)
python 05_regression.py --scores scores.csv \
    --vdem V-Dem-CD-v15.csv --macro macro_indicators.csv \
    --top-n 3 --out-prefix regression
```

## Notes

- **Era boundaries** are defined identically in `03` and `04`
  (`ERA_BOUNDARIES`). Edit both if you change the periodization.
- **Frame score** is the baseline-corrected `top{N}_{c}_rel_pll`: mean top-N
  category PLL minus mean top-N random-noun-baseline PLL.
- The random baseline is sampled once with a fixed seed (`--seed 42`) from the
  most frequent corpus nouns not present in any dictionary.
- Requires a GPU for steps 3 and 4 in practice.
