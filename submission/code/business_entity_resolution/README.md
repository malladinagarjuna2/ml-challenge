# Business Entity Resolution: reproduction guide

This regenerates `output/matching_results.tsv` and `output/candidate_pairs.tsv` from the challenge data.

## Contents

```
src/
  er_pipeline.py      # the whole pipeline as one script (exported from the notebook)
  er_pipeline.ipynb   # the same pipeline as a notebook, with the outputs of the submitted run
requirements.txt      # pinned versions
```

## Setup

```bash
python -m venv .venv && .venv/Scripts/activate      # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Run

Point `ER_ROOT` at the `student_resource/` folder (the one that contains `dataset/` and `utils/`),
and `ER_WORK` at a writable folder for the cache and outputs:

```bash
set ER_ROOT=C:/path/to/student_resource          # Linux/macOS: export ER_ROOT=/path/to/student_resource
set ER_WORK=C:/path/to/workdir
python src/er_pipeline.py
```

The outputs are written to `%ER_WORK%/output/` and checked with the official
`utils/validate_submission.py` at the end.

Runtime of the submitted run on the reference machine (Intel i9-12950HX, 24 threads, 64 GB RAM,
NVIDIA RTX A2000 8 GB used for XGBoost) was about 6 hours:

| Stage | Time |
|---|---|
| Load + normalise 23.4M records (cached to parquet afterwards) | ~7 min |
| Blocking, train (200k S1 × 10.3M S2/S3) | ~11 min |
| Features, train (8M pairs) | ~7 min |
| Stage-1 + stage-2 XGBoost on GPU (5 folds each) | ~47 min |
| Decision rule search | ~3 min |
| Test inference per country (France 60 min, India 83 min, US 75 min) | ~3.9 h |

Peak RAM is about 55 GB. A CUDA GPU is required for XGBoost (`device="cuda"`); on a CPU-only machine, set
`device="cpu"` in `GBM_PARAMS`. The pipeline only uses the provided data: no external lookups, APIs or
pretrained models.

## Pipeline steps (sections inside `er_pipeline.py`)

| Section | What it does |
|---|---|
| 0 | Configuration (`MODE = "FULL"`; `"DEV"` runs a small sample for testing) |
| 1 | Load TSVs (tab-separated, no quoting) and ground truth |
| 2a–2e | Normalisation (names, addresses, states, postcodes), 29 unit checks, native-script state names learned from training labels, parallel run over all rows, quality report |
| 3 | Train sample (200k S1) and all S2/S3 |
| 5a–5c | Blocking: country + state partitions, 4 TF-IDF views with sparse top-k, 2 exact-key joins, reciprocal-rank fusion, top 40 per S1 |
| 6 | 82 pair features |
| 7 | Exact macro F0.5 metric (and a vectorised equivalent) |
| 8–9 | Stage-1 LightGBM, then stage-2 LightGBM with cluster / competition features (GroupKFold by S1) |
| 10 | Decision rule chosen on out-of-fold predictions for macro F0.5 (one-to-one, expected-F0.5 set selection) |
| 10b | Test inference one country at a time |
| 11 | Write both TSVs and validate |
