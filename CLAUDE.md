# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Pipeline

```powershell
python3 ml_pipeline_v11.py
```

The script runs end-to-end (Optuna tuning → OOF evaluation → voting/stacking → prediction files) and takes a long time due to hyperparameter search. Key tuning constants at the top of the file control trial counts and can be reduced for faster iteration:

```python
N_TRIALS_ET  = 60   # reduce to ~10 for quick test runs
N_TRIALS_XGB = 60
N_SEEDS_ET   = 30   # ET ensemble size
N_TRIALS_BLEND = 500
```

## Architecture

Single-file ML pipeline (`ml_pipeline_v11.py`) for binary classification on a food/recipe ingredient dataset (40 ingredient features → label 1 or 2, remapped to 0/1).

**Three-phase structure:**

1. **Optuna tuning** — each of 7 base model types gets an independent `*_objective` function and `run_study()` call. All studies use `TPESampler(seed=42)` and 5-fold stratified CV.

2. **Phase 1 – OOF probabilities** — `cross_val_predict(..., method="predict_proba")` generates out-of-fold probability arrays for all 8 base models. These are stored in `oof[name]` and reused cheaply for all voting evaluations without re-fitting.

3. **Phase 2 – Voting** — `vote_on_oof()` evaluates ensemble subsets defined in `SUBSETS` using equal weights, accuracy-weighted, and Optuna-searched weights. All voting scores are computed from the already-computed OOF arrays (no re-fitting).

4. **Phase 3 – Stacking** — `StackingClassifier` with LR meta-learner on three subsets (`all8`, `trees4`, `boost3`). These do require re-fitting and are slower.

**Base models (8):**
- `ET_multi` — `MultiSeedET` custom class: averages `predict_proba` across `N_SEEDS_ET` separate `ExtraTreesClassifier` instances with different seeds
- `RF`, `SVM`, `SVM_cal` (isotonic-calibrated SVM), `kNN`, `LR` — sklearn models, SVM/kNN/LR wrapped in `StandardScaler` pipelines
- `XGB`, `LGB` — XGBoost and LightGBM, both using `tree_method="hist"` / `verbosity=-1`

**Feature engineering** (`add_features`): appends `n_ingredients` (count of nonzero), `total_mass` (row sum), and `mean_intensity` to the raw ingredient columns.

**Outputs:**
- `cv_results_v11.csv` — all model/ensemble CV accuracy scores sorted descending
- `predictions_<model>_v11.txt` — one prediction per line (values 1 or 2) for each base model and selected ensembles

## Data

- `train.csv` — 42 columns: `y` (label 1/2) + 40 ingredient features (continuous, mostly sparse)
- `test.csv` — 40 ingredient columns, no label
- Label mapping: `{1: 0, 2: 1}` internally, reversed back to `{0: 1, 1: 2}` when saving predictions
