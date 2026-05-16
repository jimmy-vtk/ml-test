"""
Best-accuracy pipeline for this specific test.csv.

Strategy:
  1. Smart dedup (majority-vote conflict resolution) → 1923 clean train rows
  2. Direct-label 930 test rows via exact feature match → near-perfect labels
  3. Augmented training: 1923 + 930 = 2853 labeled samples
  4. Optuna tuning on clean 1923 (uncontaminated CV)
  5. Fit ensemble on 2853 → predict 804 uncertain test rows
  6. Label Spreading on full 2853+804 feature graph
  7. Iterative self-training: 4 rounds (thresholds 0.95→0.90→0.85→0.80)
  8. Remaining rows: blend ensemble + Label Spreading probabilities
  9. Final: direct labels (930) + best prediction (804)
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import time
import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.semi_supervised import LabelSpreading
from sklearn.base import BaseEstimator, ClassifierMixin, clone
import xgboost as xgb
import lightgbm as lgb
# from catboost import CatBoostClassifier

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Constants (reduce *_TRIALS for faster runs) ────────────────────────────────
RANDOM_STATE   = 42
N_FOLDS        = 5
N_SEEDS_ET     = 20
N_TRIALS_CAT   = 40
N_TRIALS_XGB   = 50
N_TRIALS_LGB   = 50
N_TRIALS_ET    = 40
N_TRIALS_SVM   = 40
N_TRIALS_LR    = 30
N_TRIALS_BLEND = 300
LS_ALPHA       = 0.2    # label spreading label fidelity (lower = stronger)

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# ── 1. Load & smart dedup ──────────────────────────────────────────────────────
train_raw = pd.read_csv("train.csv")
test_raw  = pd.read_csv("test.csv")
FEAT_COLS = test_raw.columns.tolist()
Y_MAP = {1: 0, 2: 1}
Y_INV = {0: 1, 1: 2}

def smart_dedup(df):
    df = df.copy()
    df["_key"] = [hash(tuple(r)) for r in df[FEAT_COLS].values]
    majority_y = df.groupby("_key")["y"].agg(lambda s: s.mode()[0])
    deduped = df.drop_duplicates(subset="_key", keep="first").copy()
    deduped["y"] = deduped["_key"].map(majority_y)
    return deduped[FEAT_COLS + ["y"]].reset_index(drop=True)

train = smart_dedup(train_raw)
log(f"After smart dedup: {len(train)} rows | {train['y'].value_counts().to_dict()}")

# ── 2. Direct-label test rows with exact train matches ─────────────────────────
raw_lookup: dict = {}
for row in train_raw[FEAT_COLS + ["y"]].itertuples(index=False):
    key = hash(tuple(row[:-1]))
    raw_lookup.setdefault(key, []).append(row[-1])
label_lookup = {k: pd.Series(v).mode()[0] for k, v in raw_lookup.items()}

test_direct: dict = {}
for idx, row in test_raw.iterrows():
    key = hash(tuple(row[FEAT_COLS]))
    if key in label_lookup:
        test_direct[idx] = Y_MAP[label_lookup[key]]

uncertain_idx = [i for i in test_raw.index if i not in test_direct]
log(f"Direct labeled: {len(test_direct)} / {len(test_raw)}")
log(f"Uncertain (need model): {len(uncertain_idx)}")

# ── 3. Feature engineering ─────────────────────────────────────────────────────
def make_features(df: pd.DataFrame) -> np.ndarray:
    base = df[FEAT_COLS].values.astype(np.float64)
    n_ing = (base > 0).sum(axis=1)
    total = base.sum(axis=1)
    with np.errstate(invalid="ignore"):
        masked = np.where(base > 0, base, np.nan)
        nz_std = np.nan_to_num(np.nanstd(masked, axis=1))
    return np.hstack([
        base,                                           # 40 original TF-IDF
        np.log1p(base),                                 # 40 log-transform
        (base > 0).astype(np.float64),                 # 40 binary presence
        n_ing.reshape(-1, 1),                           #  1 ingredient count
        total.reshape(-1, 1),                           #  1 total mass
        (total / (n_ing + 1e-9)).reshape(-1, 1),        #  1 mean intensity
        base.max(axis=1, keepdims=True),                #  1 max TF-IDF
        nz_std.reshape(-1, 1),                          #  1 std of nonzero
    ])                                                  # = 125 features

X_train = make_features(train)
y_train = train["y"].map(Y_MAP).values
X_test  = make_features(test_raw)
log(f"Features: train={X_train.shape}, test={X_test.shape}")

# ── 4. Augmented training: add 930 matched test rows ──────────────────────────
direct_idx_arr = np.array(list(test_direct.keys()), dtype=int)
direct_y_arr   = np.array([test_direct[i] for i in direct_idx_arr], dtype=int)
X_aug = np.vstack([X_train, X_test[direct_idx_arr]])
y_aug = np.concatenate([y_train, direct_y_arr])
log(f"Augmented training: {len(X_aug)} rows | {np.bincount(y_aug).tolist()}")

spw_clean = float((y_train == 0).sum()) / float((y_train == 1).sum())
spw_aug   = float((y_aug == 0).sum())   / float((y_aug == 1).sum())

cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

# ── 5. Optuna tuning on clean 1923 ────────────────────────────────────────────
def run_study(name, objective, n_trials):
    log(f"Tuning {name} ({n_trials} trials)...")
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    log(f"  {name}: best CV={study.best_value:.4f} | {study.best_params}")
    return study.best_params

# def cat_obj(trial):
#     m = CatBoostClassifier(
#         iterations=trial.suggest_int("iterations", 300, 1200, step=100),
#         depth=trial.suggest_int("depth", 4, 10),
#         learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
#         l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
#         subsample=trial.suggest_float("subsample", 0.6, 1.0),
#         colsample_bylevel=trial.suggest_float("colsample_bylevel", 0.5, 1.0),
#         min_data_in_leaf=trial.suggest_int("min_data_in_leaf", 1, 30),
#         auto_class_weights="Balanced", random_seed=RANDOM_STATE,
#         verbose=False, thread_count=1, allow_writing_files=False,
#     )
#     return cross_val_score(m, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1).mean()
# cat_params = run_study("CatBoost", cat_obj, N_TRIALS_CAT)

def xgb_obj(trial):
    m = xgb.XGBClassifier(
        n_estimators=trial.suggest_int("n_estimators", 200, 1200, step=100),
        max_depth=trial.suggest_int("max_depth", 3, 10),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
        gamma=trial.suggest_float("gamma", 0.0, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        scale_pos_weight=spw_clean,
        random_state=RANDOM_STATE, verbosity=0, tree_method="hist", n_jobs=1,
    )
    return cross_val_score(m, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1).mean()
xgb_params = run_study("XGBoost", xgb_obj, N_TRIALS_XGB)

def lgb_obj(trial):
    m = lgb.LGBMClassifier(
        n_estimators=trial.suggest_int("n_estimators", 200, 1200, step=100),
        max_depth=trial.suggest_int("max_depth", 3, 10),
        num_leaves=trial.suggest_int("num_leaves", 20, 300),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        min_child_samples=trial.suggest_int("min_child_samples", 5, 50),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        class_weight="balanced",
        random_state=RANDOM_STATE, verbosity=-1, n_jobs=1,
    )
    return cross_val_score(m, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1).mean()
lgb_params = run_study("LightGBM", lgb_obj, N_TRIALS_LGB)

def et_obj(trial):
    m = ExtraTreesClassifier(
        n_estimators=trial.suggest_int("n_estimators", 300, 1500, step=100),
        max_depth=trial.suggest_categorical("max_depth", [None, 15, 25, 40]),
        min_samples_split=trial.suggest_int("min_samples_split", 2, 10),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 6),
        max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5]),
        class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=1,
    )
    return cross_val_score(m, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1).mean()
et_params = run_study("ExtraTrees", et_obj, N_TRIALS_ET)

def svm_obj(trial):
    m = Pipeline([
        ("sc", StandardScaler()),
        ("clf", SVC(
            C=trial.suggest_float("C", 0.05, 200.0, log=True),
            gamma=trial.suggest_float("gamma", 1e-4, 1.0, log=True),
            kernel="rbf", probability=True, class_weight="balanced",
            random_state=RANDOM_STATE,
        )),
    ])
    return cross_val_score(m, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1).mean()
svm_params = run_study("SVM", svm_obj, N_TRIALS_SVM)

def lr_obj(trial):
    m = Pipeline([
        ("sc", StandardScaler()),
        ("clf", LogisticRegression(
            C=trial.suggest_float("C", 1e-3, 1e2, log=True),
            l1_ratio=trial.suggest_float("l1_ratio", 0.0, 1.0),
            penalty="elasticnet", solver="saga", max_iter=3000,
            class_weight="balanced", random_state=RANDOM_STATE,
        )),
    ])
    return cross_val_score(m, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1).mean()
lr_params = run_study("LogReg", lr_obj, N_TRIALS_LR)

# ── 6. Multi-seed ExtraTrees + LGB-with-DataFrame wrappers ────────────────────
class MultiSeedET(ClassifierMixin, BaseEstimator):
    def __init__(self, params, n_seeds=N_SEEDS_ET):
        self.params  = params
        self.n_seeds = n_seeds

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self._models = [
            ExtraTreesClassifier(**self.params, class_weight="balanced",
                                 random_state=s, n_jobs=1).fit(X, y)
            for s in range(self.n_seeds)
        ]
        return self

    def predict_proba(self, X):
        return np.mean([m.predict_proba(X) for m in self._models], axis=0)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]



# ── 7. Assemble base models ────────────────────────────────────────────────────
base_models = [
    # ("cat", CatBoostClassifier(
    #     **cat_params, auto_class_weights="Balanced",
    #     random_seed=RANDOM_STATE, verbose=False,
    #     thread_count=1, allow_writing_files=False,
    # )),
    ("xgb", xgb.XGBClassifier(
        **xgb_params, scale_pos_weight=spw_aug,
        random_state=RANDOM_STATE, verbosity=0, tree_method="hist", n_jobs=1,
    )),
    ("lgb", lgb.LGBMClassifier(**lgb_params, class_weight="balanced",
                               random_state=RANDOM_STATE, verbosity=-1, n_jobs=1)),
    ("et",  MultiSeedET(et_params, n_seeds=N_SEEDS_ET)),
    ("svm", Pipeline([
        ("sc", StandardScaler()),
        ("clf", SVC(
            C=svm_params["C"], gamma=svm_params["gamma"],
            kernel="rbf", probability=True,
            class_weight="balanced", random_state=RANDOM_STATE,
        )),
    ])),
    ("lr",  Pipeline([
        ("sc", StandardScaler()),
        ("clf", LogisticRegression(
            C=lr_params["C"], l1_ratio=lr_params["l1_ratio"],
            penalty="elasticnet", solver="saga", max_iter=3000,
            class_weight="balanced", random_state=RANDOM_STATE,
        )),
    ])),
]
ALL_NAMES = [n for n, _ in base_models]

# ── 8. OOF on clean train (for CV reporting) ───────────────────────────────────
log("=" * 60)
log("OOF on clean train (1923 rows)")
log("=" * 60)
oof, oof_scores = {}, {}
for name, model in base_models:
    t0 = time.time()
    p  = cross_val_predict(model, X_train, y_train, cv=cv,
                            method="predict_proba", n_jobs=-1)
    oof[name] = p
    oof_scores[name] = float(np.mean(np.argmax(p, axis=1) == y_train))
    log(f"  {name}: {oof_scores[name]:.4f}  ({time.time()-t0:.0f}s)")

# Optuna blend weights
def vote_oof(names, weights=None):
    w = np.ones(len(names)) if weights is None else np.array(weights, float)
    blended = np.einsum("k,knc->nc", w / w.sum(), np.stack([oof[n] for n in names]))
    return float(np.mean(np.argmax(blended, axis=1) == y_train))

def blend_obj(trial):
    return vote_oof(ALL_NAMES, [trial.suggest_float(f"w_{n}", 0.0, 1.0) for n in ALL_NAMES])

log(f"Optuna blend ({N_TRIALS_BLEND} trials)...")
blend_study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
)
blend_study.optimize(blend_obj, n_trials=N_TRIALS_BLEND, show_progress_bar=False)
best_w   = np.array([blend_study.best_params[f"w_{n}"] for n in ALL_NAMES])
best_w  /= best_w.sum()
acc_blend = vote_oof(ALL_NAMES, best_w)
log(f"  Blend OOF={acc_blend:.4f} | weights: { {n: round(w, 3) for n, w in zip(ALL_NAMES, best_w)} }")

# Save CV results
results = [{"model": n, "cv_acc": s, "note": "base"} for n, s in oof_scores.items()]
results.append({"model": "blend_optuna", "cv_acc": acc_blend, "note": "blend"})
cv_df = pd.DataFrame(results).sort_values("cv_acc", ascending=False)
cv_df.to_csv("cv_results_best.csv", index=False)
log(f"\n{cv_df.to_string(index=False)}\n")

# ── 9. Fit ensemble on augmented data (2853 samples) ──────────────────────────
log("=" * 60)
log("Fitting ensemble on augmented data (2853 samples)")
log("=" * 60)
fitted = {}
for name, model in base_models:
    t0 = time.time()
    m = clone(model)
    if name == "xgb":   # update scale_pos_weight for augmented set
        m.set_params(scale_pos_weight=spw_aug)
    m.fit(X_aug, y_aug)
    fitted[name] = m
    log(f"  Fitted {name}  ({time.time()-t0:.0f}s)")

def ensemble_proba(X, weights=best_w):
    probas = np.stack([fitted[n].predict_proba(X) for n in ALL_NAMES])
    return np.einsum("k,knc->nc", weights, probas)

# ── 10. Label Spreading on 2853 labeled + 804 uncertain ───────────────────────
log("=" * 60)
log("Label Spreading")
log("=" * 60)
X_uncertain = X_test[uncertain_idx]
X_spread = np.vstack([X_aug, X_uncertain])
y_spread = np.concatenate([y_aug, np.full(len(uncertain_idx), -1, dtype=int)])

scaler_ls = StandardScaler().fit(X_spread)
X_spread_sc = scaler_ls.transform(X_spread)

ls = LabelSpreading(kernel="rbf", alpha=LS_ALPHA, max_iter=1000)
ls.fit(X_spread_sc, y_spread)

n_labeled = len(X_aug)
ls_proba_uncertain = ls.label_distributions_[n_labeled:]   # (804, 2)
ls_preds_uncertain = ls.transduction_[n_labeled:]
log(f"  LS predictions: {np.bincount(ls_preds_uncertain).tolist()} (class 0, class 1)")

# ── 11. Iterative self-training on uncertain rows ──────────────────────────────
log("=" * 60)
log("Iterative self-training on 804 uncertain rows")
log("=" * 60)

current_X = X_aug.copy()
current_y = y_aug.copy()
remaining  = list(uncertain_idx)
settled    = {}   # idx → predicted label

for round_num, threshold in enumerate([0.95, 0.90, 0.85, 0.80]):
    if not remaining:
        break

    # Refit ensemble on current labeled set
    round_fitted = {}
    for name, model in base_models:
        m = clone(model)
        if name == "xgb":
            sw = float((current_y == 0).sum()) / float((current_y == 1).sum() + 1e-9)
            m.set_params(scale_pos_weight=sw)
        m.fit(current_X, current_y)
        round_fitted[name] = m

    # Predict remaining uncertain rows
    rem_X = X_test[remaining]
    probas = np.stack([round_fitted[n].predict_proba(rem_X) for n in ALL_NAMES])
    round_proba = np.einsum("k,knc->nc", best_w, probas)

    new_remaining, added = [], 0
    for i, test_i in enumerate(remaining):
        p = round_proba[i]
        if p.max() >= threshold:
            pred = int(np.argmax(p))
            settled[test_i] = pred
            current_X = np.vstack([current_X, X_test[test_i:test_i + 1]])
            current_y = np.append(current_y, pred)
            added += 1
        else:
            new_remaining.append(test_i)

    log(f"  Round {round_num+1} (threshold={threshold}): +{added} settled, {len(new_remaining)} remaining")
    remaining = new_remaining

# For truly ambiguous remaining rows: blend ensemble + Label Spreading
uncertain_pos = {idx: pos for pos, idx in enumerate(uncertain_idx)}
if remaining:
    log(f"  {len(remaining)} rows still ambiguous — blending ensemble + LabelSpreading")
    ens_proba_full = ensemble_proba(X_test)
    for test_i in remaining:
        pos     = uncertain_pos[test_i]
        ens_p   = ens_proba_full[test_i]
        ls_p    = ls_proba_uncertain[pos]
        combined = 0.6 * ens_p + 0.4 * ls_p
        settled[test_i] = int(np.argmax(combined))

# ── 12. Assemble final predictions ────────────────────────────────────────────
final_preds = np.full(len(test_raw), -1, dtype=int)
for idx, label in test_direct.items():
    final_preds[idx] = label
for idx in uncertain_idx:
    final_preds[idx] = settled[idx]

assert (final_preds == -1).sum() == 0, "BUG: some predictions missing"

# ── 13. Save outputs ───────────────────────────────────────────────────────────
out = pd.Series([Y_INV[p] for p in final_preds])
out.to_csv("predictions_best.txt", index=False, header=False)
log("Saved predictions_best.txt")

log("=" * 60)
log(f"Direct-labeled:  {len(test_direct)} / {len(test_raw)} ({len(test_direct)/len(test_raw):.1%})")
log(f"Self-trained:    {len(settled) - len(remaining) if remaining else len(settled)} rows")
log(f"Blend-fallback:  {len(remaining)} rows")
log(f"Pred dist: {out.value_counts().to_dict()}")
log(f"Best individual CV:  {max(oof_scores.values()):.4f}")
log(f"Blend CV:           {acc_blend:.4f}")
log("Done.")
