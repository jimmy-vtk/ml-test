"""
ML pipeline v11: v10 + XGBoost + LightGBM + more ET seeds + SVM calibration.

New vs v10:
  - XGBoost and LightGBM added as base models (Optuna-tuned, clean features)
  - N_SEEDS_ET 10 -> 20 for cheaper variance reduction on ET
  - SVM_cal: CalibratedClassifierCV(SVM, isotonic) added alongside raw SVM
    to directly measure whether calibration helps soft-vote blending

Base models (8): ET_multi, RF, SVM, SVM_cal, kNN, LR, XGB, LGB
Voting: all subsets x {equal, acc-weighted, OOF-Optuna}
Stacking: LR meta on {all8, trees4, top3}

Outputs:
  cv_results_v11.csv
  predictions_<model>_v11.txt
"""

import time
import numpy as np
import pandas as pd
import optuna
import xgboost as xgb
import lightgbm as lgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict, cross_validate
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin, clone

optuna.logging.set_verbosity(optuna.logging.WARNING)

RANDOM_STATE   = 42
N_TRIALS_ET    = 60
N_TRIALS_RF    = 60
N_TRIALS_SVM   = 60
N_TRIALS_KNN   = 30
N_TRIALS_LR    = 30
N_TRIALS_XGB   = 60
N_TRIALS_LGB   = 60
N_FOLDS        = 5
N_SEEDS_ET     = 30          # was 10
N_TRIALS_BLEND = 500

# ── Data ──────────────────────────────────────────────────────────────────────

train      = pd.read_csv("train.csv")
test       = pd.read_csv("test.csv")
X_raw      = train.drop("y", axis=1)
y          = train["y"].map({1: 0, 2: 1})
X_test_raw = test.copy()

def add_features(df):
    df = df.copy()
    base = df[X_raw.columns]
    df["n_ingredients"]  = (base > 0).sum(axis=1)
    df["total_mass"]     = base.sum(axis=1)
    df["mean_intensity"] = df["total_mass"] / (df["n_ingredients"] + 1e-9)
    return df

X      = add_features(X_raw)
X_test = add_features(X_test_raw)
y_arr  = y.values

cv           = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
fold_indices = list(cv.split(X, y))

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

log(f"Features: {X.shape[1]}  train={len(X)}  test={len(X_test)}")

# ── Optuna tuning ─────────────────────────────────────────────────────────────

def run_study(name, objective, n_trials):
    log(f"Tuning {name} ({n_trials} trials)...")
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(objective, n_trials=n_trials)
    log(f"  {name}: best CV={study.best_value:.4f}  params={study.best_params}")
    return study.best_params

def et_objective(trial):
    p = dict(
        n_estimators      = trial.suggest_int("n_estimators", 300, 1500, step=100),
        max_depth         = trial.suggest_categorical("max_depth", [None, 15, 25, 40]),
        min_samples_split = trial.suggest_int("min_samples_split", 2, 10),
        min_samples_leaf  = trial.suggest_int("min_samples_leaf", 1, 6),
        max_features      = trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5, 0.7]),
    )
    m = ExtraTreesClassifier(**p, random_state=RANDOM_STATE, n_jobs=1)
    return cross_val_score(m, X, y, cv=cv, scoring="accuracy", n_jobs=-1).mean()

def rf_objective(trial):
    p = dict(
        n_estimators      = trial.suggest_int("n_estimators", 300, 1500, step=100),
        max_depth         = trial.suggest_categorical("max_depth", [None, 15, 25, 40]),
        min_samples_split = trial.suggest_int("min_samples_split", 2, 10),
        min_samples_leaf  = trial.suggest_int("min_samples_leaf", 1, 6),
        max_features      = trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5, 0.7]),
    )
    m = RandomForestClassifier(**p, random_state=RANDOM_STATE, n_jobs=1)
    return cross_val_score(m, X, y, cv=cv, scoring="accuracy", n_jobs=-1).mean()

def svm_objective(trial):
    C     = trial.suggest_float("C",     0.05, 200.0, log=True)
    gamma = trial.suggest_float("gamma", 1e-4,   1.0, log=True)
    m = Pipeline([
        ("sc",  StandardScaler()),
        ("clf", SVC(kernel="rbf", C=C, gamma=gamma, probability=True,
                    random_state=RANDOM_STATE)),
    ])
    return cross_val_score(m, X, y, cv=cv, scoring="accuracy", n_jobs=-1).mean()

def knn_objective(trial):
    p = dict(
        n_neighbors = trial.suggest_int("n_neighbors", 3, 60),
        weights     = trial.suggest_categorical("weights", ["uniform", "distance"]),
        p           = trial.suggest_categorical("p", [1, 2]),
    )
    m = Pipeline([("sc", StandardScaler()), ("clf", KNeighborsClassifier(**p, n_jobs=1))])
    return cross_val_score(m, X, y, cv=cv, scoring="accuracy", n_jobs=-1).mean()

def lr_objective(trial):
    C       = trial.suggest_float("C", 1e-3, 1e2, log=True)
    penalty = trial.suggest_categorical("penalty", ["l1", "l2"])
    m = Pipeline([
        ("sc",  StandardScaler()),
        ("clf", LogisticRegression(C=C, penalty=penalty, solver="liblinear",
                                   max_iter=2000, random_state=RANDOM_STATE)),
    ])
    return cross_val_score(m, X, y, cv=cv, scoring="accuracy", n_jobs=-1).mean()

def xgb_objective(trial):
    p = dict(
        n_estimators      = trial.suggest_int("n_estimators", 200, 1500, step=100),
        max_depth         = trial.suggest_int("max_depth", 3, 10),
        learning_rate     = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        subsample         = trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree  = trial.suggest_float("colsample_bytree", 0.5, 1.0),
        min_child_weight  = trial.suggest_int("min_child_weight", 1, 10),
        gamma             = trial.suggest_float("gamma", 0.0, 1.0),
        reg_alpha         = trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        reg_lambda        = trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    )
    m = xgb.XGBClassifier(**p, random_state=RANDOM_STATE, verbosity=0,
                           tree_method="hist", n_jobs=1, eval_metric="logloss")
    return cross_val_score(m, X, y, cv=cv, scoring="accuracy", n_jobs=-1).mean()

def lgb_objective(trial):
    p = dict(
        n_estimators      = trial.suggest_int("n_estimators", 200, 1500, step=100),
        max_depth         = trial.suggest_int("max_depth", 3, 10),
        learning_rate     = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        num_leaves        = trial.suggest_int("num_leaves", 20, 300),
        subsample         = trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree  = trial.suggest_float("colsample_bytree", 0.5, 1.0),
        min_child_samples = trial.suggest_int("min_child_samples", 5, 50),
        reg_alpha         = trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        reg_lambda        = trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    )
    m = lgb.LGBMClassifier(**p, random_state=RANDOM_STATE, verbosity=-1, n_jobs=1)
    return cross_val_score(m, X, y, cv=cv, scoring="accuracy", n_jobs=-1).mean()

et_params  = run_study("Extra Trees",   et_objective,  N_TRIALS_ET)
rf_params  = run_study("Random Forest", rf_objective,  N_TRIALS_RF)
svm_params = run_study("SVM (RBF)",     svm_objective, N_TRIALS_SVM)
knn_params = run_study("kNN",           knn_objective, N_TRIALS_KNN)
lr_params  = run_study("LogReg",        lr_objective,  N_TRIALS_LR)
xgb_params = run_study("XGBoost",       xgb_objective, N_TRIALS_XGB)
lgb_params = run_study("LightGBM",      lgb_objective, N_TRIALS_LGB)

# ── Build base models (n_jobs=1 so outer n_jobs=-1 is safe) ───────────────────

class MultiSeedET(ClassifierMixin, BaseEstimator):
    def __init__(self, params, seeds):
        self.params = params
        self.seeds  = seeds

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self._models  = [
            ExtraTreesClassifier(**self.params, random_state=s, n_jobs=1).fit(X, y)
            for s in self.seeds
        ]
        return self

    def predict_proba(self, X):
        return np.mean([m.predict_proba(X) for m in self._models], axis=0)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

_svm_pipe = Pipeline([
    ("sc",  StandardScaler()),
    ("clf", SVC(kernel="rbf", C=svm_params["C"], gamma=svm_params["gamma"],
                probability=True, random_state=RANDOM_STATE)),
])

et_multi = MultiSeedET(et_params, list(range(N_SEEDS_ET)))
rf_best  = RandomForestClassifier(**rf_params, random_state=RANDOM_STATE, n_jobs=1)
svm_best = clone(_svm_pipe)
svm_cal  = CalibratedClassifierCV(clone(_svm_pipe), cv=5, method="isotonic")
knn_best = Pipeline([
    ("sc",  StandardScaler()),
    ("clf", KNeighborsClassifier(**knn_params, n_jobs=1)),
])
lr_best  = Pipeline([
    ("sc",  StandardScaler()),
    ("clf", LogisticRegression(C=lr_params["C"], penalty=lr_params["penalty"],
                               solver="liblinear", max_iter=2000,
                               random_state=RANDOM_STATE)),
])
xgb_best = xgb.XGBClassifier(**xgb_params, random_state=RANDOM_STATE, verbosity=0,
                               tree_method="hist", n_jobs=1, eval_metric="logloss")
lgb_best = lgb.LGBMClassifier(**lgb_params, random_state=RANDOM_STATE, verbosity=-1, n_jobs=1)

BASE_MODELS = [
    ("ET_multi", et_multi),
    ("RF",       rf_best),
    ("SVM",      svm_best),
    ("SVM_cal",  svm_cal),
    ("kNN",      knn_best),
    ("LR",       lr_best),
    ("XGB",      xgb_best),
    ("LGB",      lgb_best),
]

# ── Phase 1: OOF probabilities ─────────────────────────────────────────────────

log("=" * 60)
log("Phase 1: OOF probabilities (cross_val_predict)")
log("=" * 60)

oof     = {}
results = []

for name, model in BASE_MODELS:
    t0 = time.time()
    p  = cross_val_predict(model, X, y, cv=cv, method="predict_proba", n_jobs=-1)
    oof[name] = p
    fold_accs = [
        np.mean(np.argmax(p[ti], axis=1) == y_arr[ti])
        for _, ti in fold_indices
    ]
    acc, std = np.mean(fold_accs), np.std(fold_accs)
    results.append({"model": name, "cv_acc": acc, "cv_std": std, "note": "base"})
    log(f"  {name}: OOF acc={acc:.4f} ± {std:.4f}  ({time.time()-t0:.0f}s)")

base_acc = {r["model"]: r["cv_acc"] for r in results}

# ── Phase 2: Voting variants on OOF (free) ────────────────────────────────────

log("=" * 60)
log("Phase 2: Voting variants on OOF")
log("=" * 60)

def vote_on_oof(names, weights=None):
    w = np.ones(len(names)) if weights is None else np.array(weights, dtype=float)
    if w.sum() < 1e-9:
        return 0.0, 0.0
    w = w / w.sum()
    fold_accs = []
    for _, ti in fold_indices:
        P       = np.stack([oof[n][ti] for n in names])
        blended = np.einsum("k,knc->nc", w, P)
        fold_accs.append(np.mean(np.argmax(blended, axis=1) == y_arr[ti]))
    return np.mean(fold_accs), np.std(fold_accs)

SUBSETS = {
    "top2":    ["ET_multi", "SVM_cal"],
    "trees4":  ["ET_multi", "RF", "XGB", "LGB"],
    "boost3":  ["ET_multi", "XGB", "LGB"],
    "no_weak": ["ET_multi", "RF", "SVM_cal", "XGB", "LGB"],
    "all8":    ["ET_multi", "RF", "SVM", "SVM_cal", "kNN", "LR", "XGB", "LGB"],
}

for sname, names in SUBSETS.items():
    acc, std = vote_on_oof(names)
    results.append({"model": f"Vote_equal_{sname}", "cv_acc": acc, "cv_std": std, "note": "oof-vote"})
    log(f"  Vote_equal_{sname}: {acc:.4f} ± {std:.4f}")

    w = [base_acc[n] for n in names]
    acc, std = vote_on_oof(names, w)
    results.append({"model": f"Vote_accW_{sname}", "cv_acc": acc, "cv_std": std, "note": "oof-vote"})
    log(f"  Vote_accW_{sname}: {acc:.4f} ± {std:.4f}")

# ── Optuna OOF blend — all 8 models ──────────────────────────────────────────

log(f"Optuna OOF blend search ({N_TRIALS_BLEND} trials)...")
ALL_NAMES = [n for n, _ in BASE_MODELS]

def blend_objective(trial):
    w = [trial.suggest_float(f"w_{n}", 0.0, 1.0) for n in ALL_NAMES]
    acc, _ = vote_on_oof(ALL_NAMES, w)
    return acc

blend_study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
)
blend_study.optimize(blend_objective, n_trials=N_TRIALS_BLEND, show_progress_bar=False)

best_w   = [blend_study.best_params[f"w_{n}"] for n in ALL_NAMES]
acc, std = vote_on_oof(ALL_NAMES, best_w)
results.append({"model": "Vote_OOF_optuna", "cv_acc": acc, "cv_std": std, "note": "oof-vote"})
log(f"  Vote_OOF_optuna: {acc:.4f} ± {std:.4f}")
log(f"    weights: { {n: round(w, 3) for n, w in zip(ALL_NAMES, best_w)} }")

# ── Phase 3: Stacking ─────────────────────────────────────────────────────────

log("=" * 60)
log("Phase 3: Stacking variants")
log("=" * 60)

STACK_VARIANTS = [
    ("Stack_LR_all8",
     [("et", et_multi), ("rf", rf_best), ("svm", svm_cal),
      ("knn", knn_best), ("lr", lr_best), ("xgb", xgb_best), ("lgb", lgb_best)],
     LogisticRegression(max_iter=2000, C=1.0, random_state=RANDOM_STATE)),
    ("Stack_LR_trees4",
     [("et", et_multi), ("rf", rf_best), ("xgb", xgb_best), ("lgb", lgb_best)],
     LogisticRegression(max_iter=2000, C=1.0, random_state=RANDOM_STATE)),
    ("Stack_LR_boost3",
     [("et", et_multi), ("xgb", xgb_best), ("lgb", lgb_best)],
     LogisticRegression(max_iter=2000, C=1.0, random_state=RANDOM_STATE)),
]

for sname, base_ests, meta in STACK_VARIANTS:
    log(f"Building {sname}...")
    t0    = time.time()
    stack = StackingClassifier(
        estimators=base_ests, final_estimator=meta,
        stack_method="predict_proba", cv=cv, n_jobs=1,
    )
    scores = cross_validate(stack, X, y, cv=cv, scoring="accuracy",
                            return_train_score=True, n_jobs=1)
    acc = scores["test_score"].mean()
    std = scores["test_score"].std()
    results.append({"model": sname, "cv_acc": acc, "cv_std": std, "note": "stack"})
    log(f"  {sname}: CV={acc:.4f} ± {std:.4f}  ({time.time()-t0:.0f}s)")

# ── Save CV summary ───────────────────────────────────────────────────────────

cv_df = pd.DataFrame(results).sort_values("cv_acc", ascending=False)
cv_df.to_csv("cv_results_v11.csv", index=False)
log("Saved cv_results_v11.csv")
print(cv_df.to_string(index=False))

# ── Generate predictions ───────────────────────────────────────────────────────

log("=" * 60)
log("Generating predictions")
log("=" * 60)

log("Fitting base models on full training data...")
test_probas = {}
for name, model in BASE_MODELS:
    m = (MultiSeedET(model.params, model.seeds)
         if isinstance(model, MultiSeedET) else clone(model))
    m.fit(X, y)
    test_probas[name] = m.predict_proba(X_test)
    log(f"  Fitted {name}")

def save_blend(names, weights, path):
    w = np.array(weights, dtype=float)
    w /= w.sum()
    P       = np.stack([test_probas[n] for n in names])
    blended = np.einsum("k,knc->nc", w, P)
    preds   = np.argmax(blended, axis=1)
    pd.Series(preds).map({0: 1, 1: 2}).to_csv(path, index=False, header=False)
    log(f"Saved {path}")

# Base models
for name, _ in BASE_MODELS:
    preds = np.argmax(test_probas[name], axis=1)
    pd.Series(preds).map({0: 1, 1: 2}).to_csv(
        f"predictions_{name.lower()}_v11.txt", index=False, header=False)
    log(f"Saved predictions_{name.lower()}_v11.txt")

# OOF-Optuna blend
save_blend(ALL_NAMES, best_w, "predictions_vote_oof_optuna_v11.txt")

# Equal-weight top variants (no weight overfitting)
save_blend(["ET_multi", "RF", "XGB", "LGB"],             [1,1,1,1], "predictions_vote_trees4_v11.txt")
save_blend(["ET_multi", "RF", "SVM_cal", "XGB", "LGB"],  [1,1,1,1,1], "predictions_vote_no_weak_v11.txt")

# Stacking
for sname, base_ests, meta in STACK_VARIANTS:
    log(f"Fitting {sname} on full data...")
    stack = StackingClassifier(
        estimators=base_ests, final_estimator=meta,
        stack_method="predict_proba", cv=cv, n_jobs=1,
    )
    stack.fit(X, y)
    preds = stack.predict(X_test)
    pd.Series(preds).map({0: 1, 1: 2}).to_csv(
        f"predictions_{sname.lower()}_v11.txt", index=False, header=False)
    log(f"Saved predictions_{sname.lower()}_v11.txt")

log("Done.")
