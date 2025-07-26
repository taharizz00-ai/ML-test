"""
outer_cv.py – twin‑selector version (2025‑06‑01)
================================================
* one selector for **regression** (band‑gap)
* another selector for **classification** (band‑type)

Down‑stream modules must now treat `fold_selectors[fold]` as a `dict`
with keys `'reg'` and `'cls'`.  See the doc‑string at the end of this
file for the two small edits required in `pipeline_steps.select_final_model`
and `pipeline_steps.evaluate_on_test_set`.
"""

from __future__ import annotations

import time
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import (
    SelectFromModel, SelectKBest,
    f_regression, f_classif,
    mutual_info_regression, mutual_info_classif,
)
from sklearn.metrics import (
    r2_score, mean_absolute_error,
    accuracy_score, roc_auc_score,
)

import lightgbm as lgb
from sklearn.model_selection import KFold

# ------------------------ config & helpers -------------------------------
from config import (
    RANDOM_STATE, N_SPLITS_OUTER_CV,
    FEATURE_SELECTION_METHOD, K_BEST_FEATURES,
    OPTUNA_TRIALS_MAIN, OPTUNA_TRIALS_OTHER,
    STACKING_CV_FOLDS,
)

# ------------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------------

def run_outer_cv_loop(
    X_train_val: pd.DataFrame,
    y_train_val_reg: pd.Series,
    y_train_val_cls: pd.Series,
    kf_outer: KFold,
    feature_names: List[str],
    *,
    # tuning / stacking plumbing injected from main.py -------------------
    MODEL_REGRESSORS: Dict[str, object],
    MODEL_CLASSIFIERS: Dict[str, object],
    STACKING_META_REGRESSOR_CANDIDATES: Dict[str, object],
    STACKING_META_CLASSIFIER_CANDIDATES: Dict[str, object],
    OPTIMIZATION_FUNCTIONS_REG: Dict[str, object],
    OPTIMIZATION_FUNCTIONS_CLS: Dict[str, object],
    run_optuna_study,  # callable
    select_best_stack,  # callable
    get_compute_device_params,  # callable
    TUNE_ALL_BASE_MODELS: bool,
) -> Tuple[
        Dict[str, List[float]],  # outer_fold_results_reg
        Dict[str, List[float]],  # outer_fold_results_cls
        List[Dict[str, List[str]]],  # fold_selected_features_list
        Dict[str, Dict[int, Dict]],  # fold_best_params_reg
        Dict[str, Dict[int, Dict]],  # fold_best_params_cls
        List[StandardScaler],        # fold_scalers
        List[Dict[str, Optional[object]]],  # fold_selectors
        List[object],  # best reg stacks per fold
        List[object],  # best cls stacks per fold
    ]:
    """Outer **K‑fold** loop with separate selectors per task."""

    # ---- master collectors ------------------------------------------------
    outer_fold_results_reg: Dict[str, List[float]] = defaultdict(list)
    outer_fold_results_cls: Dict[str, List[float]] = defaultdict(list)

    fold_selected_features_list: List[Dict[str, List[str]]] = []
    fold_best_params_reg: Dict[str, Dict[int, Dict]] = defaultdict(lambda: defaultdict(dict))
    fold_best_params_cls: Dict[str, Dict[int, Dict]] = defaultdict(lambda: defaultdict(dict))
    fold_scalers: List[StandardScaler] = []
    fold_selectors: List[Dict[str, Optional[object]]] = []        # {'reg': sel_reg, 'cls': sel_cls}
    all_fold_models_reg: List[object] = []
    all_fold_models_cls: List[object] = []

    COMPUTE_PARAMS = get_compute_device_params()

    t0_cv = time.time()

    for fold, (idx_tr, idx_va) in enumerate(kf_outer.split(X_train_val, y_train_val_reg), 1):
        print(f"\n===== Outer Fold {fold}/{N_SPLITS_OUTER_CV} =====")
        fold_start = time.time()

        # ------------------------------------------------------------ data split
        X_tr, X_va = X_train_val.iloc[idx_tr], X_train_val.iloc[idx_va]
        y_tr_reg, y_va_reg = y_train_val_reg.iloc[idx_tr], y_train_val_reg.iloc[idx_va]
        y_tr_cls, y_va_cls = y_train_val_cls.iloc[idx_tr], y_train_val_cls.iloc[idx_va]

        print(f"  Train shape: {X_tr.shape}  |  Val shape: {X_va.shape}")
        print(f"  Class distribution: {Counter(y_tr_cls)} -> {Counter(y_va_cls)}")

        # ------------------------------------------------------------ scaling
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_va_scaled = scaler.transform(X_va)
        fold_scalers.append(scaler)

        # ------------------------------------------------------------ twin selectors
        sel_reg = sel_cls = None
        X_tr_reg = X_tr_cls = X_tr_scaled  # fall‑backs if no FS
        X_va_reg = X_va_cls = X_va_scaled
        feats_reg = feats_cls = feature_names

        if FEATURE_SELECTION_METHOD == 'lgbm':
            params = dict(random_state=RANDOM_STATE, n_estimators=100, device='cpu', verbosity=-1)
            sel_reg = SelectFromModel(lgb.LGBMRegressor(**params), threshold='median')
            sel_reg.fit(X_tr_scaled, y_tr_reg)

            sel_cls = SelectFromModel(lgb.LGBMClassifier(**params), threshold='median')
            sel_cls.fit(X_tr_scaled, y_tr_cls)

        elif FEATURE_SELECTION_METHOD == 'kbest_f_both':
            k = min(K_BEST_FEATURES, X_tr_scaled.shape[1])
            sel_reg = SelectKBest(f_regression, k=k).fit(X_tr_scaled, y_tr_reg)
            sel_cls = SelectKBest(f_classif, k=k).fit(X_tr_scaled, y_tr_cls)

        elif FEATURE_SELECTION_METHOD == 'kbest_mutual_info':
            k = min(K_BEST_FEATURES, X_tr_scaled.shape[1])
            sel_reg = SelectKBest(mutual_info_regression, k=k).fit(X_tr_scaled, y_tr_reg)
            sel_cls = SelectKBest(mutual_info_classif, k=k).fit(X_tr_scaled, y_tr_cls)

        elif FEATURE_SELECTION_METHOD == 'none':
            print("    No feature selection requested.")
        else:
            print(f"    Unknown FEATURE_SELECTION_METHOD '{FEATURE_SELECTION_METHOD}'. Skipping FS.")

        if sel_reg is not None:
            X_tr_reg = sel_reg.transform(X_tr_scaled)
            X_va_reg = sel_reg.transform(X_va_scaled)
            feats_reg = [feature_names[i] for i in sel_reg.get_support(indices=True)]
            print(f"    Regression selector kept {len(feats_reg)} features.")
        if sel_cls is not None:
            X_tr_cls = sel_cls.transform(X_tr_scaled)
            X_va_cls = sel_cls.transform(X_va_scaled)
            feats_cls = [feature_names[i] for i in sel_cls.get_support(indices=True)]
            print(f"    Classification selector kept {len(feats_cls)} features.")

        fold_selected_features_list.append({'reg': feats_reg, 'cls': feats_cls})
        fold_selectors.append({'reg': sel_reg, 'cls': sel_cls})

        # ------------------------------------------------------------ Optuna tuning
        tuned_params_reg: Dict[str, Dict] = {}
        tuned_params_cls: Dict[str, Dict] = {}

        # -------- regressors
        for name, objective_fn in OPTIMIZATION_FUNCTIONS_REG.items():
            is_main = name in ['LGBM', 'XGB']
            if not (is_main or TUNE_ALL_BASE_MODELS):
                continue
            trials = OPTUNA_TRIALS_MAIN if is_main else OPTUNA_TRIALS_OTHER
            direction = 'minimize' if name in ['LGBM', 'XGB'] else 'maximize'
            print(f"    Tuning REG {name} ({trials} trials, {direction}).")
            study = run_optuna_study(
                objective_fn, X_tr_reg, y_tr_reg, X_va_reg, y_va_reg,
                n_trials=trials, direction=direction,
                study_name=f"fold{fold}_reg_{name}", timeout=None,
            )
            tuned_params_reg[name] = study.best_params if study and study.best_trial else {}
            fold_best_params_reg[name][fold-1] = tuned_params_reg[name]

        # -------- classifiers
        for name, objective_fn in OPTIMIZATION_FUNCTIONS_CLS.items():
            is_main = name in ['LGBM', 'XGB']
            if not (is_main or TUNE_ALL_BASE_MODELS):
                continue
            trials = OPTUNA_TRIALS_MAIN if is_main else OPTUNA_TRIALS_OTHER
            print(f"    Tuning CLS {name} ({trials} trials).")
            study = run_optuna_study(
                objective_fn, X_tr_cls, y_tr_cls, X_va_cls, y_va_cls,
                n_trials=trials, direction='maximize',
                study_name=f"fold{fold}_cls_{name}", timeout=None,
            )
            tuned_params_cls[name] = study.best_params if study and study.best_trial else {}
            fold_best_params_cls[name][fold-1] = tuned_params_cls[name]

        # ------------------------------------------------------------ stacking
        best_stack_reg, *_ = select_best_stack(
            MODEL_REGRESSORS, tuned_params_reg,
            STACKING_META_REGRESSOR_CANDIDATES,
            X_tr_reg, y_tr_reg, X_va_reg, y_va_reg,
            task='regression', cv_folds=STACKING_CV_FOLDS,
            random_state=RANDOM_STATE
        )
        all_fold_models_reg.append(best_stack_reg)

        best_stack_cls, *_ = select_best_stack(
            MODEL_CLASSIFIERS, tuned_params_cls,
            STACKING_META_CLASSIFIER_CANDIDATES,
            X_tr_cls, y_tr_cls, X_va_cls, y_va_cls,
            task='classification', cv_folds=STACKING_CV_FOLDS,
            random_state=RANDOM_STATE
        )
        all_fold_models_cls.append(best_stack_cls)

        # ------------------------------------------------------------ fold metrics
        if best_stack_reg is not None:
            y_pred_reg = best_stack_reg.predict(X_va_reg)
            outer_fold_results_reg['R2'].append(r2_score(y_va_reg, y_pred_reg))
            outer_fold_results_reg['MAE'].append(mean_absolute_error(y_va_reg, y_pred_reg))
        else:
            outer_fold_results_reg['R2'].append(np.nan)
            outer_fold_results_reg['MAE'].append(np.nan)

        if best_stack_cls is not None:
            y_prob_cls = best_stack_cls.predict_proba(X_va_cls)[:, 1]
            y_pred_cls = (y_prob_cls >= 0.5).astype(int)
            outer_fold_results_cls['Accuracy'].append(accuracy_score(y_va_cls, y_pred_cls))
            try:
                outer_fold_results_cls['ROC-AUC'].append(roc_auc_score(y_va_cls, y_prob_cls))
            except ValueError:
                outer_fold_results_cls['ROC-AUC'].append(np.nan)
        else:
            outer_fold_results_cls['Accuracy'].append(np.nan)
            outer_fold_results_cls['ROC-AUC'].append(np.nan)

        print(f"    Fold {fold} done in {time.time()-fold_start:.1f}s")

    # ---------------------------------------------------------------- total time
    print(f"\n--- Outer CV finished in {time.time()-t0_cv:.1f}s ---")

    return (
        outer_fold_results_reg,
        outer_fold_results_cls,
        fold_selected_features_list,
        fold_best_params_reg,
        fold_best_params_cls,
        fold_scalers,
        fold_selectors,
        all_fold_models_reg,
        all_fold_models_cls,
    )


# ------------------------------------------------------------------------
#  ✱  IMPORTANT FOR DOWN‑STREAM CODE
# ------------------------------------------------------------------------
"""
▲ `pipeline_steps.select_final_model`
   Replace:
       final_selector = fold_selectors[best_fold_idx_reg]
   With:
       final_selector = fold_selectors[best_fold_idx_reg]['reg']

▲ `pipeline_steps.evaluate_on_test_set`
   Expect `final_selector` to be a *regression* selector (call it on X).
   If you want separate transforms for classification, add an optional
   `final_selector_cls` argument sourced from
       fold_selectors[best_fold_idx_cls]['cls']
"""
