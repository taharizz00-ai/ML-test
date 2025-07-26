# -- coding: utf-8 --
# Script: Advanced Machine Learning Pipeline with Cross-Validation and Stacking
# Author: [Tahreezz Murdifin/ with AI Assistant]
# Date: 2025-06-03 (Refactored)
# Description: Implements a robust ML pipeline with:
#              - Configuration section
#              - Train/Validation/Test split + Outer K-Fold Cross-Validation
#              - Multiple feature selection options
#              - Optuna hyperparameter tuning for *all* base models (Placeholders used for some)
#              - Stacking ensemble (Regressor & Classifier) with meta-model selection
#              - Cross-validated performance reporting
#              - SHAP interpretability for both final models
# -----------------------------------------------------------------------------

# =============================================================================
# 1. LIBRARY IMPORTS
# =============================================================================
# Standard library imports
import pandas as pd
import time
from collections import defaultdict

# Scientific and ML imports
import numpy as np
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, roc_auc_score

# Deep learning frameworks
import lightgbm
import xgboost
import shap

# Local imports
from utils import run_optuna_study, get_compute_device_params
from config import *
from model_definitions import (
    get_base_regressors, get_base_classifiers,
    get_meta_regressor_candidates, get_meta_classifier_candidates,
    # Optimization functions - regression
    optimize_lgbm_reg, optimize_xgb_reg, optimize_ridge_reg,
    optimize_rf_reg, optimize_knn_reg, optimize_mlp_reg,
    # Optimization functions - classification
    optimize_lgbm_cls, optimize_xgb_cls, optimize_logistic_regression_cls,
    optimize_rf_cls, optimize_mlp_cls, optimize_svc_cls,
    # best stacking model
    select_best_stack
)
from outer_cv import run_outer_cv_loop
from pipeline_steps import (
    aggregate_cv_results,
    select_final_model,
    evaluate_on_test_set,
    run_shap_analysis,
    save_artifacts
)

# Initialize timing
start_time = time.time()

# =============================================================================
# 1. CONFIGURATION AND SETUP
# =============================================================================
print("--- 1. Initializing ML Pipeline ---")

# =============================================================================
# 2. DATA LOADING AND INITIAL SPLIT
# =============================================================================
print(f"--- 2. Loading Data from {DATA_FILE} ---")
df = pd.read_csv(DATA_FILE)

# Assuming last two columns are targets (regression then classification)
X = df.iloc[:, :-2]
y_reg = df.iloc[:, -2]
y_cls = df.iloc[:, -1]
feature_names = X.columns.tolist() # Store original feature names

# --- Hold-out Test Set ---
# This test set is NEVER used for training, tuning, or selection
# Stratify if classification target has more than 1 class
stratify_targets = y_cls if y_cls.nunique() > 1 else None
X_train_val, X_test, y_train_val_reg, y_test_reg, y_train_val_cls, y_test_cls = train_test_split(
    X, y_reg, y_cls, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_cls if STRATIFY_SPLIT else None
)

print(f"Initial split: Train/Val data = {X_train_val.shape[0]} samples, Test data = {X_test.shape[0]} samples")

COMPUTE_PARAMS = get_compute_device_params()

# =============================================================================
# 3. OUTER CROSS-VALIDATION SETUP
# =============================================================================
print(f"\n--- 3. Setting up Outer {N_SPLITS_OUTER_CV}-Fold Cross-Validation ---")

def make_outer_splitter(y_cls, n_splits, random_state):
    # If classification target has ≥2 classes use stratification
    if y_cls.nunique() > 1 and y_cls.value_counts().min() > 1:
        return StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state)
    return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

kf_outer = make_outer_splitter(y_train_val_cls, N_SPLITS_OUTER_CV, RANDOM_STATE)

# Initialize containers for results and artifacts from the loop
outer_fold_results_reg = defaultdict(list)
outer_fold_results_cls = defaultdict(list)
fold_selected_features_list = []
fold_best_params_reg = defaultdict(lambda: defaultdict(list))
fold_best_params_cls = defaultdict(lambda: defaultdict(list))
fold_scalers = []
fold_selectors = []
all_fold_models_reg = []
all_fold_models_cls = []


# =============================================================================
# 4. RUN OUTER CROSS-VALIDATION LOOP (Refactored)
# =============================================================================
print(f"\n--- 4. Running Outer {N_SPLITS_OUTER_CV}-Fold Cross-Validation Loop ---")

# Prepare dictionaries of optimization functions to pass
OPTIMIZATION_FUNCTIONS_REG = {
    'LGBM': optimize_lgbm_reg,
    'XGB': optimize_xgb_reg,
    'Ridge': optimize_ridge_reg,
    'RandomForest': optimize_rf_reg,
    'MLP': optimize_mlp_reg,
    'KNN': optimize_knn_reg, 
}
OPTIMIZATION_FUNCTIONS_CLS = {
    'LGBM': optimize_lgbm_cls,
    'XGB': optimize_xgb_cls,
    'LogisticRegression': optimize_logistic_regression_cls, # Corrected function name
    'RandomForest': optimize_rf_cls,
    'MLP': optimize_mlp_cls,
    'SVC': optimize_svc_cls,
}

# Get model definitions and meta candidates
MODEL_REGRESSORS = get_base_regressors(RANDOM_STATE)
MODEL_CLASSIFIERS = get_base_classifiers(RANDOM_STATE)
STACKING_META_REGRESSOR_CANDIDATES = get_meta_regressor_candidates(RANDOM_STATE)
STACKING_META_CLASSIFIER_CANDIDATES = get_meta_classifier_candidates(RANDOM_STATE)

# Call the outer loop function
(
    outer_fold_results_reg,
    outer_fold_results_cls,
    fold_selected_features_list,
    fold_best_params_reg,
    fold_best_params_cls,
    fold_scalers,
    fold_selectors,
    all_fold_models_reg,
    all_fold_models_cls
) = run_outer_cv_loop(
    X_train_val=X_train_val,
    y_train_val_reg=y_train_val_reg,
    y_train_val_cls=y_train_val_cls,
    kf_outer=kf_outer,
    feature_names=feature_names,
    FEATURE_SELECTION_METHOD=FEATURE_SELECTION_METHOD,
    K_BEST_FEATURES=K_BEST_FEATURES,
    TUNE_ALL_BASE_MODELS=TUNE_ALL_BASE_MODELS,
    MODEL_REGRESSORS=MODEL_REGRESSORS,
    MODEL_CLASSIFIERS=MODEL_CLASSIFIERS,
    STACKING_META_REGRESSOR_CANDIDATES=STACKING_META_REGRESSOR_CANDIDATES,
    STACKING_META_CLASSIFIER_CANDIDATES=STACKING_META_CLASSIFIER_CANDIDATES,
    get_compute_device_params=get_compute_device_params,
    run_optuna_study=run_optuna_study,
    select_best_stack=select_best_stack,
    OPTIMIZATION_FUNCTIONS_REG=OPTIMIZATION_FUNCTIONS_REG,
    OPTIMIZATION_FUNCTIONS_CLS=OPTIMIZATION_FUNCTIONS_CLS
)

# =============================================================================
# 6. AGGREGATE CROSS-VALIDATION RESULTS
# =============================================================================
mean_r2, std_r2, mean_mae, std_mae, mean_acc, std_acc, mean_roc_auc, std_roc_auc = aggregate_cv_results(
    outer_fold_results_reg,
    outer_fold_results_cls
)

# Regression Results
print("\nRegression:") 
print(f" Mean R2: {mean_r2:.4f} +/- {std_r2:.4f}") 
print(f" Mean MAE: {mean_mae:.4f} +/- {std_mae:.4f}") 

# Classification Results
print("\nClassification:") 
print(f" Mean Accuracy: {mean_acc:.4f} +/- {std_acc:.4f}") 
print(f" Mean ROC-AUC: {mean_roc_auc:.4f} +/- {std_roc_auc:.4f}") 

# =============================================================================
# 7. SELECT FINAL MODEL BASED ON CV PERFORMANCE
# =============================================================================
(
    final_regressor, final_classifier, final_scaler, final_selectors,
    selected_features_final, best_fold_idx_reg, best_fold_idx_cls
) = select_final_model(
    outer_fold_results_reg,
    outer_fold_results_cls,
    all_fold_models_reg,
    all_fold_models_cls,
    fold_scalers,
    fold_selectors,
    fold_selected_features_list,
    N_SPLITS_OUTER_CV # Pass the number of splits for logging
)

# =============================================================================
# 8. EVALUATE FINAL MODEL ON THE TEST SET
# =============================================================================
X_test_scaled_df, X_test_sel_reg_df, X_test_sel_cls_df,y_pred_reg_test, y_pred_cls_test = evaluate_on_test_set(
    X_test,
    y_test_reg,
    y_test_cls,
    final_regressor,
    final_classifier,
    final_scaler,
    final_selectors,
    selected_features_final,
    feature_names # Pass original feature names
)

# ── 8.2  Handle evaluation result ─────────────────────────────────────
if (
    X_test_scaled_df is None
    or X_test_sel_reg_df is None
    or X_test_sel_cls_df is None
):
    # We reached this point only if evaluate_on_test_set()
    # returned some Nones (e.g. missing scaler, selector, etc.)
    raise RuntimeError(
        "Evaluation on the hold-out test set failed – "
        "SHAP analysis and artefact-saving skipped."
    )

# =============================================================================
# 9. SHAP INTERPRETABILITY
# =============================================================================
run_shap_analysis(
    final_regressor,
    X_test_sel_reg_df, # Pass the preprocessed test data with selected features
    selected_features_final['reg'],
    X_train_val, # Pass the original combined train/val data
    y_train_val_reg, # Pass the target needed for potential selector refitting
    kf_outer, # Pass the CV splitter
    best_fold_idx_reg, # Index of the best fold used for scaler/selector
    final_scaler, # Pass the chosen scaler
    final_selectors['reg'], # Pass the chosen selector
    FEATURE_SELECTION_METHOD # Pass the method name
)
run_shap_analysis(
    final_classifier,
    X_test_sel_cls_df, # Pass the preprocessed test data with selected features
    selected_features_final['cls'],
    X_train_val, # Pass the original combined train/val data
    y_train_val_cls, # Pass the target needed for potential selector refitting
    kf_outer, # Pass the CV splitter
    best_fold_idx_cls, # Index of the best fold used for scaler/selector
    final_scaler, # Pass the chosen scaler
    final_selectors['cls'], # Pass the chosen selector
    FEATURE_SELECTION_METHOD # Pass the method name
)

# =============================================================================
# 10. OPTIONAL: SAVE MODELS, RESULTS, ETC.
# =============================================================================
save_artifacts(
    final_regressor,
    final_classifier,
    selected_features_final,
    outer_fold_results_reg,
    outer_fold_results_cls,
    output_dir=OUTPUT_DIR,
    save_models=SAVE_MODELS,  # Fixed variable name
    save_features=SAVE_FEATURES,
    save_results=SAVE_RESULTS
)

# =============================================================================
# 11. END OF SCRIPT
# =============================================================================
end_time = time.time()
print("\n--- Pipeline Execution Finished ---")
print(f"Total Execution Time: {end_time - start_time:.2f} seconds")