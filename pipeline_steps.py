# -*- coding: utf-8 -*-
"""
pipeline_steps.py

Contains functions defining distinct steps of the ML pipeline,
called from main.py.
"""

import numpy as np
import pandas as pd
import shap
import joblib
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, roc_auc_score, mean_squared_error
from typing import Any, Dict, List, Tuple, Optional
from sklearn.base import BaseEstimator

# Import necessary config variables
from config import (
    SAVE_MODELS,
    SAVE_FEATURES,
    SAVE_RESULTS,
    SHAP_BACKGROUND_SAMPLES,
    SHAP_EXPLAIN_SAMPLES,
    OUTPUT_DIR
)
# Utility imports could also be added here if needed

from model_definitions import get_final_model_instance


def aggregate_cv_results(outer_fold_results_reg, outer_fold_results_cls):
    """
    Calculates and prints the aggregated cross-validation results.

    Args:
        outer_fold_results_reg (dict): Dictionary with lists of regression metrics per fold.
        outer_fold_results_cls (dict): Dictionary with lists of classification metrics per fold.

    Returns:
        tuple: (mean_r2, std_r2, mean_mae, std_mae, mean_acc, std_acc, mean_roc_auc, std_roc_auc)
               Returns NaNs if metrics couldn't be calculated for any fold.
    """
    print("\n--- 6. Aggregating Cross-Validation Results ---")

    # Regression Results
    r2_scores = outer_fold_results_reg.get('R2', [])
    mae_scores = outer_fold_results_reg.get('MAE', [])
    mean_r2, std_r2, mean_mae, std_mae = np.nan, np.nan, np.nan, np.nan # Defaults

    if r2_scores and not all(np.isnan(r2_scores)):
        mean_r2 = np.nanmean(r2_scores)
        std_r2 = np.nanstd(r2_scores)
        print(f"  Average Outer CV R2: {mean_r2:.4f} +/- {std_r2:.4f}")
    else:
        print("  Average Outer CV R2: N/A (No valid scores)")

    if mae_scores and not all(np.isnan(mae_scores)):
        mean_mae = np.nanmean(mae_scores)
        std_mae = np.nanstd(mae_scores)
        print(f"  Average Outer CV MAE: {mean_mae:.4f} +/- {std_mae:.4f}")
    else:
        print("  Average Outer CV MAE: N/A (No valid scores)")

    # Classification Results
    acc_scores = outer_fold_results_cls.get('Accuracy', [])
    roc_auc_scores = outer_fold_results_cls.get('ROC-AUC', [])
    mean_acc, std_acc, mean_roc_auc, std_roc_auc = np.nan, np.nan, np.nan, np.nan # Defaults

    if acc_scores and not all(np.isnan(acc_scores)):
        mean_acc = np.nanmean(acc_scores)
        std_acc = np.nanstd(acc_scores)
        print(f"  Average Outer CV Accuracy: {mean_acc:.4f} +/- {std_acc:.4f}")
    else:
        print("  Average Outer CV Accuracy: N/A (No valid scores)")

    if roc_auc_scores and not all(np.isnan(roc_auc_scores)):
        mean_roc_auc = np.nanmean(roc_auc_scores)
        std_roc_auc = np.nanstd(roc_auc_scores)
        print(f"  Average Outer CV ROC-AUC: {mean_roc_auc:.4f} +/- {std_roc_auc:.4f}")
    else:
        print("  Average Outer CV ROC-AUC: N/A (No valid scores)")

    return mean_r2, std_r2, mean_mae, std_mae, mean_acc, std_acc, mean_roc_auc, std_roc_auc


def select_final_model(
    outer_fold_results_reg: Dict[str, List[float]],
    outer_fold_results_cls: Dict[str, List[float]],
    all_fold_models_reg: List[BaseEstimator],
    all_fold_models_cls: List[BaseEstimator],
    fold_scalers: List[Any],
    fold_selectors: List[Dict[str, Any]],  # now a dict per fold: {'reg': sel_reg, 'cls': sel_cls}
    fold_selected_features_list: List[Dict[str, List[str]]], # same dict structure
    n_splits_outer_cv: int,
) -> Tuple[
    Optional[BaseEstimator],  # final_regressor
    Optional[BaseEstimator],  # final_classifier
    Optional[Any],            # final_scaler (common for both tasks)
    Dict[str, Optional[Any]], # final_selectors  {'reg': selector_reg, 'cls': selector_cls}
    Dict[str, List[str]],     # selected_features_final {'reg': [...], 'cls': [...]} 
    int, int                  # best_fold_idx_reg, best_fold_idx_cls
]:
    """Pick the best CV fold separately for regression (by R²) and
    classification (by ROC‑AUC) and return the **already‑fitted** models
    and their preprocessing objects.

    Returns a **dict** of selectors & selected‑feature lists so down‑stream
    code can keep using `['reg']` / `['cls']`.
    """

    print("\n--- 7. Selecting Final Model (Twin‑selector aware) ---")

    # ------------------------------------------------------------------
    # Initialise return containers
    # ------------------------------------------------------------------
    final_regressor: Optional[BaseEstimator] = None
    final_classifier: Optional[BaseEstimator] = None
    final_scaler: Optional[Any] = None  # same scaler for both tasks
    final_selectors: Dict[str, Optional[Any]] = {"reg": None, "cls": None}
    selected_features_final: Dict[str, List[str]] = {"reg": [], "cls": []}
    best_fold_idx_reg = -1
    best_fold_idx_cls = -1

    # ------------------------------------------------------------------
    # 1) Regression – choose fold with highest R²
    # ------------------------------------------------------------------
    try:
        r2_scores = outer_fold_results_reg.get("R2", [])
        if r2_scores and not all(np.isnan(r2_scores)):
            best_fold_idx_reg = int(np.nanargmax(r2_scores))
            final_regressor = all_fold_models_reg[best_fold_idx_reg]
            final_selectors["reg"] = fold_selectors[best_fold_idx_reg]["reg"]
            selected_features_final["reg"] = fold_selected_features_list[best_fold_idx_reg]["reg"]
            # use the scaler from the regression‑winning fold for *both* tasks
            final_scaler = fold_scalers[best_fold_idx_reg]

            print(
                f"  Best Regression Fold: {best_fold_idx_reg + 1}/{n_splits_outer_cv} | "
                f"R² = {r2_scores[best_fold_idx_reg]:.4f} | "
                f"{len(selected_features_final['reg'])} features"
            )
        else:
            print("  No valid regression scores – skipping regressor selection.")
    except Exception as e:
        print(f"  Regression model selection failed: {e}")

    # ------------------------------------------------------------------
    # 2) Classification – choose fold with highest ROC‑AUC
    # ------------------------------------------------------------------
    try:
        roc_auc_scores = outer_fold_results_cls.get("ROC-AUC", [])
        if roc_auc_scores and not all(np.isnan(roc_auc_scores)):
            best_fold_idx_cls = int(np.nanargmax(roc_auc_scores))
            final_classifier = all_fold_models_cls[best_fold_idx_cls]
            final_selectors["cls"] = fold_selectors[best_fold_idx_cls]["cls"]
            selected_features_final["cls"] = fold_selected_features_list[best_fold_idx_cls]["cls"]

            print(
                f"  Best Classification Fold: {best_fold_idx_cls + 1}/{n_splits_outer_cv} | "
                f"ROC-AUC = {roc_auc_scores[best_fold_idx_cls]:.4f} | "
                f"{len(selected_features_final['cls'])} features"
            )
        else:
            print("  No valid classification scores – skipping classifier selection.")
    except Exception as e:
        print(f"  Classification model selection failed: {e}")

    # ------------------------------------------------------------------
    # Sanity: if we found no scaler yet but classification chose a fold
    # ------------------------------------------------------------------
    if final_scaler is None and best_fold_idx_cls != -1:
        final_scaler = fold_scalers[best_fold_idx_cls]

    return (
        final_regressor,
        final_classifier,
        final_scaler,
        final_selectors,
        selected_features_final,
        best_fold_idx_reg,
        best_fold_idx_cls,
    )



def evaluate_on_test_set(
    X_test: pd.DataFrame,
    y_test_reg: pd.Series,
    y_test_cls: pd.Series,
    final_regressor: Optional[BaseEstimator],
    final_classifier: Optional[BaseEstimator],
    final_scaler: Any,
    final_selectors: Dict[str, Any],  # {'reg': sel_or_None, 'cls': sel_or_None}
    selected_features_final: Dict[str, List[str]],  # {'reg': [...], 'cls': [...]}
    feature_names: List[str],
) -> Tuple[
    pd.DataFrame,  # X_test_scaled_df
    pd.DataFrame,  # X_test_sel_reg_df
    pd.DataFrame,  # X_test_sel_cls_df
    Optional[np.ndarray],  # y_pred_reg_test
    Optional[np.ndarray],  # y_pred_cls_test
]:
    """Apply final preprocessing, run both models, print metrics, and return
    processed matrices + predictions for SHAP/plots.
    """
    print("\n--- 8. Final Evaluation on Unseen Test Set ---")

    # Safety checks ----------------------------------------------------
    if final_scaler is None:
        print("  Error: Final scaler is not available. Cannot evaluate on test set.")
        return None, None, None, None, None
    if not selected_features_final:
        print("  Error: Final selected features dict is empty. Cannot evaluate on test set.")
        return None, None, None, None, None

    try:
        # 1️⃣  Scale test data -----------------------------------------
        X_test_scaled = final_scaler.transform(X_test)
        X_test_scaled_df = pd.DataFrame(X_test_scaled, index=X_test.index, columns=feature_names)
        print(f"  Scaled test data shape: {X_test_scaled_df.shape}")

       # 2️⃣  Feature selection – REGRESSION -------------------------
        if final_selectors.get("reg") is not None:
            X_test_sel_reg = final_selectors["reg"].transform(X_test_scaled)

            # Robust column-name recovery
            if hasattr(final_selectors["reg"], "get_feature_names_out"):
                reg_cols = final_selectors["reg"].get_feature_names_out()
            elif hasattr(final_selectors["reg"], "get_support"):
                reg_cols = X_test_scaled_df.columns[
                    final_selectors["reg"].get_support()
                ]
            else:
                reg_cols = [f"f{i}" for i in range(X_test_sel_reg.shape[1])]

            X_test_sel_reg_df = pd.DataFrame(
                X_test_sel_reg, index=X_test.index, columns=reg_cols
            )
        else:
            # ⬇︎--- guard: empty ⇒ use all scaled features
            reg_cols = selected_features_final.get("reg") or X_test_scaled_df.columns
            X_test_sel_reg_df = X_test_scaled_df[reg_cols]

        print(f"  Regression-feature test data shape: {X_test_sel_reg_df.shape}")

        # --- Feature selection – CLASSIFICATION ---------------------
        if final_selectors.get("cls") is not None:
            X_test_sel_cls = final_selectors["cls"].transform(X_test_scaled)

            if hasattr(final_selectors["cls"], "get_feature_names_out"):
                cls_cols = final_selectors["cls"].get_feature_names_out()
            elif hasattr(final_selectors["cls"], "get_support"):
                cls_cols = X_test_scaled_df.columns[
                    final_selectors["cls"].get_support()
                ]
            else:
                cls_cols = [f"f{i}" for i in range(X_test_sel_cls.shape[1])]

            X_test_sel_cls_df = pd.DataFrame(
                X_test_sel_cls, index=X_test.index, columns=cls_cols
            )
        else:
            cls_cols = selected_features_final.get("cls") or X_test_scaled_df.columns
            X_test_sel_cls_df = X_test_scaled_df[cls_cols]

        print(f"  Classification-feature test data shape: {X_test_sel_cls_df.shape}")

        # 3️⃣  Evaluate Regressor -------------------------------------
        y_pred_reg_test = None
        if final_regressor is not None:
            print("\n  Evaluating Final Regressor on Test Set…")
            try:
                y_pred_reg_test = final_regressor.predict(X_test_sel_reg_df)
                test_r2 = r2_score(y_test_reg, y_pred_reg_test)
                test_mae = mean_absolute_error(y_test_reg, y_pred_reg_test)
                test_rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_reg_test))
                print(f"    Test R²  : {test_r2:.4f}")
                print(f"    Test MAE : {test_mae:.4f}")
                print(f"    Test RMSE: {test_rmse:.4f}")
            except Exception as e:
                print(f"    Error during regression evaluation: {e}")
        else:
            print("\n  Skipping regression evaluation – no final_regressor provided.")

        # 4️⃣  Evaluate Classifier ------------------------------------
        y_pred_cls_test = None
        if final_classifier is not None:
            print("\n  Evaluating Final Classifier on Test Set…")
            try:
                y_pred_cls_test = final_classifier.predict(X_test_sel_cls_df)
                test_accuracy = accuracy_score(y_test_cls, y_pred_cls_test)
                print(f"    Test Accuracy: {test_accuracy:.4f}")

                if hasattr(final_classifier, "predict_proba"):
                    try:
                        y_pred_proba_cls_test = final_classifier.predict_proba(X_test_sel_cls_df)[:, 1]
                        test_roc_auc = roc_auc_score(y_test_cls, y_pred_proba_cls_test)
                        print(f"    Test ROC‑AUC   : {test_roc_auc:.4f}")
                    except Exception as e:
                        print(f"    Error calculating ROC‑AUC: {e}")
                else:
                    print("    ROC‑AUC not available – model lacks predict_proba.")
            except Exception as e:
                print(f"    Error during classification evaluation: {e}")
        else:
            print("\n  Skipping classification evaluation – no final_classifier provided.")

        # 5️⃣  Return artefacts for SHAP or downstream plots -----------
        return (
            X_test_scaled_df,
            X_test_sel_reg_df,
            X_test_sel_cls_df,
            y_pred_reg_test,
            y_pred_cls_test,
        )

    except Exception as e:
        print(f"Error during test set evaluation: {e}")
        return None, None, None, None, None


try:
    from xgboost import XGBClassifier, XGBRegressor
except ImportError:  # XGBoost optional
    XGBClassifier, XGBRegressor = (), ()
try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except ImportError:  # LightGBM optional
    LGBMClassifier, LGBMRegressor = (), ()
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
)
_TREE_TYPES = (
    XGBClassifier, XGBRegressor,
    LGBMClassifier, LGBMRegressor,
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
)

# ---------------------------------------------------------------------------
# Main helper
# ---------------------------------------------------------------------------

def run_shap_analysis(
        model,
        X_test_sel_df,
        selected_features,
        X_train_val,
        y_train_val,
        cv_splitter,
        best_fold_idx,
        scaler,
        selector,
        feature_selection_method):
    """Generate SHAP values and summary plots for *one* fitted model.

    Parameters
    ----------
    model : estimator
        Trained regressor or classifier for the current task.
    X_test_sel_df : pandas.DataFrame
        Test set **after** task‑specific feature selection.
    selected_features : list[str]
        Column names present in ``X_test_sel_df``.
    X_train_val : pandas.DataFrame
        Full train + validation data *before* selection.
    y_train_val : array-like
        Target aligned with ``X_train_val``.
    cv_splitter : sklearn splitter
        Outer CV splitter (e.g. ``KFold``) to reproduce folds.
    best_fold_idx : int
        Index of the outer fold whose scaler / selector became "final".
    scaler : sklearn Transformer
        Scaler fitted on that fold (shared or task‑specific).
    selector : sklearn Transformer or None
        Feature selector fitted on that fold. ``None`` if no selection.
    feature_selection_method : str
        Name of the feature‑selection strategy (for logging only).
    """
    print("\n--- 9. SHAP Interpretability --- ")

    # ------------------------------------------------------------------
    # Basic safety guards
    # ------------------------------------------------------------------
    if best_fold_idx == -1:
        print("  Skipping SHAP: Cannot determine the best fold for background data.")
        return
    if model is None:
        print("  Skipping SHAP: No model supplied.")
        return

    # ------------------------------------------------------------------
    # 1. Prepare background data (reference set)
    # ------------------------------------------------------------------
    try:
        print(f"  Preparing SHAP background data from best fold's training set (Fold {best_fold_idx + 1})")

        train_idx, _ = list(cv_splitter.split(X_train_val, y_train_val))[best_fold_idx]
        X_train_fold = X_train_val.iloc[train_idx]

        # Sub‑sample for performance
        bg_size = min(SHAP_BACKGROUND_SAMPLES, len(X_train_fold))
        bg_indices = np.random.choice(X_train_fold.index, size=bg_size, replace=False)
        X_bg_raw = X_train_fold.loc[bg_indices]

        # Scale
        X_bg_scaled = scaler.transform(X_bg_raw)

        # Select features
        if selector is not None:
            X_bg_sel = selector.transform(X_bg_scaled)
        elif feature_selection_method == "none":
            X_bg_sel = X_bg_scaled
        else:
            print("  Error: could not apply feature selection to SHAP background data.")
            return

        # Wrap to DataFrame for nicer column handling
        X_bg_df = pd.DataFrame(X_bg_sel, index=X_bg_raw.index, columns=selected_features)
        print(f"  Background data shape for SHAP: {X_bg_df.shape}")

    except Exception as e:
        print(f"  Error preparing SHAP background data: {e}")
        return

    # ------------------------------------------------------------------
    # 2. Sample test data that we actually explain
    # ------------------------------------------------------------------
    test_size = min(SHAP_EXPLAIN_SAMPLES, len(X_test_sel_df))
    test_idx = np.random.choice(X_test_sel_df.index, size=test_size, replace=False)
    X_test_sample_df = X_test_sel_df.loc[test_idx]
    print(f"  Calculating SHAP values for {test_size} test samples.")

    # ------------------------------------------------------------------
    # 3. Choose explainer and compute SHAP values
    # ------------------------------------------------------------------
    try:
        # Fast path: tree‑based models
        if isinstance(model, _TREE_TYPES):
            explainer = shap.TreeExplainer(model, X_bg_df)
            shap_values = explainer.shap_values(X_test_sample_df)
        else:
            # Define prediction function for model‑agnostic KernelExplainer
            if hasattr(model, "predict_proba"):
                predict_fn = lambda d: model.predict_proba(pd.DataFrame(d, columns=selected_features))[:, 1]
            else:
                predict_fn = lambda d: model.predict(pd.DataFrame(d, columns=selected_features))

            explainer = shap.KernelExplainer(predict_fn, X_bg_df)
            shap_values = explainer.shap_values(X_test_sample_df)

        # Summary plot (renders inline in notebooks / Colab)
        shap.summary_plot(shap_values, X_test_sample_df, feature_names=selected_features, show=False)
        print("  SHAP summary generated.\n")

        # Return for downstream use if needed
        return shap_values

    except Exception as e:
        print(f"  Error computing SHAP values: {e}")
        return



def save_artifacts(
    final_regressor, final_classifier,
    selected_features_final,
    outer_fold_results_reg, outer_fold_results_cls,
    output_dir=OUTPUT_DIR, # Use config variable as default
    save_models=SAVE_MODELS,
    save_features=SAVE_FEATURES,
    save_results=SAVE_RESULTS
):
    """
    Saves the final models, selected features, and CV results to disk.

    Args:
        final_regressor: The final trained regression model.
        final_classifier: The final trained classification model.
        selected_features_final (dict or list): Final selected feature names.
            Can be a dictionary with separate entries for regression and
            classification tasks or a simple list when features are shared.
        outer_fold_results_reg (dict): Aggregated regression CV results.
        outer_fold_results_cls (dict): Aggregated classification CV results.
        output_dir (str): Directory to save the artifacts.
        save_models (bool): Whether to save the final models.
        save_features (bool): Whether to save the list of selected features.
        save_results (bool): Whether to save the cross-validation results.

    Returns:
        None
    """
    import os
    print(f"\n--- 10. Saving Artifacts (Output Dir: {output_dir}) --- ")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # --- Save Final Models --- #
    if save_models:
        if final_regressor:
            try:
                reg_filename = os.path.join(output_dir, 'final_regressor.joblib')
                joblib.dump(final_regressor, reg_filename)
                print(f"  Saved final regressor to: {reg_filename}")
            except Exception as e:
                print(f"  Error saving final regressor: {e}")
        else:
            print("  Skipping saving final regressor (not available).")

        if final_classifier:
            try:
                cls_filename = os.path.join(output_dir, 'final_classifier.joblib')
                joblib.dump(final_classifier, cls_filename)
                print(f"  Saved final classifier to: {cls_filename}")
            except Exception as e:
                print(f"  Error saving final classifier: {e}")
        else:
            print("  Skipping saving final classifier (not available).")

    # --- Save Selected Features --- #
    if save_features:
        if not selected_features_final:
            print("  Skipping saving selected features (object is empty or None).")
        else:
            # Handle the two possible shapes: dict vs list/tuple
            feature_objects = (
                selected_features_final.items()
                if isinstance(selected_features_final, dict)
                else [("all", selected_features_final)]
            )

            for task, feat_list in feature_objects:
                try:
                    fname = (
                        f"selected_features_{task}.txt"
                        if task != "all" else "selected_features.txt"
                    )
                    features_filename = os.path.join(output_dir, fname)
                    with open(features_filename, "w") as f:
                        for feature in feat_list:
                            f.write(f"{feature}\n")
                    print(
                        f"  Saved {len(feat_list)} selected feature(s) "
                        f"to: {features_filename}"
                    )
                except Exception as e:
                    print(f"  Error saving selected features for task '{task}': {e}")

    # --- Save CV Results --- #
    if save_results:
        try:
            results_df_reg = pd.DataFrame(outer_fold_results_reg)
            results_df_cls = pd.DataFrame(outer_fold_results_cls)
            results_filename_reg = os.path.join(output_dir, 'cv_results_regression.csv')
            results_filename_cls = os.path.join(output_dir, 'cv_results_classification.csv')
            results_df_reg.to_csv(results_filename_reg, index=False)
            results_df_cls.to_csv(results_filename_cls, index=False)
            print(f"  Saved regression CV results to: {results_filename_reg}")
            print(f"  Saved classification CV results to: {results_filename_cls}")
        except Exception as e:
            print(f"  Error saving CV results: {e}")


# --- Add other pipeline step functions below ---
