import numpy as np
import lightgbm as lgb
import xgboost as xgb
from optuna import Trial
from optuna.exceptions import TrialPruned
from functools import partial
from utils import get_compute_device_params
from typing import Dict, Any, Tuple, Optional, Union
from sklearn.base import BaseEstimator
from inspect import signature
from functools import lru_cache


# scikit-learn imports
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    StackingRegressor, StackingClassifier
)
from sklearn.linear_model import Ridge, LogisticRegression, BayesianRidge
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, roc_auc_score
)

# Constants
RANDOM_STATE = 42   

# Get compute parameters once
COMPUTE_PARAMS = get_compute_device_params()

# =============================================================================
# 4. OPTUNA OBJECTIVE FUNCTIONS (Defined before the loop)
# =============================================================================
# Objective functions return the metric to be optimized (e.g., R2, -MSE, ROC-AUC)

# --- Custom Optuna Pruning Callback for LightGBM ---
def optuna_lgbm_pruning_callback(trial, metric, env):
    """
    Custom Optuna pruning callback for LightGBM.
    Monitors the specified metric on the first validation set ('valid_0').
    Reports the metric to Optuna and checks for pruning.
    """
    current_score = None
    # env.evaluation_result_list format: [('data_name', 'metric_name', value, is_higher_better), ...]
    for data_name, metric_name, value, _ in env.evaluation_result_list:
        if data_name == 'valid_0' and metric_name == metric:
            current_score = value
            break

    if current_score is None:
        # Metric not found, could happen in initial iterations or if eval_set is different
        # print(f"Warning: Metric '{metric}' not found at iteration {env.iteration} in results: {env.evaluation_result_list}")
        return # Continue training if metric not available yet

    trial.report(current_score, step=env.iteration)

    if trial.should_prune():
        message = f"Trial was pruned at iteration {env.iteration} with score {current_score:.4f}."
        raise TrialPruned(message)


# --- Regression Objectives ---
def optimize_lgbm_reg(trial: Trial, X_tr: np.ndarray, y_tr: np.ndarray, 
                     X_va: np.ndarray, y_va: np.ndarray) -> float:
    """
    Optimize LightGBM regressor hyperparameters.
    
    Args:
        trial: Optuna trial object for hyperparameter optimization
        X_tr: Training features
        y_tr: Training targets
        X_va: Validation features
        y_va: Validation targets
    
    Returns:
        float: Mean absolute error score
    """
    params = {
        'objective': 'regression_l1', 'metric': 'l1', 'n_estimators': 1000,
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'max_depth': trial.suggest_int('max_depth', 5, 30),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'device': COMPUTE_PARAMS['lgbm_device'], 'random_state': RANDOM_STATE, 'n_jobs': -1, 'verbosity': -1
    }
    model = lgb.LGBMRegressor(**params)
    try:
        # Create the partial function for the custom callback
        pruning_callback = partial(optuna_lgbm_pruning_callback, trial, 'l1')

        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric='l1',
                  callbacks=[pruning_callback, # Use custom pruning callback
                             lgb.early_stopping(stopping_rounds=50, verbose=False)])
        preds = model.predict(X_va)
        score = mean_absolute_error(y_va, preds)
        if np.isnan(score): raise TrialPruned("MAE score is NaN")
        return score
    except TrialPruned as e: # Catch pruned trials specifically
        print(f"Trial pruned during optimization: {e}")
        raise # Re-raise TrialPruned for Optuna
    except Exception as e:
        print(f"Trial failed for LGBM Reg: {e}")
        raise TrialPruned() # Prune on other errors

def optimize_xgb_reg(trial: Trial, X_tr: np.ndarray, y_tr: np.ndarray,
                    X_va: np.ndarray, y_va: np.ndarray) -> float:
    """
    Optimize XGBoost regressor hyperparameters.
    """
    params = {
        'objective': 'reg:squarederror', 'n_estimators': 1000,
        'eta': trial.suggest_float('eta', 0.005, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'tree_method': COMPUTE_PARAMS['xgb_tree_method'],
        'random_state': RANDOM_STATE,
        'verbosity': 0
    }
    xgb_params = params.copy()
    xgb_params['alpha'] = xgb_params.pop('reg_alpha')
    xgb_params['lambda'] = xgb_params.pop('reg_lambda')

    model = xgb.XGBRegressor(**xgb_params)
    try:
        # Add early stopping to prevent overfitting and reduce computation time
        model.fit(
            X_tr, y_tr, 
            eval_set=[(X_va, y_va)],
            eval_metric='rmse',
            early_stopping_rounds=50,
            verbose=False
        )
        preds = model.predict(X_va)
        mse = mean_squared_error(y_va, preds)
        score = np.sqrt(mse)
        if np.isnan(score):
            print(f"Trial pruned: RMSE score is NaN")
            raise TrialPruned("RMSE score is NaN")
        return score
    except TrialPruned as e:
        print(f"Trial pruned during optimization: {e}")
        raise
    except Exception as e:
        print(f"Trial failed for XGB Reg: {e}")
        raise TrialPruned()

def optimize_rf_reg(trial: Trial, X_tr: np.ndarray, y_tr: np.ndarray,
                   X_va: np.ndarray, y_va: np.ndarray) -> float:
    """
    Optimize Random Forest regressor hyperparameters.
    """
    try:
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 5, 50, log=True),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
            'max_features': trial.suggest_float('max_features', 0.1, 1.0),
            'random_state': RANDOM_STATE,
            'n_jobs': -1
        }
        model = RandomForestRegressor(**params)
        model.fit(X_tr, y_tr)
        preds = model.predict(X_va)
        score = r2_score(y_va, preds)
        if np.isnan(score):
            print(f"Trial pruned: R2 score is NaN")
            raise TrialPruned("R2 score is NaN")
        return score
    except TrialPruned as e:
        print(f"Trial pruned during optimization: {e}")
        raise
    except Exception as e:
        print(f"Trial failed for RF Reg: {e}")
        raise TrialPruned()

def optimize_ridge_reg(trial: Trial, X_tr: np.ndarray, y_tr: np.ndarray,
                  X_va: np.ndarray, y_va: np.ndarray) -> float:
    """
    Optimize Ridge regression hyperparameters.
    """
    try:
        alpha = trial.suggest_float('alpha', 1e-4, 1e2, log=True)
        model = Ridge(alpha=alpha, random_state=RANDOM_STATE)
        model.fit(X_tr, y_tr)
        preds = model.predict(X_va)
        score = r2_score(y_va, preds)
        if np.isnan(score):
            print(f"Trial pruned: R2 score is NaN")
            raise TrialPruned("R2 score is NaN")
        return score
    except TrialPruned as e:
        print(f"Trial pruned during optimization: {e}")
        raise
    except Exception as e:
        print(f"Trial failed for Ridge: {e}")
        raise TrialPruned()

def optimize_mlp_reg(trial: Trial, X_tr: np.ndarray, y_tr: np.ndarray,
                    X_va: np.ndarray, y_va: np.ndarray) -> float:
    """
    Optimize Multi-layer Perceptron regressor hyperparameters.
    
    Args:
        trial: Optuna trial object for hyperparameter optimization
        X_tr: Training features
        y_tr: Training targets
        X_va: Validation features
        y_va: Validation targets
    
    Returns:
        float: R2 score on validation set
    """
    try:
        hidden_layer_sizes = (
            trial.suggest_int('n_units_l1', 32, 128),
            trial.suggest_int('n_units_l2', 16, 64)
        )
        activation = trial.suggest_categorical('activation', ['relu', 'tanh'])
        solver = trial.suggest_categorical('solver', ['adam', 'sgd'])
        alpha = trial.suggest_float('alpha', 1e-5, 1e-1, log=True)
        learning_rate_init = trial.suggest_float('learning_rate_init', 1e-4, 1e-2, log=True)
        
        model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            alpha=alpha,
            learning_rate_init=learning_rate_init,
            max_iter=500,
            early_stopping=True,
            random_state=RANDOM_STATE
        )
        
        model.fit(X_tr, y_tr)
        preds = model.predict(X_va)
        score = r2_score(y_va, preds)
        if np.isnan(score):
            print(f"Trial pruned: R2 score is NaN")
            raise TrialPruned("R2 score is NaN")
        return score
    except TrialPruned as e:
        print(f"Trial pruned during optimization: {e}")
        raise
    except Exception as e:
        print(f"Trial failed for MLP Reg: {e}")
        raise TrialPruned()

def optimize_knn_reg(trial: Trial, X_tr: np.ndarray, y_tr: np.ndarray,
                    X_va: np.ndarray, y_va: np.ndarray) -> float:
    """
    Optimize K-Nearest Neighbors regressor hyperparameters.
    
    Args:
        trial: Optuna trial object for hyperparameter optimization
        X_tr: Training features
        y_tr: Training targets
        X_va: Validation features
        y_va: Validation targets
    
    Returns:
        float: R2 score on validation set
    """
    try:
        params = {
            'n_neighbors': trial.suggest_int('n_neighbors', 3, 30),
            'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
            'p': trial.suggest_int('p', 1, 2),
            'n_jobs': -1
        }
        model = KNeighborsRegressor(**params)
        model.fit(X_tr, y_tr)
        preds = model.predict(X_va)
        score = r2_score(y_va, preds)
        if np.isnan(score):
            print(f"Trial pruned: R2 score is NaN")
            raise TrialPruned("R2 score is NaN")
        return score
    except TrialPruned as e:
        print(f"Trial pruned during optimization: {e}")
        raise
    except Exception as e:
        print(f"Trial failed for KNN Reg: {e}")
        raise TrialPruned()

# --- Classification Objectives ---
def optimize_lgbm_cls(trial: Trial, X_tr: np.ndarray, y_tr: np.ndarray,
                     X_va: np.ndarray, y_va: np.ndarray) -> float:
    """
    Optimize LightGBM classifier hyperparameters.
    
    Args:
        trial: Optuna trial object for hyperparameter optimization
        X_tr: Training features
        y_tr: Training targets
        X_va: Validation features
        y_va: Validation targets
    
    Returns:
        float: ROC-AUC score (optimization metric)
        
    Note:
        While we optimize for ROC-AUC, we also track accuracy for interpretability.
    """
    params = {
        'objective': 'binary', 'metric': 'auc', 'n_estimators': 1000,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
        'device': COMPUTE_PARAMS['lgbm_device'], 'random_state': RANDOM_STATE, 'n_jobs': -1, 'verbosity': -1
    }
    model = lgb.LGBMClassifier(**params)
    try:
        # Create the partial function for the custom callback
        pruning_callback = partial(optuna_lgbm_pruning_callback, trial, 'auc')

        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric='auc',
                  callbacks=[pruning_callback,
                             lgb.early_stopping(stopping_rounds=50, verbose=False)])
        
        # Get both probability and class predictions
        y_pred_proba = model.predict_proba(X_va)[:, 1]
        y_pred = model.predict(X_va)
        
        # Calculate both metrics
        roc_auc = roc_auc_score(y_va, y_pred_proba)
        acc = accuracy_score(y_va, y_pred)
        
        print(f"    Trial metrics - ROC-AUC: {roc_auc:.4f}, Accuracy: {acc:.4f}")
        
        if np.isnan(roc_auc): 
            raise TrialPruned("ROC-AUC score is NaN")
        return roc_auc
        
    except TrialPruned as e:
        print(f"Trial pruned during optimization: {e}")
        raise
    except Exception as e:
        print(f"Trial failed for LGBM Cls: {e}")
        raise TrialPruned()

def optimize_xgb_cls(trial: Trial, X_tr: np.ndarray, y_tr: np.ndarray,
                    X_va: np.ndarray, y_va: np.ndarray) -> float:
    """
    Optimize XGBoost classifier hyperparameters.
    
    Note:
        While we optimize for ROC-AUC, we also track accuracy for interpretability.
    """
    params = {
        'objective': 'binary:logistic', 'n_estimators': 1000,
        'eta': trial.suggest_float('eta', 0.005, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'tree_method': COMPUTE_PARAMS['xgb_tree_method'],
        'random_state': RANDOM_STATE,
        'verbosity': 0
    }
    xgb_params = params.copy()
    xgb_params['alpha'] = xgb_params.pop('reg_alpha')
    xgb_params['lambda'] = xgb_params.pop('reg_lambda')

    model = xgb.XGBClassifier(**xgb_params)
    try:
        model.fit(X_tr, y_tr)
        
        # Get both probability and class predictions
        y_pred_proba = model.predict_proba(X_va)[:, 1]
        y_pred = model.predict(X_va)
        
        # Calculate both metrics
        roc_auc = roc_auc_score(y_va, y_pred_proba)
        acc = accuracy_score(y_va, y_pred)
        
        print(f"    Trial metrics - ROC-AUC: {roc_auc:.4f}, Accuracy: {acc:.4f}")
        
        if np.isnan(roc_auc):
            print(f"Trial pruned: ROC-AUC score is NaN")
            raise TrialPruned("ROC-AUC score is NaN")
        return roc_auc
    except TrialPruned as e:
        print(f"Trial pruned during optimization: {e}")
        raise
    except Exception as e:
        print(f"Trial failed for XGB Cls: {e}")
        raise TrialPruned()

def optimize_rf_cls(trial: Trial, X_tr: np.ndarray, y_tr: np.ndarray,
                   X_va: np.ndarray, y_va: np.ndarray) -> float:
    """
    Optimize Random Forest classifier hyperparameters.
    
    Note:
        While we optimize for ROC-AUC, we also track accuracy for interpretability.
    """
    try:
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 5, 50, log=True),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
            'max_features': trial.suggest_float('max_features', 0.1, 1.0),
            'class_weight': trial.suggest_categorical('class_weight', ['balanced', None]),
            'random_state': RANDOM_STATE,
            'n_jobs': -1
        }
        model = RandomForestClassifier(**params)
        model.fit(X_tr, y_tr)
        
        # Get both probability and class predictions
        y_pred_proba = model.predict_proba(X_va)[:, 1]
        y_pred = model.predict(X_va)
        
        # Calculate both metrics
        roc_auc = roc_auc_score(y_va, y_pred_proba)
        acc = accuracy_score(y_va, y_pred)
        
        print(f"    Trial metrics - ROC-AUC: {roc_auc:.4f}, Accuracy: {acc:.4f}")
        
        if np.isnan(roc_auc):
            raise TrialPruned("ROC-AUC score is NaN")
        return roc_auc
    except TrialPruned as e:
        print(f"Trial pruned during optimization: {e}")
        raise
    except Exception as e:
        print(f"Trial failed for RF Cls: {e}")
        raise TrialPruned()

def optimize_logistic_regression_cls(trial: Trial, X_tr: np.ndarray, y_tr: np.ndarray,
                                   X_va: np.ndarray, y_va: np.ndarray) -> float:
    """
    Optimize Logistic Regression hyperparameters.
    
    Note:
        While we optimize for ROC-AUC, we also track accuracy for interpretability.
    """
    C = trial.suggest_float('C', 1e-4, 1e2, log=True)
    solver = trial.suggest_categorical('solver', ['liblinear', 'saga'])
    penalty = trial.suggest_categorical('penalty', ['l1', 'l2'])
    valid_combo = False
    if solver == 'liblinear' and penalty in ['l1', 'l2']: valid_combo = True
    elif solver == 'saga' and penalty in ['l1', 'l2']: valid_combo = True

    if not valid_combo:
        raise TrialPruned(f"Invalid combo: solver={solver}, penalty={penalty}")

    model = LogisticRegression(C=C, solver=solver, penalty=penalty, max_iter=2000,
                               random_state=RANDOM_STATE, n_jobs=-1, class_weight='balanced')
    try:
        model.fit(X_tr, y_tr)
        
        # Get both probability and class predictions
        y_pred_proba = model.predict_proba(X_va)[:, 1]
        y_pred = model.predict(X_va)
        
        # Calculate both metrics
        roc_auc = roc_auc_score(y_va, y_pred_proba)
        acc = accuracy_score(y_va, y_pred)
        
        print(f"    Trial metrics - ROC-AUC: {roc_auc:.4f}, Accuracy: {acc:.4f}")
        
        if np.isnan(roc_auc):
            raise TrialPruned("ROC-AUC score is NaN")
        return roc_auc
    except ValueError as e:
        print(f"Trial failed for LR: {e}")
        raise TrialPruned()

def optimize_mlp_cls(trial: Trial, X_tr: np.ndarray, y_tr: np.ndarray,
                    X_va: np.ndarray, y_va: np.ndarray) -> float:
    """
    Optimize Multi-layer Perceptron classifier hyperparameters.
    
    Note:
        While we optimize for ROC-AUC, we also track accuracy for interpretability.
    """
    hidden_layer_sizes = (trial.suggest_int('n_units_l1', 32, 128), trial.suggest_int('n_units_l2', 16, 64))
    activation = trial.suggest_categorical('activation', ['relu', 'tanh', 'logistic'])
    solver = trial.suggest_categorical('solver', ['adam', 'sgd'])
    alpha = trial.suggest_float('alpha', 1e-5, 1e-1, log=True)
    learning_rate_init = trial.suggest_float('learning_rate_init', 1e-4, 1e-2, log=True)
    params = {
        'hidden_layer_sizes': hidden_layer_sizes,
        'activation': activation,
        'solver': solver,
        'alpha': alpha,
        'learning_rate_init': learning_rate_init,
        'max_iter': 500,
        'early_stopping': True,
        'random_state': RANDOM_STATE
    }
    model = MLPClassifier(**params)
    try:
        model.fit(X_tr, y_tr)
        
        # Get both probability and class predictions
        y_pred_proba = model.predict_proba(X_va)[:, 1]
        y_pred = model.predict(X_va)
        
        # Calculate both metrics
        roc_auc = roc_auc_score(y_va, y_pred_proba)
        acc = accuracy_score(y_va, y_pred)
        
        print(f"    Trial metrics - ROC-AUC: {roc_auc:.4f}, Accuracy: {acc:.4f}")
        
        if np.isnan(roc_auc):
            raise TrialPruned("ROC-AUC score is NaN")
        return roc_auc
    except Exception as e:
        print(f"Trial failed for MLP Cls: {e}")
        raise TrialPruned()

def optimize_svc_cls(trial: Trial, X_tr: np.ndarray, y_tr: np.ndarray,
                    X_va: np.ndarray, y_va: np.ndarray) -> float:
    """
    Optimize Support Vector classifier hyperparameters.
    
    Note:
        While we optimize for ROC-AUC, we also track accuracy for interpretability.
    """
    kernel = trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly'])
    
    if kernel == 'linear':
        gamma = 'scale'
        degree = 3
    elif kernel == 'poly':
        gamma = trial.suggest_float('gamma', 1e-4, 1e0, log=True)
        degree = trial.suggest_int('degree', 2, 5)
    else:  
        gamma = trial.suggest_float('gamma', 1e-4, 1e0, log=True)
        degree = 3

    params = {
        'C': trial.suggest_float('C', 1e-2, 1e2, log=True),
        'kernel': kernel,
        'gamma': gamma,
        'degree': degree,
        'probability': True,
        'random_state': RANDOM_STATE
    }

    model = SVC(**params)
    try:
        model.fit(X_tr, y_tr)
        
        # Get both probability and class predictions
        y_pred_proba = model.predict_proba(X_va)[:, 1]
        y_pred = model.predict(X_va)
        
        # Calculate both metrics
        roc_auc = roc_auc_score(y_va, y_pred_proba)
        acc = accuracy_score(y_va, y_pred)
        
        print(f"    Trial metrics - ROC-AUC: {roc_auc:.4f}, Accuracy: {acc:.4f}")
        
        if np.isnan(roc_auc):
            raise TrialPruned("ROC-AUC score is NaN")
        return roc_auc
    except Exception as e:
        print(f"Trial failed for SVC: {e}")
        raise TrialPruned()

# --- Helper Function for Model Instantiation ---
# ----------------------------------------------------------------------
@lru_cache(maxsize=None)
def _valid_constructor_keys(model_cls: type) -> set[str]:
    """Return the set of keyword names accepted by an estimator’s __init__()."""
    return set(signature(model_cls.__init__).parameters)


def get_final_model_instance(
    ModelClass: type,
    best_params: Optional[Dict[str, Any]],
    **kwargs
) -> Tuple[BaseEstimator, Optional[Dict[str, Any]]]:
    """
    Instantiate a model with |best_params| (from Optuna) + extra kwargs,
    but filter out anything the constructor doesn’t understand.
    Falls back to defaults if the filtered set still mis-matches.
    """
    # --------------- 1. merge candidate parameters --------------------
    candidate_params: Dict[str, Any] = {}
    if best_params:
        candidate_params.update(best_params)
    candidate_params.update(kwargs)

    # --------------- 2. inject common knobs *only if supported* -------
    accepted_keys = _valid_constructor_keys(ModelClass)

    if 'random_state' in accepted_keys:
        candidate_params['random_state'] = RANDOM_STATE
    if 'n_jobs' in accepted_keys:
        candidate_params.setdefault('n_jobs', -1)        # don’t overwrite user arg
    if ModelClass is SVC and 'probability' in accepted_keys:
        candidate_params['probability'] = True

    # --------------- 3. final filtering -------------------------------
    filtered_params = {k: v for k, v in candidate_params.items()
                       if k in accepted_keys}

    # --------------- 4. try to build the model ------------------------
    try:
        model = ModelClass(**filtered_params)
        return model, best_params
    except TypeError as e:
        # Something slipped through (rare but possible, e.g. library version drift)
        print(f"[WARN] {ModelClass.__name__}: {e}. Falling back to defaults.")
        safe_kwargs = {}
        if 'random_state' in accepted_keys:
            safe_kwargs['random_state'] = RANDOM_STATE
        return ModelClass(**safe_kwargs), None


# --- Model Definition Functions ---
def get_base_regressors(random_state: int = 42) -> Dict[str, BaseEstimator]:
    """
    Get dictionary of base regression models with default parameters.
    
    Args:
        random_state: Random seed for reproducibility
    
    Returns:
        dict: Dictionary of initialized regression models
    """
    return {
        # Tree-based learners
        'LGBM': lgb.LGBMRegressor(
            objective='regression_l1', 
            metric='l1', 
            n_estimators=1000,
            learning_rate=0.05,  # Middle of range 0.005-0.1
            num_leaves=60,       # Middle of range 20-100
            max_depth=17,        # Middle of range 5-30
            subsample=0.8,       # Middle of range 0.6-1.0
            colsample_bytree=0.8, # Middle of range 0.6-1.0
            reg_alpha=0.1,       # Middle of log range 1e-8-10.0
            reg_lambda=0.1,      # Middle of log range 1e-8-10.0
            min_child_samples=50, # Middle of range 5-100
            device=COMPUTE_PARAMS['lgbm_device'],
            random_state=random_state, 
            n_jobs=-1, 
            verbosity=-1
        ),
        'XGB': xgb.XGBRegressor(
            objective='reg:squarederror', 
            n_estimators=1000,
            eta=0.05,            # Middle of log range 0.005-0.1
            max_depth=6,         # Middle of range 3-10
            subsample=0.75,      # Middle of range 0.5-1.0
            colsample_bytree=0.75, # Middle of range 0.5-1.0
            reg_alpha=0.1,           # Middle of log range 1e-8-10.0
            reg_lambda=0.1,         # Middle of log range 1e-8-10.0
            tree_method=COMPUTE_PARAMS['xgb_tree_method'],
            random_state=random_state, 
            verbosity=0
        ),
        'RandomForest': RandomForestRegressor(
            n_estimators=175,    # Middle of range 50-300
            max_depth=25,        # Middle of log range 5-50
            min_samples_split=10, # Middle of range 2-20
            min_samples_leaf=10,  # Middle of range 1-20
            max_features=0.5,    # Middle of range 0.1-1.0
            random_state=random_state, 
            n_jobs=-1
        ),

        # Linear model
        'Ridge': Ridge(
            alpha=1.0,           # Middle of log range 1e-4-1e2
            random_state=random_state
        ),

        # Neural network
        'MLP': MLPRegressor(
            hidden_layer_sizes=(80, 40),  # Middle of ranges (32-128, 16-64)
            activation='relu',            # Default from optimization
            solver='adam',                # Default from optimization
            alpha=1e-3,                  # Middle of log range 1e-5-1e-1
            learning_rate_init=1e-3,     # Middle of log range 1e-4-1e-2
            max_iter=500,
            early_stopping=True,
            random_state=random_state
        ),

        # Distance-based
        'KNN': KNeighborsRegressor(
            n_neighbors=15,      # Middle of range 3-30
            weights='distance',  # Better default than 'uniform'
            p=2,                 # Euclidean distance
            n_jobs=-1
        ),
    }


def get_base_classifiers(random_state=42):
    """
    Returns a dictionary of base classification models (6 total).
    Each is instantiated with optimized default parameters, 
    plus a random_state for reproducibility.
    """
    return {
        # Tree-based learners
        'LGBM': lgb.LGBMClassifier(
            objective='binary', 
            metric='auc', 
            n_estimators=1000,
            learning_rate=0.1,   # Middle of log range 0.01-0.3
            num_leaves=160,      # Middle of range 20-300
            max_depth=7,         # Middle of range 3-12
            subsample=0.8,       # Middle of range 0.6-1.0
            colsample_bytree=0.8, # Middle of range 0.6-1.0
            min_child_samples=50, # Middle of range 5-100
            reg_alpha=0.1,       # Middle of log range 1e-8-1.0
            reg_lambda=0.1,      # Middle of log range 1e-8-1.0
            device=COMPUTE_PARAMS['lgbm_device'],
            random_state=random_state, 
            n_jobs=-1, 
            verbosity=-1
        ),
        'XGB': xgb.XGBClassifier(
            objective='binary:logistic', 
            n_estimators=1000,
            eta=0.05,            # Middle of log range 0.005-0.1
            max_depth=6,         # Middle of range 3-10
            subsample=0.75,      # Middle of range 0.5-1.0
            colsample_bytree=0.75, # Middle of range 0.5-1.0
            reg_alpha=0.1,           # Middle of log range 1e-8-10.0
            reg_lambda=0.1,         # Middle of log range 1e-8-10.0
            tree_method=COMPUTE_PARAMS['xgb_tree_method'],
            random_state=random_state, 
            verbosity=0
        ),
        'RandomForest': RandomForestClassifier(
            n_estimators=275,    # Middle of range 50-500
            max_depth=25,        # Middle of log range 5-50
            min_samples_split=10, # Middle of range 2-20
            min_samples_leaf=10,  # Middle of range 1-20
            max_features=0.5,    # Middle of range 0.1-1.0
            class_weight='balanced', # Better default for classification
            random_state=random_state, 
            n_jobs=-1 
        ),

        # Linear model
        'LogisticRegression': LogisticRegression(
            C=1.0,               # Middle of log range 1e-4-1e2
            solver='liblinear',  # Default from optimization
            penalty='l2',        # Default from optimization
            max_iter=2000,
            class_weight='balanced',
            random_state=random_state, 
            n_jobs=1 
        ),

        # Neural network
        'MLP': MLPClassifier(
            hidden_layer_sizes=(80, 40),  # Middle of ranges (32-128, 16-64)
            activation='relu',            # Default from optimization
            solver='adam',                # Default from optimization
            alpha=1e-3,                  # Middle of log range 1e-5-1e-1
            learning_rate_init=1e-3,     # Middle of log range 1e-4-1e-2
            max_iter=500,
            early_stopping=True,
            random_state=random_state
        ),

        # Support Vector
        'SVC': SVC(
            C=1.0,               # Middle of log range 1e-2-1e2
            kernel='rbf',        # Default from optimization
            gamma='scale',       # Default for 'rbf' kernel
            probability=True,
            random_state=random_state
        ),
    }


def get_meta_regressor_candidates(random_state=42):
    """
    Get meta-regressor candidates with default parameters.
    Returns simple models suitable for meta-learning.
    """
    return {
        # One tree-based option
        'XGB': xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            verbosity=0
        ),

        # Linear options
        'Ridge': Ridge(
            alpha=1.0,
            random_state=random_state
        ),
        
        'BayesianRidge': BayesianRidge(
            n_iter=300,
            alpha_1=1e-6,
            alpha_2=1e-6,
            lambda_1=1e-6,
            lambda_2=1e-6,
            random_state=random_state
        )
    }


def get_meta_classifier_candidates(random_state=42):
    """
    Returns a dictionary of meta classification models (3 total).
    Each model is well-suited as a meta-learner to stack base classifiers.
    """
    return {
        # Linear meta-learner
        'LogisticRegression': LogisticRegression(
            random_state=random_state, 
            class_weight='balanced', 
            max_iter=1000, 
            n_jobs=-1
        ),

        # Tree-based meta-learner
        'RandomForest': RandomForestClassifier(
            random_state=random_state, 
            n_jobs=-1, 
            n_estimators=100
        ),

        # Another tree-based or boosting option
        'XGB': xgb.XGBClassifier(random_state=random_state, verbosity=0),
    }

# select_best_stack is defined in this file and passed as an arg to outer_cv.py
def select_best_stack(
    base_model_definitions: Dict[str, BaseEstimator],
    tuned_base_params: Dict[str, Dict[str, Any]],
    meta_candidates: Dict[str, BaseEstimator],
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    task: str = 'regression',
    cv_folds: int = 5,
    random_state: int = 42
) -> Tuple[Optional[Union[StackingRegressor, StackingClassifier]], 
           Dict[str, BaseEstimator],
           Optional[BaseEstimator]]:
    """
    Build and select the best stacking ensemble.
    
    Args:
        base_model_definitions: Dictionary of base model classes
        tuned_base_params: Dictionary of tuned parameters for base models
        meta_candidates: Dictionary of meta-learner candidates
        X_tr: Training features
        y_tr: Training targets
        X_va: Validation features
        y_va: Validation targets
        task: Type of task ('regression' or 'classification')
        cv_folds: Number of cross-validation folds
        random_state: Random seed for reproducibility
        compute_params: Dictionary of compute-related parameters
    
    Returns:
        tuple: (Best stacking ensemble, Dictionary of tuned base models, Best meta-learner)
    """
    print(f"  Building stacking ensemble for {task}...")
    
    # Instantiate tuned base models
    tuned_base_models = {}
    for model_name, model_instance in base_model_definitions.items():
        if model_name in tuned_base_params and tuned_base_params[model_name]:
            print(f"    Instantiating tuned {model_name}...")
            try:
                # Use helper function to get correct instance with tuned parameters
                model, _ = get_final_model_instance(model_instance.__class__, tuned_base_params[model_name], **(COMPUTE_PARAMS or {}))
                tuned_base_models[model_name] = model
            except Exception as e:
                print(f"    Error instantiating {model_name}: {e}")
                # Fallback to default model
                tuned_base_models[model_name] = model_instance
        else:
            print(f"    Using default {model_name} (no tuned params)...")
            tuned_base_models[model_name] = model_instance

    # Try each meta-learner and select the best one
    best_score = -np.inf if task == 'regression' else 0 # Initialize regression best score to negative infinity
    best_stack = None
    best_meta = None
    best_meta_name = None
    
    for meta_name, meta_model in meta_candidates.items():
        print(f"    Trying meta-learner: {meta_name}...")
        
        try:
            # Create named tuned estimators for the stack
            named_estimators = [(name, model) for name, model in tuned_base_models.items()]
            
            # Create and fit the stacking ensemble
            if task == 'regression':
                stack = StackingRegressor(
                    estimators=named_estimators,
                    final_estimator=meta_model,
                    cv=cv_folds,
                    n_jobs=-1,
                    passthrough=False
                )
            else:  # classification
                stack = StackingClassifier(
                    estimators=named_estimators,
                    final_estimator=meta_model,
                    cv=cv_folds,
                    n_jobs=-1,
                    passthrough=False
                )
            
            # Fit on training data
            print(f"      Fitting stack with {meta_name}...")
            stack.fit(X_tr, y_tr)
            print(f"      Stack fitting finished for {meta_name}.")

            # Predict and evaluate on validation data
            if task == 'regression':
                y_pred = stack.predict(X_va)
                score = r2_score(y_va, y_pred)
                print(f"      Validation R2 for {meta_name}: {score:.4f} (is_finite: {np.isfinite(score)})") # Added finiteness check
            else:
                y_pred_proba = stack.predict_proba(X_va)[:, 1]
                score = roc_auc_score(y_va, y_pred_proba)
                print(f"      Validation ROC-AUC for {meta_name}: {score:.4f} (is_finite: {np.isfinite(score)})") # Added finiteness check

            # Update best if improved and score is valid
            if np.isfinite(score): # Only update if score is a valid number
                if (task == 'regression' and score > best_score) or (task == 'classification' and score > best_score):
                    best_score = score
                    best_stack = stack
                    best_meta = meta_model
                    best_meta_name = meta_name
                    print(f"      >>> New best meta-learner: {meta_name} (Score: {score:.4f})")
            else:
                 print(f"      Score for {meta_name} is not finite. Skipping update.")

        except Exception as e:
            import traceback # Import traceback for detailed error info
            print(f"      ERROR with meta-learner {meta_name}: {e}")
            print(f"      Traceback: {traceback.format_exc()}") # Print full traceback
            continue

    if best_stack is None:
        print("  Failed to build any valid stacking ensemble. Returning None.")
        return None, {}, None
    
    print(f"  Selected best stacking ensemble with meta-learner: {best_meta_name} (Score: {best_score:.4f})")
    return best_stack, tuned_base_models, best_meta
