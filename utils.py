"""
Utility functions for the ML pipeline.
"""

import optuna
import time
import warnings # Ensure warnings is imported
import uuid
import os
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import optuna.visualization as vis
import matplotlib.pyplot as plt


# Suppress Optuna trial INFO logging & specific warnings
# optuna.logging.set_verbosity(optuna.logging.WARNING)
optuna.logging.set_verbosity(optuna.logging.WARNING) # Change to INFO for suppressing warnings
warnings.filterwarnings('ignore', category=UserWarning, message='.*Starting from version 2.2.1.*')
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*divide by zero.*')
warnings.filterwarnings('ignore', category=optuna.exceptions.ExperimentalWarning) # For LightGBMPruningCallback

def get_compute_device_params():
    """
    Returns hardcoded CPU parameters for model training.
    """
    return {
        'xgb_tree_method': 'hist',  # for XGBoost
        'lgbm_device': "cpu"        # for LightGBM
    }

def run_optuna_study(
    objective, X_tr, y_tr, X_va, y_va, n_trials, direction, study_name, timeout=None
):
    """
    Runs an Optuna study with fault-tolerant SQLite loading and fallback.
    """
    start_time = time.time()
    print(f"Starting Optuna study '{study_name}' at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Create a unique storage name for this study
    storage_name = f"sqlite:///{study_name}.db"

    # Wrap the objective function with error handling
    def objective_with_error_handling(trial):
        try:
            return objective(trial, X_tr, y_tr, X_va, y_va)
        except Exception as e:
            print(f"Trial failed with error: {e}")
            # Return a very bad score instead of crashing
            return float('inf') if direction == 'minimize' else float('-inf')

    # Try to load the study, create a new one if it doesn't exist
    try:
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_name,
            direction=direction,
            load_if_exists=True,
            pruner=MedianPruner(
                n_startup_trials=5,  # Number of trials to run before pruning
                n_warmup_steps=10,   # Number of steps to wait before pruning
                interval_steps=1      # Interval between pruning checks
            ),
            sampler=TPESampler(seed=42)  # Use TPE sampler with fixed seed
        )
    except Exception as e:
        print(f"Error loading study: {e}")
        print("Creating a new study with fallback storage...")
        fallback_storage = f"sqlite:///optuna_{uuid.uuid4().hex[:8]}.db"
        study = optuna.create_study(
            study_name=study_name,
            storage=fallback_storage,
            direction=direction,
            pruner=MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=10,
                interval_steps=1
            ),
            sampler=TPESampler(seed=42)
        )

    # Study metadata tracking
    if not study.user_attrs.get('last_run_timestamp') or study.user_attrs.get('run_count', 0) == 0:
        study.set_user_attr('creation_timestamp', time.strftime('%Y-%m-%d %H:%M:%S'))
    study.set_user_attr('last_run_timestamp', time.strftime('%Y-%m-%d %H:%M:%S'))
    study.set_user_attr('run_count', study.user_attrs.get('run_count', 0) + 1)

    # Only run remaining trials
    completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    to_run = max(0, n_trials - completed)
    if to_run > 0:
        print(f"Running {to_run} new trials (target: {n_trials})...")
        # Optuna 4.x removed EarlyStoppingCallback. Run optimization without it.
        study.optimize(
            objective_with_error_handling,
            n_trials=to_run,
            timeout=timeout,
            n_jobs=-1,
            callbacks=None,
        )
    else:
        print(f"Study already has {completed} completed trials.")

    print(f"Optuna study finished in {time.time() - start_time:.2f} seconds.")
    print(f"Best trial: Value={study.best_value:.4f}, Params={study.best_params}")
    
    # Analyze hyperparameter importance
    analyze_hyperparameter_importance(study)
    
    # Visualize study results
    visualize_study(study)

    return study

def analyze_hyperparameter_importance(study):
    """Analyze and print hyperparameter importance."""
    try:
        importance = optuna.importance.get_param_importances(study)
        
        print("\nHyperparameter Importance:")
        for param, score in sorted(importance.items(), key=lambda x: x[1], reverse=True):
            print(f"  {param}: {score:.4f}")
        
        return importance
    except Exception as e:
        print(f"Could not analyze hyperparameter importance: {e}")
        return {}

def visualize_study(study, output_dir='results'):
    """Visualize Optuna study results and save to files."""
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot optimization history
        fig = vis.plot_optimization_history(study)
        try:
            fig.write_image(f"{output_dir}/{study.study_name}_optimization_history.png")
        except Exception as e:
            print(f"Could not save optimization history plot: {e}")
        
        # Plot parameter importances
        fig = vis.plot_param_importances(study)
        try:
            fig.write_image(f"{output_dir}/{study.study_name}_param_importances.png")
        except Exception as e:
            print(f"Could not save parameter importances plot: {e}")
        
        # Plot parallel coordinate
        fig = vis.plot_parallel_coordinate(study)
        try:
            fig.write_image(f"{output_dir}/{study.study_name}_parallel_coordinate.png")
        except Exception as e:
            print(f"Could not save parallel coordinate plot: {e}")
        
        # Plot slice
        fig = vis.plot_slice(study)
        try:
            fig.write_image(f"{output_dir}/{study.study_name}_slice.png")
        except Exception as e:
            print(f"Could not save slice plot: {e}")
        
        print(f"Study visualizations saved to {output_dir}/")
    except Exception as e:
        print(f"Could not visualize study: {e}")
