import numpy as np
import sys
import types
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Provide a minimal optuna.callbacks module so that pipeline imports succeed
callbacks = types.ModuleType("optuna.callbacks")
callbacks.EarlyStoppingCallback = object
sys.modules.setdefault("optuna.callbacks", callbacks)

# Stub matplotlib and matplotlib.pyplot which are imported by utils
mpl_mod = types.ModuleType("matplotlib")
plt_mod = types.ModuleType("matplotlib.pyplot")
mpl_mod.pyplot = plt_mod
sys.modules.setdefault("matplotlib", mpl_mod)
sys.modules.setdefault("matplotlib.pyplot", plt_mod)

# Stub shap to avoid heavy imports
shap_mod = types.ModuleType("shap")
sys.modules.setdefault("shap", shap_mod)

from pipeline_steps import aggregate_cv_results

def test_aggregate_cv_results_simple():
    reg_results = {"R2": [0.5, 0.7], "MAE": [1.0, 0.8]}
    cls_results = {"Accuracy": [0.9, 0.8], "ROC-AUC": [0.95, 0.85]}
    mean_r2, std_r2, mean_mae, std_mae, mean_acc, std_acc, mean_roc_auc, std_roc_auc = aggregate_cv_results(reg_results, cls_results)
    assert np.isclose(mean_r2, np.mean([0.5, 0.7]))
    assert np.isclose(std_r2, np.std([0.5, 0.7]))
    assert np.isclose(mean_mae, np.mean([1.0, 0.8]))
    assert np.isclose(std_mae, np.std([1.0, 0.8]))
    assert np.isclose(mean_acc, np.mean([0.9, 0.8]))
    assert np.isclose(std_acc, np.std([0.9, 0.8]))
    assert np.isclose(mean_roc_auc, np.mean([0.95, 0.85]))
    assert np.isclose(std_roc_auc, np.std([0.95, 0.85]))




