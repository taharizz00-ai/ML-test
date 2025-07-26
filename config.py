"""
Configuration settings for the ML pipeline.
"""

# Data and General Settings
DATA_FILE = "synthetic_data.csv" # Updated to use synthetic data. ### CHANGE BACK TO "A2BBO6_matched.csv" FOR REAL DATA ###
TEST_SIZE = 0.2  # Proportion of data held out for the final unseen test set
N_SPLITS_OUTER_CV = 2 # Number of folds for outer cross-validation. ### CHANGE BACK TO 5 for real data ###
RANDOM_STATE = 42   # Set random seed for reproducibility

# Feature Selection ('lgbm', 'kbest_f_both', 'kbest_mutual_info', 'none')
# Note: kbest methods require setting 'K_BEST_FEATURES'
FEATURE_SELECTION_METHOD = 'lgbm'
K_BEST_FEATURES = 50 # Number of features for SelectKBest

# === Stacking Configuration ===
STACKING_CV_FOLDS = 5 # Number of folds for inner CV within stacking meta-learner training

# Set to True to tune all base models, False to only tune LGBM/XGB
TUNE_ALL_BASE_MODELS = True # Set this according to preference

# === Optuna Configuration ===
OPTUNA_TRIALS_MAIN = 20  # Number of trials for key models (LGBM/XGB)
OPTUNA_TRIALS_OTHER = 10 # Number of trials for other base models
OPTUNA_TIMEOUT = None    # Timeout in seconds for each Optuna study (None for no timeout)
LGBM_DEVICE = "cpu"

# === Meta-Model Configuration ===
# Define potential meta-models (final estimator in stacking)
# These are NOT tuned by Optuna in this setup, but could be
# Note: Ensure GPU params are set if needed (e.g., device='gpu' for LGBM)

# Reduced for faster testing on synthetic data
SHAP_BACKGROUND_SAMPLES = 50
SHAP_EXPLAIN_SAMPLES = 20

# Output Directory
OUTPUT_DIR = 'results'
SAVE_MODELS = True # Save the best model from each fold and the final test models
SAVE_FEATURES = True
SAVE_RESULTS = True

# Stratification
STRATIFY_SPLIT = False # Set to True for real data if appropriate



# Removed imports for Optuna Objective Functions and BASE_REGRESSORS/BASE_CLASSIFIERS definitions.
# These should be handled within the main/test scripts where they are used.