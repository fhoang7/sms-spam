#%% Imports
import pandas as pd
import numpy as np
import os
import json
import joblib
import optuna
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

#%% Configuration
print("="*70)
print("OPTUNA HYPERPARAMETER OPTIMIZATION - EMBEDDING ML")
print("="*70)

DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(DIR, 'data')
GRAPHS_DIR = os.path.join(DIR, 'graphs')

# Optuna configuration
N_TRIALS = 100
N_JOBS = -1  # Use all CPU cores
RANDOM_STATE = 42

print(f"\nConfiguration:")
print(f"  Trials: {N_TRIALS}")
print(f"  Optimization metric: Recall")
print(f"  Cross-validation: 5-fold")

#%% Load Data
print("\n" + "="*70)
print("LOADING DATA")
print("="*70)

# Load embeddings
X_train_embeddings = np.load(os.path.join(DB_PATH, 'train_embeddings.npy'))
X_test_embeddings = np.load(os.path.join(DB_PATH, 'test_embeddings.npy'))
y_train = np.load(os.path.join(DB_PATH, 'y_train.npy'))
y_test = np.load(os.path.join(DB_PATH, 'y_test.npy'))

print(f"OK Loaded training embeddings: {X_train_embeddings.shape}")
print(f"OK Loaded test embeddings: {X_test_embeddings.shape}")

# Convert to DataFrame
X_train_df = pd.DataFrame(X_train_embeddings)
X_test_df = pd.DataFrame(X_test_embeddings)

#%% Load Best Model Information
print("\n" + "="*70)
print("IDENTIFYING BEST MODEL")
print("="*70)

results_file = os.path.join(GRAPHS_DIR, 'embedding_ml_results.csv')
if not os.path.exists(results_file):
    print("ERROR: embedding_ml_results.csv not found")
    print("Please run embedding_classifier.py first")
    exit(1)

results = pd.read_csv(results_file, index_col=0)
best_model_name = results.sort_values('F1 Score', ascending=False).index[0]
best_baseline_f1 = results.loc[best_model_name, 'F1 Score']

print(f"\nBest baseline model: {best_model_name}")
print(f"  Baseline F1-Score: {best_baseline_f1:.4f}")

#%% Define Objective Function
print("\n" + "="*70)
print("SETTING UP OPTUNA OPTIMIZATION")
print("="*70)

def objective(trial):
    """Optuna objective function optimized for recall."""

    if best_model_name == 'SVC':
        from sklearn.svm import SVC

        params = {
            'C': trial.suggest_float('C', 1e-2, 100, log=True),
            'kernel': trial.suggest_categorical('kernel', ['rbf', 'poly', 'sigmoid']),
            'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
            'random_state': RANDOM_STATE
        }

        if params['kernel'] == 'poly':
            params['degree'] = trial.suggest_int('degree', 2, 5)

        model = SVC(**params)

    elif best_model_name == 'LogisticRegression':
        from sklearn.linear_model import LogisticRegression

        params = {
            'C': trial.suggest_float('C', 1e-3, 100, log=True),
            'penalty': trial.suggest_categorical('penalty', ['l1', 'l2']),
            'solver': trial.suggest_categorical('solver', ['liblinear', 'saga']),
            'max_iter': 1000,
            'random_state': RANDOM_STATE
        }

        if params['penalty'] == 'l1' and params['solver'] not in ['liblinear', 'saga']:
            params['solver'] = 'liblinear'

        model = LogisticRegression(**params)

    elif best_model_name == 'LinearDiscriminantAnalysis':
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

        params = {
            'solver': trial.suggest_categorical('solver', ['svd', 'lsqr', 'eigen']),
        }

        if params['solver'] in ['lsqr', 'eigen']:
            params['shrinkage'] = trial.suggest_categorical('shrinkage', [None, 'auto'])

        model = LinearDiscriminantAnalysis(**{k: v for k, v in params.items() if v is not None})

    elif best_model_name == 'ExtraTreesClassifier':
        from sklearn.ensemble import ExtraTreesClassifier

        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 10, 50),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'random_state': RANDOM_STATE,
            'n_jobs': N_JOBS
        }

        model = ExtraTreesClassifier(**params)

    elif best_model_name == 'RandomForestClassifier':
        from sklearn.ensemble import RandomForestClassifier

        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 10, 50),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'random_state': RANDOM_STATE,
            'n_jobs': N_JOBS
        }

        model = RandomForestClassifier(**params)

    elif best_model_name == 'GradientBoostingClassifier':
        from sklearn.ensemble import GradientBoostingClassifier

        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'random_state': RANDOM_STATE
        }

        model = GradientBoostingClassifier(**params)

    elif best_model_name == 'KNeighborsClassifier':
        from sklearn.neighbors import KNeighborsClassifier

        params = {
            'n_neighbors': trial.suggest_int('n_neighbors', 3, 15),
            'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
            'p': trial.suggest_int('p', 1, 2),
            'n_jobs': N_JOBS
        }

        model = KNeighborsClassifier(**params)

    else:
        # Fallback to LogisticRegression
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)

    # Cross-validation optimized for recall
    scores = cross_val_score(model, X_train_df, y_train, cv=5, scoring='recall', n_jobs=N_JOBS)

    return scores.mean()

#%% Run Optimization
print(f"\nStarting Optuna optimization with {N_TRIALS} trials...")
print("This may take several minutes...\n")

study = optuna.create_study(
    direction='maximize',
    study_name='embedding_ml_recall_optimization',
    sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE)
)

study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

print("\n" + "="*70)
print("OPTIMIZATION COMPLETE")
print("="*70)

print(f"\nBest trial:")
print(f"  Trial number: {study.best_trial.number}")
print(f"  Recall (CV): {study.best_trial.value:.4f}")
print(f"\nBest hyperparameters:")
for key, value in study.best_trial.params.items():
    print(f"  {key}: {value}")

#%% Train Final Model
print("\n" + "="*70)
print("TRAINING OPTIMIZED MODEL")
print("="*70)

# Train model with best parameters on full training set
if best_model_name == 'SVC':
    from sklearn.svm import SVC
    best_params = study.best_trial.params.copy()
    best_params['random_state'] = RANDOM_STATE
    optimized_model = SVC(**best_params)

elif best_model_name == 'LogisticRegression':
    from sklearn.linear_model import LogisticRegression
    best_params = study.best_trial.params.copy()
    best_params['max_iter'] = 1000
    best_params['random_state'] = RANDOM_STATE
    optimized_model = LogisticRegression(**best_params)

elif best_model_name == 'LinearDiscriminantAnalysis':
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    best_params = {k: v for k, v in study.best_trial.params.items() if v is not None}
    optimized_model = LinearDiscriminantAnalysis(**best_params)

elif best_model_name == 'ExtraTreesClassifier':
    from sklearn.ensemble import ExtraTreesClassifier
    best_params = study.best_trial.params.copy()
    best_params['random_state'] = RANDOM_STATE
    best_params['n_jobs'] = N_JOBS
    optimized_model = ExtraTreesClassifier(**best_params)

elif best_model_name == 'RandomForestClassifier':
    from sklearn.ensemble import RandomForestClassifier
    best_params = study.best_trial.params.copy()
    best_params['random_state'] = RANDOM_STATE
    best_params['n_jobs'] = N_JOBS
    optimized_model = RandomForestClassifier(**best_params)

elif best_model_name == 'GradientBoostingClassifier':
    from sklearn.ensemble import GradientBoostingClassifier
    best_params = study.best_trial.params.copy()
    best_params['random_state'] = RANDOM_STATE
    optimized_model = GradientBoostingClassifier(**best_params)

elif best_model_name == 'KNeighborsClassifier':
    from sklearn.neighbors import KNeighborsClassifier
    best_params = study.best_trial.params.copy()
    best_params['n_jobs'] = N_JOBS
    optimized_model = KNeighborsClassifier(**best_params)

else:
    from sklearn.linear_model import LogisticRegression
    optimized_model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)

print("Training optimized model on full training set...")
optimized_model.fit(X_train_df, y_train)
print("OK Model trained")

#%% Evaluate on Test Set
print("\n" + "="*70)
print("EVALUATING OPTIMIZED MODEL")
print("="*70)

y_pred = optimized_model.predict(X_test_df)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print(f"\nTest Set Performance:")
print(f"  Accuracy:  {accuracy:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  F1-Score:  {f1:.4f}")

print(f"\nComparison with baseline:")
print(f"  Baseline F1: {best_baseline_f1:.4f}")
print(f"  Optimized F1: {f1:.4f}")
print(f"  Improvement: {((f1 - best_baseline_f1) / best_baseline_f1 * 100):+.2f}%")

#%% Save Results
print("\n" + "="*70)
print("SAVING OPTIMIZED MODEL")
print("="*70)

# Save optimized model
optimized_model_file = os.path.join(DB_PATH, 'stage2_optimized_model.pkl')
joblib.dump(optimized_model, optimized_model_file)
print(f"OK Saved optimized model to {optimized_model_file}")

# Save optimization results
optimization_results = {
    'model_name': best_model_name,
    'baseline_f1': float(best_baseline_f1),
    'optimized_metrics': {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1)
    },
    'best_trial': {
        'number': study.best_trial.number,
        'cv_recall': float(study.best_trial.value),
        'params': study.best_trial.params
    },
    'optimization_config': {
        'n_trials': N_TRIALS,
        'metric': 'recall',
        'cv_folds': 5
    }
}

results_file = os.path.join(GRAPHS_DIR, 'embedding_optimization_results.json')
with open(results_file, 'w') as f:
    json.dump(optimization_results, f, indent=2)
print(f"OK Saved optimization results to {results_file}")

# Save Optuna study
study_file = os.path.join(DB_PATH, 'embedding_optuna_study.pkl')
joblib.dump(study, study_file)
print(f"OK Saved Optuna study to {study_file}")

print("\n" + "="*70)
print("EMBEDDING ML OPTIMIZATION COMPLETE")
print("="*70)
print(f"\nOptimized {best_model_name} for recall")
print(f"  Test Recall: {recall:.4f}")
print(f"  Test F1-Score: {f1:.4f}")
print(f"  Trials completed: {N_TRIALS}")
print("\n" + "="*70 + "\n")

# %%
