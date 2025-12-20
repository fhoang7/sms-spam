"""Unified Optuna hyperparameter optimization for traditional and embedding classifiers."""

import optuna
import numpy as np
from sklearn.model_selection import cross_val_score
from ..utils.constants import OPTUNA_N_TRIALS, OPTUNA_METRIC, RANDOM_STATE
from ..utils.io import save_model, save_json
import os


class OptunaOptimizer:
    """
    Unified Optuna hyperparameter optimizer for spam classifiers.

    Supports both TF-IDF-based and embedding-based classifiers.
    """

    def __init__(
        self,
        model_class_name,
        n_trials=OPTUNA_N_TRIALS,
        metric=OPTUNA_METRIC,
        random_state=RANDOM_STATE,
        verbose=True
    ):
        """
        Initialize Optuna optimizer.

        Args:
            model_class_name (str): Name of model class to optimize
            n_trials (int): Number of optimization trials
            metric (str): Metric to optimize ('recall', 'f1', 'precision')
            random_state (int): Random seed
            verbose (bool): Print progress
        """
        self.model_class_name = model_class_name
        self.n_trials = n_trials
        self.metric = metric
        self.random_state = random_state
        self.verbose = verbose

        self.best_params = None
        self.best_score = None
        self.study = None

    def _get_search_space(self, trial):
        """
        Get hyperparameter search space for the model.

        Args:
            trial: Optuna trial object

        Returns:
            dict: Hyperparameters for this trial
        """
        model_name = self.model_class_name

        # Common parameters
        params = {'random_state': self.random_state}

        if model_name == 'LogisticRegression':
            params.update({
                'C': trial.suggest_float('C', 0.01, 100, log=True),
                'penalty': trial.suggest_categorical('penalty', ['l1', 'l2']),
                'solver': trial.suggest_categorical('solver', ['liblinear', 'saga']),
                'max_iter': 1000,
                'class_weight': 'balanced'
            })

        elif model_name == 'RandomForestClassifier':
            params.update({
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'class_weight': 'balanced'
            })

        elif model_name == 'GradientBoostingClassifier':
            params.update({
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0)
            })

        elif model_name == 'ExtraTreesClassifier':
            params.update({
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'class_weight': 'balanced'
            })

        else:
            # Default: minimal parameters
            if self.verbose:
                print(f"⚠ No specific search space for {model_name}, using defaults")

        return params

    def optimize(self, X_train, y_train, cv=5):
        """
        Run hyperparameter optimization.

        Args:
            X_train: Training features
            y_train: Training labels
            cv (int): Cross-validation folds

        Returns:
            dict: Best parameters and study results
        """
        from ..classifiers.traditional import create_model_from_name

        if self.verbose:
            print(f"\n{'='*70}")
            print(f"OPTUNA OPTIMIZATION - {self.model_class_name}")
            print(f"{'='*70}")
            print(f"Metric: {self.metric}")
            print(f"Trials: {self.n_trials}")
            print(f"CV Folds: {cv}\n")

        def objective(trial):
            # Get hyperparameters
            params = self._get_search_space(trial)

            # Create model
            try:
                model = create_model_from_name(self.model_class_name, **params)
            except Exception as e:
                if self.verbose:
                    print(f"Failed to create model: {e}")
                return 0.0

            # Cross-validation
            try:
                scores = cross_val_score(
                    model, X_train, y_train,
                    cv=cv,
                    scoring=self.metric,
                    n_jobs=-1
                )
                return scores.mean()
            except Exception as e:
                if self.verbose:
                    print(f"Trial failed: {e}")
                return 0.0

        # Create and run study
        self.study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=self.random_state)
        )

        self.study.optimize(
            objective,
            n_trials=self.n_trials,
            show_progress_bar=self.verbose
        )

        self.best_params = self.study.best_params
        self.best_score = self.study.best_value

        if self.verbose:
            print(f"\n✓ Optimization complete!")
            print(f"Best {self.metric}: {self.best_score:.4f}")
            print(f"Best parameters: {self.best_params}")

        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'n_trials': len(self.study.trials)
        }

    def get_best_model(self):
        """
        Create model instance with best parameters.

        Returns:
            sklearn model with optimized parameters
        """
        if self.best_params is None:
            raise ValueError("No optimization results. Run optimize() first.")

        from ..classifiers.traditional import create_model_from_name

        params = self.best_params.copy()
        params['random_state'] = self.random_state

        model = create_model_from_name(self.model_class_name, **params)

        if self.verbose:
            print(f"✓ Created optimized {self.model_class_name}")

        return model


def optimize_and_save_model(
    X_train,
    y_train,
    model_class_name,
    output_model_path,
    output_results_path=None,
    n_trials=OPTUNA_N_TRIALS,
    metric=OPTUNA_METRIC
):
    """
    Optimize model and save results.

    Args:
        X_train: Training features
        y_train: Training labels
        model_class_name (str): Model class name
        output_model_path (str): Path to save optimized model
        output_results_path (str, optional): Path to save optimization results
        n_trials (int): Number of trials
        metric (str): Metric to optimize

    Returns:
        tuple: (optimized_model, results_dict)
    """
    # Run optimization
    optimizer = OptunaOptimizer(
        model_class_name=model_class_name,
        n_trials=n_trials,
        metric=metric,
        verbose=True
    )

    results = optimizer.optimize(X_train, y_train)

    # Get and train best model
    model = optimizer.get_best_model()
    model.fit(X_train, y_train)

    # Save model
    save_model(model, output_model_path)

    # Save results
    if output_results_path:
        save_json(results, output_results_path)

    return model, results
