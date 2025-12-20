"""Traditional ML classifiers using TF-IDF features."""

import pandas as pd
import numpy as np
from lazypredict.Supervised import LazyClassifier
from sklearn.linear_model import LogisticRegression
from .base import BaseClassifier
from ..evaluation.metrics import evaluate_classifier, print_evaluation


class TraditionalMLClassifier(BaseClassifier):
    """
    Classifier using TF-IDF features with traditional ML models.
    """

    def __init__(self, model=None, model_name="LogisticRegression"):
        """
        Initialize traditional ML classifier.

        Args:
            model: sklearn model instance (if None, creates LogisticRegression)
            model_name (str): Name of the model
        """
        super().__init__(model_name)

        if model is None:
            # Default to LogisticRegression
            self.model = LogisticRegression(
                max_iter=1000,
                random_state=42,
                class_weight='balanced'
            )
        else:
            self.model = model

    def fit(self, X, y):
        """
        Train the classifier.

        Args:
            X: TF-IDF features (sparse matrix or DataFrame)
            y: Labels (0=ham, 1=spam)

        Returns:
            self
        """
        print(f"\nTraining {self.model_name}...")

        self.model.fit(X, y)
        self.is_fitted = True

        print(f"✓ {self.model_name} trained successfully")
        return self

    def predict(self, X):
        """
        Predict class labels.

        Args:
            X: TF-IDF features

        Returns:
            np.ndarray: Predicted labels
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        return self.model.predict(X)

    def predict_proba(self, X):
        """
        Predict class probabilities.

        Args:
            X: TF-IDF features

        Returns:
            np.ndarray: Predicted probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        return self.model.predict_proba(X)[:, 1]  # Return probability of spam class

    def evaluate(self, X_test, y_test):
        """
        Evaluate the classifier.

        Args:
            X_test: Test features
            y_test: True labels

        Returns:
            dict: Evaluation metrics
        """
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)

        print_evaluation(y_test, y_pred, title=f"{self.model_name} Evaluation")

        return evaluate_classifier(y_test, y_pred, y_proba, self.model_name)


def run_lazypredict_comparison(X_train, X_test, y_train, y_test, verbose=True):
    """
    Run LazyPredict to compare multiple traditional ML models.

    Args:
        X_train: Training TF-IDF features (sparse matrix)
        X_test: Test TF-IDF features (sparse matrix)
        y_train: Training labels
        y_test: Test labels
        verbose (bool): Print progress

    Returns:
        tuple: (models_df, predictions_dict)
            - models_df: DataFrame with model comparison results
            - predictions_dict: Dictionary of predictions for each model
    """
    if verbose:
        print("\n" + "="*70)
        print("LAZYPREDICT - BASELINE MODEL COMPARISON")
        print("="*70)
        print("\nConverting sparse matrices to DataFrames for LazyPredict...")

    # LazyPredict requires DataFrame input
    X_train_df = pd.DataFrame(X_train.toarray() if hasattr(X_train, 'toarray') else X_train)
    X_test_df = pd.DataFrame(X_test.toarray() if hasattr(X_test, 'toarray') else X_test)

    if verbose:
        print(f"✓ Converted to DataFrames: {X_train_df.shape}, {X_test_df.shape}")
        print("\nTesting 30+ classification models...")
        print("(This may take a few minutes...)\n")

    # Run LazyPredict
    clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
    models, predictions = clf.fit(X_train_df, X_test_df, y_train, y_test)

    # Calculate additional metrics
    if verbose:
        print("\nCalculating Precision and Recall for all models...")

    from sklearn.metrics import precision_score, recall_score

    precision_scores = []
    recall_scores = []

    for model_name in models.index:
        y_pred = predictions[model_name]
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        precision_scores.append(precision)
        recall_scores.append(recall)

    models['Precision'] = precision_scores
    models['Recall'] = recall_scores

    # Sort by F1 Score
    models = models.sort_values('F1 Score', ascending=False)

    if verbose:
        print(f"\n✓ Tested {len(models)} models")
        print(f"\nTop 5 Models (by F1 Score):")
        print(models[['Accuracy', 'Precision', 'Recall', 'F1 Score']].head())

    return models, predictions


def get_best_model_from_lazypredict(models_df, metric='F1 Score'):
    """
    Get the best performing model from LazyPredict results.

    Args:
        models_df (pd.DataFrame): LazyPredict results
        metric (str): Metric to use for selection

    Returns:
        str: Name of best model
    """
    best_model_name = models_df[metric].idxmax()
    best_score = models_df.loc[best_model_name, metric]

    print(f"\nBest Model: {best_model_name}")
    print(f"{metric}: {best_score:.4f}")

    return best_model_name


def create_model_from_name(model_name, **kwargs):
    """
    Create a sklearn model instance from its name.

    Args:
        model_name (str): Name of the model
        **kwargs: Additional parameters for the model

    Returns:
        sklearn model instance
    """
    from sklearn.linear_model import LogisticRegression, SGDClassifier
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.ensemble import (
        RandomForestClassifier, ExtraTreesClassifier,
        GradientBoostingClassifier, AdaBoostClassifier
    )
    from sklearn.naive_bayes import MultinomialNB, BernoulliNB
    from sklearn.svm import LinearSVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier

    model_map = {
        'LogisticRegression': LogisticRegression,
        'LinearDiscriminantAnalysis': LinearDiscriminantAnalysis,
        'SGDClassifier': SGDClassifier,
        'RandomForestClassifier': RandomForestClassifier,
        'ExtraTreesClassifier': ExtraTreesClassifier,
        'GradientBoostingClassifier': GradientBoostingClassifier,
        'AdaBoostClassifier': AdaBoostClassifier,
        'MultinomialNB': MultinomialNB,
        'BernoulliNB': BernoulliNB,
        'LinearSVC': LinearSVC,
        'DecisionTreeClassifier': DecisionTreeClassifier,
        'KNeighborsClassifier': KNeighborsClassifier,
    }

    if model_name not in model_map:
        raise ValueError(f"Unknown model: {model_name}")

    # Set some sensible defaults
    if 'random_state' not in kwargs and hasattr(model_map[model_name](), 'random_state'):
        kwargs['random_state'] = 42

    return model_map[model_name](**kwargs)
