"""Embedding-based ML classifiers."""

import pandas as pd
import numpy as np
from lazypredict.Supervised import LazyClassifier
from sklearn.linear_model import LogisticRegression
from .base import BaseClassifier
from ..evaluation.metrics import evaluate_classifier, print_evaluation


class EmbeddingClassifier(BaseClassifier):
    """
    Classifier using sentence transformer embeddings with traditional ML models.
    """

    def __init__(self, model=None, model_name="LogisticRegression"):
        """
        Initialize embedding-based classifier.

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
            X: Embedding features (numpy array or DataFrame)
            y: Labels (0=ham, 1=spam)

        Returns:
            self
        """
        print(f"\nTraining {self.model_name} on embeddings...")

        self.model.fit(X, y)
        self.is_fitted = True

        print(f"✓ {self.model_name} trained successfully")
        return self

    def predict(self, X):
        """
        Predict class labels.

        Args:
            X: Embedding features

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
            X: Embedding features

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

        print_evaluation(y_test, y_pred, title=f"{self.model_name} (Embeddings) Evaluation")

        return evaluate_classifier(y_test, y_pred, y_proba, self.model_name)


def run_lazypredict_on_embeddings(train_embeddings, test_embeddings, y_train, y_test, verbose=True):
    """
    Run LazyPredict to compare multiple ML models on embeddings.

    Args:
        train_embeddings: Training embeddings (numpy array)
        test_embeddings: Test embeddings (numpy array)
        y_train: Training labels
        y_test: Test labels
        verbose (bool): Print progress

    Returns:
        tuple: (models_df, predictions_dict)
    """
    if verbose:
        print("\n" + "="*70)
        print("LAZYPREDICT - EMBEDDING-BASED MODEL COMPARISON")
        print("="*70)
        print("\nConverting embeddings to DataFrames for LazyPredict...")

    # Convert to DataFrame
    X_train_df = pd.DataFrame(train_embeddings)
    X_test_df = pd.DataFrame(test_embeddings)

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
