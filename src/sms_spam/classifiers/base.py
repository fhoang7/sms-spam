"""Base classifier interface for SMS spam detection."""

from abc import ABC, abstractmethod
from ..utils.io import save_model, load_model


class BaseClassifier(ABC):
    """
    Abstract base class for all spam classifiers.

    Defines the interface that all classifiers must implement.
    """

    def __init__(self, model_name="BaseClassifier"):
        """
        Initialize the classifier.

        Args:
            model_name (str): Name of the classifier
        """
        self.model_name = model_name
        self.model = None
        self.is_fitted = False

    @abstractmethod
    def fit(self, X, y):
        """
        Train the classifier.

        Args:
            X: Training features
            y: Training labels

        Returns:
            self: The fitted classifier
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        Predict class labels.

        Args:
            X: Features to predict

        Returns:
            np.ndarray: Predicted labels (0=ham, 1=spam)
        """
        pass

    @abstractmethod
    def predict_proba(self, X):
        """
        Predict class probabilities.

        Args:
            X: Features to predict

        Returns:
            np.ndarray: Predicted probabilities for each class
        """
        pass

    def save(self, filepath):
        """
        Save the fitted model to disk.

        Args:
            filepath (str): Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model. Call fit() first.")

        save_model(self.model, filepath)

    def load(self, filepath):
        """
        Load a fitted model from disk.

        Args:
            filepath (str): Path to the saved model

        Returns:
            self: The classifier with loaded model
        """
        self.model = load_model(filepath)
        self.is_fitted = True
        return self

    def __repr__(self):
        """String representation."""
        status = "fitted" if self.is_fitted else "not fitted"
        return f"{self.model_name} ({status})"
