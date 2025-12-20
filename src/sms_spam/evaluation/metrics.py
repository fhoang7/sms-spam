"""Evaluation metrics for spam classification."""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score,
    classification_report, confusion_matrix,
    precision_recall_curve
)
from ..utils.constants import CLASS_NAMES


def evaluate_classifier(y_true, y_pred, y_proba=None, model_name="Model"):
    """
    Comprehensive evaluation metrics for binary classification.

    Args:
        y_true (array-like): Ground truth labels (0=ham, 1=spam)
        y_pred (array-like): Predicted labels
        y_proba (array-like, optional): Prediction probabilities for class 1
        model_name (str): Name for display

    Returns:
        dict: Evaluation metrics including accuracy, precision, recall, F1, ROC AUC
    """
    results = {
        'model': model_name,
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0)
    }

    if y_proba is not None:
        try:
            results['roc_auc'] = roc_auc_score(y_true, y_proba)
            results['avg_precision'] = average_precision_score(y_true, y_proba)
        except ValueError:
            # Handle cases where ROC AUC cannot be computed
            results['roc_auc'] = None
            results['avg_precision'] = None

    return results


def print_evaluation(y_true, y_pred, target_names=None, title=None):
    """
    Print formatted classification report and confusion matrix.

    Args:
        y_true (array-like): Ground truth labels
        y_pred (array-like): Predicted labels
        target_names (list, optional): Class names for display
        title (str, optional): Title to display above report
    """
    if target_names is None:
        target_names = CLASS_NAMES

    if title:
        print(f"\n{'='*70}")
        print(f"{title:^70}")
        print('='*70)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(f"{'':15} Predicted Ham  Predicted Spam")
    print(f"Actual Ham         {cm[0,0]:>6}          {cm[0,1]:>6}")
    print(f"Actual Spam        {cm[1,0]:>6}          {cm[1,1]:>6}")
    print('='*70 + "\n")


def find_optimal_threshold(y_true, y_proba, metric='f1'):
    """
    Find optimal classification threshold.

    Args:
        y_true (array-like): True labels
        y_proba (array-like): Prediction probabilities
        metric (str): Metric to optimize ('f1', 'precision', 'recall')

    Returns:
        tuple: (optimal_threshold, best_score)
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)

    if metric == 'f1':
        # Calculate F1-scores for each threshold
        scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    elif metric == 'precision':
        scores = precisions
    elif metric == 'recall':
        scores = recalls
    else:
        raise ValueError(f"Unknown metric: {metric}")

    # Find threshold with maximum score
    best_idx = np.argmax(scores)
    optimal_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    best_score = scores[best_idx]

    return optimal_threshold, best_score


def calculate_spam_recall(y_true, y_pred):
    """
    Calculate recall for spam class specifically.

    This is critical for spam detection where missing spam (false negatives)
    is more costly than false positives.

    Args:
        y_true (array-like): True labels (0=ham, 1=spam)
        y_pred (array-like): Predicted labels

    Returns:
        float: Spam recall (0.0 to 1.0)
    """
    return recall_score(y_true, y_pred, pos_label=1, zero_division=0)


def calculate_ham_precision(y_true, y_pred):
    """
    Calculate precision for ham class specifically.

    Args:
        y_true (array-like): True labels (0=ham, 1=spam)
        y_pred (array-like): Predicted labels

    Returns:
        float: Ham precision (0.0 to 1.0)
    """
    return precision_score(y_true, y_pred, pos_label=0, zero_division=0)
