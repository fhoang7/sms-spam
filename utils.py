"""Utility functions for SMS spam detection project"""

import re
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score,
    classification_report, confusion_matrix
)


def preprocess_text(text):
    """
    Clean and normalize SMS text.

    Args:
        text (str): Raw SMS message

    Returns:
        str: Cleaned message

    Note:
        - Preserves case for features like 'FREE' vs 'free'
        - Does NOT remove stopwords (they're signals for spam)
        - Keeps basic structure for semantic meaning
    """
    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)

    # Remove special characters (keep alphanumeric and spaces)
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


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
        results['roc_auc'] = roc_auc_score(y_true, y_proba)
        results['avg_precision'] = average_precision_score(y_true, y_proba)

    return results


def print_evaluation(y_true, y_pred, target_names=['ham', 'spam'], title=None):
    """
    Print formatted classification report and confusion matrix.

    Args:
        y_true (array-like): Ground truth labels
        y_pred (array-like): Predicted labels
        target_names (list): Class names for display
        title (str, optional): Title to display above report
    """
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


def get_message_stats(messages, labels=None):
    """
    Calculate message length statistics.

    Args:
        messages (pd.Series): SMS messages
        labels (pd.Series, optional): Labels for class-wise statistics

    Returns:
        pd.DataFrame: Statistics including char count, word count by class
    """
    char_counts = messages.str.len()
    word_counts = messages.str.split().str.len()

    if labels is None:
        stats = {
            'char_mean': char_counts.mean(),
            'char_std': char_counts.std(),
            'char_median': char_counts.median(),
            'char_max': char_counts.max(),
            'word_mean': word_counts.mean(),
            'word_std': word_counts.std(),
            'word_median': word_counts.median(),
            'word_max': word_counts.max()
        }
        return pd.DataFrame([stats])
    else:
        # Calculate stats by class
        df = pd.DataFrame({
            'message': messages,
            'label': labels,
            'char_count': char_counts,
            'word_count': word_counts
        })

        stats = df.groupby('label').agg({
            'char_count': ['mean', 'std', 'median', 'max'],
            'word_count': ['mean', 'std', 'median', 'max']
        }).round(2)

        return stats


def display_sample_messages(data, n_samples=5):
    """
    Display sample messages from each class.

    Args:
        data (pd.DataFrame): DataFrame with 'label' and 'message' columns
        n_samples (int): Number of samples per class
    """
    print("\n" + "="*70)
    print("SAMPLE MESSAGES")
    print("="*70)

    for label in data['label'].unique():
        print(f"\n{label.upper()} Messages:")
        print("-" * 70)
        samples = data[data['label'] == label].sample(min(n_samples, sum(data['label'] == label)))
        for idx, (_, row) in enumerate(samples.iterrows(), 1):
            msg = row['message']
            if len(msg) > 100:
                msg = msg[:97] + "..."
            print(f"{idx}. {msg}")

    print("="*70)


def find_optimal_threshold(y_true, y_proba):
    """
    Find optimal classification threshold using F1-score.

    Args:
        y_true (array-like): True labels
        y_proba (array-like): Prediction probabilities

    Returns:
        tuple: (optimal_threshold, best_f1_score)
    """
    from sklearn.metrics import precision_recall_curve

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)

    # Calculate F1-scores for each threshold
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)

    # Find threshold with maximum F1
    best_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    best_f1 = f1_scores[best_idx]

    return optimal_threshold, best_f1
