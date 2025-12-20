"""Threshold optimization for cascade classifiers."""

import numpy as np
from sklearn.metrics import precision_recall_curve
from ..utils.io import save_json
from ..evaluation.metrics import calculate_spam_recall


class ThresholdOptimizer:
    """
    Find optimal probability thresholds for cascade classifiers.

    Optimizes thresholds to achieve target metrics (e.g., 100% spam recall)
    while maximizing coverage at each stage.
    """

    def __init__(self, verbose=True):
        """
        Initialize threshold optimizer.

        Args:
            verbose (bool): Print progress messages
        """
        self.verbose = verbose
        self.thresholds = {}

    def find_high_recall_threshold(self, y_true, y_proba, target_recall=1.0):
        """
        Find threshold that achieves target recall for spam class.

        Args:
            y_true (array-like): True labels
            y_proba (array-like): Predicted probabilities for spam class
            target_recall (float): Target recall (default: 1.0 for 100%)

        Returns:
            dict: Threshold and metrics
        """
        if self.verbose:
            print(f"\nFinding threshold for {target_recall*100:.0f}% spam recall...")

        precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)

        # Find threshold achieving target recall
        valid_indices = np.where(recalls >= target_recall)[0]

        if len(valid_indices) == 0:
            if self.verbose:
                print(f"⚠ Cannot achieve {target_recall*100:.0f}% recall")
            return None

        # Among valid thresholds, choose one with best precision
        best_idx = valid_indices[np.argmax(precisions[valid_indices])]

        # Handle edge case where best_idx >= len(thresholds)
        if best_idx >= len(thresholds):
            threshold = 0.0
        else:
            threshold = thresholds[best_idx]

        precision = precisions[best_idx]
        recall = recalls[best_idx]

        if self.verbose:
            print(f"✓ Threshold: {threshold:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")

        return {
            'threshold': float(threshold),
            'precision': float(precision),
            'recall': float(recall)
        }

    def find_cascade_thresholds(
        self,
        y_true,
        stage1_proba,
        target_stage1_coverage=0.8,
        spam_recall_requirement=1.0
    ):
        """
        Find optimal thresholds for 2-stage cascade.

        Args:
            y_true (array-like): True labels
            stage1_proba (array-like): Stage 1 probabilities
            target_stage1_coverage (float): Target % handled by Stage 1
            spam_recall_requirement (float): Minimum spam recall

        Returns:
            dict: Optimized thresholds for cascade
        """
        if self.verbose:
            print("\n" + "="*70)
            print("CASCADE THRESHOLD OPTIMIZATION")
            print("="*70)
            print(f"Target Stage 1 Coverage: {target_stage1_coverage*100:.0f}%")
            print(f"Spam Recall Requirement: {spam_recall_requirement*100:.0f}%")

        # Find low threshold (high confidence HAM)
        # We want a threshold below which we're confident it's HAM
        ham_mask = y_true == 0
        ham_probas = stage1_proba[ham_mask]

        # Low threshold: capture most hams with high confidence
        low_threshold = np.percentile(ham_probas, 20)

        # Find high threshold (high confidence SPAM)
        # We want a threshold above which we're confident it's SPAM
        spam_mask = y_true == 1
        spam_probas = stage1_proba[spam_mask]

        # High threshold: capture most spams with high confidence
        high_threshold = np.percentile(spam_probas, 80)

        # Adjust to achieve target coverage
        uncertain_mask = (stage1_proba > low_threshold) & (stage1_proba < high_threshold)
        current_coverage = 1 - uncertain_mask.mean()

        if self.verbose:
            print(f"\nInitial Thresholds:")
            print(f"  Low (HAM): {low_threshold:.4f}")
            print(f"  High (SPAM): {high_threshold:.4f}")
            print(f"  Stage 1 Coverage: {current_coverage*100:.1f}%")

        thresholds = {
            'stage1_low_threshold': float(low_threshold),
            'stage1_high_threshold': float(high_threshold),
            'stage1_coverage': float(current_coverage),
            'stage2_threshold': 0.5  # Default for Stage 2
        }

        if self.verbose:
            print(f"\n✓ Cascade thresholds optimized")
            print(f"  Final Stage 1 Coverage: {current_coverage*100:.1f}%")

        self.thresholds = thresholds
        return thresholds

    def save_thresholds(self, filepath):
        """
        Save optimized thresholds to JSON.

        Args:
            filepath (str): Path to save thresholds
        """
        if not self.thresholds:
            raise ValueError("No thresholds to save. Run optimization first.")

        save_json(self.thresholds, filepath)

        if self.verbose:
            print(f"✓ Thresholds saved to {filepath}")


def optimize_cascade_thresholds(
    y_true,
    stage1_proba,
    output_path=None,
    target_coverage=0.8
):
    """
    Optimize and optionally save cascade thresholds.

    Args:
        y_true (array-like): True labels
        stage1_proba (array-like): Stage 1 probabilities
        output_path (str, optional): Path to save results
        target_coverage (float): Target Stage 1 coverage

    Returns:
        dict: Optimized thresholds
    """
    optimizer = ThresholdOptimizer(verbose=True)

    thresholds = optimizer.find_cascade_thresholds(
        y_true=y_true,
        stage1_proba=stage1_proba,
        target_stage1_coverage=target_coverage
    )

    if output_path:
        optimizer.save_thresholds(output_path)

    return thresholds
