"""2-Stage Hybrid Cascade Classifier (TF-IDF → Embeddings)."""

import numpy as np
from .base import BaseClassifier
from ..utils.constants import (
    CASCADE_STAGE1_LOW_THRESHOLD,
    CASCADE_STAGE1_HIGH_THRESHOLD,
    CASCADE_STAGE2_THRESHOLD
)
from ..evaluation.metrics import evaluate_classifier, print_evaluation


class HybridCascadeClassifier(BaseClassifier):
    """
    2-Stage Cascade Classifier for spam detection.

    Stage 1: Fast TF-IDF-based classification (handles 80-90% of messages)
        - High confidence spam (>= high_threshold) → SPAM
        - High confidence ham (<= low_threshold) → HAM
        - Uncertain cases → Send to Stage 2

    Stage 2: Embedding-based classification (handles remaining 10-20%)
        - Uses semantic embeddings for final decision
    """

    def __init__(
        self,
        stage1_model,
        stage2_model,
        stage1_low_threshold=CASCADE_STAGE1_LOW_THRESHOLD,
        stage1_high_threshold=CASCADE_STAGE1_HIGH_THRESHOLD,
        stage2_threshold=CASCADE_STAGE2_THRESHOLD,
        verbose=True
    ):
        """
        Initialize 2-stage cascade classifier.

        Args:
            stage1_model: Fitted TF-IDF-based classifier
            stage2_model: Fitted embedding-based classifier
            stage1_low_threshold (float): Below this → HAM (Stage 1)
            stage1_high_threshold (float): Above this → SPAM (Stage 1)
            stage2_threshold (float): Decision threshold for Stage 2
            verbose (bool): Print progress messages
        """
        super().__init__("HybridCascade")

        self.stage1 = stage1_model
        self.stage2 = stage2_model

        self.stage1_low_threshold = stage1_low_threshold
        self.stage1_high_threshold = stage1_high_threshold
        self.stage2_threshold = stage2_threshold

        self.verbose = verbose
        self.is_fitted = True  # Models are already fitted

        if verbose:
            print("\n2-Stage Hybrid Cascade Classifier Initialized")
            print(f"  Stage 1 Low Threshold (HAM): {stage1_low_threshold}")
            print(f"  Stage 1 High Threshold (SPAM): {stage1_high_threshold}")
            print(f"  Stage 2 Threshold: {stage2_threshold}")

    def fit(self, X, y):
        """
        Cascade uses pre-fitted models.

        Args:
            X: Features (not used)
            y: Labels (not used)

        Returns:
            self
        """
        print("ℹ Cascade uses pre-fitted stage models. No fitting needed.")
        return self

    def predict(self, X_stage1, X_stage2):
        """
        Predict using 2-stage cascade.

        Args:
            X_stage1: TF-IDF features for Stage 1
            X_stage2: Embedding features for Stage 2

        Returns:
            np.ndarray: Final predictions
        """
        results = self.predict_with_metadata(X_stage1, X_stage2)
        return results['predictions']

    def predict_proba(self, X_stage1, X_stage2):
        """
        Get final probabilities from cascade.

        Args:
            X_stage1: TF-IDF features
            X_stage2: Embedding features

        Returns:
            np.ndarray: Final probabilities (spam class)
        """
        results = self.predict_with_metadata(X_stage1, X_stage2)

        # Return final probabilities
        final_probas = np.zeros(len(results['metadata']))
        for i, meta in enumerate(results['metadata']):
            if meta['stage_used'] == 1:
                final_probas[i] = meta['stage1_proba']
            else:
                final_probas[i] = meta['stage2_proba']

        return final_probas

    def predict_with_metadata(self, X_stage1, X_stage2):
        """
        Predict with detailed metadata about which stage was used.

        Args:
            X_stage1: TF-IDF features
            X_stage2: Embedding features

        Returns:
            dict: {
                'predictions': np.ndarray of predictions,
                'metadata': list of dicts with decision details,
                'stage_stats': dict with stage usage statistics
            }
        """
        n_samples = X_stage1.shape[0]

        if self.verbose:
            print(f"\nProcessing {n_samples} samples through 2-stage cascade...")

        # Stage 1: TF-IDF predictions
        stage1_probas = self.stage1.predict_proba(X_stage1)

        predictions = np.zeros(n_samples, dtype=int)
        metadata = []

        stage1_count = 0
        stage2_count = 0

        for i in range(n_samples):
            stage1_proba = stage1_probas[i]

            # High confidence decisions from Stage 1
            if stage1_proba <= self.stage1_low_threshold:
                # High confidence HAM
                predictions[i] = 0
                stage1_count += 1
                metadata.append({
                    'prediction': 0,
                    'confidence': 1 - stage1_proba,
                    'stage_used': 1,
                    'stage1_proba': stage1_proba,
                    'stage2_proba': None
                })

            elif stage1_proba >= self.stage1_high_threshold:
                # High confidence SPAM
                predictions[i] = 1
                stage1_count += 1
                metadata.append({
                    'prediction': 1,
                    'confidence': stage1_proba,
                    'stage_used': 1,
                    'stage1_proba': stage1_proba,
                    'stage2_proba': None
                })

            else:
                # Uncertain → Send to Stage 2
                stage2_proba = self.stage2.predict_proba(X_stage2[i:i+1])[0]
                predictions[i] = 1 if stage2_proba >= self.stage2_threshold else 0
                stage2_count += 1
                metadata.append({
                    'prediction': int(predictions[i]),
                    'confidence': stage2_proba if predictions[i] == 1 else (1 - stage2_proba),
                    'stage_used': 2,
                    'stage1_proba': stage1_proba,
                    'stage2_proba': stage2_proba
                })

        stage_stats = {
            'total': n_samples,
            'stage1_count': stage1_count,
            'stage2_count': stage2_count,
            'stage1_pct': (stage1_count / n_samples) * 100,
            'stage2_pct': (stage2_count / n_samples) * 100
        }

        if self.verbose:
            print(f"\nCascade Stage Statistics:")
            print(f"  Stage 1 (TF-IDF): {stage1_count} ({stage_stats['stage1_pct']:.1f}%)")
            print(f"  Stage 2 (Embeddings): {stage2_count} ({stage_stats['stage2_pct']:.1f}%)")

        return {
            'predictions': predictions,
            'metadata': metadata,
            'stage_stats': stage_stats
        }

    def evaluate(self, X_stage1, X_stage2, y_test):
        """
        Evaluate cascade performance with stage breakdown.

        Args:
            X_stage1: TF-IDF test features
            X_stage2: Embedding test features
            y_test: True labels

        Returns:
            dict: Comprehensive evaluation results
        """
        results = self.predict_with_metadata(X_stage1, X_stage2)

        y_pred = results['predictions']
        metadata = results['metadata']
        stage_stats = results['stage_stats']

        # Overall evaluation
        print_evaluation(y_test, y_pred, title="2-Stage Cascade Evaluation")

        # Calculate final probabilities
        final_probas = np.array([
            meta['stage2_proba'] if meta['stage_used'] == 2 else meta['stage1_proba']
            for meta in metadata
        ])

        overall_metrics = evaluate_classifier(
            y_test, y_pred, final_probas, "HybridCascade"
        )

        # Stage-wise analysis
        stage1_indices = [i for i, meta in enumerate(metadata) if meta['stage_used'] == 1]
        stage2_indices = [i for i, meta in enumerate(metadata) if meta['stage_used'] == 2]

        stage1_metrics = None
        stage2_metrics = None

        if stage1_indices:
            stage1_pred = y_pred[stage1_indices]
            stage1_true = y_test[stage1_indices]
            stage1_metrics = evaluate_classifier(stage1_true, stage1_pred, model_name="Stage1")

        if stage2_indices:
            stage2_pred = y_pred[stage2_indices]
            stage2_true = y_test[stage2_indices]
            stage2_metrics = evaluate_classifier(stage2_true, stage2_pred, model_name="Stage2")

        return {
            'overall': overall_metrics,
            'stage_stats': stage_stats,
            'stage1_metrics': stage1_metrics,
            'stage2_metrics': stage2_metrics,
            'metadata': metadata
        }

    def save(self, directory):
        """
        Save cascade models and configuration.

        Args:
            directory (str): Directory to save cascade components
        """
        import os
        from ..utils.io import ensure_dir, save_json

        ensure_dir(directory)

        # Save models
        self.stage1.save(os.path.join(directory, 'stage1_model.pkl'))
        self.stage2.save(os.path.join(directory, 'stage2_model.pkl'))

        # Save configuration
        config = {
            'stage1_low_threshold': self.stage1_low_threshold,
            'stage1_high_threshold': self.stage1_high_threshold,
            'stage2_threshold': self.stage2_threshold
        }
        save_json(config, os.path.join(directory, 'cascade_config.json'))

        print(f"✓ Cascade saved to {directory}/")

    @classmethod
    def load(cls, directory):
        """
        Load cascade from directory.

        Args:
            directory (str): Directory containing cascade components

        Returns:
            HybridCascadeClassifier: Loaded cascade
        """
        import os
        from ..utils.io import load_json
        from .traditional import TraditionalMLClassifier
        from .embedding import EmbeddingClassifier

        # Load models
        stage1 = TraditionalMLClassifier()
        stage1.load(os.path.join(directory, 'stage1_model.pkl'))

        stage2 = EmbeddingClassifier()
        stage2.load(os.path.join(directory, 'stage2_model.pkl'))

        # Load configuration
        config = load_json(os.path.join(directory, 'cascade_config.json'))

        return cls(
            stage1_model=stage1,
            stage2_model=stage2,
            **config
        )
