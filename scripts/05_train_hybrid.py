#!/usr/bin/env python3
"""
05_train_hybrid.py - Train 2-Stage Hybrid Cascade

Creates a 2-stage cascade classifier:
  Stage 1: TF-IDF (fast, handles 80-90%)
  Stage 2: Embeddings (semantic, handles remaining 10-20%)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from sms_spam.data.preprocessing import load_processed_data
from sms_spam.features.tfidf import load_tfidf_features
from sms_spam.data.embeddings import load_embeddings
from sms_spam.classifiers.traditional import TraditionalMLClassifier
from sms_spam.classifiers.embedding import EmbeddingClassifier
from sms_spam.classifiers.hybrid import HybridCascadeClassifier
from sms_spam.optimization.threshold_tuner import optimize_cascade_thresholds
from sms_spam.utils.io import ensure_dir, load_json, save_json


def main():
    """Train 2-stage hybrid cascade."""
    print("\n" + "="*70)
    print("2-STAGE HYBRID CASCADE CLASSIFIER")
    print("="*70)

    # Load data
    X_train, X_test, y_train, y_test = load_processed_data()
    X_train_tfidf, X_test_tfidf = load_tfidf_features()
    train_embeddings, test_embeddings = load_embeddings()

    # Load or use baseline models
    print("\nLoading Stage 1 (TF-IDF) model...")
    stage1 = TraditionalMLClassifier()
    try:
        # Try optimized first
        stage1.load("models/traditional/optimized/optimized_model.pkl")
        print("✓ Loaded optimized traditional model")
    except:
        # Fall back to baseline
        stage1.load("models/traditional/baseline/best_model.pkl")
        print("✓ Loaded baseline traditional model")

    print("\nLoading Stage 2 (Embeddings) model...")
    stage2 = EmbeddingClassifier()
    try:
        # Try optimized first
        stage2.load("models/embedding/optimized/optimized_model.pkl")
        print("✓ Loaded optimized embedding model")
    except:
        # Fall back to baseline
        stage2.load("models/embedding/baseline/best_model.pkl")
        print("✓ Loaded baseline embedding model")

    # Optimize thresholds for cascade
    print("\nOptimizing cascade thresholds...")
    stage1_train_proba = stage1.predict_proba(X_train_tfidf)

    thresholds = optimize_cascade_thresholds(
        y_true=y_train,
        stage1_proba=stage1_train_proba,
        output_path="results/metrics/cascade_thresholds.json",
        target_coverage=0.8
    )

    # Create cascade classifier
    cascade = HybridCascadeClassifier(
        stage1_model=stage1,
        stage2_model=stage2,
        stage1_low_threshold=thresholds['stage1_low_threshold'],
        stage1_high_threshold=thresholds['stage1_high_threshold'],
        stage2_threshold=thresholds['stage2_threshold'],
        verbose=True
    )

    # Evaluate cascade
    print("\nEvaluating 2-stage cascade on test set...")
    results = cascade.evaluate(X_test_tfidf, test_embeddings, y_test)

    # Save cascade
    ensure_dir("models/hybrid")
    cascade.save("models/hybrid")

    # Save results
    ensure_dir("results/metrics")
    save_json(results, "results/metrics/hybrid_cascade_results.json")

    print("\n" + "="*70)
    print("2-STAGE CASCADE TRAINING COMPLETE")
    print("="*70)
    print(f"\nOverall Performance:")
    print(f"  Accuracy: {results['overall']['accuracy']:.4f}")
    print(f"  Precision: {results['overall']['precision']:.4f}")
    print(f"  Recall: {results['overall']['recall']:.4f}")
    print(f"  F1 Score: {results['overall']['f1']:.4f}")

    print(f"\nStage Distribution:")
    print(f"  Stage 1 (TF-IDF): {results['stage_stats']['stage1_pct']:.1f}%")
    print(f"  Stage 2 (Embeddings): {results['stage_stats']['stage2_pct']:.1f}%")

    print(f"\nSaved files:")
    print(f"  - models/hybrid/")
    print(f"  - results/metrics/cascade_thresholds.json")
    print(f"  - results/metrics/hybrid_cascade_results.json")
    print(f"\nNext step: Run scripts/06_evaluate.py for comprehensive comparison")


if __name__ == "__main__":
    main()
