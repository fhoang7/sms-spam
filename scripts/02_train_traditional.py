#!/usr/bin/env python3
"""
02_train_traditional.py - Traditional ML with TF-IDF

Loads TF-IDF features and runs LazyPredict comparison
to find the best traditional ML models.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from sms_spam.data.preprocessing import load_processed_data
from sms_spam.features.tfidf import load_tfidf_features
from sms_spam.classifiers.traditional import (
    run_lazypredict_comparison,
    get_best_model_from_lazypredict,
    create_model_from_name,
    TraditionalMLClassifier
)
from sms_spam.utils.io import save_csv, save_model, ensure_dir


def main():
    """Train traditional ML classifiers."""
    print("\n" + "="*70)
    print("TRADITIONAL ML - TF-IDF FEATURES")
    print("="*70)

    # Load data
    X_train, X_test, y_train, y_test = load_processed_data()
    X_train_tfidf, X_test_tfidf = load_tfidf_features()

    # Run LazyPredict comparison
    models_df, predictions = run_lazypredict_comparison(
        X_train_tfidf, X_test_tfidf, y_train, y_test
    )

    # Save results
    ensure_dir("results/metrics")
    save_csv(models_df, "results/metrics/traditional_ml_results.csv")

    # Train and save best model
    best_model_name = get_best_model_from_lazypredict(models_df)

    print(f"\nTraining best model: {best_model_name}")
    best_sklearn_model = create_model_from_name(best_model_name)
    classifier = TraditionalMLClassifier(best_sklearn_model, best_model_name)
    classifier.fit(X_train_tfidf, y_train)

    # Evaluate
    results = classifier.evaluate(X_test_tfidf, y_test)

    # Save
    ensure_dir("models/traditional/baseline")
    classifier.save("models/traditional/baseline/best_model.pkl")

    print("\n" + "="*70)
    print("TRADITIONAL ML TRAINING COMPLETE")
    print("="*70)
    print(f"\nBest model: {best_model_name}")
    print(f"F1 Score: {results['f1']:.4f}")
    print(f"\nSaved files:")
    print(f"  - results/metrics/traditional_ml_results.csv")
    print(f"  - models/traditional/baseline/best_model.pkl")
    print(f"\nNext step: Run scripts/03_train_embeddings.py")


if __name__ == "__main__":
    main()
