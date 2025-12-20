#!/usr/bin/env python3
"""
03_train_embeddings.py - Embedding-based ML

Loads sentence transformer embeddings and runs LazyPredict comparison
to find the best embedding-based models.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from sms_spam.data.preprocessing import load_processed_data
from sms_spam.data.embeddings import load_embeddings
from sms_spam.classifiers.embedding import (
    run_lazypredict_on_embeddings,
    EmbeddingClassifier
)
from sms_spam.classifiers.traditional import (
    get_best_model_from_lazypredict,
    create_model_from_name
)
from sms_spam.utils.io import save_csv, ensure_dir


def main():
    """Train embedding-based classifiers."""
    print("\n" + "="*70)
    print("EMBEDDING-BASED ML - SENTENCE TRANSFORMERS")
    print("="*70)

    # Load data
    _, _, y_train, y_test = load_processed_data()
    train_embeddings, test_embeddings = load_embeddings()

    # Run LazyPredict comparison
    models_df, predictions = run_lazypredict_on_embeddings(
        train_embeddings, test_embeddings, y_train, y_test
    )

    # Save results
    ensure_dir("results/metrics")
    save_csv(models_df, "results/metrics/embedding_ml_results.csv")

    # Train and save best model
    best_model_name = get_best_model_from_lazypredict(models_df)

    print(f"\nTraining best model: {best_model_name}")
    best_sklearn_model = create_model_from_name(best_model_name)
    classifier = EmbeddingClassifier(best_sklearn_model, best_model_name)
    classifier.fit(train_embeddings, y_train)

    # Evaluate
    results = classifier.evaluate(test_embeddings, y_test)

    # Save
    ensure_dir("models/embedding/baseline")
    classifier.save("models/embedding/baseline/best_model.pkl")

    print("\n" + "="*70)
    print("EMBEDDING-BASED ML TRAINING COMPLETE")
    print("="*70)
    print(f"\nBest model: {best_model_name}")
    print(f"F1 Score: {results['f1']:.4f}")
    print(f"\nSaved files:")
    print(f"  - results/metrics/embedding_ml_results.csv")
    print(f"  - models/embedding/baseline/best_model.pkl")
    print(f"\nNext step: Run scripts/05_train_hybrid.py (or 04_optimize.py for tuning)")


if __name__ == "__main__":
    main()
