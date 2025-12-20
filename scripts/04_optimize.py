#!/usr/bin/env python3
"""
04_optimize.py - Hyperparameter Optimization (Optional)

Uses Optuna to optimize hyperparameters for both traditional
and embedding-based classifiers.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from sms_spam.data.preprocessing import load_processed_data
from sms_spam.features.tfidf import load_tfidf_features
from sms_spam.data.embeddings import load_embeddings
from sms_spam.optimization.optuna_tuner import optimize_and_save_model
from sms_spam.classifiers.traditional import TraditionalMLClassifier
from sms_spam.classifiers.embedding import EmbeddingClassifier
from sms_spam.utils.io import load_csv


def main():
    """Optimize hyperparameters."""
    print("\n" + "="*70)
    print("HYPERPARAMETER OPTIMIZATION WITH OPTUNA")
    print("="*70)

    # Load data
    X_train, X_test, y_train, y_test = load_processed_data()
    X_train_tfidf, X_test_tfidf = load_tfidf_features()
    train_embeddings, test_embeddings = load_embeddings()

    # Get best model names from previous runs
    print("\nLoading best models from baseline results...")
    trad_results = load_csv("results/metrics/traditional_ml_results.csv", index_col=0)
    best_trad_model = trad_results['F1 Score'].idxmax()

    emb_results = load_csv("results/metrics/embedding_ml_results.csv", index_col=0)
    best_emb_model = emb_results['F1 Score'].idxmax()

    print(f"  Traditional: {best_trad_model}")
    print(f"  Embedding: {best_emb_model}")

    # Optimize traditional model
    print("\n" + "-"*70)
    print("OPTIMIZING TRADITIONAL MODEL")
    print("-"*70)

    trad_model, trad_results = optimize_and_save_model(
        X_train_tfidf, y_train,
        model_class_name=best_trad_model,
        output_model_path="models/traditional/optimized/optimized_model.pkl",
        output_results_path="results/metrics/traditional_optimization.json",
        n_trials=100
    )

    # Evaluate on test set
    classifier_trad = TraditionalMLClassifier(trad_model, best_trad_model)
    classifier_trad.is_fitted = True
    results_trad = classifier_trad.evaluate(X_test_tfidf, y_test)

    # Optimize embedding model
    print("\n" + "-"*70)
    print("OPTIMIZING EMBEDDING MODEL")
    print("-"*70)

    emb_model, emb_results = optimize_and_save_model(
        train_embeddings, y_train,
        model_class_name=best_emb_model,
        output_model_path="models/embedding/optimized/optimized_model.pkl",
        output_results_path="results/metrics/embedding_optimization.json",
        n_trials=100
    )

    # Evaluate on test set
    classifier_emb = EmbeddingClassifier(emb_model, best_emb_model)
    classifier_emb.is_fitted = True
    results_emb = classifier_emb.evaluate(test_embeddings, y_test)

    print("\n" + "="*70)
    print("OPTIMIZATION COMPLETE")
    print("="*70)
    print(f"\nTraditional Model:")
    print(f"  Model: {best_trad_model}")
    print(f"  Test F1: {results_trad['f1']:.4f}")

    print(f"\nEmbedding Model:")
    print(f"  Model: {best_emb_model}")
    print(f"  Test F1: {results_emb['f1']:.4f}")

    print(f"\nSaved files:")
    print(f"  - models/traditional/optimized/optimized_model.pkl")
    print(f"  - models/embedding/optimized/optimized_model.pkl")
    print(f"\nNext step: Run scripts/05_train_hybrid.py")


if __name__ == "__main__":
    main()
