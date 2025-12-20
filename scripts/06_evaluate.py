#!/usr/bin/env python3
"""
06_evaluate.py - Comprehensive Evaluation and Comparison

Compares all approaches:
  - Traditional ML (TF-IDF)
  - Embedding-based ML
  - 2-Stage Hybrid Cascade

Generates visualizations and reports.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
from sms_spam.data.preprocessing import load_processed_data
from sms_spam.features.tfidf import load_tfidf_features
from sms_spam.data.embeddings import load_embeddings
from sms_spam.classifiers.traditional import TraditionalMLClassifier
from sms_spam.classifiers.embedding import EmbeddingClassifier
from sms_spam.classifiers.hybrid import HybridCascadeClassifier
from sms_spam.evaluation.metrics import evaluate_classifier
from sms_spam.evaluation.visualization import (
    plot_model_comparison,
    plot_cascade_analysis,
    plot_confusion_matrix
)
from sms_spam.utils.io import ensure_dir, load_json


def main():
    """Comprehensive evaluation."""
    print("\n" + "="*70)
    print("COMPREHENSIVE EVALUATION AND COMPARISON")
    print("="*70)

    # Load data
    _, X_test, _, y_test = load_processed_data()
    X_test_tfidf, _ = load_tfidf_features()
    _, test_embeddings = load_embeddings()

    # Load models
    print("\nLoading models...")

    traditional = TraditionalMLClassifier()
    traditional.load("models/traditional/baseline/best_model.pkl")

    embedding = EmbeddingClassifier()
    embedding.load("models/embedding/baseline/best_model.pkl")

    cascade = HybridCascadeClassifier.load("models/hybrid")

    print("✓ All models loaded")

    # Evaluate all models
    print("\nEvaluating all approaches...")

    # Traditional ML
    y_pred_trad = traditional.predict(X_test_tfidf)
    y_proba_trad = traditional.predict_proba(X_test_tfidf)
    results_trad = evaluate_classifier(y_test, y_pred_trad, y_proba_trad, "Traditional-ML")

    # Embedding ML
    y_pred_emb = embedding.predict(test_embeddings)
    y_proba_emb = embedding.predict_proba(test_embeddings)
    results_emb = evaluate_classifier(y_test, y_pred_emb, y_proba_emb, "Embedding-ML")

    # Hybrid Cascade
    cascade_results = cascade.evaluate(X_test_tfidf, test_embeddings, y_test)
    results_cascade = cascade_results['overall']
    y_pred_cascade = cascade.predict(X_test_tfidf, test_embeddings)

    # Create comparison DataFrame
    comparison_df = pd.DataFrame([results_trad, results_emb, results_cascade])
    comparison_df = comparison_df.set_index('model')

    print("\n" + "-"*70)
    print("RESULTS COMPARISON")
    print("-"*70)
    print(comparison_df[['accuracy', 'precision', 'recall', 'f1', 'roc_auc']])

    # Save comparison
    ensure_dir("results/metrics")
    comparison_df.to_csv("results/metrics/final_comparison.csv")

    # Generate visualizations
    print("\nGenerating visualizations...")
    ensure_dir("results/visualizations")

    # Model comparison plot
    plot_model_comparison(
        comparison_df,
        metrics=['accuracy', 'precision', 'recall', 'f1'],
        title="Model Performance Comparison",
        output_path="results/visualizations/model_comparison.png"
    )

    # Cascade analysis
    plot_cascade_analysis(
        cascade_results,
        output_path="results/visualizations/cascade_analysis.png"
    )

    # Confusion matrices
    plot_confusion_matrix(
        y_test, y_pred_cascade,
        title="Hybrid Cascade Confusion Matrix",
        output_path="results/visualizations/cascade_confusion_matrix.png"
    )

    # Generate text report
    print("\nGenerating report...")
    ensure_dir("results/reports")

    with open("results/reports/evaluation_report.txt", 'w') as f:
        f.write("="*70 + "\n")
        f.write("SMS SPAM DETECTION - FINAL EVALUATION REPORT\n")
        f.write("="*70 + "\n\n")

        f.write("APPROACH COMPARISON\n")
        f.write("-"*70 + "\n")
        f.write(comparison_df.to_string())
        f.write("\n\n")

        f.write("2-STAGE CASCADE DETAILS\n")
        f.write("-"*70 + "\n")
        f.write(f"Stage 1 (TF-IDF): {cascade_results['stage_stats']['stage1_pct']:.1f}%\n")
        f.write(f"Stage 2 (Embeddings): {cascade_results['stage_stats']['stage2_pct']:.1f}%\n")
        f.write("\n")

        if cascade_results['stage1_metrics']:
            f.write("Stage 1 Performance:\n")
            for key, val in cascade_results['stage1_metrics'].items():
                if key != 'model':
                    f.write(f"  {key}: {val:.4f}\n")
            f.write("\n")

        if cascade_results['stage2_metrics']:
            f.write("Stage 2 Performance:\n")
            for key, val in cascade_results['stage2_metrics'].items():
                if key != 'model':
                    f.write(f"  {key}: {val:.4f}\n")

    print("✓ Report saved to results/reports/evaluation_report.txt")

    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)
    print(f"\nBest Approach: {comparison_df['f1'].idxmax()}")
    print(f"Best F1 Score: {comparison_df['f1'].max():.4f}")

    print(f"\nGenerated files:")
    print(f"  - results/metrics/final_comparison.csv")
    print(f"  - results/visualizations/*.png (3 plots)")
    print(f"  - results/reports/evaluation_report.txt")

    print(f"\n✓ All done! SMS spam detection system is ready.")


if __name__ == "__main__":
    main()
