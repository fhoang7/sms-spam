#!/usr/bin/env python3
"""
01_preprocess.py - Data Loading and Preprocessing

Loads spam.csv, preprocesses text, creates train/test split,
generates TF-IDF features and embeddings, and saves everything.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from sms_spam.data.loader import load_spam_dataset, validate_dataset, get_dataset_info, display_sample_messages
from sms_spam.data.preprocessing import (
    preprocess_dataset, show_preprocessing_examples,
    create_train_test_split, save_processed_data, get_message_stats
)
from sms_spam.data.embeddings import generate_and_save_embeddings
from sms_spam.features.tfidf import extract_and_save_tfidf_features
from sms_spam.features.chromadb_storage import setup_chromadb_storage


def main():
    """Main preprocessing pipeline."""
    print("\n" + "="*70)
    print("SMS SPAM DETECTION - DATA PREPROCESSING")
    print("="*70)

    # 1. Load dataset
    data = load_spam_dataset("spam.csv")
    data = validate_dataset(data)
    get_dataset_info(data)
    display_sample_messages(data, n_samples=3)

    # 2. Text preprocessing
    data = preprocess_dataset(data, verbose=True)
    show_preprocessing_examples(data, n=3)

    # 3. Message statistics
    print("\n" + "-"*70)
    print("MESSAGE LENGTH STATISTICS")
    print("-"*70)
    stats = get_message_stats(data['message'], data['label'])
    print(stats)

    # 4. Train/test split
    X_train, X_test, y_train, y_test = create_train_test_split(
        data,
        test_size=0.2,
        random_state=42,
        verbose=True
    )

    # 5. Save preprocessed data
    save_processed_data(X_train, X_test, y_train, y_test)

    # 6. Generate TF-IDF features
    X_train_tfidf, X_test_tfidf, featurizer = extract_and_save_tfidf_features(
        X_train, X_test
    )

    # 7. Generate embeddings
    train_embeddings, test_embeddings = generate_and_save_embeddings(
        X_train, X_test
    )

    # 8. Optional: Setup ChromaDB (for semantic search demos)
    print("\nℹ Setting up ChromaDB for semantic search (optional)...")
    try:
        storage = setup_chromadb_storage(
            train_embeddings, test_embeddings,
            X_train, X_test,
            y_train, y_test
        )
        print("✓ ChromaDB setup complete")
    except Exception as e:
        print(f"⚠ ChromaDB setup skipped: {e}")
        print("  (This is optional and doesn't affect classification)")

    # Summary
    print("\n" + "="*70)
    print("PREPROCESSING COMPLETE")
    print("="*70)
    print(f"\nGenerated files:")
    print(f"  - data/processed/: Preprocessed text (CSV, NPY)")
    print(f"  - data/features/: TF-IDF features (NPZ) and embeddings (NPY)")
    print(f"  - data/vectorizers/: Fitted TF-IDF vectorizer (PKL)")
    print(f"  - data/chromadb/: ChromaDB vector storage (optional)")
    print(f"\nNext step: Run scripts/02_train_traditional.py")


if __name__ == "__main__":
    main()
