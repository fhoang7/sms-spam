"""TF-IDF feature extraction for SMS spam detection."""

from sklearn.feature_extraction.text import TfidfVectorizer
from ..utils.constants import (
    TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE,
    TFIDF_MIN_DF, TFIDF_MAX_DF
)
from ..utils.io import save_model, load_model, ensure_dir
import os
import scipy.sparse as sp


class TFIDFFeaturizer:
    """
    TF-IDF vectorization wrapper for SMS messages.

    Converts text messages into TF-IDF feature vectors using sklearn's TfidfVectorizer.
    """

    def __init__(
        self,
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=TFIDF_NGRAM_RANGE,
        min_df=TFIDF_MIN_DF,
        max_df=TFIDF_MAX_DF,
        sublinear_tf=True,
        verbose=True
    ):
        """
        Initialize TF-IDF vectorizer.

        Args:
            max_features (int): Maximum number of features
            ngram_range (tuple): Range of n-grams (min_n, max_n)
            min_df (int): Minimum document frequency
            max_df (float): Maximum document frequency (proportion)
            sublinear_tf (bool): Apply sublinear tf scaling (log)
            verbose (bool): Print configuration
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=sublinear_tf,
            use_idf=True
        )
        self.verbose = verbose

        if verbose:
            print("\nTF-IDF Configuration:")
            print(f"  - max_features: {max_features}")
            print(f"  - ngram_range: {ngram_range} - {'unigrams' if ngram_range[0]==1 else 'bigrams'}")
            if ngram_range[1] > 1:
                print(f"    (includes n-grams from {ngram_range[0]} to {ngram_range[1]} words)")
            print(f"  - min_df: {min_df} - ignore terms in < {min_df} documents")
            print(f"  - max_df: {max_df} - ignore terms in > {int(max_df*100)}% documents")
            print(f"  - sublinear_tf: {sublinear_tf} - {'use log scaling' if sublinear_tf else 'linear'}")

    def fit_transform(self, texts):
        """
        Fit vectorizer and transform texts to TF-IDF features.

        Args:
            texts (list or pd.Series): Training texts

        Returns:
            scipy.sparse.csr_matrix: TF-IDF feature matrix
        """
        if self.verbose:
            print(f"\nFitting TF-IDF vectorizer on {len(texts)} documents...")

        X_tfidf = self.vectorizer.fit_transform(texts)

        if self.verbose:
            print(f"✓ Features shape: {X_tfidf.shape}")
            print(f"✓ Vocabulary size: {len(self.vectorizer.vocabulary_)}")

        return X_tfidf

    def transform(self, texts):
        """
        Transform texts to TF-IDF features using fitted vectorizer.

        Args:
            texts (list or pd.Series): Texts to transform

        Returns:
            scipy.sparse.csr_matrix: TF-IDF feature matrix
        """
        if self.verbose:
            print(f"\nTransforming {len(texts)} documents to TF-IDF features...")

        X_tfidf = self.vectorizer.transform(texts)

        if self.verbose:
            print(f"✓ Features shape: {X_tfidf.shape}")

        return X_tfidf

    def get_feature_names(self):
        """
        Get feature names (vocabulary terms).

        Returns:
            np.ndarray: Array of feature names
        """
        return self.vectorizer.get_feature_names_out()

    def get_vocabulary_size(self):
        """
        Get vocabulary size.

        Returns:
            int: Number of features
        """
        return len(self.vectorizer.vocabulary_)

    def save(self, filepath):
        """
        Save fitted vectorizer to disk.

        Args:
            filepath (str): Path to save the vectorizer
        """
        save_model(self.vectorizer, filepath)

    @classmethod
    def load(cls, filepath):
        """
        Load fitted vectorizer from disk.

        Args:
            filepath (str): Path to saved vectorizer

        Returns:
            TFIDFFeaturizer: Instance with loaded vectorizer
        """
        instance = cls(verbose=False)
        instance.vectorizer = load_model(filepath)
        return instance


def extract_and_save_tfidf_features(
    X_train,
    X_test,
    output_dir="data/features",
    vectorizer_dir="data/vectorizers",
    **tfidf_kwargs
):
    """
    Extract TF-IDF features and save to disk.

    Args:
        X_train (pd.Series or list): Training texts
        X_test (pd.Series or list): Test texts
        output_dir (str): Directory to save feature matrices
        vectorizer_dir (str): Directory to save fitted vectorizer
        **tfidf_kwargs: Additional arguments for TFIDFFeaturizer

    Returns:
        tuple: (X_train_tfidf, X_test_tfidf, featurizer)
    """
    print("\n" + "="*70)
    print("TF-IDF FEATURE EXTRACTION")
    print("="*70)

    # Create featurizer
    featurizer = TFIDFFeaturizer(**tfidf_kwargs)

    # Fit and transform
    X_train_tfidf = featurizer.fit_transform(X_train)
    X_test_tfidf = featurizer.transform(X_test)

    # Save feature matrices
    ensure_dir(output_dir)
    sp.save_npz(os.path.join(output_dir, 'X_train_tfidf.npz'), X_train_tfidf)
    sp.save_npz(os.path.join(output_dir, 'X_test_tfidf.npz'), X_test_tfidf)
    print(f"✓ Feature matrices saved to {output_dir}/")

    # Save vectorizer
    ensure_dir(vectorizer_dir)
    featurizer.save(os.path.join(vectorizer_dir, 'tfidf_vectorizer.pkl'))

    # Show sample features
    feature_names = featurizer.get_feature_names()
    print(f"\nSample features (first 20): {list(feature_names[:20])}")

    print("\n" + "="*70)
    print("TF-IDF FEATURE EXTRACTION COMPLETE")
    print("="*70)

    return X_train_tfidf, X_test_tfidf, featurizer


def load_tfidf_features(features_dir="data/features"):
    """
    Load TF-IDF features from disk.

    Args:
        features_dir (str): Directory containing feature files

    Returns:
        tuple: (X_train_tfidf, X_test_tfidf)
    """
    X_train_tfidf = sp.load_npz(os.path.join(features_dir, 'X_train_tfidf.npz'))
    X_test_tfidf = sp.load_npz(os.path.join(features_dir, 'X_test_tfidf.npz'))

    print(f"✓ Loaded TF-IDF features from {features_dir}/")
    print(f"  Training features: {X_train_tfidf.shape}")
    print(f"  Test features: {X_test_tfidf.shape}")

    return X_train_tfidf, X_test_tfidf
