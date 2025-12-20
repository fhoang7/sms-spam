"""Sentence transformer embedding generation."""

import numpy as np
from sentence_transformers import SentenceTransformer
from ..utils.constants import EMBEDDING_MODEL_NAME, EMBEDDING_BATCH_SIZE
from ..utils.io import save_numpy, load_numpy, ensure_dir
import os


class EmbeddingGenerator:
    """
    Generate sentence transformer embeddings for SMS messages.

    Uses the sentence-transformers library to create semantic embeddings
    that capture the meaning of text messages.
    """

    def __init__(self, model_name=EMBEDDING_MODEL_NAME, verbose=True):
        """
        Initialize the embedding generator.

        Args:
            model_name (str): Name of the sentence transformer model
            verbose (bool): Print loading messages
        """
        self.model_name = model_name
        self.verbose = verbose

        if verbose:
            print(f"\nLoading embedding model: {model_name}")
            print("(This may take a moment on first run...)")

        self.model = SentenceTransformer(model_name)

        if verbose:
            print("✓ Model loaded successfully")

    def generate(self, texts, batch_size=EMBEDDING_BATCH_SIZE, show_progress=True):
        """
        Generate embeddings for a list of texts.

        Args:
            texts (list or pd.Series): Text messages to embed
            batch_size (int): Batch size for encoding
            show_progress (bool): Show progress bar

        Returns:
            np.ndarray: Embedding matrix of shape (n_samples, embedding_dim)
        """
        if self.verbose:
            print(f"\nGenerating embeddings for {len(texts)} messages...")

        embeddings = self.model.encode(
            texts.tolist() if hasattr(texts, 'tolist') else texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )

        if self.verbose:
            print(f"✓ Generated embeddings: shape {embeddings.shape}")

        return embeddings

    def get_embedding_dim(self):
        """
        Get the embedding dimension of the model.

        Returns:
            int: Embedding dimension
        """
        return self.model.get_sentence_embedding_dimension()


def generate_and_save_embeddings(
    X_train,
    X_test,
    output_dir="data/features",
    model_name=EMBEDDING_MODEL_NAME,
    batch_size=EMBEDDING_BATCH_SIZE
):
    """
    Generate and save train/test embeddings.

    Args:
        X_train (pd.Series or list): Training messages
        X_test (pd.Series or list): Test messages
        output_dir (str): Directory to save embeddings
        model_name (str): Sentence transformer model name
        batch_size (int): Batch size for encoding

    Returns:
        tuple: (train_embeddings, test_embeddings)
    """
    print("\n" + "="*70)
    print("EMBEDDING GENERATION")
    print("="*70)

    # Initialize generator
    generator = EmbeddingGenerator(model_name=model_name, verbose=True)

    # Generate embeddings
    train_embeddings = generator.generate(X_train, batch_size=batch_size)
    test_embeddings = generator.generate(X_test, batch_size=batch_size)

    # Save to disk
    ensure_dir(output_dir)
    save_numpy(train_embeddings, os.path.join(output_dir, 'train_embeddings.npy'))
    save_numpy(test_embeddings, os.path.join(output_dir, 'test_embeddings.npy'))

    print(f"\nEmbedding Summary:")
    print(f"  Model: {model_name}")
    print(f"  Embedding dimension: {train_embeddings.shape[1]}")
    print(f"  Training embeddings: {train_embeddings.shape}")
    print(f"  Test embeddings: {test_embeddings.shape}")

    print("\n" + "="*70)
    print("EMBEDDING GENERATION COMPLETE")
    print("="*70)

    return train_embeddings, test_embeddings


def load_embeddings(input_dir="data/features"):
    """
    Load pre-generated embeddings from disk.

    Args:
        input_dir (str): Directory containing embedding files

    Returns:
        tuple: (train_embeddings, test_embeddings)
    """
    train_embeddings = load_numpy(os.path.join(input_dir, 'train_embeddings.npy'))
    test_embeddings = load_numpy(os.path.join(input_dir, 'test_embeddings.npy'))

    print(f"✓ Loaded embeddings from {input_dir}/")
    print(f"  Training embeddings: {train_embeddings.shape}")
    print(f"  Test embeddings: {test_embeddings.shape}")

    return train_embeddings, test_embeddings
