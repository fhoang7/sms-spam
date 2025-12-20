"""
ChromaDB vector storage for SMS embeddings (optional).

This module provides utilities for storing and querying embeddings using ChromaDB.
Note: ChromaDB is optional and not required for classification. Primary embeddings
are stored as .npy files and used directly by classifiers.
"""

import chromadb
from chromadb.config import Settings
import os


class EmbeddingStorage:
    """
    ChromaDB-based storage for sentence embeddings.

    Provides semantic search capabilities over SMS messages using their embeddings.
    """

    def __init__(self, db_path="data/chromadb", collection_name="sms_spam_embeddings"):
        """
        Initialize ChromaDB storage.

        Args:
            db_path (str): Path to ChromaDB database directory
            collection_name (str): Name of the collection
        """
        self.db_path = db_path
        self.collection_name = collection_name

        # Create persistent client
        self.client = chromadb.PersistentClient(path=db_path)

        print(f"✓ ChromaDB client initialized at {db_path}")

    def create_collection(self, delete_existing=False):
        """
        Create or get collection.

        Args:
            delete_existing (bool): Delete existing collection if it exists

        Returns:
            chromadb.Collection: The collection
        """
        if delete_existing:
            try:
                self.client.delete_collection(name=self.collection_name)
                print(f"✓ Deleted existing collection: {self.collection_name}")
            except:
                pass

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "SMS spam/ham embeddings"}
        )

        print(f"✓ Collection '{self.collection_name}' ready")
        return self.collection

    def add_embeddings(self, embeddings, texts, labels, ids=None, split="train"):
        """
        Add embeddings to the collection.

        Args:
            embeddings (np.ndarray): Embedding vectors
            texts (list): Original text messages
            labels (list or np.ndarray): Labels (0=ham, 1=spam)
            ids (list, optional): Document IDs (auto-generated if None)
            split (str): Dataset split ('train' or 'test')
        """
        if not hasattr(self, 'collection'):
            self.create_collection()

        # Prepare data
        n = len(embeddings)
        if ids is None:
            ids = [f"{split}_{i}" for i in range(n)]

        # Convert labels to strings
        label_names = ['ham' if l == 0 else 'spam' for l in labels]

        # Add to collection
        self.collection.add(
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=[{"label": label, "split": split} for label in label_names],
            ids=ids
        )

        print(f"✓ Added {n} {split} embeddings to collection")

    def query(self, query_embedding, n_results=5):
        """
        Query similar messages by embedding.

        Args:
            query_embedding (np.ndarray): Query embedding vector
            n_results (int): Number of results to return

        Returns:
            dict: Query results with ids, documents, distances, metadatas
        """
        if not hasattr(self, 'collection'):
            raise ValueError("Collection not initialized. Call create_collection() first.")

        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results
        )

        return results

    def get_stats(self):
        """
        Get collection statistics.

        Returns:
            dict: Statistics about the collection
        """
        if not hasattr(self, 'collection'):
            self.collection = self.client.get_collection(name=self.collection_name)

        count = self.collection.count()
        print(f"\nCollection Statistics:")
        print(f"  Name: {self.collection_name}")
        print(f"  Total documents: {count}")

        return {"name": self.collection_name, "count": count}


def setup_chromadb_storage(
    train_embeddings,
    test_embeddings,
    X_train,
    X_test,
    y_train,
    y_test,
    db_path="data/chromadb",
    collection_name="sms_spam_embeddings"
):
    """
    Set up ChromaDB storage with train/test embeddings.

    Args:
        train_embeddings (np.ndarray): Training embeddings
        test_embeddings (np.ndarray): Test embeddings
        X_train (list or pd.Series): Training messages
        X_test (list or pd.Series): Test messages
        y_train (np.ndarray): Training labels
        y_test (np.ndarray): Test labels
        db_path (str): Path to ChromaDB database
        collection_name (str): Collection name

    Returns:
        EmbeddingStorage: Initialized storage with data
    """
    print("\n" + "="*70)
    print("CHROMADB VECTOR STORAGE SETUP")
    print("="*70)

    # Initialize storage
    storage = EmbeddingStorage(db_path=db_path, collection_name=collection_name)
    storage.create_collection(delete_existing=True)

    # Convert to lists if needed
    X_train_list = X_train.tolist() if hasattr(X_train, 'tolist') else list(X_train)
    X_test_list = X_test.tolist() if hasattr(X_test, 'tolist') else list(X_test)

    # Add train embeddings
    storage.add_embeddings(
        embeddings=train_embeddings,
        texts=X_train_list,
        labels=y_train,
        split="train"
    )

    # Add test embeddings
    storage.add_embeddings(
        embeddings=test_embeddings,
        texts=X_test_list,
        labels=y_test,
        split="test"
    )

    # Show stats
    storage.get_stats()

    print("\n" + "="*70)
    print("CHROMADB SETUP COMPLETE")
    print("="*70)

    return storage
