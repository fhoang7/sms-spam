"""Data loading utilities for SMS spam dataset."""

import pandas as pd
import os
from pathlib import Path


def load_spam_dataset(filepath="spam.csv", encoding="latin"):
    """
    Load the SMS spam dataset.

    Args:
        filepath (str): Path to spam.csv file
        encoding (str): File encoding (must be 'latin' for spam.csv)

    Returns:
        pd.DataFrame: DataFrame with 'label' and 'message' columns

    Raises:
        FileNotFoundError: If spam.csv not found

    Note:
        The spam.csv file must be read with latin-1 encoding, NOT UTF-8.
        It contains columns 'v1' (label) and 'v2' (message) which are renamed.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset not found: {filepath}\n"
            f"Please ensure spam.csv is in the correct location."
        )

    print(f"Loading SMS Spam Dataset from {filepath}...")

    # Read CSV with latin encoding
    data = pd.read_csv(filepath, encoding=encoding)

    # Select and rename columns
    data = data[['v1', 'v2']]
    data.columns = ['label', 'message']

    print(f"✓ Loaded {len(data)} messages")

    return data


def validate_dataset(data):
    """
    Validate and clean the spam dataset.

    Args:
        data (pd.DataFrame): Raw dataset

    Returns:
        pd.DataFrame: Cleaned dataset

    Performs:
        - Removes rows with missing values
        - Removes duplicate messages
        - Ensures proper data types
        - Reports statistics
    """
    initial_count = len(data)

    # Drop rows with missing messages or labels
    data = data.dropna(subset=['message', 'label'])
    dropped_na = initial_count - len(data)
    if dropped_na > 0:
        print(f"⚠ Dropped {dropped_na} rows with missing values")

    # Remove duplicates
    data = data.drop_duplicates(subset=['message'])
    dropped_dupes = (initial_count - dropped_na) - len(data)
    if dropped_dupes > 0:
        print(f"⚠ Dropped {dropped_dupes} duplicate messages")

    # Ensure message column is string type
    data['message'] = data['message'].astype(str)

    # Verify labels
    valid_labels = {'ham', 'spam'}
    invalid_labels = set(data['label'].unique()) - valid_labels
    if invalid_labels:
        raise ValueError(f"Invalid labels found: {invalid_labels}. Expected 'ham' or 'spam'.")

    print(f"✓ Final dataset: {len(data)} messages")

    return data


def get_dataset_info(data):
    """
    Display comprehensive dataset information.

    Args:
        data (pd.DataFrame): Dataset with 'label' and 'message' columns
    """
    print("\n" + "="*70)
    print("DATASET INFORMATION")
    print("="*70)

    # Basic info
    print(f"\nDataset Shape: {data.shape}")
    print(f"Columns: {list(data.columns)}")
    print(f"Memory Usage: {data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    # Class distribution
    print("\nClass Distribution:")
    class_counts = data['label'].value_counts()
    print(class_counts)

    print(f"\nClass Percentages:")
    for label, count in class_counts.items():
        percentage = (count / len(data)) * 100
        print(f"  {label}: {count} ({percentage:.2f}%)")

    # Imbalance ratio
    if 'ham' in class_counts and 'spam' in class_counts:
        ham_count = class_counts['ham']
        spam_count = class_counts['spam']
        ratio = ham_count / spam_count
        print(f"\nClass Imbalance Ratio (Ham:Spam): {ratio:.2f}:1")

    print("="*70)


def display_sample_messages(data, n_samples=5):
    """
    Display sample messages from each class.

    Args:
        data (pd.DataFrame): DataFrame with 'label' and 'message' columns
        n_samples (int): Number of samples per class to display
    """
    print("\n" + "="*70)
    print("SAMPLE MESSAGES")
    print("="*70)

    for label in sorted(data['label'].unique()):
        print(f"\n{label.upper()} Messages:")
        print("-" * 70)

        label_data = data[data['label'] == label]
        n = min(n_samples, len(label_data))
        samples = label_data.sample(n)

        for idx, (_, row) in enumerate(samples.iterrows(), 1):
            msg = row['message']
            if len(msg) > 100:
                msg = msg[:97] + "..."
            print(f"{idx}. {msg}")

    print("="*70)
