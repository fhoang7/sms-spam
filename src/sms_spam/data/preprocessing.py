"""Text preprocessing and train/test split utilities."""

import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from ..utils.constants import RANDOM_STATE, TEST_SIZE, LABEL_MAP


def preprocess_text(text):
    """
    Clean and normalize SMS text.

    Args:
        text (str): Raw SMS message

    Returns:
        str: Cleaned message

    Processing steps:
        - Convert to lowercase
        - Remove URLs
        - Remove special characters (keep alphanumeric and spaces)
        - Remove extra whitespace

    Note:
        - Does NOT remove stopwords (they're signals for spam)
        - Keeps basic structure for semantic meaning
    """
    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)

    # Remove special characters (keep alphanumeric and spaces)
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def preprocess_dataset(data, verbose=True):
    """
    Apply text preprocessing to the entire dataset.

    Args:
        data (pd.DataFrame): DataFrame with 'message' column
        verbose (bool): Print progress messages

    Returns:
        pd.DataFrame: DataFrame with added 'message_clean' column

    Note:
        Removes messages that become empty after preprocessing.
    """
    if verbose:
        print("\nPreprocessing text messages...")

    # Ensure message column is string type
    data = data.copy()
    data['message'] = data['message'].astype(str)

    # Apply preprocessing
    data['message_clean'] = data['message'].apply(preprocess_text)

    # Remove messages that became empty after cleaning
    before_filter = len(data)
    data = data[data['message_clean'].str.strip() != '']
    after_filter = len(data)

    if verbose and before_filter > after_filter:
        print(f"⚠ Dropped {before_filter - after_filter} messages that became empty after preprocessing")

    if verbose:
        print(f"✓ Text preprocessing complete ({len(data)} messages remain)")

    return data


def show_preprocessing_examples(data, n=3):
    """
    Display preprocessing examples.

    Args:
        data (pd.DataFrame): DataFrame with 'message' and 'message_clean' columns
        n (int): Number of examples to show
    """
    print("\nPreprocessing Examples:")
    print("-"*70)

    for i in range(min(n, len(data))):
        original = data['message'].iloc[i]
        cleaned = data['message_clean'].iloc[i]

        print(f"\nOriginal: {original[:100]}")
        print(f"Cleaned:  {cleaned[:100]}")


def create_train_test_split(
    data,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    label_col='label',
    text_col='message_clean',
    verbose=True
):
    """
    Create stratified train/test split.

    Args:
        data (pd.DataFrame): Dataset with labels and messages
        test_size (float): Proportion of test set (default: 0.2)
        random_state (int): Random seed for reproducibility
        label_col (str): Name of label column
        text_col (str): Name of text column
        verbose (bool): Print split statistics

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
            - X_train/X_test: pd.Series of text messages
            - y_train/y_test: np.ndarray of binary labels (0=ham, 1=spam)
    """
    if verbose:
        print("\n" + "="*70)
        print("TRAIN/TEST SPLIT")
        print("="*70)

    # Prepare features and labels
    X = data[text_col]
    y = data[label_col].map(LABEL_MAP)  # Binary encoding: ham=0, spam=1

    # Stratified split to maintain class distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    if verbose:
        print(f"\nTotal Samples: {len(data)}")
        print(f"Training Set: {len(X_train)} ({len(X_train)/len(data)*100:.1f}%)")
        print(f"Test Set: {len(X_test)} ({len(X_test)/len(data)*100:.1f}%)")

        print(f"\nTraining Set Distribution:")
        print(f"  Ham: {sum(y_train == 0)} ({sum(y_train == 0)/len(y_train)*100:.1f}%)")
        print(f"  Spam: {sum(y_train == 1)} ({sum(y_train == 1)/len(y_train)*100:.1f}%)")

        print(f"\nTest Set Distribution:")
        print(f"  Ham: {sum(y_test == 0)} ({sum(y_test == 0)/len(y_test)*100:.1f}%)")
        print(f"  Spam: {sum(y_test == 1)} ({sum(y_test == 1)/len(y_test)*100:.1f}%)")

    return X_train, X_test, y_train.values, y_test.values


def get_message_stats(messages, labels=None):
    """
    Calculate message length statistics.

    Args:
        messages (pd.Series): SMS messages
        labels (pd.Series, optional): Labels for class-wise statistics

    Returns:
        pd.DataFrame: Statistics including char count, word count by class
    """
    char_counts = messages.str.len()
    word_counts = messages.str.split().str.len()

    if labels is None:
        # Overall statistics
        stats = {
            'char_mean': char_counts.mean(),
            'char_std': char_counts.std(),
            'char_median': char_counts.median(),
            'char_max': char_counts.max(),
            'word_mean': word_counts.mean(),
            'word_std': word_counts.std(),
            'word_median': word_counts.median(),
            'word_max': word_counts.max()
        }
        return pd.DataFrame([stats])
    else:
        # Class-wise statistics
        df = pd.DataFrame({
            'message': messages,
            'label': labels,
            'char_count': char_counts,
            'word_count': word_counts
        })

        stats = df.groupby('label').agg({
            'char_count': ['mean', 'std', 'median', 'max'],
            'word_count': ['mean', 'std', 'median', 'max']
        }).round(2)

        return stats


def save_processed_data(X_train, X_test, y_train, y_test, output_dir="data/processed"):
    """
    Save preprocessed train/test data to disk.

    Args:
        X_train (pd.Series): Training messages
        X_test (pd.Series): Test messages
        y_train (np.ndarray): Training labels
        y_test (np.ndarray): Test labels
        output_dir (str): Output directory path
    """
    from ..utils.io import ensure_dir, save_csv, save_numpy
    import os

    ensure_dir(output_dir)

    # Save text data as CSV
    save_csv(
        pd.DataFrame({'message': X_train}),
        os.path.join(output_dir, 'X_train.csv'),
        index=False
    )
    save_csv(
        pd.DataFrame({'message': X_test}),
        os.path.join(output_dir, 'X_test.csv'),
        index=False
    )

    # Save labels as numpy arrays
    save_numpy(y_train, os.path.join(output_dir, 'y_train.npy'))
    save_numpy(y_test, os.path.join(output_dir, 'y_test.npy'))

    print(f"✓ All preprocessed data saved to {output_dir}/")


def load_processed_data(input_dir="data/processed"):
    """
    Load preprocessed train/test data from disk.

    Args:
        input_dir (str): Input directory path

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    from ..utils.io import load_csv, load_numpy
    import os

    X_train = load_csv(os.path.join(input_dir, 'X_train.csv'))['message']
    X_test = load_csv(os.path.join(input_dir, 'X_test.csv'))['message']
    y_train = load_numpy(os.path.join(input_dir, 'y_train.npy'))
    y_test = load_numpy(os.path.join(input_dir, 'y_test.npy'))

    print(f"✓ Loaded preprocessed data from {input_dir}/")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")

    return X_train, X_test, y_train, y_test
