"""File I/O utilities for the SMS spam detection project."""

import os
import joblib
import json
import pandas as pd
import numpy as np
from pathlib import Path


def ensure_dir(directory):
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        directory (str or Path): Directory path

    Returns:
        Path: Path object for the directory
    """
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_model(model, filepath):
    """
    Save a sklearn model using joblib.

    Args:
        model: sklearn model or pipeline
        filepath (str): Path to save the model
    """
    ensure_dir(os.path.dirname(filepath))
    joblib.dump(model, filepath)
    print(f"✓ Model saved to {filepath}")


def load_model(filepath):
    """
    Load a sklearn model using joblib.

    Args:
        filepath (str): Path to the saved model

    Returns:
        Loaded model
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found: {filepath}")
    model = joblib.load(filepath)
    print(f"✓ Model loaded from {filepath}")
    return model


def save_json(data, filepath):
    """
    Save data as JSON.

    Args:
        data (dict): Data to save
        filepath (str): Path to save the JSON file
    """
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"✓ JSON saved to {filepath}")


def load_json(filepath):
    """
    Load data from JSON file.

    Args:
        filepath (str): Path to JSON file

    Returns:
        dict: Loaded data
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"JSON file not found: {filepath}")
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data


def save_numpy(array, filepath):
    """
    Save numpy array.

    Args:
        array (np.ndarray): Array to save
        filepath (str): Path to save the array
    """
    ensure_dir(os.path.dirname(filepath))
    np.save(filepath, array)
    print(f"✓ Array saved to {filepath}")


def load_numpy(filepath):
    """
    Load numpy array.

    Args:
        filepath (str): Path to the array file

    Returns:
        np.ndarray: Loaded array
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Numpy file not found: {filepath}")
    return np.load(filepath)


def save_csv(df, filepath, **kwargs):
    """
    Save pandas DataFrame as CSV.

    Args:
        df (pd.DataFrame): DataFrame to save
        filepath (str): Path to save the CSV
        **kwargs: Additional arguments for pd.DataFrame.to_csv()
    """
    ensure_dir(os.path.dirname(filepath))
    df.to_csv(filepath, **kwargs)
    print(f"✓ CSV saved to {filepath}")


def load_csv(filepath, **kwargs):
    """
    Load CSV file as pandas DataFrame.

    Args:
        filepath (str): Path to CSV file
        **kwargs: Additional arguments for pd.read_csv()

    Returns:
        pd.DataFrame: Loaded DataFrame
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"CSV file not found: {filepath}")
    return pd.read_csv(filepath, **kwargs)
