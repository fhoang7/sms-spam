#%% Imports
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from utils import preprocess_text, get_message_stats, display_sample_messages

#%% Create local storage path
DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(DIR, 'data')
if not os.path.exists(DB_PATH):
    os.makedirs(DB_PATH)

#%% Read in data
print("Loading SMS Spam Dataset...")
data = pd.read_csv('spam.csv', encoding='latin')
data = data[['v1', 'v2']]
data.columns = ['label', 'message']
print(f"✓ Loaded {len(data)} messages")

#%% Exploratory Data Analysis
print("\n" + "="*70)
print("EXPLORATORY DATA ANALYSIS")
print("="*70)

# Dataset info
print(f"\nDataset Shape: {data.shape}")
print(f"Columns: {list(data.columns)}")
print(f"\nMemory Usage: {data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# Check for missing values
print(f"\nMissing Values:\n{data.isnull().sum()}")

# Check for duplicates
duplicates = data.duplicated().sum()
print(f"\nDuplicate Messages: {duplicates}")

# Class distribution
print("\nClass Distribution:")
class_counts = data['label'].value_counts()
print(class_counts)
print(f"\nClass Percentages:")
for label, count in class_counts.items():
    percentage = (count / len(data)) * 100
    print(f"  {label}: {percentage:.2f}%")

ham_count = class_counts['ham']
spam_count = class_counts['spam']
ratio = ham_count / spam_count
print(f"\nClass Imbalance Ratio (Ham:Spam): {ratio:.2f}:1")

# Message length statistics
print("\n" + "-"*70)
print("MESSAGE LENGTH STATISTICS")
print("-"*70)
stats = get_message_stats(data['message'], data['label'])
print(stats)

# Display sample messages
display_sample_messages(data, n_samples=3)

#%% Text Preprocessing
print("\nPreprocessing text messages...")
data['message_clean'] = data['message'].apply(preprocess_text)
print("✓ Text preprocessing complete")

# Show preprocessing examples
print("\nPreprocessing Examples:")
print("-"*70)
for i in range(3):
    print(f"\nOriginal: {data['message'].iloc[i][:100]}")
    print(f"Cleaned:  {data['message_clean'].iloc[i][:100]}")

#%% Train/Test Split
print("\n" + "="*70)
print("TRAIN/TEST SPLIT")
print("="*70)

# Prepare features and labels
X = data['message_clean']
y = data['label'].map({'ham': 0, 'spam': 1})  # Binary encoding

# 80/20 split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"\nTotal Samples: {len(data)}")
print(f"Training Set: {len(X_train)} ({len(X_train)/len(data)*100:.1f}%)")
print(f"Test Set: {len(X_test)} ({len(X_test)/len(data)*100:.1f}%)")

print(f"\nTraining Set Distribution:")
print(f"  Ham: {sum(y_train == 0)} ({sum(y_train == 0)/len(y_train)*100:.1f}%)")
print(f"  Spam: {sum(y_train == 1)} ({sum(y_train == 1)/len(y_train)*100:.1f}%)")

print(f"\nTest Set Distribution:")
print(f"  Ham: {sum(y_test == 0)} ({sum(y_test == 0)/len(y_test)*100:.1f}%)")
print(f"  Spam: {sum(y_test == 1)} ({sum(y_test == 1)/len(y_test)*100:.1f}%)")

#%% Save preprocessed data
print("\nSaving preprocessed data...")
X_train.to_csv(os.path.join(DB_PATH, 'X_train.csv'), index=False, header=['message'])
X_test.to_csv(os.path.join(DB_PATH, 'X_test.csv'), index=False, header=['message'])
np.save(os.path.join(DB_PATH, 'y_train.npy'), y_train.values)
np.save(os.path.join(DB_PATH, 'y_test.npy'), y_test.values)
print(f"✓ Saved to {DB_PATH}/")

print("\n" + "="*70)
print("DATA PREPROCESSING COMPLETE")
print("="*70)

# %%
