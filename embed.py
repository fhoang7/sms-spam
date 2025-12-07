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

#%% Generate Embeddings with Sentence Transformers
print("\n" + "="*70)
print("EMBEDDING GENERATION")
print("="*70)

from sentence_transformers import SentenceTransformer

# Initialize model
model_name = 'all-MiniLM-L6-v2'
print(f"\nLoading embedding model: {model_name}")
print("(This may take a moment on first run...)")
embedding_model = SentenceTransformer(model_name)
print("✓ Model loaded successfully")

# Test embedding
sample_text = "Win a free iPhone now!"
sample_embedding = embedding_model.encode(sample_text)
print(f"\nEmbedding dimensions: {sample_embedding.shape[0]}")
print(f"Sample embedding (first 5 values): {sample_embedding[:5]}")

# Generate embeddings for training set
print(f"\nGenerating embeddings for {len(X_train)} training messages...")
train_embeddings = embedding_model.encode(
    X_train.tolist(),
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True
)
print(f"✓ Training embeddings shape: {train_embeddings.shape}")

# Generate embeddings for test set
print(f"\nGenerating embeddings for {len(X_test)} test messages...")
test_embeddings = embedding_model.encode(
    X_test.tolist(),
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True
)
print(f"✓ Test embeddings shape: {test_embeddings.shape}")

# Save embeddings
print("\nSaving embeddings...")
np.save(os.path.join(DB_PATH, 'train_embeddings.npy'), train_embeddings)
np.save(os.path.join(DB_PATH, 'test_embeddings.npy'), test_embeddings)
print(f"✓ Embeddings saved to {DB_PATH}/")

#%% ChromaDB Integration
print("\n" + "="*70)
print("CHROMADB INTEGRATION")
print("="*70)

import chromadb
from chromadb.config import Settings

# Initialize ChromaDB client
print("\nInitializing ChromaDB...")
client = chromadb.PersistentClient(
    path=DB_PATH,
    settings=Settings(anonymized_telemetry=False)
)
print("✓ ChromaDB client initialized")

# Create or get collection
collection_name = "sms_spam_embeddings"
print(f"\nCreating collection: {collection_name}")

# Delete existing collection if it exists (for fresh start)
try:
    client.delete_collection(name=collection_name)
    print("  (Deleted existing collection)")
except:
    pass

collection = client.create_collection(
    name=collection_name,
    metadata={
        "model": model_name,
        "description": "SMS spam detection embeddings",
        "embedding_dim": str(train_embeddings.shape[1])
    }
)
print("✓ Collection created")

# Add training data to ChromaDB
print(f"\nAdding {len(X_train)} training samples to ChromaDB...")
train_ids = [f"train_{i}" for i in range(len(X_train))]
train_metadata = [
    {"label": int(label), "split": "train"}
    for label in y_train.values
]

collection.add(
    embeddings=train_embeddings.tolist(),
    documents=X_train.tolist(),
    metadatas=train_metadata,
    ids=train_ids
)
print("✓ Training data added")

# Add test data to ChromaDB
print(f"\nAdding {len(X_test)} test samples to ChromaDB...")
test_ids = [f"test_{i}" for i in range(len(X_test))]
test_metadata = [
    {"label": int(label), "split": "test"}
    for label in y_test.values
]

collection.add(
    embeddings=test_embeddings.tolist(),
    documents=X_test.tolist(),
    metadatas=test_metadata,
    ids=test_ids
)
print("✓ Test data added")

# Verify collection
print(f"\nCollection Statistics:")
print(f"  Total documents: {collection.count()}")
print(f"  Expected: {len(X_train) + len(X_test)}")

#%% Demonstrate Semantic Search
print("\n" + "="*70)
print("SEMANTIC SEARCH DEMONSTRATION")
print("="*70)

# Query with a spam-like message
query_text = "Congratulations! You've won a free prize"
print(f"\nQuery: '{query_text}'")
query_embedding = embedding_model.encode([query_text])

results = collection.query(
    query_embeddings=query_embedding.tolist(),
    n_results=5,
    where={"split": "train"}
)

print("\nTop 5 Similar Training Messages:")
print("-"*70)
for i, (doc, distance, metadata) in enumerate(zip(
    results['documents'][0],
    results['distances'][0],
    results['metadatas'][0]
), 1):
    label_name = "SPAM" if metadata['label'] == 1 else "HAM"
    msg_preview = doc[:80] + "..." if len(doc) > 80 else doc
    print(f"{i}. [{label_name}] Distance: {distance:.4f}")
    print(f"   {msg_preview}\n")

print("\n" + "="*70)
print("EMBEDDING GENERATION AND CHROMADB SETUP COMPLETE")
print("="*70)
print(f"\nSaved files in {DB_PATH}/:")
print("  - X_train.csv, X_test.csv")
print("  - y_train.npy, y_test.npy")
print("  - train_embeddings.npy, test_embeddings.npy")
print("  - chroma.sqlite3 (ChromaDB database)")

# %%
