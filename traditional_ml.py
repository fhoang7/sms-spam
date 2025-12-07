#%% Imports
import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_score, recall_score
from lazypredict.Supervised import LazyClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from utils import print_evaluation

#%% Load preprocessed data
print("Loading preprocessed data...")
DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(DIR, 'data')

X_train = pd.read_csv(os.path.join(DB_PATH, 'X_train.csv'))['message']
X_test = pd.read_csv(os.path.join(DB_PATH, 'X_test.csv'))['message']
y_train = np.load(os.path.join(DB_PATH, 'y_train.npy'))
y_test = np.load(os.path.join(DB_PATH, 'y_test.npy'))

print(f"✓ Loaded {len(X_train)} training and {len(X_test)} test samples")

#%% TF-IDF Feature Extraction
print("\n" + "="*70)
print("TF-IDF FEATURE EXTRACTION")
print("="*70)

print("\nConfiguration:")
print("  - max_features: 3000")
print("  - ngram_range: (1, 2) - unigrams + bigrams")
print("  - min_df: 2 - ignore terms in < 2 documents")
print("  - max_df: 0.95 - ignore terms in > 95% documents")
print("  - sublinear_tf: True - use log scaling")

vectorizer = TfidfVectorizer(
    max_features=3000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)

print("\nFitting vectorizer on training data...")
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print(f"✓ Training features shape: {X_train_tfidf.shape}")
print(f"✓ Test features shape: {X_test_tfidf.shape}")
print(f"✓ Vocabulary size: {len(vectorizer.vocabulary_)}")

# Show some feature names
feature_names = vectorizer.get_feature_names_out()
print(f"\nSample features (first 20):")
print(feature_names[:20])

#%% LazyPredict - Quick Model Comparison
print("\n" + "="*70)
print("LAZYPREDICT - BASELINE MODEL COMPARISON")
print("="*70)

print("\nConverting sparse matrices to DataFrames for LazyPredict...")
# LazyPredict requires DataFrame input, not sparse matrices
X_train_df = pd.DataFrame(X_train_tfidf.toarray())
X_test_df = pd.DataFrame(X_test_tfidf.toarray())
print(f"✓ Converted to DataFrames: {X_train_df.shape}, {X_test_df.shape}")

print("\nTesting 30+ classification models...")
print("(This may take a few minutes...)\n")

clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
models, predictions = clf.fit(X_train_df, X_test_df, y_train, y_test)

# Calculate Precision and Recall for each model from predictions
print("\nCalculating Precision and Recall for all models...")
precision_scores = []
recall_scores = []

for model_name in models.index:
    try:
        # Get predictions for this model
        y_pred = predictions[model_name].values
        precision = precision_score(y_test, y_pred, average='binary', zero_division=0)
        recall = recall_score(y_test, y_pred, average='binary', zero_division=0)
        precision_scores.append(precision)
        recall_scores.append(recall)
    except KeyError:
        # Model failed and has no predictions
        precision_scores.append(np.nan)
        recall_scores.append(np.nan)
    except Exception as e:
        # Other errors
        print(f"⚠ Error processing {model_name}: {e}")
        precision_scores.append(np.nan)
        recall_scores.append(np.nan)

# Add Precision and Recall columns to models DataFrame
models['Precision'] = precision_scores
models['Recall'] = recall_scores

print(f"✓ Evaluated {len(models)} models with Accuracy, F1-Score, ROC AUC, Precision, and Recall")

# Display results sorted by F1-Score
print("\nTop 15 Models by F1-Score:")
print("="*70)
top_models = models.sort_values('F1 Score', ascending=False).head(15)
print(top_models.to_string())

#%% Visualize LazyPredict Results
print("\n" + "="*70)
print("MODEL PERFORMANCE VISUALIZATION")
print("="*70)

# Create graphs directory
graphs_dir = 'graphs'
if not os.path.exists(graphs_dir):
    os.makedirs(graphs_dir)
    print(f"✓ Created {graphs_dir}/ directory")

# Plot top 10 models by different metrics
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']
top_n = 10

for idx, metric in enumerate(metrics):
    ax = axes[idx // 3, idx % 3]

    # Get top 10 models by this metric
    top_data = models.sort_values(metric, ascending=False).head(top_n)

    # Create horizontal bar plot
    y_pos = np.arange(len(top_data))
    ax.barh(y_pos, top_data[metric], color='steelblue', alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_data.index, fontsize=9)
    ax.set_xlabel(metric, fontsize=10)
    ax.set_title(f'Top {top_n} Models by {metric}', fontsize=12, fontweight='bold')

    # Dynamically adjust x-axis for better visibility
    # If minimum value is > 0.9, zoom in to show differences better
    min_val = top_data[metric].min()
    max_val = top_data[metric].max()

    if min_val > 0.9:
        # Zoom in to show differences in high-performing models
        x_min = max(0, min_val - 0.03)
        x_max = min(1.0, max_val + 0.01)
        ax.set_xlim([x_min, x_max])
    elif min_val > 0.85:
        # Moderate zoom for good performers
        x_min = max(0, min_val - 0.05)
        x_max = min(1.0, max_val + 0.02)
        ax.set_xlim([x_min, x_max])
    else:
        # Use standard 0-1 range for more varied scores
        ax.set_xlim([0, 1])

    ax.grid(axis='x', alpha=0.3)
    ax.invert_yaxis()

# Hide the empty subplot (we have 5 metrics in a 2x3 grid)
axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(graphs_dir, 'traditional_ml_comparison.png'), dpi=300, bbox_inches='tight')
print(f"✓ Saved visualization to {graphs_dir}/traditional_ml_comparison.png")
plt.close()

#%% Identify Top Performers
print("\n" + "="*70)
print("TOP PERFORMERS ANALYSIS")
print("="*70)

# Get top 3 models by F1-score
top_3 = models.sort_values('F1 Score', ascending=False).head(3)

print("\nTop 3 Models for Fine-Tuning:")
for i, (model_name, scores) in enumerate(top_3.iterrows(), 1):
    print(f"\n{i}. {model_name}")
    print(f"   Accuracy: {scores['Accuracy']:.4f}")
    print(f"   Precision: {scores['Precision']:.4f}")
    print(f"   Recall: {scores['Recall']:.4f}")
    print(f"   F1-Score: {scores['F1 Score']:.4f}")
    print(f"   ROC AUC: {scores['ROC AUC']:.4f}")

# Save results to CSV in graphs directory
models.to_csv(os.path.join(graphs_dir, 'traditional_ml_results.csv'))
print(f"\n✓ Full results saved to {graphs_dir}/traditional_ml_results.csv")

#%% Detailed Evaluation of Best Model
print("\n" + "="*70)
print("BEST MODEL DETAILED EVALUATION")
print("="*70)

best_model_name = models.sort_values('F1 Score', ascending=False).index[0]
print(f"\nBest Model: {best_model_name}")

# Get predictions from LazyPredict
if best_model_name in predictions.columns:
    best_predictions = predictions[best_model_name]
    # Print detailed evaluation
    print_evaluation(y_test, best_predictions, title=f"{best_model_name} Performance")
else:
    print(f"⚠ Cannot evaluate {best_model_name} - predictions not available")

#%% Save TF-IDF vectorizer for later use
print("\n" + "="*70)
print("SAVING ARTIFACTS")
print("="*70)

import joblib
joblib.dump(vectorizer, os.path.join(DB_PATH, 'tfidf_vectorizer.pkl'))
print(f"✓ Saved TF-IDF vectorizer to {DB_PATH}/tfidf_vectorizer.pkl")

# Save TF-IDF matrices for use in other scripts
from scipy import sparse
sparse.save_npz(os.path.join(DB_PATH, 'X_train_tfidf.npz'), X_train_tfidf)
sparse.save_npz(os.path.join(DB_PATH, 'X_test_tfidf.npz'), X_test_tfidf)
print(f"✓ Saved TF-IDF matrices to {DB_PATH}/")

print("\n" + "="*70)
print("TRADITIONAL ML BASELINE COMPLETE")
print("="*70)
print(f"\nNext steps:")
print("  1. Run class imbalance handling techniques")
print("  2. Fine-tune top models with Optuna")
print("  3. Compare with embedding-based approaches")

# %%
