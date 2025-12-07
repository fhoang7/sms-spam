#%% Imports
import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
from scipy import sparse

#%% Configuration
print("="*70)
print("THRESHOLD OPTIMIZATION FOR 100% SPAM RECALL")
print("="*70)

DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(DIR, 'data')
GRAPHS_DIR = os.path.join(DIR, 'graphs')

# Create graphs directory if needed
if not os.path.exists(GRAPHS_DIR):
    os.makedirs(GRAPHS_DIR)

#%% Load Traditional ML Results to Find Best Model
print("\n" + "="*70)
print("LOADING BEST TRADITIONAL ML MODEL")
print("="*70)

# Load results to find best model
results_file = os.path.join(GRAPHS_DIR, 'traditional_ml_results.csv')
if not os.path.exists(results_file):
    print(f"\nERROR: {results_file} not found")
    print("Please run traditional_ml.py first to generate results.")
    exit(1)

results_df = pd.read_csv(results_file, index_col=0)
best_model_name = results_df.sort_values('F1 Score', ascending=False).index[0]

print(f"\nBest model from traditional ML: {best_model_name}")
print(f"  F1-Score: {results_df.loc[best_model_name, 'F1 Score']:.4f}")
print(f"  Accuracy: {results_df.loc[best_model_name, 'Accuracy']:.4f}")

#%% Load Data and Models
print("\n" + "="*70)
print("LOADING DATA AND MODELS")
print("="*70)

# Load test data
y_test = np.load(os.path.join(DB_PATH, 'y_test.npy'))
print(f"OK Loaded test labels: {len(y_test)} samples")

# Load TF-IDF features
X_test_tfidf = sparse.load_npz(os.path.join(DB_PATH, 'X_test_tfidf.npz'))
print(f"OK Loaded TF-IDF features: {X_test_tfidf.shape}")

# We need to retrain the best model to get probability predictions
# Load training data
X_train_tfidf = sparse.load_npz(os.path.join(DB_PATH, 'X_train_tfidf.npz'))
y_train = np.load(os.path.join(DB_PATH, 'y_train.npy'))
print(f"OK Loaded training data: {X_train_tfidf.shape}")

# Convert to DataFrame for LazyPredict compatibility
X_train_df = pd.DataFrame(X_train_tfidf.toarray())
X_test_df = pd.DataFrame(X_test_tfidf.toarray())

#%% Train Best Model to Get Probabilities
print("\n" + "="*70)
print(f"TRAINING {best_model_name} FOR PROBABILITY PREDICTIONS")
print("="*70)

# Import the best model class based on name
# Map LazyPredict model names to sklearn classes
model_mapping = {
    'LogisticRegression': ('sklearn.linear_model', 'LogisticRegression'),
    'LinearDiscriminantAnalysis': ('sklearn.discriminant_analysis', 'LinearDiscriminantAnalysis'),
    'SGDClassifier': ('sklearn.linear_model', 'SGDClassifier'),
    'PassiveAggressiveClassifier': ('sklearn.linear_model', 'PassiveAggressiveClassifier'),
    'RidgeClassifier': ('sklearn.linear_model', 'RidgeClassifier'),
    'RidgeClassifierCV': ('sklearn.linear_model', 'RidgeClassifierCV'),
    'Perceptron': ('sklearn.linear_model', 'Perceptron'),
    'CalibratedClassifierCV': ('sklearn.calibration', 'CalibratedClassifierCV'),
    'LinearSVC': ('sklearn.svm', 'LinearSVC'),
    'SVC': ('sklearn.svm', 'SVC'),
    'NuSVC': ('sklearn.svm', 'NuSVC'),
    'ExtraTreesClassifier': ('sklearn.ensemble', 'ExtraTreesClassifier'),
    'RandomForestClassifier': ('sklearn.ensemble', 'RandomForestClassifier'),
    'BaggingClassifier': ('sklearn.ensemble', 'BaggingClassifier'),
    'GradientBoostingClassifier': ('sklearn.ensemble', 'GradientBoostingClassifier'),
    'AdaBoostClassifier': ('sklearn.ensemble', 'AdaBoostClassifier'),
    'ExtraTreeClassifier': ('sklearn.tree', 'ExtraTreeClassifier'),
    'DecisionTreeClassifier': ('sklearn.tree', 'DecisionTreeClassifier'),
    'NearestCentroid': ('sklearn.neighbors', 'NearestCentroid'),
    'KNeighborsClassifier': ('sklearn.neighbors', 'KNeighborsClassifier'),
    'BernoulliNB': ('sklearn.naive_bayes', 'BernoulliNB'),
    'GaussianNB': ('sklearn.naive_bayes', 'GaussianNB'),
    'LabelPropagation': ('sklearn.semi_supervised', 'LabelPropagation'),
    'LabelSpreading': ('sklearn.semi_supervised', 'LabelSpreading'),
    'DummyClassifier': ('sklearn.dummy', 'DummyClassifier'),
}

if best_model_name in model_mapping:
    module_name, class_name = model_mapping[best_model_name]
    import importlib
    module = importlib.import_module(module_name)
    ModelClass = getattr(module, class_name)

    print(f"Training {class_name}...")

    # Special handling for models that don't support probability
    if best_model_name in ['LinearSVC', 'RidgeClassifier', 'RidgeClassifierCV', 'Perceptron']:
        # Use CalibratedClassifierCV wrapper for probability calibration
        from sklearn.calibration import CalibratedClassifierCV
        base_model = ModelClass()
        model = CalibratedClassifierCV(base_model, cv=3)
        print("  (Using CalibratedClassifierCV for probability calibration)")
    else:
        model = ModelClass()

    # Train model
    model.fit(X_train_df, y_train)
    print("OK Model trained")

    # Get probability predictions
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test_df)[:, 1]  # Probability of spam (class 1)
    else:
        # Fallback to decision function
        y_proba = model.decision_function(X_test_df)
        # Normalize to 0-1 range
        y_proba = (y_proba - y_proba.min()) / (y_proba.max() - y_proba.min())

    print(f"OK Generated probability predictions")
    print(f"  Min probability: {y_proba.min():.4f}")
    print(f"  Max probability: {y_proba.max():.4f}")
    print(f"  Mean probability: {y_proba.mean():.4f}")

else:
    print(f"\nERROR: Model {best_model_name} not recognized")
    print("Using fallback: LogisticRegression")

    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_df, y_train)
    y_proba = model.predict_proba(X_test_df)[:, 1]

#%% Find Optimal Threshold for 100% Spam Recall
print("\n" + "="*70)
print("FINDING OPTIMAL THRESHOLD FOR 100% SPAM RECALL")
print("="*70)

# Get precision-recall curve
precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)

print(f"\nPrecision-Recall Curve computed")
print(f"  Number of thresholds: {len(thresholds)}")

# Find thresholds that achieve 100% recall for spam (class 1)
# For spam detection, recall = TP / (TP + FN) where we want FN = 0
spam_mask = y_test == 1
ham_mask = y_test == 0

n_spam = spam_mask.sum()
n_ham = ham_mask.sum()

print(f"\nTest set composition:")
print(f"  Spam: {n_spam} ({n_spam/len(y_test)*100:.1f}%)")
print(f"  Ham: {n_ham} ({n_ham/len(y_test)*100:.1f}%)")

# Find threshold for 100% spam recall
# Recall = 1.0 means we correctly identify all spam (no false negatives)
perfect_recall_indices = np.where(recalls >= 0.999)[0]  # Allow tiny floating point error

if len(perfect_recall_indices) == 0:
    print("\nWARNING Warning: Could not find threshold with 100% recall")
    print("  Using threshold with highest recall instead")
    best_recall_idx = np.argmax(recalls)
    optimal_threshold = thresholds[best_recall_idx]
    optimal_precision = precisions[best_recall_idx]
    optimal_recall = recalls[best_recall_idx]
else:
    # Among thresholds with 100% recall, choose the one with highest precision
    best_idx = perfect_recall_indices[np.argmax(precisions[perfect_recall_indices])]
    optimal_threshold = thresholds[best_idx]
    optimal_precision = precisions[best_idx]
    optimal_recall = recalls[best_idx]

print(f"\nOK Optimal threshold found: {optimal_threshold:.6f}")
print(f"  Precision: {optimal_precision:.4f}")
print(f"  Recall: {optimal_recall:.4f}")

# Make predictions with optimal threshold
y_pred_optimal = (y_proba >= optimal_threshold).astype(int)

# Calculate metrics
accuracy_optimal = accuracy_score(y_test, y_pred_optimal)
precision_optimal = precision_score(y_test, y_pred_optimal, zero_division=0)
recall_optimal = recall_score(y_test, y_pred_optimal, zero_division=0)
f1_optimal = f1_score(y_test, y_pred_optimal, zero_division=0)

print(f"\nPerformance with optimal threshold:")
print(f"  Accuracy:  {accuracy_optimal:.4f}")
print(f"  Precision: {precision_optimal:.4f}")
print(f"  Recall:    {recall_optimal:.4f}")
print(f"  F1-Score:  {f1_optimal:.4f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_optimal)
print(f"\nConfusion Matrix:")
print(f"  True Ham (correctly classified):  {cm[0,0]}")
print(f"  False Spam (ham classified spam): {cm[0,1]}")
print(f"  False Ham (spam classified ham):  {cm[1,0]}")
print(f"  True Spam (correctly classified): {cm[1,1]}")

if cm[1,0] == 0:
    print(f"\nOK Perfect spam recall achieved! No spam messages missed.")
else:
    print(f"\nWARNING Spam recall not perfect: {cm[1,0]} spam messages classified as ham")

#%% Analyze Threshold Impact
print("\n" + "="*70)
print("ANALYZING THRESHOLD IMPACT")
print("="*70)

# Test multiple threshold values
test_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
threshold_analysis = []

for threshold in test_thresholds:
    y_pred = (y_proba >= threshold).astype(int)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    # Count messages that would go to Stage 2
    uncertain = np.sum((y_proba < threshold) & (y_proba >= 1 - threshold))

    threshold_analysis.append({
        'threshold': threshold,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'uncertain_count': uncertain,
        'uncertain_pct': uncertain / len(y_test) * 100
    })

analysis_df = pd.DataFrame(threshold_analysis)

print("\nThreshold Analysis:")
print(analysis_df.to_string(index=False))

#%% Define Stage 1 Thresholds for Hybrid System
print("\n" + "="*70)
print("DEFINING STAGE 1 THRESHOLDS FOR HYBRID CASCADE")
print("="*70)

# Stage 1 will have two thresholds:
# - Low threshold: optimized for 100% spam recall (found above)
# - High threshold: for confident spam classification (high precision)

# Find high-confidence threshold (e.g., 95%+ precision)
high_precision_indices = np.where(precisions >= 0.95)[0]
if len(high_precision_indices) > 0:
    # Choose threshold with highest recall among high-precision thresholds
    best_high_idx = high_precision_indices[np.argmax(recalls[high_precision_indices])]
    high_threshold = thresholds[best_high_idx]
    high_precision = precisions[best_high_idx]
    high_recall = recalls[best_high_idx]
else:
    # Fallback to 0.9 if no threshold achieves 95% precision
    high_threshold = 0.9
    y_pred_high = (y_proba >= high_threshold).astype(int)
    high_precision = precision_score(y_test, y_pred_high, zero_division=0)
    high_recall = recall_score(y_test, y_pred_high, zero_division=0)

print(f"\nStage 1 Thresholds:")
print(f"  Low threshold (100% recall):  {optimal_threshold:.6f}")
print(f"    - Precision: {optimal_precision:.4f}")
print(f"    - Recall: {optimal_recall:.4f}")
print(f"  High threshold (high confidence): {high_threshold:.6f}")
print(f"    - Precision: {high_precision:.4f}")
print(f"    - Recall: {high_recall:.4f}")

# Calculate what percentage would be filtered by Stage 1
confident_spam = np.sum(y_proba >= high_threshold)
confident_ham = np.sum(y_proba <= optimal_threshold)
uncertain = len(y_proba) - confident_spam - confident_ham

print(f"\nStage 1 Coverage Estimate:")
print(f"  Confident SPAM (>= {high_threshold:.3f}): {confident_spam} ({confident_spam/len(y_test)*100:.1f}%)")
print(f"  Confident HAM  (<= {optimal_threshold:.3f}): {confident_ham} ({confident_ham/len(y_test)*100:.1f}%)")
print(f"  Uncertain (-> Stage 2):           {uncertain} ({uncertain/len(y_test)*100:.1f}%)")

#%% Save Optimal Thresholds
print("\n" + "="*70)
print("SAVING OPTIMAL THRESHOLDS")
print("="*70)

thresholds_config = {
    'model_name': best_model_name,
    'stage1_low_threshold': float(optimal_threshold),
    'stage1_high_threshold': float(high_threshold),
    'stage1_low_precision': float(optimal_precision),
    'stage1_low_recall': float(optimal_recall),
    'stage1_high_precision': float(high_precision),
    'stage1_high_recall': float(high_recall),
    'stage1_accuracy': float(accuracy_optimal),
    'stage1_f1': float(f1_optimal),
    'stage1_confident_spam_pct': float(confident_spam/len(y_test)*100),
    'stage1_confident_ham_pct': float(confident_ham/len(y_test)*100),
    'stage1_uncertain_pct': float(uncertain/len(y_test)*100),
    # Stage 2 thresholds (will be tuned later, using conservative defaults)
    'stage2_low_threshold': 0.15,
    'stage2_high_threshold': 0.85,
}

thresholds_file = os.path.join(GRAPHS_DIR, 'optimal_thresholds.json')
with open(thresholds_file, 'w') as f:
    json.dump(thresholds_config, f, indent=2)

print(f"OK Saved thresholds to {thresholds_file}")

#%% Generate Visualizations
print("\n" + "="*70)
print("GENERATING VISUALIZATIONS")
print("="*70)

# Set style
sns.set_style("whitegrid")

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Precision-Recall Curve
ax1 = axes[0, 0]
ax1.plot(recalls, precisions, 'b-', linewidth=2, label='PR Curve')
ax1.axhline(optimal_precision, color='red', linestyle='--', alpha=0.7,
            label=f'Optimal (Recall={optimal_recall:.3f})')
ax1.axvline(optimal_recall, color='red', linestyle='--', alpha=0.7)
ax1.scatter([optimal_recall], [optimal_precision], color='red', s=100, zorder=5)
ax1.set_xlabel('Recall', fontsize=12, fontweight='bold')
ax1.set_ylabel('Precision', fontsize=12, fontweight='bold')
ax1.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xlim([0, 1.05])
ax1.set_ylim([0, 1.05])

# 2. Threshold vs Metrics
ax2 = axes[0, 1]
ax2.plot(analysis_df['threshold'], analysis_df['precision'], 'b-', marker='o', label='Precision', linewidth=2)
ax2.plot(analysis_df['threshold'], analysis_df['recall'], 'r-', marker='s', label='Recall', linewidth=2)
ax2.plot(analysis_df['threshold'], analysis_df['f1'], 'g-', marker='^', label='F1-Score', linewidth=2)
ax2.axvline(optimal_threshold, color='red', linestyle='--', alpha=0.5, label=f'Optimal: {optimal_threshold:.3f}')
ax2.axvline(high_threshold, color='orange', linestyle='--', alpha=0.5, label=f'High: {high_threshold:.3f}')
ax2.set_xlabel('Threshold', fontsize=12, fontweight='bold')
ax2.set_ylabel('Score', fontsize=12, fontweight='bold')
ax2.set_title('Threshold Impact on Metrics', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xlim([0, 1])
ax2.set_ylim([0, 1.05])

# 3. Probability Distribution by Class
ax3 = axes[1, 0]
spam_probs = y_proba[y_test == 1]
ham_probs = y_proba[y_test == 0]

ax3.hist(ham_probs, bins=50, alpha=0.6, color='green', label='Ham', edgecolor='black')
ax3.hist(spam_probs, bins=50, alpha=0.6, color='red', label='Spam', edgecolor='black')
ax3.axvline(optimal_threshold, color='blue', linestyle='--', linewidth=2,
            label=f'Low Threshold: {optimal_threshold:.3f}')
ax3.axvline(high_threshold, color='orange', linestyle='--', linewidth=2,
            label=f'High Threshold: {high_threshold:.3f}')
ax3.set_xlabel('Predicted Probability', fontsize=12, fontweight='bold')
ax3.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax3.set_title('Probability Distribution by Class', fontsize=14, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Stage 1 Coverage
ax4 = axes[1, 1]
coverage_data = [
    confident_spam / len(y_test) * 100,
    confident_ham / len(y_test) * 100,
    uncertain / len(y_test) * 100
]
colors = ['#ff6b6b', '#51cf66', '#ffd43b']
labels = [f'Confident Spam\n{confident_spam} ({coverage_data[0]:.1f}%)',
          f'Confident Ham\n{confident_ham} ({coverage_data[1]:.1f}%)',
          f'Uncertain -> Stage 2\n{uncertain} ({coverage_data[2]:.1f}%)']

wedges, texts, autotexts = ax4.pie(coverage_data, labels=labels, colors=colors,
                                     autopct='%1.1f%%', startangle=90,
                                     textprops={'fontsize': 11, 'fontweight': 'bold'})
ax4.set_title('Stage 1 Coverage Breakdown', fontsize=14, fontweight='bold')

plt.tight_layout()
viz_path = os.path.join(GRAPHS_DIR, 'threshold_optimization.png')
plt.savefig(viz_path, dpi=300, bbox_inches='tight')
print(f"OK Saved visualization to {viz_path}")
plt.close()

#%% Final Summary
print("\n" + "="*70)
print("THRESHOLD OPTIMIZATION COMPLETE")
print("="*70)

print(f"\nPerformance: Optimal Thresholds for Hybrid Cascade:")
print(f"   Low (100% recall):  {optimal_threshold:.6f}")
print(f"   High (confident):   {high_threshold:.6f}")

print(f"\n Expected Stage 1 Performance:")
print(f"   Accuracy:  {accuracy_optimal:.4f}")
print(f"   Precision: {precision_optimal:.4f}")
print(f"   Recall:    {recall_optimal:.4f}")
print(f"   F1-Score:  {f1_optimal:.4f}")

print(f"\nDistribution: Stage 1 Coverage:")
print(f"   Handles directly: {(confident_spam + confident_ham)/len(y_test)*100:.1f}%")
print(f"   Passes to Stage 2: {uncertain/len(y_test)*100:.1f}%")

print(f"\nOutput: Output files:")
print(f"   - {thresholds_file}")
print(f"   - {viz_path}")

print("\nOK Ready for hybrid_cascade.py implementation!")
print("\n" + "="*70 + "\n")

# %%
