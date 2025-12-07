#%% Imports
import pandas as pd
import numpy as np
import os
import json
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from scipy import sparse
import joblib
from sentence_transformers import SentenceTransformer
from utils import print_evaluation

#%% Configuration
print("="*70)
print("HYBRID CASCADE SPAM CLASSIFIER")
print("="*70)

DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(DIR, 'data')
GRAPHS_DIR = os.path.join(DIR, 'graphs')

# Create graphs directory if needed
if not os.path.exists(GRAPHS_DIR):
    os.makedirs(GRAPHS_DIR)

#%% Load Optimal Thresholds
print("\n" + "="*70)
print("LOADING CONFIGURATION")
print("="*70)

thresholds_file = os.path.join(GRAPHS_DIR, 'optimal_thresholds.json')
if not os.path.exists(thresholds_file):
    print(f"\nERROR: {thresholds_file} not found")
    print("Please run threshold_optimizer.py first.")
    exit(1)

with open(thresholds_file, 'r') as f:
    config = json.load(f)

print(f"OK Loaded threshold configuration")
print(f"\nStage 1 (TF-IDF) Thresholds:")
print(f"  Low threshold:  {config['stage1_low_threshold']:.6f} (100% recall)")
print(f"  High threshold: {config['stage1_high_threshold']:.6f} (high confidence)")

print(f"\nStage 2 (Embeddings) Thresholds:")
print(f"  Low threshold:  {config['stage2_low_threshold']:.6f}")
print(f"  High threshold: {config['stage2_high_threshold']:.6f}")

#%% Load Test Data
print("\n" + "="*70)
print("LOADING TEST DATA")
print("="*70)

# Load labels
y_test = np.load(os.path.join(DB_PATH, 'y_test.npy'))
print(f"OK Loaded test labels: {len(y_test)} samples")
print(f"  Spam: {sum(y_test == 1)} ({sum(y_test == 1)/len(y_test)*100:.1f}%)")
print(f"  Ham: {sum(y_test == 0)} ({sum(y_test == 0)/len(y_test)*100:.1f}%)")

# Load messages
X_test_messages = pd.read_csv(os.path.join(DB_PATH, 'X_test.csv'))['message']
print(f"OK Loaded test messages")

#%% Load Stage 1: TF-IDF Model
print("\n" + "="*70)
print("LOADING STAGE 1: TF-IDF MODEL")
print("="*70)

# Load TF-IDF features
X_test_tfidf = sparse.load_npz(os.path.join(DB_PATH, 'X_test_tfidf.npz'))
print(f"OK Loaded TF-IDF features: {X_test_tfidf.shape}")

# Load training data to retrain model
X_train_tfidf = sparse.load_npz(os.path.join(DB_PATH, 'X_train_tfidf.npz'))
y_train = np.load(os.path.join(DB_PATH, 'y_train.npy'))
print(f"OK Loaded training data: {X_train_tfidf.shape}")

# Train the model identified as best
model_name = config['model_name']
print(f"\nTraining {model_name}...")

# Convert to DataFrame
X_train_df = pd.DataFrame(X_train_tfidf.toarray())
X_test_df = pd.DataFrame(X_test_tfidf.toarray())

# Import and train model
model_mapping = {
    'LogisticRegression': ('sklearn.linear_model', 'LogisticRegression'),
    'LinearDiscriminantAnalysis': ('sklearn.discriminant_analysis', 'LinearDiscriminantAnalysis'),
    'SGDClassifier': ('sklearn.linear_model', 'SGDClassifier'),
    'PassiveAggressiveClassifier': ('sklearn.linear_model', 'PassiveAggressiveClassifier'),
    'RidgeClassifier': ('sklearn.linear_model', 'RidgeClassifier'),
    'RidgeClassifierCV': ('sklearn.linear_model', 'RidgeClassifierCV'),
    'ExtraTreesClassifier': ('sklearn.ensemble', 'ExtraTreesClassifier'),
    'RandomForestClassifier': ('sklearn.ensemble', 'RandomForestClassifier'),
    'BaggingClassifier': ('sklearn.ensemble', 'BaggingClassifier'),
    'GradientBoostingClassifier': ('sklearn.ensemble', 'GradientBoostingClassifier'),
    'LinearSVC': ('sklearn.svm', 'LinearSVC'),
}

if model_name in model_mapping:
    import importlib
    module_name, class_name = model_mapping[model_name]
    module = importlib.import_module(module_name)
    ModelClass = getattr(module, class_name)

    if model_name in ['LinearSVC', 'RidgeClassifier', 'RidgeClassifierCV']:
        from sklearn.calibration import CalibratedClassifierCV
        base_model = ModelClass()
        stage1_model = CalibratedClassifierCV(base_model, cv=3)
    else:
        stage1_model = ModelClass()

    stage1_model.fit(X_train_df, y_train)
    print("OK Stage 1 model trained")
else:
    print(f"WARNING Model {model_name} not recognized, using LogisticRegression")
    from sklearn.linear_model import LogisticRegression
    stage1_model = LogisticRegression(max_iter=1000)
    stage1_model.fit(X_train_df, y_train)

#%% Load Stage 2: Embedding Model
print("\n" + "="*70)
print("LOADING STAGE 2: EMBEDDING MODEL")
print("="*70)

# Load pre-computed embeddings
X_test_embeddings = np.load(os.path.join(DB_PATH, 'test_embeddings.npy'))
X_train_embeddings = np.load(os.path.join(DB_PATH, 'train_embeddings.npy'))
print(f"OK Loaded embeddings")
print(f"  Train: {X_train_embeddings.shape}")
print(f"  Test: {X_test_embeddings.shape}")

# Load embedding results to find best model
embedding_results_file = os.path.join(GRAPHS_DIR, 'embedding_ml_results.csv')
if os.path.exists(embedding_results_file):
    embedding_results = pd.read_csv(embedding_results_file, index_col=0)
    best_embedding_model_name = embedding_results.sort_values('F1 Score', ascending=False).index[0]
    print(f"OK Best embedding model: {best_embedding_model_name}")

    # Train best embedding model
    if best_embedding_model_name in model_mapping:
        module_name, class_name = model_mapping[best_embedding_model_name]
        module = importlib.import_module(module_name)
        ModelClass = getattr(module, class_name)

        if best_embedding_model_name in ['LinearSVC', 'RidgeClassifier', 'RidgeClassifierCV']:
            from sklearn.calibration import CalibratedClassifierCV
            base_model = ModelClass()
            stage2_model = CalibratedClassifierCV(base_model, cv=3)
        else:
            stage2_model = ModelClass()

        X_train_emb_df = pd.DataFrame(X_train_embeddings)
        X_test_emb_df = pd.DataFrame(X_test_embeddings)

        print(f"Training {best_embedding_model_name}...")
        stage2_model.fit(X_train_emb_df, y_train)
        print("OK Stage 2 model trained")
    else:
        print(f"WARNING Model {best_embedding_model_name} not recognized, using LogisticRegression")
        from sklearn.linear_model import LogisticRegression
        stage2_model = LogisticRegression(max_iter=1000)
        X_train_emb_df = pd.DataFrame(X_train_embeddings)
        X_test_emb_df = pd.DataFrame(X_test_embeddings)
        stage2_model.fit(X_train_emb_df, y_train)
else:
    print("WARNING Embedding results not found, using LogisticRegression")
    from sklearn.linear_model import LogisticRegression
    stage2_model = LogisticRegression(max_iter=1000)
    X_train_emb_df = pd.DataFrame(X_train_embeddings)
    X_test_emb_df = pd.DataFrame(X_test_embeddings)
    stage2_model.fit(X_train_emb_df, y_train)

#%% Check for Stage 3: LLM Results
print("\n" + "="*70)
print("CHECKING STAGE 3: LLM RESULTS")
print("="*70)

llm_cache_file = os.path.join(GRAPHS_DIR, 'llm_results_cache.json')
llm_available = False
llm_predictions = {}

if os.path.exists(llm_cache_file):
    with open(llm_cache_file, 'r') as f:
        llm_cache = json.load(f)

    if 'results' in llm_cache and len(llm_cache['results']) > 0:
        llm_available = True
        # Create lookup dict by message index
        for result in llm_cache['results']:
            llm_predictions[result['index']] = result['predicted_label']

        print(f"OK Loaded cached LLM predictions")
        print(f"  Available for {len(llm_predictions)} messages")
    else:
        print("WARNING LLM cache exists but contains no results")
else:
    print("WARNING No LLM results available (run llm_classifier.py to generate)")

print(f"\nStage 3 status: {'Available' if llm_available else 'Not available - will use Stage 2 fallback'}")

#%% Run Hybrid Cascade Classification
print("\n" + "="*70)
print("RUNNING HYBRID CASCADE CLASSIFICATION")
print("="*70)

# Initialize tracking
results = []
stage_counts = {1: 0, 2: 0, 3: 0}
stage1_time = 0
stage2_time = 0
stage3_time = 0

print(f"\nProcessing {len(X_test_messages)} test messages...")

for idx in range(len(X_test_messages)):
    message = X_test_messages.iloc[idx]
    true_label = int(y_test[idx])

    result = {
        'index': idx,
        'message': message,
        'true_label': true_label,
        'stages_used': [],
        'confidences': {},
        'final_prediction': None,
        'final_stage': None,
        'cost': 0.0,
        'latency_ms': 0.0
    }

    # STAGE 1: TF-IDF Classification
    start_time = time.time()

    # Get probability from Stage 1
    tfidf_features = X_test_df.iloc[[idx]]
    if hasattr(stage1_model, 'predict_proba'):
        stage1_prob = stage1_model.predict_proba(tfidf_features)[0, 1]
    else:
        # Use decision function
        stage1_score = stage1_model.decision_function(tfidf_features)[0]
        stage1_prob = 1 / (1 + np.exp(-stage1_score))  # Sigmoid

    stage1_latency = (time.time() - start_time) * 1000
    stage1_time += stage1_latency

    result['stages_used'].append(1)
    result['confidences']['stage1'] = float(stage1_prob)
    result['latency_ms'] += stage1_latency

    # Decision logic for Stage 1
    if stage1_prob >= config['stage1_high_threshold']:
        # High confidence: SPAM
        result['final_prediction'] = 1
        result['final_stage'] = 1
        stage_counts[1] += 1

    elif stage1_prob <= config['stage1_low_threshold']:
        # High confidence: HAM
        result['final_prediction'] = 0
        result['final_stage'] = 1
        stage_counts[1] += 1

    else:
        # Uncertain - proceed to Stage 2
        # STAGE 2: Embedding Classification
        start_time = time.time()

        embedding_features = X_test_emb_df.iloc[[idx]]
        if hasattr(stage2_model, 'predict_proba'):
            stage2_prob = stage2_model.predict_proba(embedding_features)[0, 1]
        else:
            stage2_score = stage2_model.decision_function(embedding_features)[0]
            stage2_prob = 1 / (1 + np.exp(-stage2_score))

        stage2_latency = (time.time() - start_time) * 1000
        stage2_time += stage2_latency

        result['stages_used'].append(2)
        result['confidences']['stage2'] = float(stage2_prob)
        result['latency_ms'] += stage2_latency

        # Decision logic for Stage 2
        if stage2_prob >= config['stage2_high_threshold']:
            # High confidence: SPAM
            result['final_prediction'] = 1
            result['final_stage'] = 2
            stage_counts[2] += 1

        elif stage2_prob <= config['stage2_low_threshold']:
            # High confidence: HAM
            result['final_prediction'] = 0
            result['final_stage'] = 2
            stage_counts[2] += 1

        else:
            # Still uncertain - proceed to Stage 3 (LLM)
            if llm_available and idx in llm_predictions:
                # Use cached LLM prediction
                result['stages_used'].append(3)
                result['final_prediction'] = llm_predictions[idx]
                result['final_stage'] = 3
                stage_counts[3] += 1

                # Add estimated cost and latency for LLM
                result['cost'] = 0.0001  # Estimated cost per LLM call
                result['latency_ms'] += 200  # Estimated LLM latency
                stage3_time += 200

            else:
                # No LLM available - use Stage 2 prediction as fallback
                result['final_prediction'] = 1 if stage2_prob >= 0.5 else 0
                result['final_stage'] = 2
                result['confidences']['stage2_fallback'] = True
                # Don't increment stage2_counts again, but note this was a fallback

    results.append(result)

    # Progress update
    if (idx + 1) % 100 == 0:
        print(f"  Processed {idx + 1}/{len(X_test_messages)} messages...")

print(f"\nOK Classification complete!")

#%% Calculate Performance Metrics
print("\n" + "="*70)
print("PERFORMANCE EVALUATION")
print("="*70)

# Extract predictions
y_pred = np.array([r['final_prediction'] for r in results])
y_true = np.array([r['true_label'] for r in results])

# Overall metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)

print(f"\nOverall Performance:")
print(f"  Accuracy:  {accuracy:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  F1-Score:  {f1:.4f}")

# Detailed evaluation
print_evaluation(y_true, y_pred, title="Hybrid Cascade Classifier")

# Stage usage statistics
total_messages = len(results)
print(f"\nStage Usage Statistics:")
print(f"  Stage 1 (TF-IDF):     {stage_counts[1]:4d} ({stage_counts[1]/total_messages*100:5.1f}%)")
print(f"  Stage 2 (Embeddings): {stage_counts[2]:4d} ({stage_counts[2]/total_messages*100:5.1f}%)")
print(f"  Stage 3 (LLM):        {stage_counts[3]:4d} ({stage_counts[3]/total_messages*100:5.1f}%)")

#%% Cost and Latency Analysis
print("\n" + "="*70)
print("COST AND LATENCY ANALYSIS")
print("="*70)

# Calculate total cost
total_cost = sum(r['cost'] for r in results)
avg_cost_per_message = total_cost / len(results)

print(f"\nCost Analysis:")
print(f"  Total cost:           ${total_cost:.6f}")
print(f"  Avg cost per message: ${avg_cost_per_message:.8f}")
print(f"  Cost per 1K messages: ${avg_cost_per_message * 1000:.4f}")
print(f"  Cost per 1M messages: ${avg_cost_per_message * 1_000_000:.2f}")

# Latency statistics
latencies = [r['latency_ms'] for r in results]
print(f"\nLatency Analysis:")
print(f"  Mean:   {np.mean(latencies):.2f}ms")
print(f"  Median: {np.median(latencies):.2f}ms")
print(f"  P50:    {np.percentile(latencies, 50):.2f}ms")
print(f"  P95:    {np.percentile(latencies, 95):.2f}ms")
print(f"  P99:    {np.percentile(latencies, 99):.2f}ms")

# Breakdown by stage
if stage_counts[1] > 0:
    avg_stage1_latency = stage1_time / total_messages
    print(f"\nAverage latency by stage:")
    print(f"  Stage 1: {avg_stage1_latency:.2f}ms")

if stage_counts[2] > 0:
    avg_stage2_latency = stage2_time / stage_counts[2]
    print(f"  Stage 2: {avg_stage2_latency:.2f}ms (for {stage_counts[2]} messages)")

if stage_counts[3] > 0:
    avg_stage3_latency = stage3_time / stage_counts[3]
    print(f"  Stage 3: {avg_stage3_latency:.2f}ms (for {stage_counts[3]} messages)")

#%% Save Results
print("\n" + "="*70)
print("SAVING RESULTS")
print("="*70)

# Save detailed results
results_df = pd.DataFrame(results)
results_csv = os.path.join(GRAPHS_DIR, 'hybrid_cascade_results.csv')
results_df.to_csv(results_csv, index=False)
print(f"OK Saved detailed results to {results_csv}")

# Save summary
summary = {
    'model_config': {
        'stage1_model': config['model_name'],
        'stage2_model': best_embedding_model_name if 'best_embedding_model_name' in locals() else 'LogisticRegression',
        'stage3_available': llm_available
    },
    'thresholds': config,
    'performance': {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1)
    },
    'stage_usage': {
        'stage1_count': int(stage_counts[1]),
        'stage1_pct': float(stage_counts[1]/total_messages*100),
        'stage2_count': int(stage_counts[2]),
        'stage2_pct': float(stage_counts[2]/total_messages*100),
        'stage3_count': int(stage_counts[3]),
        'stage3_pct': float(stage_counts[3]/total_messages*100)
    },
    'cost': {
        'total_cost': float(total_cost),
        'avg_cost_per_message': float(avg_cost_per_message),
        'cost_per_1k': float(avg_cost_per_message * 1000),
        'cost_per_1m': float(avg_cost_per_message * 1_000_000)
    },
    'latency': {
        'mean_ms': float(np.mean(latencies)),
        'median_ms': float(np.median(latencies)),
        'p50_ms': float(np.percentile(latencies, 50)),
        'p95_ms': float(np.percentile(latencies, 95)),
        'p99_ms': float(np.percentile(latencies, 99))
    }
}

summary_file = os.path.join(GRAPHS_DIR, 'hybrid_cascade_summary.json')
with open(summary_file, 'w') as f:
    json.dump(summary, f, indent=2)
print(f"OK Saved summary to {summary_file}")

#%% Generate Visualizations
print("\n" + "="*70)
print("GENERATING VISUALIZATIONS")
print("="*70)

sns.set_style("whitegrid")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Stage Usage Pie Chart
ax1 = axes[0, 0]
stage_data = [stage_counts[1], stage_counts[2], stage_counts[3]]
stage_labels = [f'Stage 1: TF-IDF\n{stage_counts[1]} ({stage_counts[1]/total_messages*100:.1f}%)',
                f'Stage 2: Embeddings\n{stage_counts[2]} ({stage_counts[2]/total_messages*100:.1f}%)',
                f'Stage 3: LLM\n{stage_counts[3]} ({stage_counts[3]/total_messages*100:.1f}%)']
colors = ['#51cf66', '#ffd43b', '#ff6b6b']

wedges, texts, autotexts = ax1.pie(stage_data, labels=stage_labels, colors=colors,
                                     autopct='%1.1f%%', startangle=90,
                                     textprops={'fontsize': 10, 'fontweight': 'bold'})
ax1.set_title('Stage Usage Distribution', fontsize=14, fontweight='bold')

# 2. Latency Distribution
ax2 = axes[0, 1]
ax2.hist(latencies, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
ax2.axvline(np.mean(latencies), color='red', linestyle='--', linewidth=2,
            label=f'Mean: {np.mean(latencies):.1f}ms')
ax2.axvline(np.median(latencies), color='green', linestyle='--', linewidth=2,
            label=f'Median: {np.median(latencies):.1f}ms')
ax2.set_xlabel('Latency (ms)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax2.set_title('Latency Distribution', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Performance Metrics
ax3 = axes[1, 0]
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
values = [accuracy, precision, recall, f1]
colors_metrics = ['steelblue', 'forestgreen', 'orange', 'purple']

bars = ax3.bar(metrics, values, color=colors_metrics, alpha=0.7, edgecolor='black')
ax3.set_ylabel('Score', fontsize=12, fontweight='bold')
ax3.set_title('Hybrid Cascade Performance', fontsize=14, fontweight='bold')
ax3.set_ylim([0, 1.05])
ax3.grid(True, alpha=0.3, axis='y')

for bar, value in zip(bars, values):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
             f'{value:.4f}',
             ha='center', va='bottom', fontsize=10, fontweight='bold')

# 4. Cost Comparison (projected)
ax4 = axes[1, 1]
approaches = ['TF-IDF\nOnly', 'Embeddings\nOnly', 'LLM\nOnly', 'Hybrid\nCascade']
costs_per_1k = [
    0,  # TF-IDF
    0,  # Embeddings
    100,  # LLM only (estimated $0.10 per 1K at ~$0.0001 per msg)
    avg_cost_per_message * 1000  # Hybrid
]

bars = ax4.bar(approaches, costs_per_1k, color=['green', 'yellow', 'red', 'blue'], alpha=0.7, edgecolor='black')
ax4.set_ylabel('Cost per 1K Messages ($)', fontsize=12, fontweight='bold')
ax4.set_title('Cost Comparison (per 1K messages)', fontsize=14, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')

for bar, cost in zip(bars, costs_per_1k):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
             f'${cost:.2f}' if cost > 0.01 else f'${cost:.4f}',
             ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
viz_path = os.path.join(GRAPHS_DIR, 'hybrid_cascade_analysis.png')
plt.savefig(viz_path, dpi=300, bbox_inches='tight')
print(f"OK Saved visualization to {viz_path}")
plt.close()

#%% Final Summary
print("\n" + "="*70)
print("HYBRID CASCADE COMPLETE")
print("="*70)

print(f"\nPerformance: Performance:")
print(f"   Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")

print(f"\nDistribution: Stage Distribution:")
print(f"   Stage 1: {stage_counts[1]/total_messages*100:.1f}% | Stage 2: {stage_counts[2]/total_messages*100:.1f}% | Stage 3: {stage_counts[3]/total_messages*100:.1f}%")

print(f"\nCost: Cost Efficiency:")
print(f"   Avg: ${avg_cost_per_message:.8f}/msg | Per 1K: ${avg_cost_per_message*1000:.4f}")

print(f"\nSpeed: Speed:")
print(f"   Mean: {np.mean(latencies):.1f}ms | P95: {np.percentile(latencies, 95):.1f}ms")

print(f"\nOutput: Output files:")
print(f"   - {results_csv}")
print(f"   - {summary_file}")
print(f"   - {viz_path}")

if not llm_available:
    print(f"\nWARNING Note: Stage 3 (LLM) was not available")
    print(f"  Run llm_classifier.py to enable full 3-stage cascade")
else:
    print(f"\nOK Full 3-stage cascade operational!")

print("\n" + "="*70 + "\n")

# %%
