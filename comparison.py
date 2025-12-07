#%% Imports
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import json

#%% Configuration
print("="*70)
print("TRADITIONAL ML vs EMBEDDING-BASED COMPARISON")
print("="*70)

DIR = os.path.dirname(os.path.abspath(__file__))
GRAPHS_DIR = os.path.join(DIR, 'graphs')

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'

#%% Load Results
print("\nLoading results from both approaches...")

# Load traditional ML results
try:
    traditional_results = pd.read_csv(os.path.join(GRAPHS_DIR, 'traditional_ml_results.csv'), index_col=0)
    print(f"✓ Loaded {len(traditional_results)} traditional ML models")
except FileNotFoundError:
    print("✗ traditional_ml_results.csv not found. Run traditional_ml.py first.")
    exit(1)

# Load embedding-based results
try:
    embedding_results = pd.read_csv(os.path.join(GRAPHS_DIR, 'embedding_ml_results.csv'), index_col=0)
    print(f"✓ Loaded {len(embedding_results)} embedding-based models")
except FileNotFoundError:
    print("✗ embedding_ml_results.csv not found. Run embedding_classifier.py first.")
    exit(1)

# Load embedding summary
try:
    with open(os.path.join(GRAPHS_DIR, 'embedding_summary.json'), 'r') as f:
        embedding_summary = json.load(f)
    print(f"✓ Loaded embedding summary")
except FileNotFoundError:
    embedding_summary = None
    print("⚠ embedding_summary.json not found")

#%% Best Model Comparison
print("\n" + "="*70)
print("BEST MODEL COMPARISON")
print("="*70)

# Get best models from each approach
best_traditional = traditional_results.sort_values('F1 Score', ascending=False).iloc[0]
best_embedding = embedding_results.sort_values('F1 Score', ascending=False).iloc[0]

print("\n" + "-"*70)
print("BEST TRADITIONAL ML MODEL (TF-IDF)")
print("-"*70)
print(f"Model:      {best_traditional.name}")
print(f"Accuracy:   {best_traditional['Accuracy']:.4f}")
print(f"Precision:  {best_traditional['Precision']:.4f}")
print(f"Recall:     {best_traditional['Recall']:.4f}")
print(f"F1-Score:   {best_traditional['F1 Score']:.4f}")
print(f"ROC AUC:    {best_traditional['ROC AUC']:.4f}")

print("\n" + "-"*70)
print("BEST EMBEDDING-BASED MODEL (Sentence Transformers)")
print("-"*70)
print(f"Model:      {best_embedding.name}")
print(f"Accuracy:   {best_embedding['Accuracy']:.4f}")
print(f"Precision:  {best_embedding['Precision']:.4f}")
print(f"Recall:     {best_embedding['Recall']:.4f}")
print(f"F1-Score:   {best_embedding['F1 Score']:.4f}")
print(f"ROC AUC:    {best_embedding['ROC AUC']:.4f}")

# Calculate improvements
print("\n" + "-"*70)
print("PERFORMANCE IMPROVEMENTS (Embedding vs Traditional)")
print("-"*70)

metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']
improvements = {}

for metric in metrics:
    trad_value = best_traditional[metric]
    embed_value = best_embedding[metric]
    improvement = ((embed_value - trad_value) / trad_value * 100)
    improvements[metric] = improvement

    symbol = "↑" if improvement > 0 else ("↓" if improvement < 0 else "=")
    print(f"{metric:12} {symbol} {improvement:+6.2f}%")

#%% Side-by-side Bar Chart
print("\n" + "="*70)
print("GENERATING COMPARATIVE VISUALIZATIONS")
print("="*70)

# Create comparison dataframe
comparison_df = pd.DataFrame({
    'Traditional ML (TF-IDF)': [
        best_traditional['Accuracy'],
        best_traditional['Precision'],
        best_traditional['Recall'],
        best_traditional['F1 Score'],
        best_traditional['ROC AUC']
    ],
    'Embedding-based (Transformers)': [
        best_embedding['Accuracy'],
        best_embedding['Precision'],
        best_embedding['Recall'],
        best_embedding['F1 Score'],
        best_embedding['ROC AUC']
    ]
}, index=['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC'])

# Plot 1: Side-by-side comparison
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(comparison_df.index))
width = 0.35

bars1 = ax.bar(x - width/2, comparison_df['Traditional ML (TF-IDF)'],
               width, label='Traditional ML (TF-IDF)', color='steelblue', alpha=0.8)
bars2 = ax.bar(x + width/2, comparison_df['Embedding-based (Transformers)'],
               width, label='Embedding-based (Transformers)', color='forestgreen', alpha=0.8)

ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title('Traditional ML vs Embedding-based: Best Model Comparison',
             fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(comparison_df.index, fontsize=11)
ax.legend(fontsize=11, loc='lower right')
ax.set_ylim([0.85, 1.0])
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(GRAPHS_DIR, 'best_model_comparison.png'), dpi=300, bbox_inches='tight')
print(f"✓ Saved best_model_comparison.png")
plt.close()

#%% Top 5 Models Comparison
print("\nGenerating top 5 models comparison...")

# Get top 5 from each approach
top5_traditional = traditional_results.sort_values('F1 Score', ascending=False).head(5)
top5_embedding = embedding_results.sort_values('F1 Score', ascending=False).head(5)

fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# Traditional ML Top 5
ax1 = axes[0]
y_pos = np.arange(len(top5_traditional))
ax1.barh(y_pos, top5_traditional['F1 Score'], color='steelblue', alpha=0.8)
ax1.set_yticks(y_pos)
ax1.set_yticklabels(top5_traditional.index, fontsize=10)
ax1.set_xlabel('F1 Score', fontsize=11, fontweight='bold')
ax1.set_title('Top 5 Traditional ML Models (TF-IDF)', fontsize=12, fontweight='bold')
ax1.set_xlim([0.9, 1.0])
ax1.grid(axis='x', alpha=0.3)
ax1.invert_yaxis()

# Add value labels
for i, v in enumerate(top5_traditional['F1 Score']):
    ax1.text(v + 0.002, i, f'{v:.4f}', va='center', fontsize=9)

# Embedding-based Top 5
ax2 = axes[1]
y_pos = np.arange(len(top5_embedding))
ax2.barh(y_pos, top5_embedding['F1 Score'], color='forestgreen', alpha=0.8)
ax2.set_yticks(y_pos)
ax2.set_yticklabels(top5_embedding.index, fontsize=10)
ax2.set_xlabel('F1 Score', fontsize=11, fontweight='bold')
ax2.set_title('Top 5 Embedding-based Models', fontsize=12, fontweight='bold')
ax2.set_xlim([0.9, 1.0])
ax2.grid(axis='x', alpha=0.3)
ax2.invert_yaxis()

# Add value labels
for i, v in enumerate(top5_embedding['F1 Score']):
    ax2.text(v + 0.002, i, f'{v:.4f}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(GRAPHS_DIR, 'top5_models_comparison.png'), dpi=300, bbox_inches='tight')
print(f"✓ Saved top5_models_comparison.png")
plt.close()

#%% Model Distribution Comparison
print("\nGenerating model distribution comparison...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

metrics_to_plot = ['Accuracy', 'F1 Score', 'Precision', 'Recall']

for idx, metric in enumerate(metrics_to_plot):
    ax = axes[idx // 2, idx % 2]

    # Create box plots
    data_to_plot = [
        traditional_results[metric].dropna(),
        embedding_results[metric].dropna()
    ]

    bp = ax.boxplot(data_to_plot, labels=['Traditional ML', 'Embedding-based'],
                    patch_artist=True, widths=0.6)

    # Color the boxes
    bp['boxes'][0].set_facecolor('steelblue')
    bp['boxes'][0].set_alpha(0.7)
    bp['boxes'][1].set_facecolor('forestgreen')
    bp['boxes'][1].set_alpha(0.7)

    ax.set_ylabel(metric, fontsize=11, fontweight='bold')
    ax.set_title(f'{metric} Distribution Across All Models', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Add statistical summary
    trad_mean = traditional_results[metric].mean()
    embed_mean = embedding_results[metric].mean()
    trad_median = traditional_results[metric].median()
    embed_median = embedding_results[metric].median()

    # Add text with statistics
    stats_text = f'Traditional: μ={trad_mean:.3f}, M={trad_median:.3f}\n'
    stats_text += f'Embedding: μ={embed_mean:.3f}, M={embed_median:.3f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()
plt.savefig(os.path.join(GRAPHS_DIR, 'distribution_comparison.png'), dpi=300, bbox_inches='tight')
print(f"✓ Saved distribution_comparison.png")
plt.close()

#%% Improvement Heatmap
print("\nGenerating improvement heatmap...")

# Get top 10 common model types (if they exist in both)
common_models = list(set(traditional_results.index) & set(embedding_results.index))

if len(common_models) > 0:
    print(f"  Found {len(common_models)} models tested in both approaches")

    # Limit to top 10 by average F1 score
    common_f1_scores = []
    for model in common_models:
        avg_f1 = (traditional_results.loc[model, 'F1 Score'] +
                  embedding_results.loc[model, 'F1 Score']) / 2
        common_f1_scores.append((model, avg_f1))

    common_f1_scores.sort(key=lambda x: x[1], reverse=True)
    top_common = [m[0] for m in common_f1_scores[:10]]

    # Create improvement matrix
    improvement_matrix = []
    for model in top_common:
        row = []
        for metric in metrics:
            trad = traditional_results.loc[model, metric]
            embed = embedding_results.loc[model, metric]
            improvement = ((embed - trad) / trad * 100)
            row.append(improvement)
        improvement_matrix.append(row)

    improvement_df = pd.DataFrame(
        improvement_matrix,
        index=top_common,
        columns=metrics
    )

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(improvement_df, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
                cbar_kws={'label': 'Improvement (%)'}, ax=ax,
                linewidths=0.5, linecolor='gray')
    ax.set_title('Performance Improvement: Embedding vs Traditional ML (%)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
    ax.set_ylabel('Model', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHS_DIR, 'improvement_heatmap.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved improvement_heatmap.png")
    plt.close()
else:
    print("  ⚠ No common models found between approaches")

#%% Summary Statistics
print("\n" + "="*70)
print("SUMMARY STATISTICS")
print("="*70)

summary_stats = {
    'approach': ['Traditional ML', 'Embedding-based'],
    'feature_type': ['TF-IDF (3000 features)', f'Sentence Transformers ({embedding_summary.get("embedding_dim", "384")}D)'],
    'best_model': [best_traditional.name, best_embedding.name],
    'best_f1': [best_traditional['F1 Score'], best_embedding['F1 Score']],
    'best_accuracy': [best_traditional['Accuracy'], best_embedding['Accuracy']],
    'best_precision': [best_traditional['Precision'], best_embedding['Precision']],
    'best_recall': [best_traditional['Recall'], best_embedding['Recall']],
    'best_roc_auc': [best_traditional['ROC AUC'], best_embedding['ROC AUC']],
    'mean_f1': [traditional_results['F1 Score'].mean(), embedding_results['F1 Score'].mean()],
    'median_f1': [traditional_results['F1 Score'].median(), embedding_results['F1 Score'].median()],
    'std_f1': [traditional_results['F1 Score'].std(), embedding_results['F1 Score'].std()]
}

summary_df = pd.DataFrame(summary_stats)
print("\n" + summary_df.to_string(index=False))

# Save summary
summary_df.to_csv(os.path.join(GRAPHS_DIR, 'comparison_summary.csv'), index=False)
print(f"\n✓ Saved comparison_summary.csv")

#%% Generate Comprehensive Report
print("\n" + "="*70)
print("GENERATING COMPREHENSIVE REPORT")
print("="*70)

report_path = os.path.join(GRAPHS_DIR, 'comparison_report.txt')

with open(report_path, 'w') as f:
    f.write("="*70 + "\n")
    f.write("SMS SPAM DETECTION: COMPREHENSIVE COMPARISON REPORT\n")
    f.write("="*70 + "\n\n")

    f.write("PROJECT OVERVIEW\n")
    f.write("-"*70 + "\n")
    f.write("Comparing Traditional ML (TF-IDF) vs Embedding-based (Transformers)\n")
    f.write("Dataset: SMS Spam Collection (5,571 messages)\n")
    f.write("  - Ham: 86.6%\n")
    f.write("  - Spam: 13.4%\n\n")

    f.write("BEST MODEL COMPARISON\n")
    f.write("-"*70 + "\n")
    f.write(f"Traditional ML (TF-IDF):\n")
    f.write(f"  Model:      {best_traditional.name}\n")
    f.write(f"  Accuracy:   {best_traditional['Accuracy']:.4f}\n")
    f.write(f"  Precision:  {best_traditional['Precision']:.4f}\n")
    f.write(f"  Recall:     {best_traditional['Recall']:.4f}\n")
    f.write(f"  F1-Score:   {best_traditional['F1 Score']:.4f}\n")
    f.write(f"  ROC AUC:    {best_traditional['ROC AUC']:.4f}\n\n")

    f.write(f"Embedding-based (Sentence Transformers):\n")
    f.write(f"  Model:      {best_embedding.name}\n")
    f.write(f"  Accuracy:   {best_embedding['Accuracy']:.4f}\n")
    f.write(f"  Precision:  {best_embedding['Precision']:.4f}\n")
    f.write(f"  Recall:     {best_embedding['Recall']:.4f}\n")
    f.write(f"  F1-Score:   {best_embedding['F1 Score']:.4f}\n")
    f.write(f"  ROC AUC:    {best_embedding['ROC AUC']:.4f}\n\n")

    f.write("PERFORMANCE IMPROVEMENTS\n")
    f.write("-"*70 + "\n")
    for metric, improvement in improvements.items():
        symbol = "↑" if improvement > 0 else ("↓" if improvement < 0 else "=")
        f.write(f"{metric:12} {symbol} {improvement:+6.2f}%\n")
    f.write("\n")

    f.write("OVERALL STATISTICS\n")
    f.write("-"*70 + "\n")
    f.write(f"Traditional ML:\n")
    f.write(f"  Models tested: {len(traditional_results)}\n")
    f.write(f"  Mean F1-Score: {traditional_results['F1 Score'].mean():.4f}\n")
    f.write(f"  Median F1-Score: {traditional_results['F1 Score'].median():.4f}\n")
    f.write(f"  Std Dev F1-Score: {traditional_results['F1 Score'].std():.4f}\n\n")

    f.write(f"Embedding-based:\n")
    f.write(f"  Models tested: {len(embedding_results)}\n")
    f.write(f"  Mean F1-Score: {embedding_results['F1 Score'].mean():.4f}\n")
    f.write(f"  Median F1-Score: {embedding_results['F1 Score'].median():.4f}\n")
    f.write(f"  Std Dev F1-Score: {embedding_results['F1 Score'].std():.4f}\n\n")

    f.write("KEY FINDINGS\n")
    f.write("-"*70 + "\n")

    # Determine winner
    if improvements['F1 Score'] > 1:
        f.write("✓ Embedding-based approach significantly outperforms Traditional ML\n")
        f.write(f"  (F1-Score improvement: {improvements['F1 Score']:+.2f}%)\n\n")
    elif improvements['F1 Score'] < -1:
        f.write("✓ Traditional ML approach outperforms Embedding-based method\n")
        f.write(f"  (F1-Score difference: {improvements['F1 Score']:+.2f}%)\n\n")
    else:
        f.write("≈ Both approaches perform similarly\n")
        f.write(f"  (F1-Score difference: {improvements['F1 Score']:+.2f}%)\n\n")

    f.write("RECOMMENDATIONS\n")
    f.write("-"*70 + "\n")
    if improvements['F1 Score'] > 1:
        f.write("1. Use embedding-based approach for production deployment\n")
        f.write("2. Consider ensemble methods combining both approaches\n")
        f.write("3. Fine-tune the best embedding model for optimal performance\n")
    else:
        f.write("1. Traditional ML (TF-IDF) is sufficient for this task\n")
        f.write("2. Lower computational cost makes it ideal for production\n")
        f.write("3. Consider embeddings only if semantic understanding is critical\n")

    f.write("\n" + "="*70 + "\n")
    f.write("Generated visualizations:\n")
    f.write("  - best_model_comparison.png\n")
    f.write("  - top5_models_comparison.png\n")
    f.write("  - distribution_comparison.png\n")
    if len(common_models) > 0:
        f.write("  - improvement_heatmap.png\n")
    f.write("="*70 + "\n")

print(f"✓ Saved comparison_report.txt")

#%% Final Summary
print("\n" + "="*70)
print("COMPARISON COMPLETE")
print("="*70)

print("\nGenerated files in graphs/:")
print("  - best_model_comparison.png")
print("  - top5_models_comparison.png")
print("  - distribution_comparison.png")
if len(common_models) > 0:
    print("  - improvement_heatmap.png")
print("  - comparison_summary.csv")
print("  - comparison_report.txt")

print("\n" + "="*70)
print("WINNER DETERMINATION")
print("="*70)

if improvements['F1 Score'] > 1:
    print(f"\n🏆 WINNER: Embedding-based Approach")
    print(f"   F1-Score Improvement: {improvements['F1 Score']:+.2f}%")
    print(f"   Best Model: {best_embedding.name}")
elif improvements['F1 Score'] < -1:
    print(f"\n🏆 WINNER: Traditional ML Approach")
    print(f"   F1-Score Advantage: {abs(improvements['F1 Score']):.2f}%")
    print(f"   Best Model: {best_traditional.name}")
else:
    print(f"\n⚖️  RESULT: Both approaches perform similarly")
    print(f"   F1-Score Difference: {improvements['F1 Score']:+.2f}%")

print("\n" + "="*70 + "\n")

# %%
