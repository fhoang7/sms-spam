"""Visualization utilities for spam detection results."""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def plot_model_comparison(
    results_df,
    metrics=None,
    title="Model Comparison",
    output_path=None,
    figsize=(12, 6)
):
    """
    Create bar plot comparing multiple models.

    Args:
        results_df (pd.DataFrame): Results with models as index
        metrics (list, optional): Metrics to plot (defaults to all numeric columns)
        title (str): Plot title
        output_path (str, optional): Path to save figure
        figsize (tuple): Figure size

    Returns:
        matplotlib.figure.Figure: The created figure
    """
    if metrics is None:
        # Use all numeric columns
        metrics = results_df.select_dtypes(include=[np.number]).columns.tolist()

    # Select top N models to avoid clutter
    top_n = min(10, len(results_df))
    plot_data = results_df.head(top_n)[metrics]

    fig, ax = plt.subplots(figsize=figsize)

    plot_data.plot(kind='bar', ax=ax)

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_ylim([0, 1])
    ax.legend(title='Metrics', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='y', alpha=0.3)

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Plot saved to {output_path}")

    return fig


def plot_cascade_analysis(
    cascade_results,
    output_path=None,
    figsize=(14, 5)
):
    """
    Visualize 2-stage cascade performance.

    Args:
        cascade_results (dict): Results from HybridCascadeClassifier.evaluate()
        output_path (str, optional): Path to save figure
        figsize (tuple): Figure size

    Returns:
        matplotlib.figure.Figure: The created figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Plot 1: Stage usage
    stage_stats = cascade_results['stage_stats']
    stages = ['Stage 1\n(TF-IDF)', 'Stage 2\n(Embeddings)']
    counts = [stage_stats['stage1_count'], stage_stats['stage2_count']]
    colors = ['#3498db', '#e74c3c']

    axes[0].bar(stages, counts, color=colors, alpha=0.7, edgecolor='black')
    axes[0].set_title('Message Distribution Across Stages', fontweight='bold')
    axes[0].set_ylabel('Number of Messages')
    axes[0].grid(axis='y', alpha=0.3)

    # Add percentage labels
    for i, (stage, count) in enumerate(zip(stages, counts)):
        pct = (count / stage_stats['total']) * 100
        axes[0].text(i, count, f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')

    # Plot 2: Stage performance comparison
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    overall = cascade_results['overall']
    stage1 = cascade_results['stage1_metrics']
    stage2 = cascade_results['stage2_metrics']

    x = np.arange(len(metrics))
    width = 0.25

    overall_scores = [overall.get(m, 0) for m in metrics]
    stage1_scores = [stage1.get(m, 0) if stage1 else 0 for m in metrics]
    stage2_scores = [stage2.get(m, 0) if stage2 else 0 for m in metrics]

    axes[1].bar(x - width, overall_scores, width, label='Overall', color='#2ecc71', alpha=0.7)
    axes[1].bar(x, stage1_scores, width, label='Stage 1', color='#3498db', alpha=0.7)
    axes[1].bar(x + width, stage2_scores, width, label='Stage 2', color='#e74c3c', alpha=0.7)

    axes[1].set_title('Performance Metrics by Stage', fontweight='bold')
    axes[1].set_ylabel('Score')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([m.capitalize() for m in metrics])
    axes[1].set_ylim([0, 1.0])
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Cascade analysis saved to {output_path}")

    return fig


def plot_confusion_matrix(
    y_true,
    y_pred,
    class_names=None,
    title="Confusion Matrix",
    output_path=None,
    figsize=(8, 6)
):
    """
    Plot confusion matrix heatmap.

    Args:
        y_true (array-like): True labels
        y_pred (array-like): Predicted labels
        class_names (list, optional): Class names
        title (str): Plot title
        output_path (str, optional): Path to save figure
        figsize (tuple): Figure size

    Returns:
        matplotlib.figure.Figure: The created figure
    """
    from sklearn.metrics import confusion_matrix

    if class_names is None:
        class_names = ['Ham', 'Spam']

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar_kws={'label': 'Count'}
    )

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Confusion matrix saved to {output_path}")

    return fig


def plot_threshold_optimization(
    precisions,
    recalls,
    thresholds,
    optimal_threshold,
    output_path=None,
    figsize=(10, 6)
):
    """
    Plot precision-recall curve with optimal threshold.

    Args:
        precisions (array-like): Precision values
        recalls (array-like): Recall values
        thresholds (array-like): Threshold values
        optimal_threshold (float): Optimal threshold to highlight
        output_path (str, optional): Path to save figure
        figsize (tuple): Figure size

    Returns:
        matplotlib.figure.Figure: The created figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(thresholds, precisions[:-1], 'b-', label='Precision', linewidth=2)
    ax.plot(thresholds, recalls[:-1], 'r-', label='Recall', linewidth=2)

    # Mark optimal threshold
    ax.axvline(optimal_threshold, color='green', linestyle='--', linewidth=2,
               label=f'Optimal Threshold: {optimal_threshold:.3f}')

    ax.set_title('Precision-Recall vs Threshold', fontsize=14, fontweight='bold')
    ax.set_xlabel('Threshold', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.legend(loc='best')
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Threshold plot saved to {output_path}")

    return fig
