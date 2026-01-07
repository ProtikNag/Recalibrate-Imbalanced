"""
Visualization utilities for VL-CAV experiments.

Creates publication-quality visualizations with academic styling:
- Training curves
- Confusion matrices
- Class distribution plots
- TCAV score comparisons
- Correlation matrices
- Performance comparisons
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import os
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')


# Academic color palette
COLORS = {
    'primary': '#2E4057',      # Dark blue-gray
    'secondary': '#048A81',    # Teal
    'accent1': '#54C6EB',      # Light blue
    'accent2': '#8EE3EF',      # Pale cyan
    'accent3': '#F7A278',      # Coral
    'accent4': '#F25C54',      # Red
    'positive': '#2E7D32',     # Green
    'negative': '#C62828',     # Red
    'neutral': '#757575',      # Gray
    'background': '#FAFAFA',   # Light gray
}

# Color palette for multiple classes
CLASS_COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5'
]


def set_academic_style():
    """Set matplotlib style for academic publications."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 14,
        'axes.linewidth': 1.0,
        'axes.edgecolor': '#333333',
        'axes.labelcolor': '#333333',
        'text.color': '#333333',
        'xtick.color': '#333333',
        'ytick.color': '#333333',
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
        'savefig.edgecolor': 'white',
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
    })


def save_figure(fig: plt.Figure, path: str, formats: List[str] = ['png', 'svg']):
    """
    Save figure in multiple formats.
    
    Args:
        fig: Matplotlib figure
        path: Base path without extension
        formats: List of formats to save
    """
    for fmt in formats:
        fig.savefig(f"{path}.{fmt}", format=fmt, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_training_curves(history: Dict[str, List[float]], 
                         output_path: str,
                         title: str = "Training Progress",
                         formats: List[str] = ['png', 'svg']):
    """
    Plot training and validation curves.
    
    Args:
        history: Dictionary with 'train_loss', 'val_loss', 'val_accuracy'
        output_path: Path to save figure (without extension)
        title: Figure title
        formats: Output formats
    """
    set_academic_style()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss plot
    ax1 = axes[0]
    ax1.plot(epochs, history['train_loss'], '-', color=COLORS['primary'], 
             linewidth=2, label='Training Loss', marker='o', markersize=4)
    if 'val_loss' in history:
        ax1.plot(epochs, history['val_loss'], '--', color=COLORS['accent3'],
                 linewidth=2, label='Validation Loss', marker='s', markersize=4)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Curves')
    ax1.legend(frameon=True, fancybox=False, edgecolor='gray')
    ax1.grid(True, alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Accuracy plot
    ax2 = axes[1]
    if 'train_accuracy' in history:
        ax2.plot(epochs, history['train_accuracy'], '-', color=COLORS['primary'],
                 linewidth=2, label='Training Accuracy', marker='o', markersize=4)
    if 'val_accuracy' in history:
        ax2.plot(epochs, history['val_accuracy'], '--', color=COLORS['secondary'],
                 linewidth=2, label='Validation Accuracy', marker='s', markersize=4)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy Curves')
    ax2.legend(frameon=True, fancybox=False, edgecolor='gray')
    ax2.grid(True, alpha=0.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_ylim([0, 1])
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    save_figure(fig, output_path, formats)


def plot_recalibration_losses(history: Dict[str, List[float]],
                              output_path: str,
                              formats: List[str] = ['png', 'svg']):
    """
    Plot recalibration training curves with separate loss components.
    
    Args:
        history: Dictionary with 'train_loss', 'train_cls_loss', 'train_align_loss'
        output_path: Path to save figure
        formats: Output formats
    """
    set_academic_style()
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Total loss
    ax1 = axes[0]
    ax1.plot(epochs, history['train_loss'], '-', color=COLORS['primary'],
             linewidth=2, marker='o', markersize=4)
    ax1.fill_between(epochs, history['train_loss'], alpha=0.2, color=COLORS['primary'])
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Total Loss')
    ax1.grid(True, alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Classification loss
    ax2 = axes[1]
    ax2.plot(epochs, history['train_cls_loss'], '-', color=COLORS['secondary'],
             linewidth=2, marker='o', markersize=4)
    ax2.fill_between(epochs, history['train_cls_loss'], alpha=0.2, color=COLORS['secondary'])
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Classification Loss')
    ax2.grid(True, alpha=0.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # Alignment loss
    ax3 = axes[2]
    ax3.plot(epochs, history['train_align_loss'], '-', color=COLORS['accent3'],
             linewidth=2, marker='o', markersize=4)
    ax3.fill_between(epochs, history['train_align_loss'], alpha=0.2, color=COLORS['accent3'])
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss')
    ax3.set_title('Alignment Loss')
    ax3.grid(True, alpha=0.3)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    fig.suptitle('Recalibration Training Progress', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    save_figure(fig, output_path, formats)


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                          class_names: List[str],
                          output_path: str,
                          title: str = "Confusion Matrix",
                          normalize: bool = True,
                          formats: List[str] = ['png', 'svg']):
    """
    Plot confusion matrix with academic styling.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        output_path: Path to save figure
        title: Figure title
        normalize: Whether to normalize the matrix
        formats: Output formats
    """
    set_academic_style()
    
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    fig, ax = plt.subplots(figsize=(8, 7))
    
    # Custom colormap
    cmap = LinearSegmentedColormap.from_list(
        'academic_blue', ['#FFFFFF', '#E3F2FD', '#90CAF9', '#42A5F5', '#1976D2', '#0D47A1']
    )
    
    sns.heatmap(cm, annot=True, fmt=fmt, cmap=cmap,
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, cbar_kws={'shrink': 0.8},
                linewidths=0.5, linecolor='white',
                annot_kws={'size': 9})
    
    ax.set_xlabel('Predicted Label', fontweight='bold')
    ax.set_ylabel('True Label', fontweight='bold')
    ax.set_title(title, fontsize=13, fontweight='bold', pad=15)
    
    # Rotate labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    save_figure(fig, output_path, formats)


def plot_confusion_matrix_comparison(y_true: np.ndarray,
                                     y_pred_before: np.ndarray,
                                     y_pred_after: np.ndarray,
                                     class_names: List[str],
                                     output_path: str,
                                     formats: List[str] = ['png', 'svg']):
    """
    Plot before/after confusion matrices side by side.
    
    Args:
        y_true: True labels
        y_pred_before: Predictions before recalibration
        y_pred_after: Predictions after recalibration
        class_names: List of class names
        output_path: Path to save figure
        formats: Output formats
    """
    set_academic_style()
    
    cm_before = confusion_matrix(y_true, y_pred_before)
    cm_after = confusion_matrix(y_true, y_pred_after)
    
    # Normalize
    cm_before_norm = cm_before.astype('float') / cm_before.sum(axis=1)[:, np.newaxis]
    cm_after_norm = cm_after.astype('float') / cm_after.sum(axis=1)[:, np.newaxis]
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    cmap = LinearSegmentedColormap.from_list(
        'academic_blue', ['#FFFFFF', '#E3F2FD', '#90CAF9', '#42A5F5', '#1976D2', '#0D47A1']
    )
    
    # Before
    sns.heatmap(cm_before_norm, annot=True, fmt='.2f', cmap=cmap,
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[0], cbar=False, linewidths=0.5, linecolor='white',
                annot_kws={'size': 8})
    axes[0].set_title('Before Recalibration', fontweight='bold')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')
    
    # After
    sns.heatmap(cm_after_norm, annot=True, fmt='.2f', cmap=cmap,
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[1], cbar=False, linewidths=0.5, linecolor='white',
                annot_kws={'size': 8})
    axes[1].set_title('After Recalibration', fontweight='bold')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')
    
    # Difference
    cm_diff = cm_after_norm - cm_before_norm
    cmap_diff = LinearSegmentedColormap.from_list(
        'diff', [COLORS['negative'], '#FFFFFF', COLORS['positive']]
    )
    
    vmax = max(abs(cm_diff.min()), abs(cm_diff.max()))
    sns.heatmap(cm_diff, annot=True, fmt='.2f', cmap=cmap_diff,
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[2], center=0, vmin=-vmax, vmax=vmax,
                linewidths=0.5, linecolor='white',
                annot_kws={'size': 8})
    axes[2].set_title('Difference (After - Before)', fontweight='bold')
    axes[2].set_xlabel('Predicted')
    axes[2].set_ylabel('True')
    
    for ax in axes:
        plt.sca(ax)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    save_figure(fig, output_path, formats)


def plot_class_distribution(class_counts: Dict[int, int],
                            class_names: List[str],
                            output_path: str,
                            imbalance_classes: Optional[List[int]] = None,
                            title: str = "Class Distribution",
                            formats: List[str] = ['png', 'svg']):
    """
    Plot class distribution bar chart.
    
    Args:
        class_counts: Dictionary mapping class index to count
        class_names: List of class names
        output_path: Path to save figure
        imbalance_classes: Indices of imbalanced classes to highlight
        title: Figure title
        formats: Output formats
    """
    set_academic_style()
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    indices = sorted(class_counts.keys())
    counts = [class_counts[i] for i in indices]
    names = [class_names[i] if i < len(class_names) else f"Class {i}" for i in indices]
    
    # Color bars based on imbalance
    colors = []
    for i in indices:
        if imbalance_classes and i in imbalance_classes:
            colors.append(COLORS['accent4'])
        else:
            colors.append(COLORS['primary'])
    
    bars = ax.bar(range(len(counts)), counts, color=colors, edgecolor='white', linewidth=1)
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.annotate(f'{count}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Class', fontweight='bold')
    ax.set_ylabel('Number of Samples', fontweight='bold')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right')
    
    # Add legend for imbalanced classes
    if imbalance_classes:
        legend_elements = [
            mpatches.Patch(facecolor=COLORS['primary'], label='Normal'),
            mpatches.Patch(facecolor=COLORS['accent4'], label='Imbalanced')
        ]
        ax.legend(handles=legend_elements, loc='upper right', frameon=True)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    save_figure(fig, output_path, formats)


def plot_per_class_accuracy(accuracy_before: Dict[int, float],
                            accuracy_after: Dict[int, float],
                            class_names: List[str],
                            output_path: str,
                            imbalance_classes: Optional[List[int]] = None,
                            formats: List[str] = ['png', 'svg']):
    """
    Plot per-class accuracy comparison.
    
    Args:
        accuracy_before: Accuracy per class before recalibration
        accuracy_after: Accuracy per class after recalibration
        class_names: List of class names
        output_path: Path to save figure
        imbalance_classes: Indices of imbalanced classes
        formats: Output formats
    """
    set_academic_style()
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    indices = sorted(accuracy_before.keys())
    names = [class_names[i] if i < len(class_names) else f"Class {i}" for i in indices]
    
    acc_before = [accuracy_before[i] for i in indices]
    acc_after = [accuracy_after[i] for i in indices]
    
    x = np.arange(len(indices))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, acc_before, width, label='Before',
                   color=COLORS['neutral'], edgecolor='white')
    bars2 = ax.bar(x + width/2, acc_after, width, label='After',
                   color=COLORS['secondary'], edgecolor='white')
    
    # Highlight imbalanced classes
    if imbalance_classes:
        for i, idx in enumerate(indices):
            if idx in imbalance_classes:
                bars1[i].set_color(COLORS['accent4'])
                bars1[i].set_alpha(0.6)
                bars2[i].set_color(COLORS['positive'])
    
    ax.set_xlabel('Class', fontweight='bold')
    ax.set_ylabel('Accuracy', fontweight='bold')
    ax.set_title('Per-Class Accuracy Comparison', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylim([0, 1.1])
    ax.legend(loc='upper right', frameon=True)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, axis='y', alpha=0.3)
    
    # Add improvement annotations
    for i, (b, a) in enumerate(zip(acc_before, acc_after)):
        diff = a - b
        color = COLORS['positive'] if diff > 0 else COLORS['negative']
        sign = '+' if diff > 0 else ''
        ax.annotate(f'{sign}{diff:.2f}',
                    xy=(x[i] + width/2, a),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8,
                    color=color, fontweight='bold')
    
    plt.tight_layout()
    
    save_figure(fig, output_path, formats)


def plot_tcav_scores(tcav_before: Dict[int, float],
                     tcav_after: Dict[int, float],
                     class_names: List[str],
                     output_path: str,
                     formats: List[str] = ['png', 'svg']):
    """
    Plot TCAV score comparison.
    
    Args:
        tcav_before: TCAV scores per class before recalibration
        tcav_after: TCAV scores per class after recalibration
        class_names: List of class names
        output_path: Path to save figure
        formats: Output formats
    """
    set_academic_style()
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    indices = sorted(tcav_before.keys())
    names = [class_names[i] if i < len(class_names) else f"Class {i}" for i in indices]
    
    scores_before = [tcav_before[i] for i in indices]
    scores_after = [tcav_after[i] for i in indices]
    
    x = np.arange(len(indices))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, scores_before, width, label='Before',
                   color=COLORS['neutral'], edgecolor='white')
    bars2 = ax.bar(x + width/2, scores_after, width, label='After',
                   color=COLORS['accent1'], edgecolor='white')
    
    ax.set_xlabel('Class', fontweight='bold')
    ax.set_ylabel('TCAV Score', fontweight='bold')
    ax.set_title('TCAV Score Comparison (Before vs After Recalibration)', 
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylim([0, 1.1])
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random baseline')
    ax.legend(loc='upper right', frameon=True)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    save_figure(fig, output_path, formats)


def plot_correlation_matrix(corr_matrix: np.ndarray,
                            layer_names: List[str],
                            output_path: str,
                            title: str = "Layer Sensitivity Correlation",
                            formats: List[str] = ['png', 'svg']):
    """
    Plot correlation matrix between layers.
    
    Args:
        corr_matrix: Correlation matrix
        layer_names: List of layer names
        output_path: Path to save figure
        title: Figure title
        formats: Output formats
    """
    set_academic_style()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create custom diverging colormap
    cmap = LinearSegmentedColormap.from_list(
        'correlation', [COLORS['negative'], '#FFFFFF', COLORS['positive']]
    )
    
    # Shorten layer names if too long
    short_names = [name.replace('features.', 'f').replace('layer', 'L') 
                   for name in layer_names]
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap=cmap,
                xticklabels=short_names, yticklabels=short_names,
                ax=ax, center=0, vmin=-1, vmax=1,
                linewidths=0.5, linecolor='white',
                annot_kws={'size': 8})
    
    ax.set_title(title, fontsize=13, fontweight='bold', pad=15)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    save_figure(fig, output_path, formats)


def plot_embedding_space(embeddings: np.ndarray,
                         labels: np.ndarray,
                         class_names: List[str],
                         output_path: str,
                         title: str = "Embedding Space Visualization",
                         method: str = "tsne",
                         formats: List[str] = ['png', 'svg']):
    """
    Plot 2D visualization of embedding space.
    
    Args:
        embeddings: Embedding vectors [N, D]
        labels: Class labels [N]
        class_names: List of class names
        output_path: Path to save figure
        title: Figure title
        method: Dimensionality reduction method ("tsne" or "pca")
        formats: Output formats
    """
    set_academic_style()
    
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    
    # Reduce dimensionality
    if method == "tsne":
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
    else:
        reducer = PCA(n_components=2, random_state=42)
    
    coords = reducer.fit_transform(embeddings)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    unique_labels = np.unique(labels)
    for i, label in enumerate(unique_labels):
        mask = labels == label
        name = class_names[label] if label < len(class_names) else f"Class {label}"
        color = CLASS_COLORS[i % len(CLASS_COLORS)]
        
        ax.scatter(coords[mask, 0], coords[mask, 1], 
                   c=color, label=name, alpha=0.7, s=30, edgecolors='white', linewidth=0.5)
    
    ax.set_xlabel(f'{method.upper()} Dimension 1', fontweight='bold')
    ax.set_ylabel(f'{method.upper()} Dimension 2', fontweight='bold')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(loc='best', frameon=True, ncol=2 if len(unique_labels) > 5 else 1)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    save_figure(fig, output_path, formats)


def plot_experiment_summary(results: Dict,
                            output_path: str,
                            formats: List[str] = ['png', 'svg']):
    """
    Create a summary dashboard with key metrics.
    
    Args:
        results: Dictionary with experiment results
        output_path: Path to save figure
        formats: Output formats
    """
    set_academic_style()
    
    fig = plt.figure(figsize=(14, 10))
    
    # Create grid
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
    
    # 1. Overall accuracy comparison
    ax1 = fig.add_subplot(gs[0, 0])
    metrics = ['Accuracy\nBefore', 'Accuracy\nAfter']
    values = [results.get('accuracy_before', 0), results.get('accuracy_after', 0)]
    colors = [COLORS['neutral'], COLORS['secondary']]
    bars = ax1.bar(metrics, values, color=colors, edgecolor='white')
    ax1.set_ylim([0, 1])
    ax1.set_title('Overall Accuracy', fontweight='bold')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    for bar, val in zip(bars, values):
        ax1.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width()/2, val),
                     xytext=(0, 3), textcoords='offset points', ha='center', fontsize=10)
    
    # 2. Imbalanced class improvement
    ax2 = fig.add_subplot(gs[0, 1])
    if 'imbalance_class_acc_before' in results and 'imbalance_class_acc_after' in results:
        metrics = ['Before', 'After']
        values = [results['imbalance_class_acc_before'], results['imbalance_class_acc_after']]
        colors = [COLORS['accent4'], COLORS['positive']]
        bars = ax2.bar(metrics, values, color=colors, edgecolor='white')
        ax2.set_ylim([0, 1])
        improvement = values[1] - values[0]
        ax2.set_title(f'Imbalanced Class Acc.\n(Δ = {improvement:+.3f})', fontweight='bold')
    else:
        ax2.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=14)
        ax2.set_title('Imbalanced Class Acc.', fontweight='bold')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # 3. Training info
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis('off')
    info_text = f"""
    Model: {results.get('model_name', 'N/A')}
    Dataset: {results.get('dataset_name', 'N/A')}
    Imbalance Ratio: {results.get('imbalance_ratio', 'N/A')}
    Alpha (text/vision): {results.get('alpha', 'N/A')}
    Bottleneck Layers: {results.get('bottleneck_layers', 'N/A')}
    """
    ax3.text(0.1, 0.5, info_text, fontsize=10, va='center', 
             fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    ax3.set_title('Experiment Configuration', fontweight='bold')
    
    # 4. Per-class accuracy (spanning two columns)
    if 'per_class_acc_before' in results and 'per_class_acc_after' in results:
        ax4 = fig.add_subplot(gs[1, :2])
        class_names = results.get('class_names', [f'C{i}' for i in range(len(results['per_class_acc_before']))])
        x = np.arange(len(class_names))
        width = 0.35
        
        acc_before = list(results['per_class_acc_before'].values())
        acc_after = list(results['per_class_acc_after'].values())
        
        ax4.bar(x - width/2, acc_before, width, label='Before', color=COLORS['neutral'])
        ax4.bar(x + width/2, acc_after, width, label='After', color=COLORS['secondary'])
        ax4.set_xticks(x)
        ax4.set_xticklabels(class_names, rotation=45, ha='right')
        ax4.set_ylabel('Accuracy')
        ax4.set_title('Per-Class Accuracy', fontweight='bold')
        ax4.legend()
        ax4.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)
        ax4.set_ylim([0, 1.1])
    
    # 5. TCAV scores
    if 'tcav_before' in results and 'tcav_after' in results:
        ax5 = fig.add_subplot(gs[1, 2])
        tcav_before = np.mean(list(results['tcav_before'].values()))
        tcav_after = np.mean(list(results['tcav_after'].values()))
        
        ax5.bar(['Before', 'After'], [tcav_before, tcav_after],
                color=[COLORS['neutral'], COLORS['accent1']], edgecolor='white')
        ax5.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        ax5.set_ylim([0, 1])
        ax5.set_title('Avg. TCAV Score', fontweight='bold')
        ax5.spines['top'].set_visible(False)
        ax5.spines['right'].set_visible(False)
    
    # 6. Training curves (spanning bottom)
    if 'train_history' in results:
        ax6 = fig.add_subplot(gs[2, :])
        history = results['train_history']
        epochs = range(1, len(history['train_loss']) + 1)
        
        ax6.plot(epochs, history['train_loss'], '-', color=COLORS['primary'],
                 linewidth=2, label='Train Loss')
        if 'val_accuracy' in history:
            ax6_twin = ax6.twinx()
            ax6_twin.plot(epochs, history['val_accuracy'], '--', color=COLORS['secondary'],
                         linewidth=2, label='Val Acc')
            ax6_twin.set_ylabel('Accuracy', color=COLORS['secondary'])
            ax6_twin.tick_params(axis='y', labelcolor=COLORS['secondary'])
            ax6_twin.set_ylim([0, 1])
        
        ax6.set_xlabel('Epoch')
        ax6.set_ylabel('Loss', color=COLORS['primary'])
        ax6.tick_params(axis='y', labelcolor=COLORS['primary'])
        ax6.set_title('Training Progress', fontweight='bold')
        ax6.spines['top'].set_visible(False)
        
        # Combined legend
        lines1, labels1 = ax6.get_legend_handles_labels()
        if 'val_accuracy' in history:
            lines2, labels2 = ax6_twin.get_legend_handles_labels()
            ax6.legend(lines1 + lines2, labels1 + labels2, loc='center right')
        else:
            ax6.legend()
    
    fig.suptitle('VL-CAV Recalibration Experiment Summary', fontsize=16, fontweight='bold', y=0.98)
    
    save_figure(fig, output_path, formats)
