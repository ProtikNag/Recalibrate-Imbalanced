#!/usr/bin/env python3
"""
Visualization module for TCAV-Based Recalibration experiments.

Supports N-class classification with:
- Loss curves (total, classification, alignment, per-class)
- Confusion matrices with percentages
- Per-class accuracy comparisons
- Metrics comparison charts
- Class distribution plots
- Misclassification analysis
- TCAV score comparisons
- Summary dashboards

All visualizations are generated in both PNG and SVG formats.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from typing import Dict, List, Optional, Any, Tuple


class ResultVisualizer:
    """
    Generate comprehensive visualizations for experiment results.
    Supports any number of classes.
    """
    
    def __init__(self, results_dir: str, style: str = 'seaborn-v0_8-whitegrid'):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
        # Try to set style, fall back if not available
        try:
            plt.style.use(style)
        except:
            try:
                plt.style.use('seaborn-whitegrid')
            except:
                plt.style.use('default')
        
        # Color scheme
        self.colors = {
            'before': '#3498db',      # Blue
            'after': '#2ecc71',       # Green
            'change_pos': '#27ae60',  # Dark green
            'change_neg': '#e74c3c',  # Red
            'highlight': '#f39c12',   # Orange
            'neutral': '#95a5a6'      # Gray
        }
        
        # Color palette for multiple classes
        self.class_colors = plt.cm.Set2.colors
    
    def _save_figure(self, fig: plt.Figure, filename: str, close: bool = True):
        """Save figure in both PNG and SVG formats."""
        base_name = os.path.splitext(filename)[0]
        
        # Save PNG
        png_path = os.path.join(self.results_dir, f"{base_name}.png")
        fig.savefig(png_path, dpi=150, bbox_inches='tight')
        
        # Save SVG
        svg_path = os.path.join(self.results_dir, f"{base_name}.svg")
        fig.savefig(svg_path, format='svg', bbox_inches='tight')
        
        if close:
            plt.close(fig)
    
    def _get_class_color(self, idx: int) -> Tuple:
        """Get color for a class index."""
        return self.class_colors[idx % len(self.class_colors)]
    
    def plot_initial_training(self, loss_history: Dict, epochs: int,
                             filename: str = "initial_training_loss"):
        """Plot initial/pre-training loss curves."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = range(1, epochs + 1)
        
        has_loss_plotted = False
        
        if 'train_loss' in loss_history:
            ax.plot(x, loss_history['train_loss'], 'b-o', linewidth=2,
                   markersize=4, label='Train Loss')
            has_loss_plotted = True
        if 'val_loss' in loss_history:
            ax.plot(x, loss_history['val_loss'], 'r-s', linewidth=2,
                   markersize=4, label='Val Loss')
            has_loss_plotted = True
        if 'loss' in loss_history and not has_loss_plotted:
            ax.plot(x, loss_history['loss'], 'b-o', linewidth=2,
                   markersize=4, label='Loss')
            has_loss_plotted = True
        if 'total' in loss_history and not has_loss_plotted:
            ax.plot(x, loss_history['total'], 'b-o', linewidth=2,
                   markersize=4, label='Total Loss')
        
        # Accuracy on secondary axis
        ax2 = None
        if 'train_acc' in loss_history:
            ax2 = ax.twinx()
            ax2.plot(x, loss_history['train_acc'], 'g--^', linewidth=2,
                    markersize=4, label='Train Acc')
            ax2.set_ylabel('Accuracy', color='g')
            ax2.tick_params(axis='y', labelcolor='g')
            ax2.set_ylim(0, 1.1)
        if 'val_acc' in loss_history:
            if ax2 is None:
                ax2 = ax.twinx()
                ax2.set_ylabel('Accuracy', color='g')
                ax2.tick_params(axis='y', labelcolor='g')
                ax2.set_ylim(0, 1.1)
            ax2.plot(x, loss_history['val_acc'], 'm--v', linewidth=2,
                    markersize=4, label='Val Acc')
        
        ax.set_title('Initial Training Progress', fontsize=14, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        
        if ax2 is not None:
            ax2.legend(loc='lower right')
        
        plt.tight_layout()
        self._save_figure(fig, filename)
    
    def plot_loss_curves(self, loss_history: Dict, epochs: int,
                        filename: str = "loss_curves"):
        """Plot recalibration loss curves."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        x = range(1, epochs + 1)
        
        # Total loss
        axes[0].plot(x, loss_history['total'], 'b-o', linewidth=2, markersize=4)
        axes[0].set_title('Total Loss', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].grid(True, alpha=0.3)
        
        # Classification loss
        axes[1].plot(x, loss_history['cls'], 'orange', marker='s', linewidth=2, markersize=4)
        axes[1].set_title('Classification Loss', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].grid(True, alpha=0.3)
        
        # Alignment loss
        axes[2].plot(x, loss_history['align'], 'g-^', linewidth=2, markersize=4)
        axes[2].set_title('Alignment Loss', fontsize=14, fontweight='bold')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Loss')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        self._save_figure(fig, filename)
        
        # Combined plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x, loss_history['total'], 'b-o', label='Total', linewidth=2)
        ax.plot(x, loss_history['cls'], 'orange', marker='s', label='Classification', linewidth=2)
        ax.plot(x, loss_history['align'], 'g-^', label='Alignment', linewidth=2)
        ax.set_title('Training Loss Components', fontsize=14, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self._save_figure(fig, 'loss_combined')
        
        # Per-class alignment losses if available (Experiment 3)
        if 'per_class_align' in loss_history and loss_history['per_class_align']:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            for i, (class_name, losses) in enumerate(loss_history['per_class_align'].items()):
                ax.plot(x, losses, marker='o', linewidth=2, markersize=4, 
                       label=class_name, color=self._get_class_color(i))
            
            ax.set_title('Per-Class Alignment Loss', fontsize=14, fontweight='bold')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Alignment Loss')
            ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            self._save_figure(fig, 'loss_per_class_align')
    
    def plot_confusion_matrices(self, cm_before: List[List], cm_after: List[List],
                               class_names: List[str],
                               filename: str = "confusion_matrices"):
        """Plot confusion matrices before and after recalibration with percentages."""
        n_classes = len(class_names)
        
        # Adjust figure size based on number of classes
        fig_width = max(14, 6 + n_classes)
        fig_height = max(6, 3 + n_classes * 0.5)
        
        fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height))
        
        cm_before = np.array(cm_before)
        cm_after = np.array(cm_after)
        
        # Normalize by row
        cm_before_norm = cm_before.astype('float') / cm_before.sum(axis=1, keepdims=True)
        cm_after_norm = cm_after.astype('float') / cm_after.sum(axis=1, keepdims=True)
        
        cm_before_norm = np.nan_to_num(cm_before_norm)
        cm_after_norm = np.nan_to_num(cm_after_norm)
        
        # Create annotations with percentages and counts
        def create_annotations(cm_norm, cm_raw):
            annot = []
            for i in range(len(cm_norm)):
                row = []
                for j in range(len(cm_norm[i])):
                    pct = cm_norm[i][j] * 100
                    cnt = int(cm_raw[i][j])
                    row.append(f'{pct:.1f}%\n({cnt})')
                annot.append(row)
            return np.array(annot)
        
        annot_before = create_annotations(cm_before_norm, cm_before)
        annot_after = create_annotations(cm_after_norm, cm_after)
        
        # Plot heatmaps
        sns.heatmap(cm_before_norm, annot=annot_before, fmt='', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names, ax=axes[0],
                   cbar_kws={'label': 'Proportion'})
        axes[0].set_title('Before Recalibration', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('True')
        
        sns.heatmap(cm_after_norm, annot=annot_after, fmt='', cmap='Greens',
                   xticklabels=class_names, yticklabels=class_names, ax=axes[1],
                   cbar_kws={'label': 'Proportion'})
        axes[1].set_title('After Recalibration', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Predicted')
        axes[1].set_ylabel('True')
        
        # Rotate labels if many classes
        if n_classes > 5:
            for ax in axes:
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
                ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        
        plt.tight_layout()
        self._save_figure(fig, filename)
        
        # Difference plot
        self._plot_confusion_difference(cm_before_norm, cm_after_norm, class_names)
    
    def _plot_confusion_difference(self, cm_before_norm: np.ndarray, 
                                   cm_after_norm: np.ndarray,
                                   class_names: List[str]):
        """Plot confusion matrix difference."""
        n_classes = len(class_names)
        fig_size = max(8, 4 + n_classes * 0.5)
        
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))
        
        diff = cm_after_norm - cm_before_norm
        
        # Custom colormap: red for negative, green for positive
        colors = ['#e74c3c', '#ffffff', '#2ecc71']
        n_bins = 100
        cmap = LinearSegmentedColormap.from_list('diff', colors, N=n_bins)
        
        # Create annotations
        annot = []
        for i in range(len(diff)):
            row = []
            for j in range(len(diff[i])):
                val = diff[i][j] * 100
                sign = '+' if val > 0 else ''
                row.append(f'{sign}{val:.1f}%')
            annot.append(row)
        annot = np.array(annot)
        
        vmax = max(abs(diff.min()), abs(diff.max()))
        
        sns.heatmap(diff, annot=annot, fmt='', cmap=cmap,
                   xticklabels=class_names, yticklabels=class_names,
                   center=0, vmin=-vmax, vmax=vmax, ax=ax,
                   cbar_kws={'label': 'Change in Proportion'})
        
        ax.set_title('Confusion Matrix Change\n(After - Before)', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        
        if n_classes > 5:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        self._save_figure(fig, 'confusion_matrix_diff')
    
    def plot_per_class_comparison(self, per_class_before: Dict, per_class_after: Dict,
                                  class_names: List[str],
                                  filename: str = "per_class_comparison"):
        """Plot per-class metrics comparison."""
        n_classes = len(class_names)
        
        # Adjust layout based on number of classes
        fig_width = max(12, 8 + n_classes * 0.5)
        fig, axes = plt.subplots(2, 2, figsize=(fig_width, 10))
        
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        titles = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        
        x = np.arange(n_classes)
        width = 0.35
        
        for ax, metric, title in zip(axes.flat, metrics, titles):
            before_vals = [per_class_before[c][metric] for c in class_names]
            after_vals = [per_class_after[c][metric] for c in class_names]
            
            bars1 = ax.bar(x - width/2, before_vals, width, label='Before',
                          color=self.colors['before'], alpha=0.8)
            bars2 = ax.bar(x + width/2, after_vals, width, label='After',
                          color=self.colors['after'], alpha=0.8)
            
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(class_names, rotation=45 if n_classes > 5 else 0, ha='right')
            ax.set_ylim(0, 1.1)
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels
            for bar in bars1:
                height = bar.get_height()
                ax.annotate(f'{height:.2f}',
                           xy=(bar.get_x() + bar.get_width()/2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)
            for bar in bars2:
                height = bar.get_height()
                ax.annotate(f'{height:.2f}',
                           xy=(bar.get_x() + bar.get_width()/2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        self._save_figure(fig, filename)
    
    def plot_accuracy_change(self, per_class_before: Dict, per_class_after: Dict,
                            class_names: List[str],
                            filename: str = "accuracy_change"):
        """Plot accuracy change by class."""
        n_classes = len(class_names)
        fig_width = max(10, 6 + n_classes * 0.5)
        
        fig, ax = plt.subplots(figsize=(fig_width, 6))
        
        changes = [per_class_after[c]['accuracy'] - per_class_before[c]['accuracy'] 
                  for c in class_names]
        
        colors = [self.colors['change_pos'] if v >= 0 else self.colors['change_neg']
                 for v in changes]
        
        bars = ax.bar(class_names, changes, color=colors, alpha=0.8, edgecolor='black')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        ax.set_title('Accuracy Change by Class', fontsize=14, fontweight='bold')
        ax.set_ylabel('Change in Accuracy')
        ax.grid(axis='y', alpha=0.3)
        
        if n_classes > 5:
            ax.set_xticks(range(len(class_names)))
            ax.set_xticklabels(class_names, rotation=45, ha='right')
        
        # Add value labels
        for bar, val in zip(bars, changes):
            height = bar.get_height()
            va = 'bottom' if height >= 0 else 'top'
            offset = 3 if height >= 0 else -3
            ax.annotate(f'{val:+.3f}',
                       xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, offset), textcoords="offset points",
                       ha='center', va=va, fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        self._save_figure(fig, filename)
    
    def plot_metrics_comparison(self, results_before: Dict, results_after: Dict,
                               tcav_before: float, tcav_after: float,
                               filename: str = "metrics_comparison"):
        """Plot overall metrics comparison."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Bar chart
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'TCAV']
        before = [
            results_before['overall']['accuracy'],
            results_before['overall']['precision'],
            results_before['overall']['recall'],
            results_before['overall']['f1'],
            tcav_before
        ]
        after = [
            results_after['overall']['accuracy'],
            results_after['overall']['precision'],
            results_after['overall']['recall'],
            results_after['overall']['f1'],
            tcav_after
        ]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = axes[0].bar(x - width/2, before, width, label='Before',
                           color=self.colors['before'], alpha=0.8)
        bars2 = axes[0].bar(x + width/2, after, width, label='After',
                           color=self.colors['after'], alpha=0.8)
        
        axes[0].set_title('Metrics Comparison', fontsize=14, fontweight='bold')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(metrics)
        axes[0].set_ylim(0, 1.1)
        axes[0].legend()
        axes[0].grid(axis='y', alpha=0.3)
        
        for bar in bars1:
            axes[0].annotate(f'{bar.get_height():.3f}',
                           xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)
        for bar in bars2:
            axes[0].annotate(f'{bar.get_height():.3f}',
                           xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)
        
        # Change chart
        changes = [a - b for a, b in zip(after, before)]
        colors = [self.colors['change_pos'] if v >= 0 else self.colors['change_neg']
                 for v in changes]
        
        bars = axes[1].bar(metrics, changes, color=colors, alpha=0.8, edgecolor='black')
        axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes[1].set_title('Metrics Change', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Change')
        axes[1].grid(axis='y', alpha=0.3)
        
        for bar, val in zip(bars, changes):
            height = bar.get_height()
            va = 'bottom' if height >= 0 else 'top'
            offset = 3 if height >= 0 else -10
            axes[1].annotate(f'{val:+.4f}',
                           xy=(bar.get_x() + bar.get_width()/2, height),
                           xytext=(0, offset), textcoords="offset points",
                           ha='center', va=va, fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        self._save_figure(fig, filename)
    
    def plot_class_distribution(self, train_counts: Dict, val_counts: Dict,
                               filename: str = "class_distribution"):
        """Plot class distribution for training and validation sets."""
        class_names = list(train_counts.keys())
        n_classes = len(class_names)
        fig_width = max(12, 8 + n_classes * 0.5)
        
        fig, axes = plt.subplots(1, 2, figsize=(fig_width, 6))
        
        # Training set
        train_values = [train_counts[c] for c in class_names]
        colors = [self._get_class_color(i) for i in range(n_classes)]
        bars = axes[0].bar(class_names, train_values, color=colors, alpha=0.8, edgecolor='black')
        axes[0].set_title('Training Set Distribution', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Number of Samples')
        axes[0].grid(axis='y', alpha=0.3)
        
        if n_classes > 5:
            axes[0].set_xticks(range(len(class_names)))
            axes[0].set_xticklabels(class_names, rotation=45, ha='right')
        
        for bar in bars:
            axes[0].annotate(f'{int(bar.get_height())}',
                           xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=10)
        
        # Validation set
        val_values = [val_counts.get(c, 0) for c in class_names]
        bars = axes[1].bar(class_names, val_values, color=colors, alpha=0.8, edgecolor='black')
        axes[1].set_title('Validation Set Distribution', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Number of Samples')
        axes[1].grid(axis='y', alpha=0.3)
        
        if n_classes > 5:
            axes[1].set_xticks(range(len(class_names)))
            axes[1].set_xticklabels(class_names, rotation=45, ha='right')
        
        for bar in bars:
            axes[1].annotate(f'{int(bar.get_height())}',
                           xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        self._save_figure(fig, filename)
    
    def plot_misclassification_analysis(self, misclass_before: Dict, misclass_after: Dict,
                                        class_names: List[str],
                                        filename: str = "misclassification_analysis"):
        """Plot misclassification analysis."""
        n_classes = len(class_names)
        fig_width = max(14, 8 + n_classes)
        
        fig, axes = plt.subplots(1, 2, figsize=(fig_width, 6))
        
        def create_misclass_matrix(misclass_dict):
            matrix = np.zeros((n_classes, n_classes))
            for true_class, pred_dict in misclass_dict.items():
                if true_class in class_names:
                    true_idx = class_names.index(true_class)
                    for pred_class, count in pred_dict.items():
                        if pred_class in class_names:
                            pred_idx = class_names.index(pred_class)
                            matrix[true_idx, pred_idx] = count
            return matrix
        
        matrix_before = create_misclass_matrix(misclass_before)
        matrix_after = create_misclass_matrix(misclass_after)
        
        # Before
        sns.heatmap(matrix_before, annot=True, fmt='.0f', cmap='Reds',
                   xticklabels=class_names, yticklabels=class_names, ax=axes[0])
        axes[0].set_title('Misclassifications Before', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Predicted As')
        axes[0].set_ylabel('True Class')
        
        # After
        sns.heatmap(matrix_after, annot=True, fmt='.0f', cmap='Reds',
                   xticklabels=class_names, yticklabels=class_names, ax=axes[1])
        axes[1].set_title('Misclassifications After', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Predicted As')
        axes[1].set_ylabel('True Class')
        
        if n_classes > 5:
            for ax in axes:
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        self._save_figure(fig, filename)
        
        # Summary bar chart
        self._plot_misclassification_summary(misclass_before, misclass_after, class_names)
    
    def _plot_misclassification_summary(self, misclass_before: Dict, misclass_after: Dict,
                                        class_names: List[str]):
        """Plot misclassification summary by class."""
        n_classes = len(class_names)
        fig_width = max(10, 6 + n_classes * 0.5)
        
        fig, ax = plt.subplots(figsize=(fig_width, 6))
        
        before_totals = []
        after_totals = []
        
        for class_name in class_names:
            before_total = sum(misclass_before.get(class_name, {}).values())
            after_total = sum(misclass_after.get(class_name, {}).values())
            before_totals.append(before_total)
            after_totals.append(after_total)
        
        x = np.arange(n_classes)
        width = 0.35
        
        bars1 = ax.bar(x - width/2, before_totals, width, label='Before',
                      color=self.colors['before'], alpha=0.8)
        bars2 = ax.bar(x + width/2, after_totals, width, label='After',
                      color=self.colors['after'], alpha=0.8)
        
        ax.set_title('Total Misclassifications by Class', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(class_names, rotation=45 if n_classes > 5 else 0, ha='right')
        ax.set_ylabel('Number of Misclassifications')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        self._save_figure(fig, 'misclassification_summary')
    
    def plot_experiment3_tcav_comparison(self, tcav_before: Dict, tcav_after: Dict,
                                         class_layer_map: Dict,
                                         filename: str = "experiment3_tcav"):
        """Plot Experiment 3 specific TCAV comparison."""
        class_names = list(class_layer_map.keys())
        n_classes = len(class_names)
        fig_width = max(14, 8 + n_classes)
        
        fig, axes = plt.subplots(1, 2, figsize=(fig_width, 6))
        
        x = np.arange(n_classes)
        width = 0.35
        
        before_vals = [tcav_before.get(c, 0) for c in class_names]
        after_vals = [tcav_after.get(c, 0) for c in class_names]
        
        bars1 = axes[0].bar(x - width/2, before_vals, width, label='Before',
                           color=self.colors['before'], alpha=0.8)
        bars2 = axes[0].bar(x + width/2, after_vals, width, label='After',
                           color=self.colors['after'], alpha=0.8)
        
        for bar in bars1:
            axes[0].annotate(f'{bar.get_height():.3f}',
                           xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)
        for bar in bars2:
            axes[0].annotate(f'{bar.get_height():.3f}',
                           xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)
        
        axes[0].set_title('Per-Class TCAV Scores', fontsize=14, fontweight='bold')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(class_names, rotation=45 if n_classes > 5 else 0, ha='right')
        axes[0].set_ylim(0, 1.1)
        axes[0].legend()
        axes[0].grid(axis='y', alpha=0.3)
        
        # Change chart
        changes = [tcav_after.get(c, 0) - tcav_before.get(c, 0) for c in class_names]
        colors = [self.colors['change_pos'] if v >= 0 else self.colors['change_neg']
                 for v in changes]
        
        bars = axes[1].bar(class_names, changes, color=colors, alpha=0.8, edgecolor='black')
        axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes[1].set_title('TCAV Score Change', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Change')
        axes[1].grid(axis='y', alpha=0.3)
        
        if n_classes > 5:
            axes[1].set_xticks(range(len(class_names)))
            axes[1].set_xticklabels(class_names, rotation=45, ha='right')
        
        for bar, val in zip(bars, changes):
            height = bar.get_height()
            va = 'bottom' if height >= 0 else 'top'
            offset = 3 if height >= 0 else -3
            axes[1].annotate(f'{val:+.4f}',
                           xy=(bar.get_x() + bar.get_width()/2, height),
                           xytext=(0, offset), textcoords="offset points",
                           ha='center', va=va, fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        self._save_figure(fig, filename)
        
        # Class-Layer assignment table
        self._plot_class_layer_assignments(tcav_before, tcav_after, class_layer_map)
    
    def _plot_class_layer_assignments(self, tcav_before: Dict, tcav_after: Dict,
                                      class_layer_map: Dict):
        """Plot class-layer assignment table."""
        class_names = list(class_layer_map.keys())
        n_classes = len(class_names)
        fig_height = max(4, 2 + n_classes * 0.5)
        
        fig, ax = plt.subplots(figsize=(12, fig_height))
        ax.axis('off')
        
        table_data = []
        headers = ['Class', 'Layer', 'TCAV Before', 'TCAV After', 'Change']
        
        for class_name in class_names:
            layer = class_layer_map[class_name]
            before = tcav_before.get(class_name, 0)
            after = tcav_after.get(class_name, 0)
            change = after - before
            table_data.append([
                class_name,
                layer,
                f'{before:.4f}',
                f'{after:.4f}',
                f'{change:+.4f}'
            ])
        
        table = ax.table(
            cellText=table_data,
            colLabels=headers,
            loc='center',
            cellLoc='center',
            colColours=['#3498db'] * 5
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        for i in range(len(headers)):
            table[(0, i)].set_text_props(color='white', fontweight='bold')
        
        ax.set_title('Experiment 3: Class-Layer Assignments', fontsize=14,
                    fontweight='bold', pad=20)
        
        plt.tight_layout()
        self._save_figure(fig, 'experiment3_assignments')
    
    def create_summary_dashboard(self, all_results: Dict):
        """Create a comprehensive summary dashboard."""
        class_names = list(all_results['results_before']['per_class'].keys())
        n_classes = len(class_names)
        
        fig_width = max(20, 16 + n_classes * 0.5)
        fig = plt.figure(figsize=(fig_width, 16))
        
        target_info = all_results.get("target_class", "multiple")
        fig.suptitle(f'Experiment {all_results["experiment"]} Summary Dashboard\n'
                    f'Model: {all_results["model_name"]} | Classes: {n_classes}',
                    fontsize=16, fontweight='bold', y=0.98)
        
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Key metrics
        ax1 = fig.add_subplot(gs[0, 0])
        metrics = ['Accuracy', 'TCAV']
        before = [all_results['results_before']['overall']['accuracy'],
                 all_results['tcav_before']]
        after = [all_results['results_after']['overall']['accuracy'],
                all_results['tcav_after']]
        
        x = np.arange(len(metrics))
        ax1.bar(x - 0.2, before, 0.4, label='Before', color=self.colors['before'])
        ax1.bar(x + 0.2, after, 0.4, label='After', color=self.colors['after'])
        ax1.set_xticks(x)
        ax1.set_xticklabels(metrics)
        ax1.set_title('Key Metrics', fontweight='bold')
        ax1.legend()
        ax1.set_ylim(0, 1.1)
        
        # 2. Class distribution
        ax2 = fig.add_subplot(gs[0, 1])
        train_counts = all_results['train_class_counts']
        colors = [self._get_class_color(i) for i in range(n_classes)]
        ax2.bar(train_counts.keys(), train_counts.values(), color=colors)
        ax2.set_title('Training Class Distribution', fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Loss curve
        ax3 = fig.add_subplot(gs[0, 2])
        epochs = range(1, all_results['epochs'] + 1)
        ax3.plot(epochs, all_results['loss_history']['total'], 'b-', label='Total')
        ax3.plot(epochs, all_results['loss_history']['cls'], 'orange', label='Cls')
        ax3.plot(epochs, all_results['loss_history']['align'], 'g-', label='Align')
        ax3.set_title('Loss Curves', fontweight='bold')
        ax3.legend()
        ax3.set_xlabel('Epoch')
        
        # 4. Per-class accuracy
        ax4 = fig.add_subplot(gs[1, :2])
        before_acc = [all_results['results_before']['per_class'][c]['accuracy']
                     for c in class_names]
        after_acc = [all_results['results_after']['per_class'][c]['accuracy']
                    for c in class_names]
        
        x = np.arange(n_classes)
        ax4.bar(x - 0.2, before_acc, 0.4, label='Before', color=self.colors['before'])
        ax4.bar(x + 0.2, after_acc, 0.4, label='After', color=self.colors['after'])
        ax4.set_xticks(x)
        ax4.set_xticklabels(class_names, rotation=45 if n_classes > 5 else 0, ha='right')
        ax4.set_title('Per-Class Accuracy', fontweight='bold')
        ax4.legend()
        ax4.set_ylim(0, 1.1)
        
        # 5. Configuration info
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.axis('off')
        
        layer_info = all_results.get('layer', 'multiple')
        concept_info = all_results.get('concept', 'multiple')
        if isinstance(concept_info, dict):
            concept_info = 'multiple'
        
        config_text = (
            f"Configuration:\n"
            f"─────────────\n"
            f"Model: {all_results['model_name']}\n"
            f"Layer: {layer_info}\n"
            f"Target: {target_info}\n"
            f"Concept: {concept_info}\n"
            f"Lambda: {all_results['lambda_align']}\n"
            f"Epochs: {all_results['epochs']}\n"
            f"Imbalance: {all_results.get('imbalance_class', 'None')}\n"
            f"Ratio: {all_results.get('imbalance_ratio', 1.0):.1%}"
        )
        ax5.text(0.1, 0.9, config_text, transform=ax5.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 6. Accuracy changes
        ax6 = fig.add_subplot(gs[2, :])
        changes = {c: after_acc[i] - before_acc[i] for i, c in enumerate(class_names)}
        colors = [self.colors['change_pos'] if v >= 0 else self.colors['change_neg']
                 for v in changes.values()]
        bars = ax6.bar(changes.keys(), changes.values(), color=colors)
        ax6.axhline(y=0, color='black', linewidth=0.5)
        ax6.set_title('Accuracy Change by Class', fontweight='bold')
        ax6.set_ylabel('Change')
        
        if n_classes > 5:
            ax6.set_xticks(range(len(class_names)))
            ax6.set_xticklabels(list(changes.keys()), rotation=45, ha='right')
        
        for bar, val in zip(bars, changes.values()):
            height = bar.get_height()
            ax6.annotate(f'{val:+.3f}',
                        xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3 if height >= 0 else -10),
                        textcoords="offset points",
                        ha='center', fontweight='bold')
        
        self._save_figure(fig, 'summary_dashboard')
