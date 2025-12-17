"""
Visualization module for TCAV-Based Recalibration experiments.

Generates:
- Loss curves
- Confusion matrices (with percentages in all cells)
- Per-class accuracy comparisons
- Metrics comparison charts
- Class distribution plots
- Misclassification analysis
- Experiment 3 specific visualizations
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from typing import Dict, List, Optional


class ResultVisualizer:
    """
    Generate comprehensive visualizations for experiment results.
    """

    def __init__(self, results_dir: str, style: str = 'seaborn-v0_8-whitegrid'):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)

        # Try to set style, fall back if not available
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')

        # Color scheme
        self.colors = {
            'before': '#3498db',  # Blue
            'after': '#2ecc71',  # Green
            'change_pos': '#27ae60',  # Dark green
            'change_neg': '#e74c3c',  # Red
            'highlight': '#f39c12',  # Orange
            'neutral': '#95a5a6'  # Gray
        }

    def plot_initial_training(self, loss_history: Dict, epochs: int,
                              filename: str = "initial_training_loss.png"):
        """Plot initial/pre-training loss curves."""
        fig, ax = plt.subplots(figsize=(10, 6))

        x = range(1, epochs + 1)

        # Handle different possible keys in loss_history
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
            has_loss_plotted = True
        if 'cls' in loss_history:
            ax.plot(x, loss_history['cls'], 'orange', marker='s', linewidth=2,
                    markersize=4, label='Classification Loss')

        # Handle accuracy on secondary axis
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
        plt.savefig(os.path.join(self.results_dir, filename), dpi=150, bbox_inches='tight')
        plt.close()

    def plot_loss_curves(self, loss_history: Dict, epochs: int,
                         filename: str = "loss_curves.png"):
        """Plot training loss curves."""
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
        plt.savefig(os.path.join(self.results_dir, filename), dpi=150, bbox_inches='tight')
        plt.close()

        # Also save combined plot
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
        plt.savefig(os.path.join(self.results_dir, 'loss_combined.png'), dpi=150, bbox_inches='tight')
        plt.close()

        # Plot per-class alignment losses if available (Experiment 3)
        if 'per_class_align' in loss_history and loss_history['per_class_align']:
            fig, ax = plt.subplots(figsize=(12, 6))

            for class_name, losses in loss_history['per_class_align'].items():
                ax.plot(x, losses, marker='o', linewidth=2, markersize=4, label=class_name)

            ax.set_title('Per-Class Alignment Loss', fontsize=14, fontweight='bold')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Alignment Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, 'loss_per_class_align.png'),
                        dpi=150, bbox_inches='tight')
            plt.close()

    def plot_confusion_matrices(self, cm_before: List[List], cm_after: List[List],
                                class_names: List[str],
                                filename: str = "confusion_matrices.png"):
        """Plot confusion matrices before and after recalibration with percentages in ALL cells."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        cm_before = np.array(cm_before)
        cm_after = np.array(cm_after)

        # Normalize by row (true labels)
        cm_before_norm = cm_before.astype('float') / cm_before.sum(axis=1, keepdims=True)
        cm_after_norm = cm_after.astype('float') / cm_after.sum(axis=1, keepdims=True)

        # Handle NaN (division by zero for empty rows)
        cm_before_norm = np.nan_to_num(cm_before_norm)
        cm_after_norm = np.nan_to_num(cm_after_norm)

        # Create annotation arrays with percentage and raw count for ALL cells
        annot_before = np.empty_like(cm_before, dtype=object)
        annot_after = np.empty_like(cm_after, dtype=object)

        for i in range(len(class_names)):
            for j in range(len(class_names)):
                pct_before = cm_before_norm[i, j] * 100
                pct_after = cm_after_norm[i, j] * 100
                annot_before[i, j] = f'{pct_before:.1f}%\n({cm_before[i, j]})'
                annot_after[i, j] = f'{pct_after:.1f}%\n({cm_after[i, j]})'

        # Before confusion matrix
        sns.heatmap(cm_before_norm, annot=annot_before, fmt='', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names, ax=axes[0],
                    cbar_kws={'label': 'Proportion'}, annot_kws={'fontsize': 10})
        axes[0].set_title('Before Recalibration', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('True')

        # After confusion matrix
        sns.heatmap(cm_after_norm, annot=annot_after, fmt='', cmap='Greens',
                    xticklabels=class_names, yticklabels=class_names, ax=axes[1],
                    cbar_kws={'label': 'Proportion'}, annot_kws={'fontsize': 10})
        axes[1].set_title('After Recalibration', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Predicted')
        axes[1].set_ylabel('True')

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, filename), dpi=150, bbox_inches='tight')
        plt.close()

        # Difference matrix with percentages
        fig, ax = plt.subplots(figsize=(8, 6))
        cm_diff = cm_after_norm - cm_before_norm

        # Create annotation for difference matrix
        annot_diff = np.empty_like(cm_diff, dtype=object)
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                diff_pct = cm_diff[i, j] * 100
                sign = '+' if diff_pct >= 0 else ''
                annot_diff[i, j] = f'{sign}{diff_pct:.1f}%'

        # Custom colormap: red for negative, white for zero, green for positive
        colors = ['#e74c3c', 'white', '#27ae60']
        cmap = LinearSegmentedColormap.from_list('diff', colors)

        sns.heatmap(cm_diff, annot=annot_diff, fmt='', cmap=cmap, center=0,
                    xticklabels=class_names, yticklabels=class_names, ax=ax,
                    cbar_kws={'label': 'Change'}, annot_kws={'fontsize': 11, 'fontweight': 'bold'})
        ax.set_title('Change in Confusion Matrix (After - Before)',
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'confusion_matrix_diff.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

    def plot_per_class_comparison(self, per_class_before: Dict, per_class_after: Dict,
                                  class_names: List[str],
                                  filename: str = "per_class_comparison.png"):
        """Plot per-class metrics comparison."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        x = np.arange(len(class_names))
        width = 0.35

        metrics = [
            ('accuracy', 'Accuracy', axes[0, 0]),
            ('precision', 'Precision', axes[0, 1]),
            ('recall', 'Recall', axes[1, 0]),
            ('f1', 'F1 Score', axes[1, 1])
        ]

        for metric, title, ax in metrics:
            before_vals = [per_class_before[c][metric] for c in class_names]
            after_vals = [per_class_after[c][metric] for c in class_names]

            bars1 = ax.bar(x - width / 2, before_vals, width, label='Before',
                           color=self.colors['before'], alpha=0.8)
            bars2 = ax.bar(x + width / 2, after_vals, width, label='After',
                           color=self.colors['after'], alpha=0.8)

            # Add value labels
            for bar in bars1:
                height = bar.get_height()
                ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=9)

            for bar in bars2:
                height = bar.get_height()
                ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=9)

            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(class_names, rotation=45, ha='right')
            ax.set_ylim(0, 1.1)
            ax.legend()
            ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, filename), dpi=150, bbox_inches='tight')
        plt.close()

        # Additional: Change plot
        fig, ax = plt.subplots(figsize=(12, 6))

        acc_changes = [per_class_after[c]['accuracy'] - per_class_before[c]['accuracy']
                       for c in class_names]

        colors = [self.colors['change_pos'] if v >= 0 else self.colors['change_neg']
                  for v in acc_changes]

        bars = ax.bar(class_names, acc_changes, color=colors, alpha=0.8, edgecolor='black')

        # Add value labels
        for bar, val in zip(bars, acc_changes):
            height = bar.get_height()
            va = 'bottom' if height >= 0 else 'top'
            offset = 3 if height >= 0 else -3
            ax.annotate(f'{val:+.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, offset), textcoords="offset points",
                        ha='center', va=va, fontsize=10, fontweight='bold')

        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_title('Per-Class Accuracy Change', fontsize=14, fontweight='bold')
        ax.set_ylabel('Change in Accuracy')
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'accuracy_change.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

    def plot_metrics_comparison(self, results_before: Dict, results_after: Dict,
                                tcav_before: float, tcav_after: float,
                                filename: str = "metrics_comparison.png"):
        """Plot overall metrics comparison."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Bar chart comparison
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'TCAV']
        before_vals = [
            results_before['overall']['accuracy'],
            results_before['overall']['precision'],
            results_before['overall']['recall'],
            results_before['overall']['f1'],
            tcav_before
        ]
        after_vals = [
            results_after['overall']['accuracy'],
            results_after['overall']['precision'],
            results_after['overall']['recall'],
            results_after['overall']['f1'],
            tcav_after
        ]

        x = np.arange(len(metrics))
        width = 0.35

        bars1 = axes[0].bar(x - width / 2, before_vals, width, label='Before',
                            color=self.colors['before'], alpha=0.8)
        bars2 = axes[0].bar(x + width / 2, after_vals, width, label='After',
                            color=self.colors['after'], alpha=0.8)

        axes[0].set_title('Metrics Before vs After', fontsize=14, fontweight='bold')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(metrics)
        axes[0].set_ylim(0, 1.1)
        axes[0].legend()
        axes[0].grid(axis='y', alpha=0.3)

        # Add value labels
        for bar in bars1:
            axes[0].annotate(f'{bar.get_height():.3f}',
                             xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                             xytext=(0, 3), textcoords="offset points",
                             ha='center', va='bottom', fontsize=9)
        for bar in bars2:
            axes[0].annotate(f'{bar.get_height():.3f}',
                             xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                             xytext=(0, 3), textcoords="offset points",
                             ha='center', va='bottom', fontsize=9)

        # Change chart
        changes = [a - b for a, b in zip(after_vals, before_vals)]
        colors = [self.colors['change_pos'] if v >= 0 else self.colors['change_neg']
                  for v in changes]

        bars = axes[1].bar(metrics, changes, color=colors, alpha=0.8, edgecolor='black')
        axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes[1].set_title('Metrics Change (After - Before)', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Change')
        axes[1].grid(axis='y', alpha=0.3)

        for bar, val in zip(bars, changes):
            height = bar.get_height()
            va = 'bottom' if height >= 0 else 'top'
            offset = 3 if height >= 0 else -3
            axes[1].annotate(f'{val:+.4f}',
                             xy=(bar.get_x() + bar.get_width() / 2, height),
                             xytext=(0, offset), textcoords="offset points",
                             ha='center', va=va, fontsize=10, fontweight='bold')

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, filename), dpi=150, bbox_inches='tight')
        plt.close()

    def plot_class_distribution(self, train_counts: Dict, val_counts: Dict,
                                filename: str = "class_distribution.png"):
        """Plot class distribution in training and validation sets."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        class_names = list(train_counts.keys())

        # Training set
        train_vals = [train_counts[c] for c in class_names]
        colors_train = plt.cm.Blues(np.linspace(0.4, 0.8, len(class_names)))

        bars1 = axes[0].bar(class_names, train_vals, color=colors_train, edgecolor='black')
        axes[0].set_title('Training Set Distribution', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Number of Images')

        for bar, val in zip(bars1, train_vals):
            axes[0].annotate(str(val), xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                             xytext=(0, 3), textcoords="offset points",
                             ha='center', va='bottom', fontsize=11, fontweight='bold')

        # Calculate percentages
        total_train = sum(train_vals)
        for i, (bar, val) in enumerate(zip(bars1, train_vals)):
            pct = val / total_train * 100
            axes[0].annotate(f'({pct:.1f}%)',
                             xy=(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2),
                             ha='center', va='center', fontsize=10, color='white',
                             fontweight='bold')

        # Validation set
        val_vals = [val_counts[c] for c in class_names]
        colors_val = plt.cm.Greens(np.linspace(0.4, 0.8, len(class_names)))

        bars2 = axes[1].bar(class_names, val_vals, color=colors_val, edgecolor='black')
        axes[1].set_title('Validation Set Distribution', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Number of Images')

        for bar, val in zip(bars2, val_vals):
            axes[1].annotate(str(val), xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                             xytext=(0, 3), textcoords="offset points",
                             ha='center', va='bottom', fontsize=11, fontweight='bold')

        total_val = sum(val_vals)
        for i, (bar, val) in enumerate(zip(bars2, val_vals)):
            pct = val / total_val * 100
            axes[1].annotate(f'({pct:.1f}%)',
                             xy=(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2),
                             ha='center', va='center', fontsize=10, color='white',
                             fontweight='bold')

        axes[0].grid(axis='y', alpha=0.3)
        axes[1].grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, filename), dpi=150, bbox_inches='tight')
        plt.close()

        # Pie chart version
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        axes[0].pie(train_vals, labels=class_names, autopct='%1.1f%%',
                    colors=colors_train, startangle=90)
        axes[0].set_title('Training Set', fontsize=14, fontweight='bold')

        axes[1].pie(val_vals, labels=class_names, autopct='%1.1f%%',
                    colors=colors_val, startangle=90)
        axes[1].set_title('Validation Set', fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'class_distribution_pie.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

    def plot_misclassification_analysis(self, misclass_before: Dict,
                                        misclass_after: Dict,
                                        class_names: List[str],
                                        filename: str = "misclassification_analysis.png"):
        """Plot misclassification analysis."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Convert to matrices
        n = len(class_names)
        matrix_before = np.zeros((n, n))
        matrix_after = np.zeros((n, n))

        for i, true_class in enumerate(class_names):
            if true_class in misclass_before:
                for pred_class, count in misclass_before[true_class].items():
                    j = class_names.index(pred_class)
                    matrix_before[i, j] = count

            if true_class in misclass_after:
                for pred_class, count in misclass_after[true_class].items():
                    j = class_names.index(pred_class)
                    matrix_after[i, j] = count

        # Before
        sns.heatmap(matrix_before, annot=True, fmt='.0f', cmap='Reds',
                    xticklabels=class_names, yticklabels=class_names, ax=axes[0])
        axes[0].set_title('Misclassifications Before', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Predicted As')
        axes[0].set_ylabel('True Class')

        # After
        sns.heatmap(matrix_after, annot=True, fmt='.0f', cmap='Oranges',
                    xticklabels=class_names, yticklabels=class_names, ax=axes[1])
        axes[1].set_title('Misclassifications After', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Predicted As')
        axes[1].set_ylabel('True Class')

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, filename), dpi=150, bbox_inches='tight')
        plt.close()

        # Summary bar chart
        fig, ax = plt.subplots(figsize=(10, 6))

        total_before = [sum(misclass_before.get(c, {}).values()) for c in class_names]
        total_after = [sum(misclass_after.get(c, {}).values()) for c in class_names]

        x = np.arange(len(class_names))
        width = 0.35

        ax.bar(x - width / 2, total_before, width, label='Before',
               color=self.colors['before'], alpha=0.8)
        ax.bar(x + width / 2, total_after, width, label='After',
               color=self.colors['after'], alpha=0.8)

        ax.set_title('Total Misclassifications by Class', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(class_names)
        ax.set_ylabel('Number of Misclassifications')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'misclassification_summary.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

    def plot_experiment3_tcav_comparison(self, tcav_before: Dict, tcav_after: Dict,
                                         class_layer_map: Dict,
                                         filename: str = "experiment3_tcav.png"):
        """
        Plot TCAV score comparison for Experiment 3.

        Args:
            tcav_before: Dict mapping class_name -> tcav_score before
            tcav_after: Dict mapping class_name -> tcav_score after
            class_layer_map: Dict mapping class_name -> layer_name
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        class_names = list(tcav_before.keys())

        # TCAV comparison bar chart
        x = np.arange(len(class_names))
        width = 0.35

        before_vals = [tcav_before[c] for c in class_names]
        after_vals = [tcav_after[c] for c in class_names]

        bars1 = axes[0].bar(x - width / 2, before_vals, width, label='Before',
                            color=self.colors['before'], alpha=0.8)
        bars2 = axes[0].bar(x + width / 2, after_vals, width, label='After',
                            color=self.colors['after'], alpha=0.8)

        # Add value labels
        for bar in bars1:
            axes[0].annotate(f'{bar.get_height():.3f}',
                             xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                             xytext=(0, 3), textcoords="offset points",
                             ha='center', va='bottom', fontsize=9)
        for bar in bars2:
            axes[0].annotate(f'{bar.get_height():.3f}',
                             xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                             xytext=(0, 3), textcoords="offset points",
                             ha='center', va='bottom', fontsize=9)

        axes[0].set_title('Per-Class TCAV Scores', fontsize=14, fontweight='bold')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(class_names, rotation=45, ha='right')
        axes[0].set_ylim(0, 1.1)
        axes[0].legend()
        axes[0].grid(axis='y', alpha=0.3)

        # TCAV change chart
        changes = [tcav_after[c] - tcav_before[c] for c in class_names]
        colors = [self.colors['change_pos'] if v >= 0 else self.colors['change_neg']
                  for v in changes]

        bars = axes[1].bar(class_names, changes, color=colors, alpha=0.8, edgecolor='black')
        axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes[1].set_title('TCAV Score Change', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Change')
        axes[1].set_xticklabels(class_names, rotation=45, ha='right')
        axes[1].grid(axis='y', alpha=0.3)

        for bar, val in zip(bars, changes):
            height = bar.get_height()
            va = 'bottom' if height >= 0 else 'top'
            offset = 3 if height >= 0 else -3
            axes[1].annotate(f'{val:+.4f}',
                             xy=(bar.get_x() + bar.get_width() / 2, height),
                             xytext=(0, offset), textcoords="offset points",
                             ha='center', va=va, fontsize=10, fontweight='bold')

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, filename), dpi=150, bbox_inches='tight')
        plt.close()

        # Class-Layer assignment table
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.axis('off')

        table_data = []
        headers = ['Class', 'Layer', 'TCAV Before', 'TCAV After', 'Change']

        for class_name in class_names:
            layer = class_layer_map[class_name]
            before = tcav_before[class_name]
            after = tcav_after[class_name]
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
        table.set_fontsize(11)
        table.scale(1.2, 1.5)

        # Color the header
        for i in range(len(headers)):
            table[(0, i)].set_text_props(color='white', fontweight='bold')

        ax.set_title('Experiment 3: Class-Layer Assignments', fontsize=14,
                     fontweight='bold', pad=20)

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'experiment3_assignments.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

    def create_summary_dashboard(self, all_results: Dict):
        """Create a comprehensive summary dashboard."""
        fig = plt.figure(figsize=(20, 16))

        # Title
        target_info = all_results.get("target_class", "multiple")
        fig.suptitle(f'Experiment {all_results["experiment"]} Summary Dashboard\n'
                     f'Model: {all_results["model_name"]} | Target: {target_info}',
                     fontsize=16, fontweight='bold', y=0.98)

        # Create grid
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
        ax2.bar(train_counts.keys(), train_counts.values(), color=plt.cm.Set2.colors)
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

        # 4-5. Per-class accuracy
        ax4 = fig.add_subplot(gs[1, :2])
        class_names = list(all_results['results_before']['per_class'].keys())
        before_acc = [all_results['results_before']['per_class'][c]['accuracy']
                      for c in class_names]
        after_acc = [all_results['results_after']['per_class'][c]['accuracy']
                     for c in class_names]

        x = np.arange(len(class_names))
        ax4.bar(x - 0.2, before_acc, 0.4, label='Before', color=self.colors['before'])
        ax4.bar(x + 0.2, after_acc, 0.4, label='After', color=self.colors['after'])
        ax4.set_xticks(x)
        ax4.set_xticklabels(class_names)
        ax4.set_title('Per-Class Accuracy', fontweight='bold')
        ax4.legend()
        ax4.set_ylim(0, 1.1)

        # 6. Configuration info
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.axis('off')

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
        ax6.text(0.1, 0.9, config_text, transform=ax6.transAxes, fontsize=12,
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # 7-9. Changes
        ax7 = fig.add_subplot(gs[2, :])
        changes = {c: after_acc[i] - before_acc[i] for i, c in enumerate(class_names)}
        colors = [self.colors['change_pos'] if v >= 0 else self.colors['change_neg']
                  for v in changes.values()]
        bars = ax7.bar(changes.keys(), changes.values(), color=colors)
        ax7.axhline(y=0, color='black', linewidth=0.5)
        ax7.set_title('Accuracy Change by Class', fontweight='bold')
        ax7.set_ylabel('Change')

        for bar, val in zip(bars, changes.values()):
            height = bar.get_height()
            ax7.annotate(f'{val:+.3f}',
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3 if height >= 0 else -10),
                         textcoords="offset points",
                         ha='center', fontweight='bold')

        plt.savefig(os.path.join(self.results_dir, 'summary_dashboard.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()
