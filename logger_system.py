"""
Comprehensive logging system for TCAV-Based Recalibration experiments.

Features:
- Detailed experiment logging
- Run history tracking
- Model-specific logging
- Result comparison logging
"""

import os
import logging
import json
from datetime import datetime
from typing import Dict, Any, Optional
import csv


class ExperimentLogger:
    """
    Comprehensive logger for TCAV recalibration experiments.
    
    Creates:
    - experiment.log: Detailed text log
    - experiment_summary.json: JSON summary
    - results_history.csv: Append-only results history
    """
    
    def __init__(self, results_dir: str, experiment_name: str):
        self.results_dir = results_dir
        self.experiment_name = experiment_name
        self.start_time = datetime.now()
        
        os.makedirs(results_dir, exist_ok=True)
        
        # Set up file logger
        self.log_file = os.path.join(results_dir, "experiment.log")
        self.logger = self._setup_logger()
        
        # JSON summary storage
        self.summary_data = {
            'experiment_name': experiment_name,
            'start_time': self.start_time.isoformat(),
            'sections': []
        }
        
        # Results history file (global)
        self.history_file = os.path.join(
            os.path.dirname(results_dir), 
            "all_experiments_history.csv"
        )
    
    def _setup_logger(self):
        """Set up Python logger."""
        logger = logging.getLogger(self.experiment_name)
        logger.setLevel(logging.DEBUG)
        
        # Clear existing handlers
        logger.handlers = []
        
        # File handler
        file_handler = logging.FileHandler(self.log_file, mode='w')
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Format
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def log_header(self, title: str):
        """Log experiment header."""
        separator = "=" * 70
        self.logger.info(separator)
        self.logger.info(f"  {title}")
        self.logger.info(f"  Experiment: {self.experiment_name}")
        self.logger.info(f"  Started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(separator)
    
    def log_section(self, title: str):
        """Log a new section."""
        self.logger.info("")
        self.logger.info("-" * 50)
        self.logger.info(f"[SECTION] {title}")
        self.logger.info("-" * 50)
        
        self.summary_data['sections'].append({
            'title': title,
            'timestamp': datetime.now().isoformat()
        })
    
    def log_info(self, message: str):
        """Log info message."""
        self.logger.info(message)
    
    def log_debug(self, message: str):
        """Log debug message."""
        self.logger.debug(message)
    
    def log_warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)
    
    def log_error(self, message: str):
        """Log error message."""
        self.logger.error(message)
    
    def log_config(self, config: Dict[str, Any]):
        """Log experiment configuration."""
        self.logger.info("Configuration:")
        for key, value in config.items():
            self.logger.info(f"  {key}: {value}")
        
        self.summary_data['config'] = config
    
    def log_epoch(self, epoch: int, total_epochs: int, 
                  total_loss: float, cls_loss: float, align_loss: float,
                  extra_info: str = None):
        """Log training epoch progress."""
        msg = (f"Epoch {epoch:3d}/{total_epochs} | "
               f"Total: {total_loss:.4f} | "
               f"Cls: {cls_loss:.4f} | "
               f"Align: {align_loss:.4f}")
        
        if extra_info:
            msg += f" | {extra_info}"
        
        self.logger.info(msg)
    
    def log_evaluation_results(self, results: Dict, phase: str, tcav_score: float):
        """
        Log detailed evaluation results.
        
        Args:
            results: Dictionary from evaluate_detailed()
            phase: "BEFORE" or "AFTER"
            tcav_score: TCAV score
        """
        self.logger.info(f"\n{'='*50}")
        self.logger.info(f"EVALUATION RESULTS - {phase}")
        self.logger.info(f"{'='*50}")
        
        # Overall metrics
        overall = results['overall']
        self.logger.info(f"\nOverall Metrics:")
        self.logger.info(f"  Accuracy:   {overall['accuracy']:.4f}")
        self.logger.info(f"  Precision:  {overall['precision']:.4f}")
        self.logger.info(f"  Recall:     {overall['recall']:.4f}")
        self.logger.info(f"  F1 Score:   {overall['f1']:.4f}")
        self.logger.info(f"  TCAV Score: {tcav_score:.4f}")
        self.logger.info(f"  Avg Conf:   {overall['avg_confidence']:.4f}")
        
        # Per-class metrics
        self.logger.info(f"\nPer-Class Metrics:")
        self.logger.info(f"  {'Class':<12} {'Total':>6} {'Correct':>8} {'Wrong':>6} "
                        f"{'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7}")
        self.logger.info(f"  {'-'*70}")
        
        for class_name, metrics in results['per_class'].items():
            self.logger.info(
                f"  {class_name:<12} {metrics['total']:>6} {metrics['correct']:>8} "
                f"{metrics['wrong']:>6} {metrics['accuracy']:>7.3f} "
                f"{metrics['precision']:>7.3f} {metrics['recall']:>7.3f} "
                f"{metrics['f1']:>7.3f}"
            )
        
        # Top misclassifications
        if results.get('top_misclassifications'):
            self.logger.info(f"\nTop Misclassifications:")
            for i, mc in enumerate(results['top_misclassifications'][:5], 1):
                self.logger.info(
                    f"  {i}. {mc['true_class']} → {mc['predicted_as']}: {mc['count']} times"
                )
        
        # Store in summary
        key = f'results_{phase.lower()}'
        self.summary_data[key] = {
            'overall': overall,
            'per_class': results['per_class'],
            'tcav_score': tcav_score
        }
    
    def log_comparison(self, results_before: Dict, results_after: Dict,
                       tcav_before: float, tcav_after: float, class_names: list):
        """Log comparison between before and after recalibration."""
        self.logger.info(f"\n{'='*70}")
        self.logger.info("COMPARISON: BEFORE vs AFTER RECALIBRATION")
        self.logger.info(f"{'='*70}")
        
        # Overall comparison
        ob = results_before['overall']
        oa = results_after['overall']
        
        self.logger.info(f"\nOverall Metrics Change:")
        self.logger.info(f"  {'Metric':<15} {'Before':>10} {'After':>10} {'Change':>10}")
        self.logger.info(f"  {'-'*45}")
        
        metrics = [
            ('Accuracy', ob['accuracy'], oa['accuracy']),
            ('Precision', ob['precision'], oa['precision']),
            ('Recall', ob['recall'], oa['recall']),
            ('F1 Score', ob['f1'], oa['f1']),
            ('TCAV Score', tcav_before, tcav_after),
            ('Avg Confidence', ob['avg_confidence'], oa['avg_confidence'])
        ]
        
        for name, before, after in metrics:
            change = after - before
            change_str = f"+{change:.4f}" if change >= 0 else f"{change:.4f}"
            self.logger.info(f"  {name:<15} {before:>10.4f} {after:>10.4f} {change_str:>10}")
        
        # Per-class comparison
        self.logger.info(f"\nPer-Class Accuracy Change:")
        self.logger.info(f"  {'Class':<12} {'Before':>10} {'After':>10} {'Change':>10}")
        self.logger.info(f"  {'-'*45}")
        
        for class_name in class_names:
            before = results_before['per_class'][class_name]['accuracy']
            after = results_after['per_class'][class_name]['accuracy']
            change = after - before
            change_str = f"+{change:.4f}" if change >= 0 else f"{change:.4f}"
            self.logger.info(f"  {class_name:<12} {before:>10.4f} {after:>10.4f} {change_str:>10}")
        
        # Store comparison
        self.summary_data['comparison'] = {
            'overall_change': {
                'accuracy': oa['accuracy'] - ob['accuracy'],
                'precision': oa['precision'] - ob['precision'],
                'recall': oa['recall'] - ob['recall'],
                'f1': oa['f1'] - ob['f1'],
                'tcav': tcav_after - tcav_before
            }
        }
    
    def log_summary(self, all_results: Dict):
        """Log final experiment summary."""
        self.logger.info(f"\n{'='*70}")
        self.logger.info("EXPERIMENT SUMMARY")
        self.logger.info(f"{'='*70}")
        
        # Key improvements
        tcav_change = all_results['tcav_after'] - all_results['tcav_before']
        acc_change = (all_results['results_after']['overall']['accuracy'] - 
                     all_results['results_before']['overall']['accuracy'])
        
        self.logger.info(f"\nKey Results:")
        self.logger.info(f"  TCAV Score: {all_results['tcav_before']:.4f} → "
                        f"{all_results['tcav_after']:.4f} ({'+' if tcav_change >= 0 else ''}{tcav_change:.4f})")
        self.logger.info(f"  Accuracy:   {all_results['results_before']['overall']['accuracy']:.4f} → "
                        f"{all_results['results_after']['overall']['accuracy']:.4f} "
                        f"({'+' if acc_change >= 0 else ''}{acc_change:.4f})")
        
        # Imbalance info
        if all_results.get('imbalance_class'):
            self.logger.info(f"\nImbalance Applied:")
            self.logger.info(f"  Class: {all_results['imbalance_class']}")
            self.logger.info(f"  Ratio: {all_results['imbalance_ratio']*100:.1f}%")
        
        # Timing
        end_time = datetime.now()
        duration = end_time - self.start_time
        self.logger.info(f"\nExecution Time: {duration}")
        
        self.summary_data['end_time'] = end_time.isoformat()
        self.summary_data['duration_seconds'] = duration.total_seconds()
        
        # Save to history
        self.save_to_history(all_results)
    
    def save_to_history(self, all_results: Dict):
        """Save results to global history CSV."""
        # Prepare row
        row = {
            'timestamp': self.start_time.isoformat(),
            'experiment': all_results['experiment'],
            'model_name': all_results['model_name'],
            'layer': all_results['layer'],
            'target_class': all_results['target_class'],
            'concept': all_results['concept'],
            'imbalance_class': all_results.get('imbalance_class', ''),
            'imbalance_ratio': all_results.get('imbalance_ratio', 1.0),
            'lambda_align': all_results['lambda_align'],
            'epochs': all_results['epochs'],
            'accuracy_before': all_results['results_before']['overall']['accuracy'],
            'accuracy_after': all_results['results_after']['overall']['accuracy'],
            'tcav_before': all_results['tcav_before'],
            'tcav_after': all_results['tcav_after'],
            'f1_before': all_results['results_before']['overall']['f1'],
            'f1_after': all_results['results_after']['overall']['f1']
        }
        
        # Append to history file
        file_exists = os.path.exists(self.history_file)
        
        with open(self.history_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
        
        self.logger.info(f"Results appended to history: {self.history_file}")
    
    def close(self):
        """Close logger and save summary."""
        # Save JSON summary
        summary_file = os.path.join(self.results_dir, "experiment_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(self.summary_data, f, indent=2, default=str)
        
        self.logger.info(f"\nSummary saved to: {summary_file}")
        self.logger.info(f"Log saved to: {self.log_file}")
        
        # Close handlers
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)


class RunHistoryManager:
    """
    Manage experiment run history across multiple experiments.
    """
    
    def __init__(self, base_results_dir: str):
        self.base_dir = base_results_dir
        self.history_file = os.path.join(base_results_dir, "all_experiments_history.csv")
        os.makedirs(base_results_dir, exist_ok=True)
    
    def load_history(self) -> list:
        """Load all experiment history."""
        if not os.path.exists(self.history_file):
            return []
        
        history = []
        with open(self.history_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                history.append(row)
        
        return history
    
    def get_best_run(self, metric='tcav_after'):
        """Get the run with the best metric value."""
        history = self.load_history()
        if not history:
            return None
        
        return max(history, key=lambda x: float(x.get(metric, 0)))
    
    def get_runs_by_model(self, model_name: str):
        """Get all runs for a specific model."""
        history = self.load_history()
        return [r for r in history if r['model_name'] == model_name]
    
    def get_runs_by_experiment(self, experiment_type: int):
        """Get all runs for a specific experiment type."""
        history = self.load_history()
        return [r for r in history if int(r['experiment']) == experiment_type]
    
    def generate_summary_report(self) -> str:
        """Generate a text summary of all runs."""
        history = self.load_history()
        if not history:
            return "No experiments recorded yet."
        
        report = []
        report.append("=" * 70)
        report.append("EXPERIMENT HISTORY SUMMARY")
        report.append("=" * 70)
        report.append(f"\nTotal runs: {len(history)}")
        
        # Group by model
        models = {}
        for run in history:
            model = run['model_name']
            if model not in models:
                models[model] = []
            models[model].append(run)
        
        report.append(f"\nRuns by model:")
        for model, runs in models.items():
            avg_tcav_change = sum(
                float(r['tcav_after']) - float(r['tcav_before']) 
                for r in runs
            ) / len(runs)
            report.append(f"  {model}: {len(runs)} runs, avg TCAV change: {avg_tcav_change:+.4f}")
        
        # Best run
        best = self.get_best_run('tcav_after')
        if best:
            report.append(f"\nBest TCAV score: {best['tcav_after']} "
                         f"(model: {best['model_name']}, exp: {best['experiment']})")
        
        return "\n".join(report)
    
    def export_to_json(self, output_file: str = None):
        """Export history to JSON format."""
        if output_file is None:
            output_file = os.path.join(self.base_dir, "history_export.json")
        
        history = self.load_history()
        with open(output_file, 'w') as f:
            json.dump(history, f, indent=2)
        
        return output_file
