"""
Experiment logging and tracking utilities.

Provides:
- Structured logging to console and file
- Experiment result saving/loading
- Metric tracking
- Progress monitoring
"""

import logging
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                           np.int16, np.int32, np.int64, np.uint8,
                           np.uint16, np.uint32, np.uint64)):
            return int(obj)
        if isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


class ExperimentLogger:
    """
    Comprehensive experiment logger.
    
    Handles:
    - Console and file logging
    - Metric tracking
    - Result serialization
    """
    
    def __init__(self, experiment_dir: str, experiment_name: str = "experiment"):
        """
        Initialize experiment logger.
        
        Args:
            experiment_dir: Directory to save logs and results
            experiment_name: Name of the experiment
        """
        self.experiment_dir = experiment_dir
        self.experiment_name = experiment_name
        
        # Create experiment directory
        os.makedirs(experiment_dir, exist_ok=True)
        
        # Setup logging
        self.log_file = os.path.join(experiment_dir, "experiment.log")
        self.logger = self._setup_logger()
        
        # Metrics storage
        self.metrics: Dict[str, List[float]] = {}
        self.results: Dict[str, Any] = {}
        
        # Timing
        self.start_time = datetime.now()
        self.timestamps: Dict[str, datetime] = {}
        
        self.info(f"Experiment started: {experiment_name}")
        self.info(f"Output directory: {experiment_dir}")
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger(self.experiment_name)
        logger.setLevel(logging.DEBUG)
        
        # Remove existing handlers
        logger.handlers = []
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)
        
        # File handler
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
        
        return logger
    
    def debug(self, message: str):
        """Log debug message."""
        self.logger.debug(message)
        
    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)
        
    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)
        
    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)
        
    def critical(self, message: str):
        """Log critical message."""
        self.logger.critical(message)
        
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log metrics for tracking.
        
        Args:
            metrics: Dictionary of metric names to values
            step: Optional step/epoch number
        """
        step_str = f"Step {step}: " if step is not None else ""
        metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.info(f"{step_str}{metric_str}")
        
        # Store for later
        for name, value in metrics.items():
            if name not in self.metrics:
                self.metrics[name] = []
            self.metrics[name].append(value)
            
    def log_config(self, config: Dict):
        """
        Log experiment configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.info("=" * 60)
        self.info("EXPERIMENT CONFIGURATION")
        self.info("=" * 60)
        for key, value in config.items():
            self.info(f"  {key}: {value}")
        self.info("=" * 60)
        
        # Save config to file
        config_path = os.path.join(self.experiment_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2, cls=NumpyEncoder)
            
    def start_phase(self, phase_name: str):
        """Mark the start of a phase."""
        self.timestamps[f"{phase_name}_start"] = datetime.now()
        self.info(f"{'='*20} Starting: {phase_name} {'='*20}")
        
    def end_phase(self, phase_name: str):
        """Mark the end of a phase and log duration."""
        end_time = datetime.now()
        self.timestamps[f"{phase_name}_end"] = end_time
        
        start_key = f"{phase_name}_start"
        if start_key in self.timestamps:
            duration = end_time - self.timestamps[start_key]
            self.info(f"{'='*20} Completed: {phase_name} (Duration: {duration}) {'='*20}")
        else:
            self.info(f"{'='*20} Completed: {phase_name} {'='*20}")
            
    def set_result(self, key: str, value: Any):
        """
        Set a result value.
        
        Args:
            key: Result key
            value: Result value
        """
        self.results[key] = value
        
    def get_result(self, key: str) -> Any:
        """Get a result value."""
        return self.results.get(key, None)
    
    def save_results(self, additional_results: Optional[Dict] = None):
        """
        Save all results to JSON file.
        
        Args:
            additional_results: Additional results to include
        """
        if additional_results:
            self.results.update(additional_results)
        
        # Add metrics
        self.results['metrics_history'] = self.metrics
        
        # Add timing info
        self.results['experiment_duration'] = str(datetime.now() - self.start_time)
        self.results['start_time'] = self.start_time.isoformat()
        self.results['end_time'] = datetime.now().isoformat()
        
        # Save to file
        results_path = os.path.join(self.experiment_dir, "results.json")
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2, cls=NumpyEncoder)
        
        self.info(f"Results saved to: {results_path}")
        
    def save_model(self, model, name: str = "model"):
        """
        Save PyTorch model.
        
        Args:
            model: PyTorch model
            name: Model name
        """
        import torch
        model_path = os.path.join(self.experiment_dir, f"{name}.pth")
        torch.save(model.state_dict(), model_path)
        self.info(f"Model saved to: {model_path}")
        
    def log_separator(self, char: str = "-", length: int = 60):
        """Log a separator line."""
        self.info(char * length)
        
    def log_dict(self, d: Dict, title: str = ""):
        """
        Log a dictionary in formatted way.
        
        Args:
            d: Dictionary to log
            title: Optional title
        """
        if title:
            self.info(f"{title}:")
        for key, value in d.items():
            if isinstance(value, float):
                self.info(f"  {key}: {value:.4f}")
            else:
                self.info(f"  {key}: {value}")
                
    def log_table(self, headers: List[str], rows: List[List], title: str = ""):
        """
        Log a formatted table.
        
        Args:
            headers: Column headers
            rows: Table rows
            title: Optional title
        """
        if title:
            self.info(f"\n{title}")
        
        # Calculate column widths
        widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                widths[i] = max(widths[i], len(str(cell)))
        
        # Format header
        header_line = " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
        separator = "-+-".join("-" * w for w in widths)
        
        self.info(header_line)
        self.info(separator)
        
        # Format rows
        for row in rows:
            row_line = " | ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row))
            self.info(row_line)
            
    def finalize(self):
        """Finalize the experiment and log summary."""
        duration = datetime.now() - self.start_time
        
        self.info("=" * 60)
        self.info("EXPERIMENT COMPLETED")
        self.info(f"Total duration: {duration}")
        self.info(f"Output directory: {self.experiment_dir}")
        self.info("=" * 60)
        
        # Save final results
        self.save_results()


class ProgressTracker:
    """
    Track progress for iterative processes.
    """
    
    def __init__(self, total: int, description: str = "Progress", 
                 log_interval: int = 10, logger: Optional[ExperimentLogger] = None):
        """
        Initialize progress tracker.
        
        Args:
            total: Total number of iterations
            description: Description of the process
            log_interval: Log every N iterations
            logger: Optional experiment logger
        """
        self.total = total
        self.description = description
        self.log_interval = log_interval
        self.logger = logger
        self.current = 0
        self.start_time = datetime.now()
        
    def update(self, n: int = 1):
        """
        Update progress.
        
        Args:
            n: Number of iterations completed
        """
        self.current += n
        
        if self.current % self.log_interval == 0 or self.current == self.total:
            elapsed = datetime.now() - self.start_time
            progress = self.current / self.total * 100
            
            if self.current > 0:
                eta = elapsed / self.current * (self.total - self.current)
            else:
                eta = "N/A"
            
            message = f"{self.description}: {self.current}/{self.total} ({progress:.1f}%) - Elapsed: {elapsed}, ETA: {eta}"
            
            if self.logger:
                self.logger.info(message)
            else:
                print(message)
                
    def done(self):
        """Mark as complete."""
        elapsed = datetime.now() - self.start_time
        message = f"{self.description}: Complete! Total time: {elapsed}"
        
        if self.logger:
            self.logger.info(message)
        else:
            print(message)
