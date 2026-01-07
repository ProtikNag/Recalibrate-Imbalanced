#!/usr/bin/env python3
"""
Unit Tests for TCAV-Based Recalibration

Run with pytest:
    pytest test_components.py -v
    pytest test_components.py -v -k "test_model"  # Run specific tests
    pytest test_components.py -v --tb=short       # Short traceback
"""

import os
import sys
import pytest
import numpy as np
import torch
import torch.nn as nn
import tempfile
import shutil

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def device():
    """Get compute device."""
    return 'cuda' if torch.cuda.is_available() else 'cpu'


@pytest.fixture
def temp_dir():
    """Create temporary directory for test outputs."""
    tmp = tempfile.mkdtemp(prefix="tcav_test_")
    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture
def dummy_images():
    """Create dummy image batch."""
    return torch.randn(4, 3, 224, 224)


@pytest.fixture
def dummy_labels():
    """Create dummy labels."""
    return torch.randint(0, 3, (4,))


@pytest.fixture
def synthetic_activations():
    """Create synthetic activations for CAV testing."""
    np.random.seed(42)
    concept_acts = np.random.randn(50, 256) + 1.0
    random_acts = np.random.randn(50, 256) - 1.0
    return concept_acts, random_acts


# ============================================================================
# Model Tests
# ============================================================================

class TestModels:
    """Test model loading and forward pass."""
    
    @pytest.mark.parametrize("model_name,num_classes", [
        ("custom_cnn", 3),
        ("custom_cnn", 5),
        ("custom_cnn", 10),
        ("custom_cnn_small", 3),
        ("custom_cnn_small", 5),
        ("custom_cnn_large", 5),
    ])
    def test_custom_model_creation(self, model_name, num_classes, device):
        """Test custom model creation with various class counts."""
        from utils import load_model
        
        model = load_model(model_name, num_classes=num_classes)
        model = model.to(device)
        
        # Check model is created
        assert model is not None
        
        # Check output dimension
        dummy = torch.randn(2, 3, 224, 224).to(device)
        model.eval()
        with torch.no_grad():
            output = model(dummy)
        
        assert output.shape == (2, num_classes)
    
    @pytest.mark.parametrize("model_name", [
        "vgg16", "resnet18", "resnet50", "mobilenet_v3_small"
    ])
    def test_pretrained_model_creation(self, model_name, device):
        """Test pretrained model loading (without weights for speed)."""
        from utils import load_model
        
        model = load_model(model_name, num_classes=5, pretrained=False)
        model = model.to(device)
        
        dummy = torch.randn(2, 3, 224, 224).to(device)
        model.eval()
        with torch.no_grad():
            output = model(dummy)
            if isinstance(output, tuple):
                output = output[0]
        
        assert output.shape == (2, 5)
    
    def test_model_training_mode(self, device):
        """Test model can switch between train/eval modes."""
        from utils import load_model
        
        model = load_model("custom_cnn_small", num_classes=3)
        model = model.to(device)
        
        model.train()
        assert model.training
        
        model.eval()
        assert not model.training
    
    def test_model_gradient_flow(self, device, dummy_images, dummy_labels):
        """Test gradients flow through model."""
        from utils import load_model
        
        model = load_model("custom_cnn_small", num_classes=3)
        model = model.to(device)
        model.train()
        
        images = dummy_images.to(device)
        labels = dummy_labels.to(device)
        
        output = model(images)
        loss = nn.CrossEntropyLoss()(output, labels)
        loss.backward()
        
        # Check some parameters have gradients
        has_gradients = False
        for param in model.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_gradients = True
                break
        
        assert has_gradients


# ============================================================================
# CAV Tests
# ============================================================================

class TestCAV:
    """Test CAV training and computation."""
    
    def test_cav_training_linear_svc(self, synthetic_activations):
        """Test CAV training with LinearSVC."""
        from utils import train_cav
        
        concept_acts, random_acts = synthetic_activations
        cav = train_cav(concept_acts, random_acts, classifier_type='LinearSVC')
        
        assert cav.shape == (256,)
        assert np.abs(np.linalg.norm(cav) - 1.0) < 0.01  # Normalized
    
    def test_cav_training_sgd(self, synthetic_activations):
        """Test CAV training with SGDClassifier."""
        from utils import train_cav
        
        concept_acts, random_acts = synthetic_activations
        cav = train_cav(concept_acts, random_acts, classifier_type='SGDClassifier')
        
        assert cav.shape == (256,)
        assert np.abs(np.linalg.norm(cav) - 1.0) < 0.01
    
    def test_cav_training_logistic(self, synthetic_activations):
        """Test CAV training with LogisticRegression."""
        from utils import train_cav
        
        concept_acts, random_acts = synthetic_activations
        cav = train_cav(concept_acts, random_acts, classifier_type='LogisticRegression')
        
        assert cav.shape == (256,)
        assert np.abs(np.linalg.norm(cav) - 1.0) < 0.01
    
    def test_cav_separates_concepts(self, synthetic_activations):
        """Test that CAV can separate concept from random activations."""
        from utils import train_cav
        
        concept_acts, random_acts = synthetic_activations
        cav = train_cav(concept_acts, random_acts, classifier_type='LinearSVC')
        
        # Project activations onto CAV
        concept_proj = concept_acts @ cav
        random_proj = random_acts @ cav
        
        # Concept projections should be higher on average
        assert np.mean(concept_proj) > np.mean(random_proj)


# ============================================================================
# Evaluation Tests
# ============================================================================

class TestEvaluation:
    """Test evaluation functions."""
    
    def test_evaluate_detailed_structure(self, device):
        """Test evaluation returns correct structure."""
        from utils import load_model, evaluate_detailed
        from torch.utils.data import TensorDataset, DataLoader
        
        model = load_model("custom_cnn_small", num_classes=3)
        model = model.to(device)
        
        # Create dummy data
        images = torch.randn(20, 3, 224, 224)
        labels = torch.randint(0, 3, (20,))
        dataset = TensorDataset(images, labels)
        loader = DataLoader(dataset, batch_size=8)
        
        class_names = ['a', 'b', 'c']
        results = evaluate_detailed(model, loader, class_names, device)
        
        # Check structure
        assert 'overall' in results
        assert 'per_class' in results
        assert 'confusion_matrix' in results
        assert 'misclassification_matrix' in results
        assert 'top_misclassifications' in results
    
    def test_evaluate_detailed_metrics(self, device):
        """Test evaluation computes valid metrics."""
        from utils import load_model, evaluate_detailed
        from torch.utils.data import TensorDataset, DataLoader
        
        model = load_model("custom_cnn_small", num_classes=3)
        model = model.to(device)
        
        images = torch.randn(30, 3, 224, 224)
        labels = torch.randint(0, 3, (30,))
        dataset = TensorDataset(images, labels)
        loader = DataLoader(dataset, batch_size=8)
        
        class_names = ['a', 'b', 'c']
        results = evaluate_detailed(model, loader, class_names, device)
        
        # Check metrics are valid
        assert 0 <= results['overall']['accuracy'] <= 1
        assert 0 <= results['overall']['precision'] <= 1
        assert 0 <= results['overall']['recall'] <= 1
        assert 0 <= results['overall']['f1'] <= 1
        
        # Check per-class metrics
        for class_name in class_names:
            assert class_name in results['per_class']
            metrics = results['per_class'][class_name]
            assert 0 <= metrics['accuracy'] <= 1
    
    def test_confusion_matrix_shape(self, device):
        """Test confusion matrix has correct shape."""
        from utils import load_model, compute_confusion_matrix
        from torch.utils.data import TensorDataset, DataLoader
        
        num_classes = 4
        model = load_model("custom_cnn_small", num_classes=num_classes)
        model = model.to(device)
        
        images = torch.randn(20, 3, 224, 224)
        labels = torch.randint(0, num_classes, (20,))
        dataset = TensorDataset(images, labels)
        loader = DataLoader(dataset, batch_size=8)
        
        class_names = [f'class_{i}' for i in range(num_classes)]
        cm = compute_confusion_matrix(model, loader, class_names, device)
        
        assert cm.shape == (num_classes, num_classes)


# ============================================================================
# Visualization Tests
# ============================================================================

class TestVisualizations:
    """Test visualization functions."""
    
    def test_loss_curves(self, temp_dir):
        """Test loss curve generation."""
        from visualizations import ResultVisualizer
        
        viz = ResultVisualizer(temp_dir)
        
        loss_history = {
            'total': [1.0, 0.8, 0.6],
            'cls': [0.8, 0.6, 0.4],
            'align': [0.2, 0.2, 0.2],
        }
        
        viz.plot_loss_curves(loss_history, epochs=3)
        
        assert os.path.exists(os.path.join(temp_dir, 'loss_curves.png'))
        assert os.path.exists(os.path.join(temp_dir, 'loss_curves.svg'))
    
    def test_confusion_matrices(self, temp_dir):
        """Test confusion matrix visualization."""
        from visualizations import ResultVisualizer
        
        viz = ResultVisualizer(temp_dir)
        
        cm_before = [[80, 20], [30, 70]]
        cm_after = [[90, 10], [20, 80]]
        class_names = ['A', 'B']
        
        viz.plot_confusion_matrices(cm_before, cm_after, class_names)
        
        assert os.path.exists(os.path.join(temp_dir, 'confusion_matrices.png'))
        assert os.path.exists(os.path.join(temp_dir, 'confusion_matrices.svg'))
    
    def test_per_class_comparison(self, temp_dir):
        """Test per-class comparison plots."""
        from visualizations import ResultVisualizer
        
        viz = ResultVisualizer(temp_dir)
        
        per_class_before = {
            'A': {'accuracy': 0.7, 'precision': 0.7, 'recall': 0.7, 'f1': 0.7},
            'B': {'accuracy': 0.8, 'precision': 0.8, 'recall': 0.8, 'f1': 0.8},
        }
        per_class_after = {
            'A': {'accuracy': 0.85, 'precision': 0.85, 'recall': 0.85, 'f1': 0.85},
            'B': {'accuracy': 0.9, 'precision': 0.9, 'recall': 0.9, 'f1': 0.9},
        }
        
        viz.plot_per_class_comparison(per_class_before, per_class_after, ['A', 'B'])
        
        assert os.path.exists(os.path.join(temp_dir, 'per_class_comparison.png'))
    
    @pytest.mark.parametrize("n_classes", [2, 3, 5, 7, 10])
    def test_visualizations_n_classes(self, temp_dir, n_classes):
        """Test visualizations work with various class counts."""
        from visualizations import ResultVisualizer
        
        viz = ResultVisualizer(temp_dir)
        
        class_names = [f'Class{i}' for i in range(n_classes)]
        
        # Create dummy data
        per_class_before = {
            c: {'accuracy': 0.7, 'precision': 0.7, 'recall': 0.7, 'f1': 0.7}
            for c in class_names
        }
        per_class_after = {
            c: {'accuracy': 0.8, 'precision': 0.8, 'recall': 0.8, 'f1': 0.8}
            for c in class_names
        }
        
        viz.plot_per_class_comparison(per_class_before, per_class_after, class_names)
        viz.plot_accuracy_change(per_class_before, per_class_after, class_names)
        
        assert os.path.exists(os.path.join(temp_dir, 'per_class_comparison.png'))
        assert os.path.exists(os.path.join(temp_dir, 'accuracy_change.png'))


# ============================================================================
# Dataloader Tests (requires network/disk)
# ============================================================================

class TestDataloader:
    """Test dataloader functionality."""
    
    def test_transforms(self):
        """Test transform creation."""
        from dataloader_caltech import get_transforms
        
        train_transform = get_transforms(224, is_train=True)
        val_transform = get_transforms(224, is_train=False)
        
        assert train_transform is not None
        assert val_transform is not None
    
    def test_parse_functions(self):
        """Test command line parsing functions."""
        from main_experiment import parse_class_concept_map
        
        # Test class concept map parsing
        mapping = parse_class_concept_map("a:concept_a,b:concept_b,c:concept_c")
        assert mapping == {'a': 'concept_a', 'b': 'concept_b', 'c': 'concept_c'}
        
        # Test empty string
        mapping = parse_class_concept_map("")
        assert mapping == {}
    
    @pytest.mark.slow
    def test_caltech_dataset_creation(self):
        """Test Caltech-101 dataset creation (slow, requires download)."""
        from dataloader_caltech import create_caltech_vehicle_dataset
        
        train_ds, val_ds = create_caltech_vehicle_dataset(
            root='./data',
            class_names=['airplanes', 'Motorbikes'],
            imbalance_class='airplanes',
            imbalance_ratio=0.5,
            download=True
        )
        
        assert len(train_ds) > 0
        assert len(val_ds) > 0
        
        # Test sample loading
        img, label = train_ds[0]
        assert img.shape == (3, 224, 224)
        assert label in [0, 1]


# ============================================================================
# Logger Tests
# ============================================================================

class TestLogger:
    """Test logging system."""
    
    def test_logger_creation(self, temp_dir):
        """Test logger creation."""
        from logger_system import ExperimentLogger
        
        logger = ExperimentLogger(temp_dir, "test_experiment")
        
        assert logger is not None
        assert os.path.exists(logger.log_file)
    
    def test_logger_sections(self, temp_dir):
        """Test logger section logging."""
        from logger_system import ExperimentLogger
        
        logger = ExperimentLogger(temp_dir, "test_experiment")
        
        logger.log_header("Test Header")
        logger.log_section("Test Section")
        logger.log_info("Test info message")
        logger.log_warning("Test warning")
        
        logger.close()
        
        # Check log file contains expected content
        with open(logger.log_file, 'r') as f:
            content = f.read()
        
        assert "Test Header" in content
        assert "Test Section" in content
        assert "Test info message" in content
        assert "Test warning" in content
    
    def test_logger_config(self, temp_dir):
        """Test logging configuration."""
        from logger_system import ExperimentLogger
        
        logger = ExperimentLogger(temp_dir, "test_experiment")
        
        config = {'model': 'test', 'epochs': 10, 'lr': 0.001}
        logger.log_config(config)
        
        logger.close()
        
        # Check JSON summary
        import json
        summary_file = os.path.join(temp_dir, 'experiment_summary.json')
        assert os.path.exists(summary_file)
        
        with open(summary_file, 'r') as f:
            summary = json.load(f)
        
        assert summary['config'] == config


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests combining multiple components."""
    
    def test_model_evaluation_pipeline(self, device):
        """Test model + evaluation pipeline."""
        from utils import load_model, evaluate_detailed
        from torch.utils.data import TensorDataset, DataLoader
        
        # Create model
        num_classes = 4
        model = load_model("custom_cnn_small", num_classes=num_classes)
        model = model.to(device)
        
        # Create data
        images = torch.randn(32, 3, 224, 224)
        labels = torch.randint(0, num_classes, (32,))
        dataset = TensorDataset(images, labels)
        loader = DataLoader(dataset, batch_size=8)
        
        class_names = [f'class_{i}' for i in range(num_classes)]
        
        # Evaluate
        results = evaluate_detailed(model, loader, class_names, device)
        
        # Verify results
        assert len(results['per_class']) == num_classes
        assert len(results['confusion_matrix']) == num_classes
    
    def test_visualization_from_results(self, device, temp_dir):
        """Test visualizations from evaluation results."""
        from utils import load_model, evaluate_detailed
        from visualizations import ResultVisualizer
        from torch.utils.data import TensorDataset, DataLoader
        
        # Get results
        num_classes = 3
        model = load_model("custom_cnn_small", num_classes=num_classes)
        model = model.to(device)
        
        images = torch.randn(30, 3, 224, 224)
        labels = torch.randint(0, num_classes, (30,))
        dataset = TensorDataset(images, labels)
        loader = DataLoader(dataset, batch_size=8)
        
        class_names = ['cat', 'dog', 'bird']
        results = evaluate_detailed(model, loader, class_names, device)
        
        # Create visualizations
        viz = ResultVisualizer(temp_dir)
        viz.plot_confusion_matrices(
            results['confusion_matrix'],
            results['confusion_matrix'],  # Same for simplicity
            class_names
        )
        
        assert os.path.exists(os.path.join(temp_dir, 'confusion_matrices.png'))


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
