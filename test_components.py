"""
Unit tests for VL-CAV recalibration framework.

Run with:
    pytest test_components.py -v
    pytest test_components.py -v -k "TestModels"
    pytest test_components.py -v --tb=short
    
Quick sanity check:
    python test_components.py
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import os
import sys
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import ExperimentConfig, get_class_concept_descriptions
from models import (
    create_model, CustomCNN, ActivationExtractor, 
    get_conv_layer_names, get_layer_output_dim,
    freeze_layers, unfreeze_layers, get_trainable_params
)
from dataloader import (
    load_dataset, create_dataloaders, get_dataset_info,
    get_transforms, ImbalancedDataset, get_class_samples,
    compute_class_weights
)
from vl_cav import CAVTrainer, SensitivityComputer, BottleneckDetector
from logger import ExperimentLogger, NumpyEncoder
from visualizations import (
    set_academic_style, plot_training_curves, 
    plot_class_distribution, plot_per_class_accuracy
)


# ============================================================================
# Test Configuration
# ============================================================================

class TestConfig:
    """Tests for configuration management."""
    
    def test_default_config_creation(self):
        """Test creating default configuration."""
        config = ExperimentConfig()
        
        assert config.experiment_name == "vl_cav_experiment"
        assert config.seed == 42
        assert config.model_name == "resnet18"
        assert config.dataset_name == "CIFAR10"
        assert config.num_classes == 10
        assert config.batch_size == 32
        
    def test_custom_config(self):
        """Test creating custom configuration."""
        config = ExperimentConfig(
            experiment_name="test_exp",
            model_name="vgg16",
            imbalance_classes=[0, 1, 2],
            imbalance_ratio=0.05,
            alpha=0.8
        )
        
        assert config.experiment_name == "test_exp"
        assert config.model_name == "vgg16"
        assert config.imbalance_classes == [0, 1, 2]
        assert config.imbalance_ratio == 0.05
        assert config.alpha == 0.8
        
    def test_config_save_load(self):
        """Test saving and loading configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ExperimentConfig(
                experiment_name="save_test",
                seed=123,
                output_dir=tmpdir
            )
            
            save_path = config.save(os.path.join(tmpdir, "test_config.json"))
            assert os.path.exists(save_path)
            
            loaded_config = ExperimentConfig.load(save_path)
            assert loaded_config.experiment_name == "save_test"
            assert loaded_config.seed == 123
            
    def test_concept_descriptions(self):
        """Test concept description generation."""
        descriptions = get_class_concept_descriptions("CIFAR10", 0, "airplane")
        
        assert len(descriptions) > 0
        assert all(isinstance(d, str) for d in descriptions)
        assert any("airplane" in d.lower() for d in descriptions)


# ============================================================================
# Test Models
# ============================================================================

class TestModels:
    """Tests for CNN models."""
    
    @pytest.fixture
    def device(self):
        return "cuda" if torch.cuda.is_available() else "cpu"
    
    def test_custom_cnn_creation(self):
        """Test CustomCNN model creation."""
        model = CustomCNN(num_classes=10)
        
        assert isinstance(model, nn.Module)
        assert hasattr(model, 'conv1')
        assert hasattr(model, 'classifier')
        
    def test_custom_cnn_forward(self, device):
        """Test CustomCNN forward pass."""
        model = CustomCNN(num_classes=10).to(device)
        x = torch.randn(2, 3, 224, 224).to(device)
        
        output = model(x)
        
        assert output.shape == (2, 10)
        
    @pytest.mark.parametrize("model_name", [
        "resnet18", "vgg16", "mobilenet_v3_small", "custom_cnn"
    ])
    def test_create_model(self, model_name, device):
        """Test model creation for different architectures."""
        model = create_model(model_name, num_classes=10, pretrained=False)
        model = model.to(device)
        
        x = torch.randn(2, 3, 224, 224).to(device)
        output = model(x)
        
        assert output.shape == (2, 10)
        
    def test_activation_extractor(self, device):
        """Test activation extraction from model."""
        model = create_model("resnet18", num_classes=10, pretrained=False).to(device)
        layer_names = ["layer1", "layer2"]
        
        extractor = ActivationExtractor(model, layer_names)
        x = torch.randn(2, 3, 224, 224).to(device)
        
        _ = model(x)
        activations = extractor.get_activations()
        
        assert "layer1" in activations
        assert "layer2" in activations
        assert activations["layer1"].shape[0] == 2
        
        extractor.remove_hooks()
        
    def test_get_conv_layer_names(self, device):
        """Test getting convolutional layer names."""
        model = create_model("resnet18", num_classes=10, pretrained=False)
        
        layer_names = get_conv_layer_names(model, "resnet18")
        
        assert len(layer_names) > 0
        assert all(isinstance(name, str) for name in layer_names)
        
    def test_freeze_layers(self, device):
        """Test layer freezing functionality."""
        model = create_model("resnet18", num_classes=10, pretrained=False).to(device)
        
        initial_trainable = get_trainable_params(model)
        
        freeze_layers(model, freeze_all_except=["fc"])
        
        frozen_trainable = get_trainable_params(model)
        
        assert frozen_trainable < initial_trainable
        
    def test_layer_output_dim(self, device):
        """Test getting layer output dimension."""
        model = create_model("resnet18", num_classes=10, pretrained=False)
        
        dim = get_layer_output_dim(model, "layer1", device=device)
        
        assert dim > 0
        assert isinstance(dim, int)


# ============================================================================
# Test Dataloaders
# ============================================================================

class TestDataloaders:
    """Tests for data loading utilities."""
    
    def test_get_dataset_info(self):
        """Test getting dataset information."""
        info = get_dataset_info("CIFAR10")
        
        assert info["num_classes"] == 10
        assert len(info["class_names"]) == 10
        assert info["channels"] == 3
        
    def test_get_transforms(self):
        """Test getting data transforms."""
        train_transform, test_transform = get_transforms("CIFAR10", image_size=224)
        
        assert train_transform is not None
        assert test_transform is not None
        
    @pytest.mark.slow
    def test_load_dataset_balanced(self):
        """Test loading balanced dataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            train_ds, test_ds, info = load_dataset(
                "CIFAR10", root=tmpdir, image_size=32, augment=False
            )
            
            assert len(train_ds) > 0
            assert len(test_ds) > 0
            assert info["num_classes"] == 10
            
    @pytest.mark.slow
    def test_load_dataset_imbalanced(self):
        """Test loading imbalanced dataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            train_ds, test_ds, info = load_dataset(
                "CIFAR10", root=tmpdir, image_size=32,
                imbalance_classes=[0, 1],
                imbalance_ratio=0.1
            )
            
            assert info["imbalanced"] == True
            assert info["imbalance_factor"] > 1.0
            
            # Check that imbalanced classes have fewer samples
            class_counts = info["class_counts"]
            assert class_counts[0] < class_counts[2]
            
    def test_imbalanced_dataset_wrapper(self):
        """Test ImbalancedDataset wrapper."""
        # Create mock dataset
        class MockDataset:
            def __init__(self):
                self.targets = list(range(10)) * 100  # 100 samples per class
                
            def __len__(self):
                return len(self.targets)
                
            def __getitem__(self, idx):
                return torch.randn(3, 32, 32), self.targets[idx]
        
        base_ds = MockDataset()
        imb_ds = ImbalancedDataset(base_ds, imbalance_classes=[0, 1], imbalance_ratio=0.1)
        
        counts = imb_ds.get_class_counts()
        
        assert counts[0] < counts[2]
        assert counts[1] < counts[3]
        assert counts[0] == int(100 * 0.1) or counts[0] == 10
        
    def test_compute_class_weights(self):
        """Test class weight computation."""
        class MockDataset:
            def __init__(self):
                self.targets = [0]*10 + [1]*50 + [2]*100
                
            def __len__(self):
                return len(self.targets)
        
        ds = MockDataset()
        weights = compute_class_weights(ds)
        
        assert len(weights) == 3
        assert weights[0] > weights[2]  # Minority class has higher weight


# ============================================================================
# Test VL-CAV Components
# ============================================================================

class TestCAV:
    """Tests for CAV training and analysis."""
    
    def test_cav_trainer_init(self):
        """Test CAV trainer initialization."""
        trainer = CAVTrainer(classifier_type="sgd")
        
        assert trainer.classifier_type == "sgd"
        assert len(trainer.cavs) == 0
        
    def test_cav_training(self):
        """Test CAV training with synthetic data."""
        trainer = CAVTrainer(classifier_type="logistic")
        
        # Create separable synthetic data
        np.random.seed(42)
        concept_acts = np.random.randn(50, 128) + 2.0
        random_acts = np.random.randn(50, 128) - 2.0
        
        cav, accuracy = trainer.train_cav(concept_acts, random_acts, "test_layer")
        
        assert cav.shape == (128,)
        assert np.isclose(np.linalg.norm(cav), 1.0)  # CAV should be normalized
        assert accuracy > 0.7  # Should be able to separate well
        
    def test_cav_retrieval(self):
        """Test retrieving trained CAV."""
        trainer = CAVTrainer()
        
        concept_acts = np.random.randn(50, 64)
        random_acts = np.random.randn(50, 64)
        
        trainer.train_cav(concept_acts, random_acts, "layer1")
        
        retrieved = trainer.get_cav("layer1")
        nonexistent = trainer.get_cav("layer99")
        
        assert retrieved is not None
        assert nonexistent is None


class TestSensitivity:
    """Tests for sensitivity computation."""
    
    @pytest.fixture
    def model_and_device(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = create_model("custom_cnn", num_classes=5, pretrained=False).to(device)
        return model, device
    
    def test_sensitivity_computer_init(self, model_and_device):
        """Test sensitivity computer initialization."""
        model, device = model_and_device
        
        computer = SensitivityComputer(model, device)
        
        assert computer.model is model
        assert computer.device == device


# ============================================================================
# Test Logger
# ============================================================================

class TestLogger:
    """Tests for experiment logging."""
    
    def test_logger_creation(self):
        """Test logger creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = ExperimentLogger(tmpdir, "test_experiment")
            
            assert os.path.exists(os.path.join(tmpdir, "experiment.log"))
            
    def test_logger_messages(self):
        """Test logging different message types."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = ExperimentLogger(tmpdir, "test")
            
            logger.info("Info message")
            logger.debug("Debug message")
            logger.warning("Warning message")
            
            # Check log file contains messages
            with open(os.path.join(tmpdir, "experiment.log"), 'r') as f:
                content = f.read()
                assert "Info message" in content
                
    def test_logger_metrics(self):
        """Test metric logging."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = ExperimentLogger(tmpdir, "test")
            
            logger.log_metrics({"loss": 0.5, "accuracy": 0.9}, step=1)
            logger.log_metrics({"loss": 0.3, "accuracy": 0.95}, step=2)
            
            assert "loss" in logger.metrics
            assert len(logger.metrics["loss"]) == 2
            
    def test_numpy_encoder(self):
        """Test NumPy JSON encoder."""
        import json
        
        data = {
            "array": np.array([1, 2, 3]),
            "float": np.float64(1.5),
            "int": np.int32(42)
        }
        
        encoded = json.dumps(data, cls=NumpyEncoder)
        decoded = json.loads(encoded)
        
        assert decoded["array"] == [1, 2, 3]
        assert decoded["float"] == 1.5
        assert decoded["int"] == 42


# ============================================================================
# Test Visualizations
# ============================================================================

class TestVisualizations:
    """Tests for visualization generation."""
    
    def test_set_academic_style(self):
        """Test setting academic matplotlib style."""
        import matplotlib.pyplot as plt
        
        set_academic_style()
        
        # Check some style settings
        assert plt.rcParams['font.family'] == ['serif']
        
    def test_plot_training_curves(self):
        """Test training curve plotting."""
        with tempfile.TemporaryDirectory() as tmpdir:
            history = {
                'train_loss': [1.0, 0.8, 0.6, 0.5],
                'val_accuracy': [0.5, 0.6, 0.7, 0.75]
            }
            
            output_path = os.path.join(tmpdir, "training_curves")
            plot_training_curves(history, output_path, formats=['png'])
            
            assert os.path.exists(f"{output_path}.png")
            
    def test_plot_class_distribution(self):
        """Test class distribution plotting."""
        with tempfile.TemporaryDirectory() as tmpdir:
            class_counts = {0: 100, 1: 50, 2: 200, 3: 150}
            class_names = ["A", "B", "C", "D"]
            
            output_path = os.path.join(tmpdir, "class_dist")
            plot_class_distribution(
                class_counts, class_names, output_path,
                imbalance_classes=[1], formats=['png']
            )
            
            assert os.path.exists(f"{output_path}.png")
            
    def test_plot_per_class_accuracy(self):
        """Test per-class accuracy plotting."""
        with tempfile.TemporaryDirectory() as tmpdir:
            acc_before = {0: 0.8, 1: 0.3, 2: 0.9}
            acc_after = {0: 0.85, 1: 0.6, 2: 0.88}
            class_names = ["A", "B", "C"]
            
            output_path = os.path.join(tmpdir, "per_class_acc")
            plot_per_class_accuracy(
                acc_before, acc_after, class_names, output_path,
                imbalance_classes=[1], formats=['png']
            )
            
            assert os.path.exists(f"{output_path}.png")


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for the complete pipeline."""
    
    @pytest.fixture
    def device(self):
        return "cuda" if torch.cuda.is_available() else "cpu"
    
    def test_model_training_loop(self, device):
        """Test a minimal training loop."""
        model = create_model("custom_cnn", num_classes=5, pretrained=False).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # Synthetic batch
        x = torch.randn(4, 3, 224, 224).to(device)
        y = torch.randint(0, 5, (4,)).to(device)
        
        # Training step
        model.train()
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        
        assert loss.item() > 0
        
    def test_activation_extraction_and_cav(self, device):
        """Test activation extraction followed by CAV training."""
        model = create_model("custom_cnn", num_classes=5, pretrained=False).to(device)
        model.eval()
        
        # Extract activations
        extractor = ActivationExtractor(model, ["conv3"])
        
        # Concept samples
        concept_x = torch.randn(10, 3, 224, 224).to(device)
        with torch.no_grad():
            _ = model(concept_x)
        concept_acts = extractor.get_activations()["conv3"]
        concept_acts_flat = concept_acts.view(concept_acts.size(0), -1).cpu().numpy()
        extractor.clear()
        
        # Random samples
        random_x = torch.randn(10, 3, 224, 224).to(device)
        with torch.no_grad():
            _ = model(random_x)
        random_acts = extractor.get_activations()["conv3"]
        random_acts_flat = random_acts.view(random_acts.size(0), -1).cpu().numpy()
        
        extractor.remove_hooks()
        
        # Train CAV
        trainer = CAVTrainer()
        cav, accuracy = trainer.train_cav(concept_acts_flat, random_acts_flat, "conv3")
        
        assert cav is not None
        assert accuracy >= 0.0 and accuracy <= 1.0


# ============================================================================
# Quick Sanity Check (run without pytest)
# ============================================================================

def run_sanity_checks():
    """Run quick sanity checks without pytest."""
    print("=" * 60)
    print("VL-CAV Framework Sanity Checks")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[1/8] Device: {device}")
    
    # Test 1: Config
    print("\n[2/8] Testing configuration...")
    config = ExperimentConfig(experiment_name="sanity_test")
    assert config.seed == 42
    print("  ✓ Configuration OK")
    
    # Test 2: Model creation
    print("\n[3/8] Testing model creation...")
    model = create_model("custom_cnn", num_classes=10, pretrained=False).to(device)
    x = torch.randn(2, 3, 224, 224).to(device)
    output = model(x)
    assert output.shape == (2, 10)
    print(f"  ✓ Model OK (output shape: {output.shape})")
    
    # Test 3: Activation extraction
    print("\n[4/8] Testing activation extraction...")
    extractor = ActivationExtractor(model, ["conv3"])
    _ = model(x)
    acts = extractor.get_activations()
    assert "conv3" in acts
    print(f"  ✓ Activation extraction OK (shape: {acts['conv3'].shape})")
    extractor.remove_hooks()
    
    # Test 4: CAV training
    print("\n[5/8] Testing CAV training...")
    trainer = CAVTrainer()
    concept_acts = np.random.randn(30, 128) + 1
    random_acts = np.random.randn(30, 128) - 1
    cav, acc = trainer.train_cav(concept_acts, random_acts, "test_layer")
    assert cav.shape == (128,)
    print(f"  ✓ CAV training OK (accuracy: {acc:.3f})")
    
    # Test 5: Dataset info
    print("\n[6/8] Testing dataset info...")
    info = get_dataset_info("CIFAR10")
    assert info["num_classes"] == 10
    print(f"  ✓ Dataset info OK (classes: {info['num_classes']})")
    
    # Test 6: Logger
    print("\n[7/8] Testing logger...")
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = ExperimentLogger(tmpdir, "sanity_test")
        logger.info("Test message")
        logger.log_metrics({"test_metric": 0.5})
        assert os.path.exists(os.path.join(tmpdir, "experiment.log"))
    print("  ✓ Logger OK")
    
    # Test 7: Visualizations
    print("\n[8/8] Testing visualizations...")
    with tempfile.TemporaryDirectory() as tmpdir:
        history = {'train_loss': [1.0, 0.8, 0.6], 'val_accuracy': [0.5, 0.6, 0.7]}
        plot_training_curves(history, os.path.join(tmpdir, "test"), formats=['png'])
        assert os.path.exists(os.path.join(tmpdir, "test.png"))
    print("  ✓ Visualizations OK")
    
    print("\n" + "=" * 60)
    print("All sanity checks passed! ✓")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--pytest":
        # Run with pytest
        pytest.main([__file__, "-v"])
    else:
        # Run quick sanity checks
        success = run_sanity_checks()
        sys.exit(0 if success else 1)
