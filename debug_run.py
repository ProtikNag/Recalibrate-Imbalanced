#!/usr/bin/env python3
"""
Debug Runner for TCAV-Based Recalibration

This script runs a minimal version of the experiment to verify everything works:
- Uses only 2-3 classes
- Limits images per class
- Runs only a few epochs
- Skips concept generation (uses synthetic data)

Usage:
    python debug_run.py                    # Full debug run
    python debug_run.py --skip-training    # Skip training, test data loading only
    python debug_run.py --test-viz         # Test visualizations with dummy data
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def print_header(title: str):
    """Print a formatted header."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


def print_success(msg: str):
    """Print success message."""
    print(f"  ✓ {msg}")


def print_fail(msg: str):
    """Print failure message."""
    print(f"  ✗ {msg}")


def print_info(msg: str):
    """Print info message."""
    print(f"  → {msg}")


def test_imports():
    """Test that all required modules can be imported."""
    print_header("Testing Imports")
    
    modules = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("numpy", "NumPy"),
        ("sklearn", "Scikit-learn"),
        ("matplotlib", "Matplotlib"),
        ("seaborn", "Seaborn"),
        ("PIL", "Pillow"),
        ("tqdm", "tqdm"),
    ]
    
    all_ok = True
    for module_name, display_name in modules:
        try:
            __import__(module_name)
            print_success(f"{display_name} imported successfully")
        except ImportError as e:
            print_fail(f"{display_name} import failed: {e}")
            all_ok = False
    
    # Test local modules
    local_modules = [
        "utils",
        "dataloader_caltech",
        "visualizations",
        "logger_system",
    ]
    
    for module_name in local_modules:
        try:
            __import__(module_name)
            print_success(f"Local module '{module_name}' imported successfully")
        except ImportError as e:
            print_fail(f"Local module '{module_name}' import failed: {e}")
            all_ok = False
    
    return all_ok


def test_models():
    """Test model loading and forward pass."""
    print_header("Testing Models")
    
    from utils import load_model
    
    models_to_test = [
        ("custom_cnn", 5),
        ("custom_cnn_small", 3),
        ("custom_cnn_large", 4),
    ]
    
    all_ok = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print_info(f"Using device: {device}")
    
    for model_name, num_classes in models_to_test:
        try:
            model = load_model(model_name, num_classes=num_classes)
            model = model.to(device)
            model.eval()
            
            # Test forward pass
            dummy_input = torch.randn(2, 3, 224, 224).to(device)
            with torch.no_grad():
                output = model(dummy_input)
            
            if isinstance(output, tuple):
                output = output[0]
            
            assert output.shape == (2, num_classes), f"Expected (2, {num_classes}), got {output.shape}"
            print_success(f"{model_name} with {num_classes} classes: output shape {output.shape}")
            
        except Exception as e:
            print_fail(f"{model_name}: {e}")
            all_ok = False
    
    return all_ok


def test_dataloader():
    """Test Caltech-101 dataloader (downloads if needed)."""
    print_header("Testing Dataloader")
    
    from dataloader_caltech import (
        Caltech101VehicleDataset,
        create_caltech_vehicle_dataset,
        get_transforms,
        print_dataset_info,
    )
    
    try:
        # Test with 2 classes for speed
        class_names = ['airplanes', 'Motorbikes']
        
        print_info(f"Testing with classes: {class_names}")
        print_info("This may download Caltech-101 (~130MB) on first run...")
        
        train_ds, val_ds = create_caltech_vehicle_dataset(
            root='./data',
            class_names=class_names,
            imbalance_class='airplanes',
            imbalance_ratio=0.5,
            download=True
        )
        
        print_success(f"Train dataset created: {len(train_ds)} samples")
        print_success(f"Val dataset created: {len(val_ds)} samples")
        
        # Print distribution
        print_info("Class distribution:")
        for cls, count in train_ds.get_class_counts().items():
            print(f"      {cls}: {count}")
        
        # Test loading a sample
        img, label = train_ds[0]
        print_success(f"Sample loaded: image shape {img.shape}, label {label}")
        
        # Test DataLoader
        from torch.utils.data import DataLoader
        loader = DataLoader(train_ds, batch_size=4, shuffle=True)
        batch_imgs, batch_labels = next(iter(loader))
        print_success(f"Batch loaded: {batch_imgs.shape}, labels {batch_labels.tolist()}")
        
        return True
        
    except Exception as e:
        print_fail(f"Dataloader test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_visualizations():
    """Test visualization functions with dummy data."""
    print_header("Testing Visualizations")
    
    from visualizations import ResultVisualizer
    import tempfile
    import shutil
    
    # Create temp directory for outputs
    temp_dir = tempfile.mkdtemp(prefix="tcav_viz_test_")
    print_info(f"Temp directory: {temp_dir}")
    
    try:
        viz = ResultVisualizer(temp_dir)
        
        # Test with 4 classes
        class_names = ['ClassA', 'ClassB', 'ClassC', 'ClassD']
        n_classes = len(class_names)
        
        # Dummy loss history
        loss_history = {
            'total': [1.0, 0.8, 0.6, 0.5, 0.4],
            'cls': [0.8, 0.6, 0.5, 0.4, 0.3],
            'align': [0.2, 0.2, 0.1, 0.1, 0.1],
            'per_class_align': {c: [0.3 - i*0.05 for i in range(5)] for c in class_names}
        }
        
        viz.plot_loss_curves(loss_history, epochs=5)
        print_success("Loss curves generated")
        
        # Dummy confusion matrices
        cm_before = np.array([
            [80, 10, 5, 5],
            [15, 70, 10, 5],
            [5, 10, 75, 10],
            [10, 5, 5, 80]
        ])
        cm_after = np.array([
            [85, 8, 4, 3],
            [10, 78, 8, 4],
            [4, 8, 80, 8],
            [8, 4, 4, 84]
        ])
        
        viz.plot_confusion_matrices(cm_before.tolist(), cm_after.tolist(), class_names)
        print_success("Confusion matrices generated")
        
        # Dummy per-class results
        per_class_before = {
            c: {'accuracy': 0.7 + np.random.rand()*0.1, 
                'precision': 0.7 + np.random.rand()*0.1,
                'recall': 0.7 + np.random.rand()*0.1,
                'f1': 0.7 + np.random.rand()*0.1}
            for c in class_names
        }
        per_class_after = {
            c: {'accuracy': 0.8 + np.random.rand()*0.1,
                'precision': 0.8 + np.random.rand()*0.1,
                'recall': 0.8 + np.random.rand()*0.1,
                'f1': 0.8 + np.random.rand()*0.1}
            for c in class_names
        }
        
        viz.plot_per_class_comparison(per_class_before, per_class_after, class_names)
        print_success("Per-class comparison generated")
        
        viz.plot_accuracy_change(per_class_before, per_class_after, class_names)
        print_success("Accuracy change plot generated")
        
        # Dummy overall results
        results_before = {
            'overall': {'accuracy': 0.75, 'precision': 0.74, 'recall': 0.73, 'f1': 0.74, 'avg_confidence': 0.8}
        }
        results_after = {
            'overall': {'accuracy': 0.85, 'precision': 0.84, 'recall': 0.83, 'f1': 0.84, 'avg_confidence': 0.9}
        }
        
        viz.plot_metrics_comparison(results_before, results_after, 0.6, 0.8)
        print_success("Metrics comparison generated")
        
        # Class distribution
        train_counts = {c: 100 + i*20 for i, c in enumerate(class_names)}
        val_counts = {c: 25 + i*5 for i, c in enumerate(class_names)}
        viz.plot_class_distribution(train_counts, val_counts)
        print_success("Class distribution generated")
        
        # Experiment 3 TCAV comparison
        tcav_before = {c: 0.5 + np.random.rand()*0.2 for c in class_names}
        tcav_after = {c: 0.7 + np.random.rand()*0.2 for c in class_names}
        class_layer_map = {c: f'layer{i}' for i, c in enumerate(class_names)}
        
        viz.plot_experiment3_tcav_comparison(tcav_before, tcav_after, class_layer_map)
        print_success("Experiment 3 TCAV comparison generated")
        
        # Check files were created
        files = os.listdir(temp_dir)
        png_files = [f for f in files if f.endswith('.png')]
        svg_files = [f for f in files if f.endswith('.svg')]
        
        print_info(f"Generated {len(png_files)} PNG files and {len(svg_files)} SVG files")
        
        return True
        
    except Exception as e:
        print_fail(f"Visualization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_cav_training():
    """Test CAV training with synthetic data."""
    print_header("Testing CAV Training")
    
    from utils import train_cav
    
    try:
        # Create synthetic activations
        np.random.seed(42)
        
        # Concept activations (cluster 1)
        concept_acts = np.random.randn(50, 256) + np.array([1.0] * 256)
        
        # Random activations (cluster 2)
        random_acts = np.random.randn(50, 256) - np.array([1.0] * 256)
        
        # Train CAV
        cav = train_cav(concept_acts, random_acts, classifier_type='LinearSVC')
        
        assert cav.shape == (256,), f"Expected CAV shape (256,), got {cav.shape}"
        assert np.abs(np.linalg.norm(cav) - 1.0) < 0.01, "CAV should be normalized"
        
        print_success(f"CAV trained successfully: shape {cav.shape}, norm {np.linalg.norm(cav):.4f}")
        
        # Test with different classifiers
        for clf_type in ['SGDClassifier', 'LogisticRegression']:
            cav = train_cav(concept_acts, random_acts, classifier_type=clf_type)
            print_success(f"CAV with {clf_type}: shape {cav.shape}")
        
        return True
        
    except Exception as e:
        print_fail(f"CAV training test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_evaluation():
    """Test evaluation functions with a small model."""
    print_header("Testing Evaluation")
    
    from utils import load_model, evaluate_detailed
    from torch.utils.data import TensorDataset, DataLoader
    
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Create small model
        model = load_model('custom_cnn_small', num_classes=3)
        model = model.to(device)
        model.eval()
        
        # Create synthetic dataset
        n_samples = 30
        images = torch.randn(n_samples, 3, 224, 224)
        labels = torch.randint(0, 3, (n_samples,))
        
        dataset = TensorDataset(images, labels)
        loader = DataLoader(dataset, batch_size=8)
        
        class_names = ['cat', 'dog', 'bird']
        
        # Run evaluation
        results = evaluate_detailed(model, loader, class_names, device)
        
        # Check results structure
        assert 'overall' in results
        assert 'per_class' in results
        assert 'confusion_matrix' in results
        
        print_success(f"Overall accuracy: {results['overall']['accuracy']:.4f}")
        print_success(f"Per-class results: {list(results['per_class'].keys())}")
        print_success(f"Confusion matrix shape: {len(results['confusion_matrix'])}x{len(results['confusion_matrix'][0])}")
        
        return True
        
    except Exception as e:
        print_fail(f"Evaluation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mini_training():
    """Test a minimal training loop."""
    print_header("Testing Mini Training Loop")
    
    from utils import load_model
    from torch.utils.data import TensorDataset, DataLoader
    import torch.optim as optim
    
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print_info(f"Using device: {device}")
        
        # Create small model
        model = load_model('custom_cnn_small', num_classes=3)
        model = model.to(device)
        
        # Create synthetic dataset
        n_samples = 32
        images = torch.randn(n_samples, 3, 224, 224)
        labels = torch.randint(0, 3, (n_samples,))
        
        dataset = TensorDataset(images, labels)
        loader = DataLoader(dataset, batch_size=8)
        
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        
        # Train for 2 epochs
        model.train()
        for epoch in range(2):
            total_loss = 0
            for imgs, lbls in loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                
                optimizer.zero_grad()
                outputs = model(imgs)
                loss = criterion(outputs, lbls)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(loader)
            print_info(f"Epoch {epoch+1}: loss = {avg_loss:.4f}")
        
        print_success("Mini training loop completed successfully")
        return True
        
    except Exception as e:
        print_fail(f"Mini training test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_pipeline_mini():
    """Test the full pipeline with minimal settings."""
    print_header("Testing Full Pipeline (Mini)")
    
    print_info("This test requires Caltech-101 data and may take a few minutes...")
    
    try:
        # Set minimal configuration
        import subprocess
        
        cmd = [
            sys.executable, 'main_experiment.py',
            '--experiment', '3',
            '--model_name', 'custom_cnn_small',
            '--dataset_path', './data',
            '--concept_path', './concepts_debug',
            '--class_concept_map', 'airplanes:subject,Motorbikes:subject',
            '--imbalance_class', 'airplanes',
            '--imbalance_ratio', '0.5',
            '--pretrain_epochs', '2',
            '--recalib_epochs', '2',
            '--batch_size', '8',
            '--results_path', './results_debug',
        ]
        
        print_info(f"Running: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1200  # 10 minute timeout
        )
        
        if result.returncode == 0:
            print_success("Full pipeline completed successfully!")
            print_info("Check ./results_debug/ for outputs")
            return True
        else:
            print_fail(f"Pipeline failed with return code {result.returncode}")
            print("STDOUT:", result.stdout[-1000:] if len(result.stdout) > 1000 else result.stdout)
            print("STDERR:", result.stderr[-1000:] if len(result.stderr) > 1000 else result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print_fail("Pipeline timed out (>20 minutes)")
        return False
    except Exception as e:
        print_fail(f"Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests(skip_training: bool = False, test_viz_only: bool = False, 
                  full_pipeline: bool = False):
    """Run all tests."""
    print_header("TCAV Recalibration - Debug Test Suite")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python: {sys.version}")
    
    results = {}
    
    if test_viz_only:
        # Only test visualizations
        results['imports'] = test_imports()
        results['visualizations'] = test_visualizations()
    else:
        # Run component tests
        results['imports'] = test_imports()
        
        if not results['imports']:
            print_fail("\nImport test failed. Cannot continue.")
            return results
        
        results['models'] = test_models()
        results['cav_training'] = test_cav_training()
        results['evaluation'] = test_evaluation()
        results['visualizations'] = test_visualizations()
        
        if not skip_training:
            results['dataloader'] = test_dataloader()
            results['mini_training'] = test_mini_training()
            
            if full_pipeline:
                results['full_pipeline'] = test_full_pipeline_mini()
    
    # Summary
    print_header("Test Summary")
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False
    
    print()
    if all_passed:
        print("  All tests passed! ✓")
    else:
        print("  Some tests failed. Please check the output above.")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Debug runner for TCAV-Based Recalibration"
    )
    parser.add_argument("--skip-training", action="store_true",
                       help="Skip tests that require training/data loading")
    parser.add_argument("--test-viz", action="store_true",
                       help="Only test visualizations")
    parser.add_argument("--full-pipeline", action="store_true",
                       help="Run full pipeline test (slow)")
    
    args = parser.parse_args()
    
    results = run_all_tests(
        skip_training=args.skip_training,
        test_viz_only=args.test_viz,
        full_pipeline=args.full_pipeline
    )
    
    # Exit with appropriate code
    all_passed = all(results.values())
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
