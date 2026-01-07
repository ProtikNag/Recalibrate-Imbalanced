# TCAV-Based Recalibration for Imbalanced Image Classification

A framework for improving CNN interpretability and correcting class imbalance bias using Testing with Concept Activation Vectors (TCAV).

## Overview

This project implements a concept-based recalibration framework that:

1. **Creates biased models**: Trains CNNs on imbalanced datasets to simulate real-world bias
2. **Identifies bottleneck layers**: Automatically selects optimal layers for concept alignment
3. **Applies TCAV-based recalibration**: Fine-tunes specific layers to align activations with human-defined concepts
4. **Evaluates improvements**: Measures changes in accuracy, TCAV scores, and interpretability

### Key Features

- **Caltech-101 Dataset**: Uses vehicle classes from Caltech-101 with automatic download
- **Automatic Concept Extraction**: Uses DeepLabV3 segmentation to extract subject/background
- **N-Class Support**: Works with any number of classes (not limited to 3)
- **Dynamic Layer Selection**: Automatically selects best layers based on TCAV scores
- **Comprehensive Visualizations**: Generates PNG and SVG outputs for all metrics

## Installation

```bash
# Clone or create project directory
mkdir tcav_recalibration && cd tcav_recalibration

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install torch torchvision numpy scikit-learn matplotlib seaborn tqdm pillow
```

## Project Structure

```
tcav_recalibration/
├── main_experiment.py      # Main experiment runner
├── concept_generator.py    # DeepLabV3-based concept extraction
├── dataloader_caltech.py   # Caltech-101 data loading with imbalance support
├── utils.py                # Models, CAV training, evaluation utilities
├── visualizations.py       # Result visualization (PNG + SVG)
├── logger_system.py        # Comprehensive logging
├── gpu.sh                  # SLURM batch script
├── README.md               # This file
├── data/                   # Caltech-101 data (auto-downloaded)
├── concepts/               # Generated concept images
└── results/                # Experiment outputs
```

## Quick Start

### 1. Generate Concepts (Optional - Auto-generated on first run)

```bash
python concept_generator.py \
    --dataset_path ./data/caltech101/101_ObjectCategories \
    --output_path ./concepts \
    --classes "airplanes,Motorbikes,car_side,ferry,helicopter"
```

### 2. Run Experiment

```bash
python main_experiment.py \
    --experiment 3 \
    --model_name custom_cnn \
    --dataset_path ./data \
    --concept_path ./concepts \
    --class_concept_map "airplanes:subject,Motorbikes:subject,car_side:subject,ferry:subject,helicopter:subject" \
    --imbalance_class airplanes \
    --imbalance_ratio 0.1 \
    --pretrain_epochs 50 \
    --recalib_epochs 20
```

### 3. Run on SLURM Cluster

```bash
sbatch gpu.sh
```

## Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--experiment` | int | 3 | Experiment type (1, 2, or 3) |
| `--model_name` | str | custom_cnn | CNN architecture |
| `--dataset_path` | str | ./data | Root directory for Caltech-101 |
| `--concept_path` | str | ./concepts | Path to concept images |
| `--class_concept_map` | str | required | Class:concept pairs (comma-separated) |
| `--imbalance_class` | str | None | Class to make imbalanced |
| `--imbalance_ratio` | float | None | Ratio of data to keep (0.05-1.0) |
| `--pretrain_epochs` | int | 30 | Epochs for initial training |
| `--recalib_epochs` | int | 10 | Epochs for recalibration |
| `--lambda_align` | float | 0.5 | Weight for alignment loss |
| `--batch_size` | int | 32 | Training batch size |
| `--pretrained` | flag | False | Use ImageNet pretrained weights |
| `--seed` | int | 42 | Random seed |

## Available Models

| Model | Description | Parameters |
|-------|-------------|------------|
| `custom_cnn` | Medium custom CNN | ~5M |
| `custom_cnn_small` | Small custom CNN | ~1M |
| `custom_cnn_large` | Large custom CNN | ~15M |
| `vgg16` | VGG-16 | 138M |
| `resnet18` | ResNet-18 | 11M |
| `resnet50` | ResNet-50 | 25M |
| `inception_v3` | Inception-V3 | 27M |
| `mobilenet_v3_small` | MobileNet-V3 Small | 2.5M |
| `mobilenet_v3_large` | MobileNet-V3 Large | 5.4M |

## Available Caltech-101 Vehicle Classes

| Class Name | Images | Notes |
|------------|--------|-------|
| `airplanes` | ~800 | Commercial aircraft |
| `Motorbikes` | ~798 | Motorcycles |
| `car_side` | ~123 | Side view of cars |
| `ferry` | ~67 | Ferry boats |
| `helicopter` | ~88 | Helicopters |
| `schooner` | ~63 | Sailing ships |

## Experiment Types

### Experiment 1: Single-Class Recalibration
- Focuses on one target class
- Uses data only from that class for recalibration
- Best for debugging and understanding single-class behavior

### Experiment 2: Selective Alignment
- Uses full dataset for classification loss
- Applies alignment loss only to target class
- Balances global performance with targeted improvement

### Experiment 3: Joint Multi-Class Optimization (Recommended)
- Automatically selects best layer for each class
- Ensures no layer is reused across classes
- Joint optimization of all alignment losses
- Best overall results for imbalanced datasets

## Output Structure

```
results/exp3_custom_cnn_5classes_YYYYMMDD_HHMMSS/
├── experiment.log              # Detailed text log
├── experiment_summary.json     # JSON summary
├── config.json                 # Run configuration
├── detailed_results.json       # Complete results
├── model_biased.pth           # Model before recalibration
├── model_recalibrated_exp3.pth # Model after recalibration
├── initial_training_loss.png/svg
├── loss_curves.png/svg
├── loss_combined.png/svg
├── loss_per_class_align.png/svg
├── confusion_matrices.png/svg
├── confusion_matrix_diff.png/svg
├── per_class_comparison.png/svg
├── accuracy_change.png/svg
├── metrics_comparison.png/svg
├── class_distribution.png/svg
├── misclassification_analysis.png/svg
├── misclassification_summary.png/svg
├── experiment3_tcav.png/svg
├── experiment3_assignments.png/svg
└── summary_dashboard.png/svg
```

## Example Workflows

### Study Effect of Imbalance Ratio

```bash
for ratio in 0.05 0.10 0.15 0.20 0.25 0.50; do
    python main_experiment.py \
        --experiment 3 \
        --model_name custom_cnn \
        --dataset_path ./data \
        --concept_path ./concepts \
        --class_concept_map "airplanes:subject,Motorbikes:subject,car_side:subject" \
        --imbalance_class airplanes \
        --imbalance_ratio ${ratio} \
        --pretrain_epochs 50 \
        --recalib_epochs 20
done
```

### Compare Model Architectures

```bash
for model in custom_cnn vgg16 resnet18 mobilenet_v3_small; do
    python main_experiment.py \
        --experiment 3 \
        --model_name ${model} \
        --dataset_path ./data \
        --concept_path ./concepts \
        --class_concept_map "airplanes:subject,Motorbikes:subject,car_side:subject" \
        --imbalance_class airplanes \
        --imbalance_ratio 0.1 \
        --pretrain_epochs 50 \
        --recalib_epochs 20
done
```

### Use Pretrained Weights

```bash
python main_experiment.py \
    --experiment 3 \
    --model_name vgg16 \
    --pretrained \
    --dataset_path ./data \
    --concept_path ./concepts \
    --class_concept_map "airplanes:subject,Motorbikes:subject,car_side:subject" \
    --imbalance_class airplanes \
    --imbalance_ratio 0.1 \
    --pretrain_epochs 10 \
    --recalib_epochs 5
```

## Concept Generation

The framework automatically generates concept images using DeepLabV3 segmentation on first run. The segmentation model:

1. **Identifies subjects**: Extracts the main object (vehicle) from each image
2. **Creates subject crops**: Saves cropped subject regions as concept images
3. **Extracts backgrounds**: Saves background regions as random/negative samples

Manual concept generation:

```bash
python concept_generator.py \
    --dataset_path ./data/caltech101/101_ObjectCategories \
    --output_path ./concepts \
    --classes "airplanes,Motorbikes,car_side,ferry,helicopter" \
    --max_images 200 \
    --device cuda
```

## Results Interpretation

### TCAV Score
- Measures how sensitive model predictions are to human-defined concepts
- Range: 0-1 (higher = more aligned with concept)
- Target: Increase after recalibration

### Accuracy Change
- Per-class accuracy before vs. after recalibration
- Positive change indicates improvement
- Watch for trade-offs between classes

### Confusion Matrix
- Shows classification patterns before/after
- Look for reduced confusion between similar classes
- Difference matrix highlights improvements

## Testing & Debugging

### Quick Debug Run

Test that all components work without running a full experiment:

```bash
# Test imports, models, CAV, visualizations (no data download)
python debug_run.py --skip-training

# Test everything including data loading
python debug_run.py

# Only test visualizations
python debug_run.py --test-viz

# Full pipeline test (slow, runs mini experiment)
python debug_run.py --full-pipeline
```

### Unit Tests (pytest)

```bash
# Install pytest
pip install pytest

# Run all tests
pytest test_components.py -v

# Run specific test class
pytest test_components.py -v -k "TestModels"

# Run with coverage
pip install pytest-cov
pytest test_components.py -v --cov=. --cov-report=html

# Skip slow tests (data download)
pytest test_components.py -v -m "not slow"
```

### Test Categories

| Test Class | Tests |
|------------|-------|
| `TestModels` | Model creation, forward pass, gradients |
| `TestCAV` | CAV training with different classifiers |
| `TestEvaluation` | Metric computation, confusion matrix |
| `TestVisualizations` | Plot generation, N-class support |
| `TestDataloader` | Data loading, transforms |
| `TestLogger` | Logging system functionality |
| `TestIntegration` | End-to-end component tests |

## Troubleshooting

### CUDA Out of Memory
```bash
# Reduce batch size
--batch_size 16

# Use smaller model
--model_name custom_cnn_small
```

### No Concept Images Found
```bash
# Manually generate concepts first
python concept_generator.py \
    --dataset_path ./data/caltech101/101_ObjectCategories \
    --output_path ./concepts \
    --classes "your,class,names"
```

### Class Not Found in Caltech-101
Check available classes:
```python
from torchvision.datasets import Caltech101
ds = Caltech101(root='./data', download=True)
print(ds.categories)
```

## Citation

If you use this code, please cite:

```bibtex
@article{tcav_recalibration,
  title={Targeted Layer Recalibration in CNNs: Enhancing Concept Alignment},
  author={...},
  journal={...},
  year={2024}
}
```

## License

MIT License - see LICENSE file for details.
