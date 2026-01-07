# VL-CAV: VLM-Augmented Concept Activation Vectors

A framework for improving CNN interpretability and correcting class imbalance bias using Vision-Language Model (VLM) augmented Concept Activation Vectors.

## Overview

This project implements **Idea 1: VLM-Augmented CAVs** from our research on concept-guided neural network recalibration. The core innovation is leveraging rich semantic knowledge from Vision-Language Models (like CLIP) to construct robust concept representations, especially for underrepresented classes.

### Key Innovation

Traditional TCAV requires concept images, which are scarce for minority classes. VL-CAV solves this by:

1. **Textual Concept Generation**: Using LLM-style descriptions to define concepts
2. **Multimodal Fusion**: Combining text embeddings with limited visual examples
3. **Cross-Space Projection**: Mapping CNN activations to VLM embedding space
4. **Targeted Recalibration**: Fine-tuning only bottleneck layers for concept alignment

### Mathematical Foundation

The unified concept embedding combines text and vision:

```
s̄_k* = α · (1/m) Σⱼ T(tⱼ) + (1-α) · (1/|X_k*|) Σᵢ V(xᵢ)
```

Where:
- `T(·)` is the VLM text encoder
- `V(·)` is the VLM vision encoder  
- `α` controls text-vision balance (higher = more text, useful when visual examples are scarce)
- `tⱼ` are textual concept descriptions
- `xᵢ` are visual examples

## Installation

```bash
# Clone repository
git clone https://github.com/your-repo/vl-cav-recalibration.git
cd vl-cav-recalibration

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Or install manually
pip install torch torchvision numpy scikit-learn matplotlib seaborn tqdm transformers
```

## Project Structure

```
vl_cav_recalibration/
├── main_experiment.py      # Main experiment runner
├── config.py               # Configuration management
├── models.py               # CNN architectures & utilities
├── vlm_encoder.py          # CLIP encoder & concept embeddings
├── vl_cav.py               # VL-CAV core algorithms
├── dataloader.py           # Dataset loading with imbalance support
├── visualizations.py       # Publication-quality plots
├── logger.py               # Experiment tracking
├── debug_run.py            # Quick debug/test script
├── test_components.py      # Unit tests
├── requirements.txt        # Dependencies
├── .gitignore
├── README.md               # This file
├── data/                   # Dataset storage (auto-downloaded)
└── results/                # Experiment outputs
```

## Quick Start

### 1. Sanity Check

Verify all components work:

```bash
python test_components.py
```

### 2. Debug Run

Quick test of the full pipeline:

```bash
# Full debug (downloads data, trains briefly)
python debug_run.py --output-dir ./debug_output

# Fast mode (synthetic data, no training)
python debug_run.py --fast --output-dir ./debug_output
```

### 3. Run Full Experiment

```bash
python main_experiment.py \
    --experiment_name cifar10_imbalanced \
    --model_name resnet18 \
    --dataset_name CIFAR10 \
    --imbalance_classes "0,1,2" \
    --imbalance_ratio 0.1 \
    --alpha 0.7 \
    --pretrain_epochs 30 \
    --recalib_epochs 10
```

## Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--experiment_name` | str | vl_cav_experiment | Name for the experiment |
| `--seed` | int | 42 | Random seed for reproducibility |
| `--model_name` | str | resnet18 | CNN architecture |
| `--dataset_name` | str | CIFAR10 | Dataset to use |
| `--imbalance_classes` | str | "" | Comma-separated class indices to imbalance |
| `--imbalance_ratio` | float | 0.1 | Fraction of samples to keep (0.0-1.0) |
| `--alpha` | float | 0.7 | Text-vision balance (0=vision, 1=text) |
| `--vlm_model` | str | openai/clip-vit-base-patch32 | VLM model |
| `--pretrain_epochs` | int | 30 | Initial training epochs |
| `--recalib_epochs` | int | 10 | Recalibration epochs |
| `--lambda_cls` | float | 0.4 | Classification loss weight |
| `--lambda_align` | float | 0.6 | Alignment loss weight |
| `--batch_size` | int | 32 | Batch size |
| `--learning_rate` | float | 1e-3 | Initial learning rate |
| `--recalib_lr` | float | 1e-4 | Recalibration learning rate |
| `--device` | str | cuda | Device (cuda/cpu/mps) |
| `--output_dir` | str | ./results | Output directory |
| `--pretrained` | flag | False | Use ImageNet pretrained weights |
| `--no_save_models` | flag | False | Don't save model checkpoints |
| `--no_visualizations` | flag | False | Don't generate plots |

## Available Models

| Model | Description | Params |
|-------|-------------|--------|
| `resnet18` | ResNet-18 | 11M |
| `resnet34` | ResNet-34 | 21M |
| `resnet50` | ResNet-50 | 25M |
| `vgg16` | VGG-16 | 138M |
| `vgg19` | VGG-19 | 143M |
| `mobilenet_v3_small` | MobileNetV3-Small | 2.5M |
| `mobilenet_v3_large` | MobileNetV3-Large | 5.4M |
| `custom_cnn` | Custom 5-block CNN | ~5M |

## Available Datasets

| Dataset | Classes | Images | Size |
|---------|---------|--------|------|
| `CIFAR10` | 10 | 60,000 | 32×32 |
| `CIFAR100` | 100 | 60,000 | 32×32 |
| `STL10` | 10 | 13,000 | 96×96 |
| `FashionMNIST` | 10 | 70,000 | 28×28 |

## Example Workflows

### Study Effect of Imbalance Ratio

```bash
for ratio in 0.05 0.10 0.15 0.20 0.50; do
    python main_experiment.py \
        --experiment_name "imbalance_study_${ratio}" \
        --model_name resnet18 \
        --dataset_name CIFAR10 \
        --imbalance_classes "0,1" \
        --imbalance_ratio ${ratio} \
        --alpha 0.7 \
        --pretrain_epochs 30 \
        --recalib_epochs 10
done
```

### Study Effect of Alpha (Text-Vision Balance)

```bash
for alpha in 0.3 0.5 0.7 0.9; do
    python main_experiment.py \
        --experiment_name "alpha_study_${alpha}" \
        --model_name resnet18 \
        --dataset_name CIFAR10 \
        --imbalance_classes "0" \
        --imbalance_ratio 0.1 \
        --alpha ${alpha} \
        --pretrain_epochs 30 \
        --recalib_epochs 10
done
```

### Compare Model Architectures

```bash
for model in resnet18 resnet50 vgg16 mobilenet_v3_small custom_cnn; do
    python main_experiment.py \
        --experiment_name "model_study_${model}" \
        --model_name ${model} \
        --dataset_name CIFAR10 \
        --imbalance_classes "0,1,2" \
        --imbalance_ratio 0.1 \
        --alpha 0.7 \
        --pretrain_epochs 30 \
        --recalib_epochs 10
done
```

### Use Pretrained Weights

```bash
python main_experiment.py \
    --experiment_name pretrained_exp \
    --model_name resnet18 \
    --pretrained \
    --dataset_name CIFAR10 \
    --imbalance_classes "0" \
    --imbalance_ratio 0.1 \
    --pretrain_epochs 10 \
    --recalib_epochs 5
```

### Different VLM Models

```bash
# Standard CLIP
python main_experiment.py \
    --vlm_model openai/clip-vit-base-patch32 \
    --experiment_name vlm_base

# Larger CLIP
python main_experiment.py \
    --vlm_model openai/clip-vit-large-patch14 \
    --experiment_name vlm_large
```

## Output Structure

```
results/vl_cav_experiment_20260107_143022/
├── config.json                    # Experiment configuration
├── experiment.log                 # Detailed text log
├── results.json                   # Complete results
├── model_biased.pth               # Model before recalibration
├── model_recalibrated.pth         # Model after recalibration
├── class_distribution.png/svg     # Class distribution plot
├── training_curves.png/svg        # Initial training loss/accuracy
├── recalibration_losses.png/svg   # Recalibration loss components
├── confusion_matrices.png/svg     # Before/after confusion matrices
├── per_class_accuracy.png/svg     # Per-class accuracy comparison
├── tcav_scores.png/svg            # TCAV score comparison
└── experiment_summary.png/svg     # Dashboard summary
```

## Testing

### Quick Sanity Check

```bash
python test_components.py
```

### Full Test Suite (pytest)

```bash
# Run all tests
pytest test_components.py -v

# Run specific test class
pytest test_components.py -v -k "TestModels"

# Skip slow tests (data downloads)
pytest test_components.py -v -m "not slow"

# With coverage
pip install pytest-cov
pytest test_components.py -v --cov=. --cov-report=html
```

### Debug Pipeline

```bash
# Test all components without full training
python debug_run.py --fast

# Test with real data but minimal training
python debug_run.py --skip-vlm

# Full debug run
python debug_run.py --output-dir ./debug_test
```

## Key Hyperparameters

| Parameter | Range | Effect |
|-----------|-------|--------|
| `alpha` | 0.0-1.0 | Higher = rely more on text descriptions |
| `lambda_cls` | 0.0-1.0 | Classification loss weight |
| `lambda_align` | 0.0-1.0 | Concept alignment loss weight |
| `imbalance_ratio` | 0.01-1.0 | Fraction of minority class samples |
| `recalib_lr` | 1e-5-1e-3 | Recalibration learning rate |

**Recommendations:**
- For severe imbalance (ratio < 0.1): Use `alpha=0.8-0.9` to leverage text descriptions
- For moderate imbalance (ratio 0.1-0.3): Use `alpha=0.6-0.7`
- Balance `lambda_cls` and `lambda_align` to prevent accuracy degradation

## Visualization Gallery

The framework generates publication-quality visualizations:

1. **Class Distribution**: Shows sample counts per class, highlighting imbalanced ones
2. **Training Curves**: Loss and accuracy during initial training
3. **Recalibration Losses**: Separate plots for classification and alignment losses
4. **Confusion Matrices**: Before/after comparison with difference highlighting
5. **Per-Class Accuracy**: Bar chart comparison showing improvements
6. **TCAV Scores**: Concept alignment before/after recalibration
7. **Experiment Summary**: Dashboard combining key metrics

All plots are saved in both PNG (for documents) and SVG (for papers) formats.

## Troubleshooting

### CUDA Out of Memory

```bash
# Reduce batch size
--batch_size 16

# Use smaller model
--model_name mobilenet_v3_small

# Use CPU
--device cpu
```

### VLM/CLIP Issues

```bash
# Skip VLM (uses random embeddings for testing)
python debug_run.py --skip-vlm

# Or install transformers
pip install transformers
```

### Slow Data Loading

```bash
# Reduce workers
--num_workers 2

# Or use synthetic data for debugging
python debug_run.py --skip-data
```

## Citation

If you use this code, please cite:

```bibtex
@article{vlcav2026,
  title={VLM-Augmented Concept Activation Vectors for Neural Network Recalibration},
  author={...},
  journal={...},
  year={2026}
}
```

## Related Work

- Kim et al. (2018). "Interpretability Beyond Feature Attribution: Quantitative Testing with Concept Activation Vectors (TCAV)"
- Radford et al. (2021). "Learning Transferable Visual Models From Natural Language Supervision (CLIP)"
- Srikanth et al. (2025). "Targeted Layer Recalibration in CNNs"

## License

MIT License - see LICENSE file for details.
