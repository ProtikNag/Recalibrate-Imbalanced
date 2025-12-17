# TCAV-Based Recalibration for Imbalanced Datasets - Enhanced Version

A comprehensive framework for improving CNN model performance on imbalanced datasets using TCAV-based layer recalibration.

## New Features in This Version

### 1. Custom CNN Models (Untrained)
- **custom_cnn**: Medium-sized CNN with 5 conv blocks
- **custom_cnn_small**: Lightweight CNN for limited data
- **custom_cnn_large**: Deeper CNN for complex tasks

### 2. Comprehensive Result Recording
- Per-class accuracy, precision, recall, F1
- Confusion matrices before/after recalibration (with percentages in all cells)
- Misclassification analysis (which classes are confused)
- Confidence scores for predictions

### 3. Advanced Logging System
- Detailed experiment logs (`experiment.log`)
- JSON summaries (`experiment_summary.json`)
- Global history tracking (`all_experiments_history.csv`)
- Run comparison across experiments

### 4. Rich Visualizations
- Loss curves (total, classification, alignment)
- Confusion matrix heatmaps with difference plots (percentages in all cells)
- Per-class accuracy comparison charts
- Class distribution visualizations
- Misclassification analysis plots
- Summary dashboard

### 5. Configurable Imbalance Injection
- Apply any imbalance ratio (5%, 10%, 20%, 25%, etc.)
- Select which class to make imbalanced
- **Training and validation sets share the same imbalance distribution**

### 6. Background Folder for Random/Negative Samples
- Uses a dedicated `background` folder under the concept directory for random/negative samples in CAV training
- More consistent and controlled negative sample selection

### 7. Experiment 3: Joint Multi-Class Optimization
- Automatic layer selection based on TCAV scores
- Different concept for each class
- Ensures no layer is used for multiple classes
- Joint optimization of all alignment losses

## Installation

```bash
pip install torch torchvision numpy scikit-learn matplotlib seaborn pillow
```

## Dataset Structure

```
dataset/
├── zebra/
│   ├── image1.jpg
│   └── ...
├── horse/
│   └── ...
└── deer/
    └── ...

concept/
├── background/           # NEW: Required folder for random/negative samples
│   ├── random1.jpg
│   └── ...
├── zebra/
│   ├── stripes/         # Concept folder
│   │   └── stripe_images...
│   └── other_concept/
│       └── ...
├── horse/
│   ├── mane/
│   └── ...
└── deer/
    ├── antlers/
    └── ...
```

**Important**: The `background` folder is required under the concept directory. It should contain random/negative images that don't represent any specific concept.

## Quick Start

### List Available Layers

```bash
# For custom CNN
python list_layers.py --model_name custom_cnn

# For pretrained VGG16
python list_layers.py --model_name vgg16 --pretrained

# Detailed view
python list_layers.py --model_name resnet50 --detailed
```

### Run Experiments

**Experiment 1**: Train only with target class
```bash
python main_experiment.py \
    --experiment 1 \
    --model_name custom_cnn \
    --layer features.17 \
    --target_class zebra \
    --concept stripes \
    --dataset_path ./dataset \
    --concept_path ./concept \
    --imbalance_class zebra \
    --imbalance_ratio 0.1 \
    --pretrain_epochs 30 \
    --recalib_epochs 10
```

**Experiment 2**: Train with full dataset, selective alignment
```bash
python main_experiment.py \
    --experiment 2 \
    --model_name custom_cnn \
    --layer features.17 \
    --target_class zebra \
    --concept stripes \
    --dataset_path ./dataset \
    --concept_path ./concept \
    --imbalance_class zebra \
    --imbalance_ratio 0.1 \
    --pretrain_epochs 30 \
    --recalib_epochs 10
```

**Experiment 3**: Joint optimization with multiple concepts (automatic layer selection)
```bash
python main_experiment.py \
    --experiment 3 \
    --model_name custom_cnn \
    --dataset_path ./dataset \
    --concept_path ./concept \
    --class_concept_map "zebra:stripes,horse:mane,deer:antlers" \
    --imbalance_class zebra \
    --imbalance_ratio 0.1 \
    --pretrain_epochs 30 \
    --recalib_epochs 10
```

## Command Line Arguments

### Required Arguments
| Argument | Description |
|----------|-------------|
| `--experiment` | Experiment type: 1, 2, or 3 |
| `--dataset_path` | Path to dataset directory |
| `--concept_path` | Path to concept directory (must contain `background` folder) |

### Experiment-Specific Arguments
| Argument | Required For | Description |
|----------|--------------|-------------|
| `--target_class` | Exp 1 & 2 | Target class for alignment |
| `--concept` | Exp 1 & 2 | Concept folder name (e.g., stripes) |
| `--class_concept_map` | Exp 3 | Class-concept mappings (format: "class1:concept1,class2:concept2") |

### Model Configuration
| Argument | Default | Description |
|----------|---------|-------------|
| `--model_name` | custom_cnn | Model architecture |
| `--model_path` | None | Path to pre-trained weights |
| `--pretrained` | False | Use ImageNet pretrained weights |
| `--layer` | auto | Layer for recalibration (ignored for Exp 3) |

### Imbalance Configuration
| Argument | Default | Description |
|----------|---------|-------------|
| `--imbalance_class` | None | Class to make imbalanced |
| `--imbalance_ratio` | None | Ratio of data to keep (0.05-1.0) |

**Note**: Both training and validation sets will have the same imbalance distribution.

**Imbalance Ratio Examples:**
- `0.05` = Keep only 5% (extreme imbalance)
- `0.10` = Keep only 10% (severe imbalance)
- `0.20` = Keep only 20% (moderate imbalance)
- `0.25` = Keep only 25% (mild imbalance)
- `1.0` = Keep all data (balanced)

### Training Hyperparameters
| Argument | Default | Description |
|----------|---------|-------------|
| `--pretrain_epochs` | 30 | Initial training epochs |
| `--recalib_epochs` | 10 | Recalibration epochs |
| `--pretrain_lr` | 1e-3 | Initial training learning rate |
| `--recalib_lr` | 1e-4 | Recalibration learning rate |
| `--batch_size` | 16 | Batch size |
| `--lambda_align` | 0.5 | Alignment loss weight |

### Other Options
| Argument | Default | Description |
|----------|---------|-------------|
| `--classifier_type` | LinearSVC | CAV classifier |
| `--seed` | 42 | Random seed |
| `--results_path` | ./results | Output directory |

## Experiments Overview

### Experiment 1: Target Class Only
- Recalibrates using only data from the target class
- Both alignment and classification losses computed on target class
- Best for when you want to focus solely on improving one class

### Experiment 2: Full Dataset with Selective Alignment
- Uses full dataset for classification loss
- Alignment loss only for target class
- Better maintains overall performance while improving target class

### Experiment 3: Joint Multi-Class Optimization (NEW)
- Automatically selects the best layer for each class based on TCAV scores
- Different concept for each class
- **No layer is used for multiple classes** (prevents conflicts)
- Jointly optimizes all alignment losses
- Best for improving multiple classes simultaneously

**How Experiment 3 Layer Selection Works:**
1. Computes CAV and TCAV scores for all class-layer combinations
2. Prioritizes classes with lowest TCAV scores (most need for improvement)
3. Assigns each class the layer with the lowest TCAV score
4. Ensures no layer is assigned to multiple classes

## Supported Models

### Custom Models (Untrained)
| Model | Layers | Parameters | Suggested Layers |
|-------|--------|------------|------------------|
| custom_cnn | 5 blocks | ~2.5M | features.3, features.17, features.24 |
| custom_cnn_small | 4 blocks | ~100K | conv1.0, conv3.0, conv4.0 |
| custom_cnn_large | 5 blocks | ~15M | block1.0, block3.0, block5.0 |

### Pretrained Models
| Model | Suggested Layers |
|-------|-----------------|
| vgg16 | features.7, features.14, features.24 |
| resnet50 | layer2.3.conv3, layer4.2.conv3 |
| resnet18 | layer2.1.conv2, layer4.1.conv2 |
| inception_v3 | Mixed_6e.branch1x1.conv |
| mobilenet_v3_small | features.8.block.0.0 |
| mobilenet_v3_large | features.12.block.0.0 |

## Output Structure

```
results/
└── exp1_custom_cnn_zebra_20241216_123456/
    ├── config.json                    # Experiment configuration
    ├── experiment.log                 # Detailed text log
    ├── experiment_summary.json        # JSON summary
    ├── detailed_results.json          # All metrics and predictions
    │
    ├── initial_training_loss.png     # Pre-training loss curves
    ├── loss_curves.png               # Recalibration loss plots
    ├── loss_combined.png             # Combined loss plot
    ├── loss_per_class_align.png      # Per-class alignment (Exp 3)
    ├── confusion_matrices.png        # Before/after confusion
    ├── confusion_matrix_diff.png     # Change in confusion
    ├── per_class_comparison.png      # Per-class metrics
    ├── accuracy_change.png           # Accuracy improvements
    ├── metrics_comparison.png        # Overall metrics
    ├── class_distribution.png        # Training/val distribution
    ├── misclassification_analysis.png
    ├── misclassification_summary.png
    ├── experiment3_tcav.png          # Exp 3: TCAV comparison
    ├── experiment3_assignments.png   # Exp 3: Layer assignments
    ├── summary_dashboard.png         # Overview dashboard
    │
    ├── model_biased.pth              # Model after initial training
    └── model_recalibrated_exp*.pth   # Recalibrated model weights

results/
└── all_experiments_history.csv       # Global experiment history
```

## Understanding Results

### Key Metrics
- **TCAV Score**: How much the model relies on the concept (higher = better)
- **Accuracy**: Overall classification accuracy
- **Per-class metrics**: Individual class performance

### What to Look For
1. **TCAV improvement**: Target class TCAV should increase
2. **Accuracy maintenance**: Overall accuracy shouldn't drop significantly
3. **Per-class balance**: Check if minority class improves without hurting others
4. **Misclassification changes**: Fewer errors for target class

### Interpreting Visualizations

**Confusion Matrix**:
- Diagonal = correct predictions (shown as percentages)
- Off-diagonal = misclassifications (shown as percentages)
- Each cell shows: percentage% (raw count)
- Green difference = improvements

**Loss Curves**:
- Total loss should decrease
- Balance between cls and align loss matters

## Example Workflow

```bash
# 1. Check available layers
python list_layers.py --model_name custom_cnn --detailed

# 2. Run baseline (no imbalance) with Experiment 2
python main_experiment.py \
    --experiment 2 \
    --model_name custom_cnn \
    --layer features.17 \
    --target_class zebra \
    --concept stripes \
    --dataset_path ./dataset \
    --concept_path ./concept

# 3. Run with 10% imbalance
python main_experiment.py \
    --experiment 2 \
    --model_name custom_cnn \
    --layer features.17 \
    --target_class zebra \
    --concept stripes \
    --dataset_path ./dataset \
    --concept_path ./concept \
    --imbalance_class zebra \
    --imbalance_ratio 0.1

# 4. Run Experiment 3 with multiple classes
python main_experiment.py \
    --experiment 3 \
    --model_name custom_cnn \
    --dataset_path ./dataset \
    --concept_path ./concept \
    --class_concept_map "zebra:stripes,horse:mane,deer:antlers" \
    --imbalance_class zebra \
    --imbalance_ratio 0.1

# 5. Compare results in all_experiments_history.csv
```

## Tips

### Choosing Layers (for Experiments 1 & 2)
- **Deeper layers** (later in network) often have higher concept sensitivity
- **Mid-level layers** sometimes work best for visual concepts
- Use `list_layers.py` to explore options
- For Experiment 3, layers are automatically selected

### Tuning Lambda
- `lambda_align=0.3`: Prioritize classification
- `lambda_align=0.5`: Balanced (default)
- `lambda_align=0.7`: Prioritize concept alignment

### Handling Severe Imbalance
- Start with Experiment 2 (uses all data)
- Use Experiment 3 for multi-class optimization
- Consider data augmentation for minority class

### Setting Up the Background Folder
- Include diverse images that don't represent any specific concept
- Good sources: random textures, nature scenes, unrelated objects
- Aim for 50-200 images for stable CAV training

## Troubleshooting

**"Background folder not found" error**
```bash
# Create a background folder under your concept directory
mkdir -p ./concept/background
# Add random/negative images to this folder
```

**"Layer not found" error**
```bash
python list_layers.py --model_name YOUR_MODEL --detailed
```

**Low TCAV improvement**
- Try deeper layers (for Exp 1 & 2)
- Increase lambda_align
- Check concept image quality
- For Exp 3, the system automatically selects layers with lowest TCAV scores

**Accuracy drops significantly**
- Reduce lambda_align (try 0.3)
- Use Experiment 2 instead of 1
- Increase recalib_epochs

**Out of memory**
- Reduce batch_size
- Use custom_cnn_small
- Try CPU (slower but uses less memory)

**Validation accuracy differs from training**
- This is expected behavior - both sets now share the same imbalance
- Check `class_distribution.png` to verify the distribution

## Changelog

### Version 2.0
- Added Experiment 3: Joint multi-class optimization with automatic layer selection
- Changed random/negative samples source to dedicated `background` folder
- Fixed validation set to use same imbalance as training set
- Fixed confusion matrices to show percentages in all cells
- Added per-class alignment loss tracking for Experiment 3
- Added new visualizations for Experiment 3 results
- Updated documentation and examples

## License

Apache License 2.0
