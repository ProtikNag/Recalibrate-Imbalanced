# TCAV-Based Recalibration for Imbalanced Datasets - Enhanced Version

A comprehensive framework for improving CNN model performance on imbalanced datasets using TCAV-based layer recalibration.

## New Features in This Version

### 1. Custom CNN Models (Untrained)
- **custom_cnn**: Medium-sized CNN with 5 conv blocks
- **custom_cnn_small**: Lightweight CNN for limited data
- **custom_cnn_large**: Deeper CNN for complex tasks

### 2. Comprehensive Result Recording
- Per-class accuracy, precision, recall, F1
- Confusion matrices before/after recalibration
- Misclassification analysis (which classes are confused)
- Confidence scores for predictions

### 3. Advanced Logging System
- Detailed experiment logs (`experiment.log`)
- JSON summaries (`experiment_summary.json`)
- Global history tracking (`all_experiments_history.csv`)
- Run comparison across experiments

### 4. Rich Visualizations
- Loss curves (total, classification, alignment)
- Confusion matrix heatmaps with difference plots
- Per-class accuracy comparison charts
- Class distribution visualizations
- Misclassification analysis plots
- Summary dashboard

### 5. Configurable Imbalance Injection
- Apply any imbalance ratio (5%, 10%, 20%, 25%, etc.)
- Select which class to make imbalanced
- Maintains validation set balance for fair evaluation

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
├── zebra/
│   ├── concept1/
│   │   └── stripe_images...
│   └── concept2/
│       └── ...
├── horse/
│   ├── concept1/
│   └── concept2/
└── deer/
    ├── concept1/
    └── concept2/
```

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
    --concept concept1 \
    --dataset_path ./dataset \
    --concept_path ./concept \
    --imbalance_class zebra \
    --imbalance_ratio 0.1 \
    --epochs 20
```

**Experiment 2**: Train with full dataset, selective alignment
```bash
python main_experiment.py \
    --experiment 2 \
    --model_name custom_cnn \
    --layer features.17 \
    --target_class zebra \
    --concept concept1 \
    --dataset_path ./dataset \
    --concept_path ./concept \
    --imbalance_class zebra \
    --imbalance_ratio 0.1 \
    --epochs 20
```

## Command Line Arguments

### Required Arguments
| Argument | Description |
|----------|-------------|
| `--experiment` | Experiment type: 1 (target only) or 2 (full dataset) |
| `--dataset_path` | Path to dataset directory |
| `--concept_path` | Path to concept directory |
| `--target_class` | Target class for alignment |
| `--concept` | Concept folder name (e.g., concept1) |

### Model Configuration
| Argument | Default | Description |
|----------|---------|-------------|
| `--model_name` | custom_cnn | Model architecture |
| `--model_path` | None | Path to pre-trained weights |
| `--pretrained` | False | Use ImageNet pretrained weights |
| `--layer` | auto | Layer for recalibration |

### Imbalance Configuration
| Argument | Default | Description |
|----------|---------|-------------|
| `--imbalance_class` | None | Class to make imbalanced |
| `--imbalance_ratio` | None | Ratio of data to keep (0.05-1.0) |

**Imbalance Ratio Examples:**
- `0.05` = Keep only 5% (extreme imbalance)
- `0.10` = Keep only 10% (severe imbalance)
- `0.20` = Keep only 20% (moderate imbalance)
- `0.25` = Keep only 25% (mild imbalance)
- `1.0` = Keep all data (balanced)

### Training Hyperparameters
| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | 10 | Training epochs |
| `--batch_size` | 16 | Batch size |
| `--learning_rate` | 1e-4 | Learning rate |
| `--lambda_align` | 0.5 | Alignment loss weight |

### Other Options
| Argument | Default | Description |
|----------|---------|-------------|
| `--classifier_type` | LinearSVC | CAV classifier |
| `--seed` | 42 | Random seed |
| `--results_path` | ./results | Output directory |

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
    ├── loss_curves.png               # Individual loss plots
    ├── loss_combined.png             # Combined loss plot
    ├── confusion_matrices.png        # Before/after confusion
    ├── confusion_matrix_diff.png     # Change in confusion
    ├── per_class_comparison.png      # Per-class metrics
    ├── accuracy_change.png           # Accuracy improvements
    ├── metrics_comparison.png        # Overall metrics
    ├── class_distribution.png        # Training/val distribution
    ├── misclassification_analysis.png
    ├── misclassification_summary.png
    ├── summary_dashboard.png         # Overview dashboard
    │
    └── model_exp1.pth                # Trained model weights

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
- Diagonal = correct predictions
- Off-diagonal = misclassifications
- Green difference = improvements

**Loss Curves**:
- Total loss should decrease
- Balance between cls and align loss matters

## Example Workflow

```bash
# 1. Check available layers
python list_layers.py --model_name custom_cnn --detailed

# 2. Run baseline (no imbalance)
python main_experiment.py \
    --experiment 2 \
    --model_name custom_cnn \
    --layer features.17 \
    --target_class zebra \
    --concept concept1 \
    --dataset_path ./dataset \
    --concept_path ./concept

# 3. Run with 10% imbalance
python main_experiment.py \
    --experiment 2 \
    --model_name custom_cnn \
    --layer features.17 \
    --target_class zebra \
    --concept concept1 \
    --dataset_path ./dataset \
    --concept_path ./concept \
    --imbalance_class zebra \
    --imbalance_ratio 0.1

# 4. Compare results in all_experiments_history.csv
```

## Tips

### Choosing Layers
- **Deeper layers** (later in network) often have higher concept sensitivity
- **Mid-level layers** sometimes work best for visual concepts
- Use `list_layers.py` to explore options

### Tuning Lambda
- `lambda_align=0.3`: Prioritize classification
- `lambda_align=0.5`: Balanced (default)
- `lambda_align=0.7`: Prioritize concept alignment

### Handling Severe Imbalance
- Start with Experiment 2 (uses all data)
- Use weighted sampling if accuracy drops too much
- Consider data augmentation for minority class

## Troubleshooting

**"Layer not found" error**
```bash
python list_layers.py --model_name YOUR_MODEL --detailed
```

**Low TCAV improvement**
- Try deeper layers
- Increase lambda_align
- Check concept image quality

**Accuracy drops significantly**
- Reduce lambda_align (try 0.3)
- Use Experiment 2 instead of 1
- Increase epochs

**Out of memory**
- Reduce batch_size
- Use custom_cnn_small
- Try CPU (slower but uses less memory)

## License

Apache License 2.0
