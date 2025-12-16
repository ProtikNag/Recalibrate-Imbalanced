#!/usr/bin/env python3
"""
Helper script to list available layers for different CNN models.

Supports:
- Custom CNN models (untrained)
- Pretrained models (VGG16, ResNet, etc.)

Usage:
    python list_layers.py --model_name custom_cnn
    python list_layers.py --model_name vgg16 --filter conv
"""

import argparse
import torch.nn as nn
from utils_imbalanced import (
    load_model, 
    get_model_layers, 
    get_suggested_layers,
    CustomCNN,
    CustomCNNSmall,
    CustomCNNLarge
)


def list_all_modules(model, prefix='', depth=0, max_depth=4):
    """Recursively list all modules in the model."""
    modules = []
    
    for name, module in model.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        module_type = type(module).__name__
        
        # Count parameters
        num_params = sum(p.numel() for p in module.parameters(recurse=False))
        
        modules.append({
            'name': full_name,
            'type': module_type,
            'depth': depth,
            'num_params': num_params,
            'has_params': num_params > 0
        })
        
        if depth < max_depth:
            modules.extend(list_all_modules(module, full_name, depth + 1, max_depth))
    
    return modules


def print_model_summary(model, model_name):
    """Print model architecture summary."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n{'='*70}")
    print(f"MODEL SUMMARY: {model_name}")
    print(f"{'='*70}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(description="List available layers for CNN models")
    parser.add_argument("--model_name", type=str, default="custom_cnn",
                       choices=["custom_cnn", "custom_cnn_small", "custom_cnn_large",
                               "vgg16", "resnet50", "resnet18", "inception_v3", 
                               "mobilenet_v3_small", "mobilenet_v3_large"],
                       help="Model architecture")
    parser.add_argument("--filter", type=str, default=None,
                       help="Filter layers by type (e.g., conv, pool, bn)")
    parser.add_argument("--detailed", action="store_true",
                       help="Show detailed layer information")
    parser.add_argument("--num_classes", type=int, default=3,
                       help="Number of output classes")
    parser.add_argument("--pretrained", action="store_true",
                       help="Load pretrained weights (for standard models)")
    
    args = parser.parse_args()
    
    print(f"\nLoading {args.model_name} model...")
    
    # Load model
    model = load_model(
        args.model_name, 
        num_classes=args.num_classes,
        pretrained=args.pretrained
    )
    model.eval()
    
    # Print summary
    print_model_summary(model, args.model_name)
    
    if args.detailed:
        # List all modules with hierarchy
        print(f"\nDetailed Layer Hierarchy:")
        print("-" * 70)
        
        modules = list_all_modules(model)
        
        for mod in modules:
            indent = "  " * mod['depth']
            params_str = f"[{mod['num_params']:,} params]" if mod['has_params'] else ""
            
            if args.filter is None or args.filter.lower() in mod['type'].lower():
                print(f"{indent}{mod['name']:<40} ({mod['type']}) {params_str}")
    
    # List layers suitable for TCAV
    print(f"\n{'='*70}")
    print("Layers suitable for TCAV analysis (Conv2d, MaxPool2d):")
    print("-" * 70)
    
    layers = get_model_layers(model, layer_types=(nn.Conv2d, nn.MaxPool2d))
    suggested = get_suggested_layers(args.model_name)
    
    for i, layer in enumerate(layers):
        if args.filter is None or args.filter.lower() in layer.lower():
            marker = " ★ SUGGESTED" if layer in suggested else ""
            print(f"  [{i:3d}] {layer}{marker}")
    
    print(f"\nTotal: {len(layers)} layers")
    
    # Print suggested layers
    if suggested:
        print(f"\n{'='*70}")
        print(f"Recommended layers for {args.model_name}:")
        print("-" * 70)
        for layer in suggested:
            print(f"  → {layer}")
    
    # Print example usage
    print(f"\n{'='*70}")
    print("Example Commands:")
    print("-" * 70)
    
    example_layer = suggested[0] if suggested else layers[len(layers)//2] if layers else "conv1"
    
    print(f"""
  # Experiment 1: Target class only (with imbalance)
  python main_experiment.py \\
      --experiment 1 \\
      --model_name {args.model_name} \\
      --layer {example_layer} \\
      --target_class zebra \\
      --concept concept1 \\
      --dataset_path ./dataset \\
      --concept_path ./concept \\
      --imbalance_class zebra \\
      --imbalance_ratio 0.1

  # Experiment 2: Full dataset with selective alignment
  python main_experiment.py \\
      --experiment 2 \\
      --model_name {args.model_name} \\
      --layer {example_layer} \\
      --target_class zebra \\
      --concept concept1 \\
      --dataset_path ./dataset \\
      --concept_path ./concept \\
      --imbalance_class zebra \\
      --imbalance_ratio 0.25
    """)
    
    # Custom CNN specific info
    if 'custom' in args.model_name.lower():
        print(f"\n{'='*70}")
        print("Note: Custom CNN models are untrained (random weights)")
        print("You should train the model on your dataset before recalibration")
        print("or use the recalibration as part of the training process.")
        print(f"{'='*70}")


if __name__ == "__main__":
    main()
