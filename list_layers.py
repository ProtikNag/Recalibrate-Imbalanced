#!/usr/bin/env python3
"""
Utility to list all layers in a CNN model suitable for TCAV analysis.

This helps identify which layers can be targeted for recalibration.
Typically, convolutional layers in the middle-to-late stages of the
network are most suitable for concept alignment.

Usage:
    python list_layers.py --model_name vgg16
    python list_layers.py --model_name custom_cnn --num_classes 5
"""

import argparse
import torch
import torch.nn as nn
from utils import load_model


def list_all_layers(model: nn.Module, indent: int = 0) -> None:
    """Recursively list all layers in a model."""
    for name, module in model.named_children():
        print("  " * indent + f"{name}: {module.__class__.__name__}")
        if len(list(module.children())) > 0:
            list_all_layers(module, indent + 1)


def list_candidate_layers(model: nn.Module, 
                         layer_types: tuple = (nn.Conv2d, nn.MaxPool2d)) -> list:
    """
    List layers suitable for TCAV analysis.
    
    Args:
        model: PyTorch model
        layer_types: Types of layers to consider
        
    Returns:
        List of (name, module) tuples
    """
    candidates = []
    for name, module in model.named_modules():
        if isinstance(module, layer_types):
            candidates.append((name, module))
    return candidates


def analyze_layer(model: nn.Module, layer_name: str, input_size: int = 224) -> dict:
    """
    Analyze a specific layer's properties.
    
    Args:
        model: PyTorch model
        layer_name: Name of the layer to analyze
        input_size: Input image size
        
    Returns:
        Dictionary with layer properties
    """
    # Get the layer
    try:
        layer = model.get_submodule(layer_name)
    except AttributeError:
        return {"error": f"Layer '{layer_name}' not found"}
    
    # Create a hook to capture output shape
    output_shape = None
    
    def hook(module, input, output):
        nonlocal output_shape
        output_shape = output.shape
    
    handle = layer.register_forward_hook(hook)
    
    # Run a forward pass
    model.eval()
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, input_size, input_size)
        try:
            model(dummy_input)
        except Exception as e:
            handle.remove()
            return {"error": str(e)}
    
    handle.remove()
    
    # Calculate properties
    if output_shape is not None:
        activation_size = output_shape.numel()
        return {
            "name": layer_name,
            "type": layer.__class__.__name__,
            "output_shape": list(output_shape),
            "activation_size": activation_size,
            "suitable_for_cav": activation_size > 100  # Heuristic
        }
    
    return {"error": "Could not determine output shape"}


def main():
    parser = argparse.ArgumentParser(
        description="List CNN layers suitable for TCAV analysis"
    )
    parser.add_argument("--model_name", type=str, default="custom_cnn",
                       help="Model architecture name")
    parser.add_argument("--num_classes", type=int, default=5,
                       help="Number of output classes")
    parser.add_argument("--input_size", type=int, default=224,
                       help="Input image size")
    parser.add_argument("--detailed", action="store_true",
                       help="Show detailed layer analysis")
    parser.add_argument("--all", action="store_true",
                       help="Show all layers (not just candidates)")
    
    args = parser.parse_args()
    
    print(f"\n{'=' * 60}")
    print(f"Layer Analysis for {args.model_name}")
    print(f"{'=' * 60}")
    
    # Load model
    model = load_model(args.model_name, num_classes=args.num_classes)
    model.eval()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    if args.all:
        print(f"\n{'=' * 60}")
        print("All Layers (Hierarchical)")
        print(f"{'=' * 60}")
        list_all_layers(model)
    
    # List candidate layers
    print(f"\n{'=' * 60}")
    print("Candidate Layers for TCAV Analysis")
    print(f"{'=' * 60}")
    
    candidates = list_candidate_layers(model)
    
    print(f"\nFound {len(candidates)} candidate layers:\n")
    print(f"{'Name':<45} {'Type':<20}")
    print("-" * 65)
    
    for name, module in candidates:
        print(f"{name:<45} {module.__class__.__name__:<20}")
    
    if args.detailed:
        print(f"\n{'=' * 60}")
        print("Detailed Layer Analysis")
        print(f"{'=' * 60}")
        
        for name, _ in candidates[-10:]:  # Analyze last 10 candidates
            info = analyze_layer(model, name, args.input_size)
            if "error" not in info:
                print(f"\n{name}:")
                print(f"  Type: {info['type']}")
                print(f"  Output shape: {info['output_shape']}")
                print(f"  Activation size: {info['activation_size']:,}")
                print(f"  Suitable for CAV: {info['suitable_for_cav']}")
            else:
                print(f"\n{name}: {info['error']}")
    
    # Suggested layers
    print(f"\n{'=' * 60}")
    print("Suggested Layers for Recalibration")
    print(f"{'=' * 60}")
    
    # Suggest layers from the middle to late stages
    n_candidates = len(candidates)
    if n_candidates >= 5:
        suggested = [candidates[i][0] for i in 
                    [n_candidates//4, n_candidates//2, 3*n_candidates//4, n_candidates-2, n_candidates-1]]
    else:
        suggested = [c[0] for c in candidates]
    
    print("\nRecommended layers (from early to late):")
    for i, name in enumerate(suggested):
        print(f"  {i+1}. {name}")
    
    print(f"\n{'=' * 60}")


if __name__ == "__main__":
    main()
