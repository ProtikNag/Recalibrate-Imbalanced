#!/usr/bin/env python
"""
Quick debug script for VL-CAV framework.

Runs a minimal pipeline to verify all components work together.
This is faster than running the full experiment and useful for debugging.

Usage:
    python debug_run.py                    # Full debug pipeline
    python debug_run.py --skip-training    # Skip model training
    python debug_run.py --skip-vlm         # Skip VLM (uses random embeddings)
    python debug_run.py --fast             # Fastest possible run
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import sys
import tempfile
from datetime import datetime

# Import framework components
from config import ExperimentConfig, get_class_concept_descriptions
from models import create_model, ActivationExtractor, get_conv_layer_names
from dataloader import load_dataset, create_dataloaders, get_class_samples
from vl_cav import CAVTrainer
from vlm_encoder import VLMEncoder, ConceptEmbedding
from logger import ExperimentLogger
from visualizations import (
    plot_training_curves, plot_class_distribution, 
    plot_per_class_accuracy, set_academic_style
)


def parse_args():
    parser = argparse.ArgumentParser(description="VL-CAV Debug Run")
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip model training phase")
    parser.add_argument("--skip-vlm", action="store_true",
                        help="Skip VLM encoding (use random embeddings)")
    parser.add_argument("--skip-data", action="store_true",
                        help="Skip data loading (use synthetic data)")
    parser.add_argument("--fast", action="store_true",
                        help="Fastest possible run (combines all skip options)")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "cpu", "mps"],
                        help="Device to use")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: temp)")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Handle fast mode
    if args.fast:
        args.skip_training = True
        args.skip_vlm = True
        args.skip_data = True
    
    # Setup device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print("=" * 70)
    print("VL-CAV DEBUG RUN")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Skip training: {args.skip_training}")
    print(f"Skip VLM: {args.skip_vlm}")
    print(f"Skip data: {args.skip_data}")
    print("=" * 70)
    
    # Setup output directory
    if args.output_dir:
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        cleanup = False
    else:
        output_dir = tempfile.mkdtemp(prefix="vl_cav_debug_")
        cleanup = True
    
    print(f"\nOutput directory: {output_dir}")
    
    try:
        # Initialize logger
        logger = ExperimentLogger(output_dir, "debug_run")
        logger.info("Starting debug run...")
        
        # ============================================================
        # Step 1: Data
        # ============================================================
        print("\n[Step 1/6] Data Loading...")
        
        if args.skip_data:
            logger.info("Using synthetic data")
            
            # Create synthetic dataset
            num_classes = 5
            class_names = [f"class_{i}" for i in range(num_classes)]
            
            # Synthetic training data
            train_x = torch.randn(100, 3, 32, 32)
            train_y = torch.randint(0, num_classes, (100,))
            
            test_x = torch.randn(20, 3, 32, 32)
            test_y = torch.randint(0, num_classes, (20,))
            
            train_dataset = torch.utils.data.TensorDataset(train_x, train_y)
            test_dataset = torch.utils.data.TensorDataset(test_x, test_y)
            
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16)
            
            class_counts = {i: 20 for i in range(num_classes)}
            class_counts[0] = 5  # Simulate imbalance
            
            print(f"  ✓ Synthetic data created ({len(train_dataset)} train, {len(test_dataset)} test)")
        else:
            logger.info("Loading CIFAR-10 dataset")
            
            train_dataset, test_dataset, info = load_dataset(
                "CIFAR10",
                root="./data",
                image_size=32,
                imbalance_classes=[0],
                imbalance_ratio=0.1,
                seed=42
            )
            
            train_loader, test_loader = create_dataloaders(
                train_dataset, test_dataset,
                batch_size=32, num_workers=0
            )
            
            num_classes = info["num_classes"]
            class_names = info["class_names"]
            class_counts = info["class_counts"]
            
            print(f"  ✓ CIFAR-10 loaded ({len(train_dataset)} train, {len(test_dataset)} test)")
        
        # Plot class distribution
        plot_class_distribution(
            class_counts, class_names,
            os.path.join(output_dir, "class_distribution"),
            imbalance_classes=[0],
            formats=["png"]
        )
        print("  ✓ Class distribution plot saved")
        
        # ============================================================
        # Step 2: Model
        # ============================================================
        print("\n[Step 2/6] Model Creation...")
        
        model = create_model("custom_cnn", num_classes=num_classes, pretrained=False)
        model = model.to(device)
        
        logger.info(f"Model: custom_cnn, Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  ✓ Model created ({sum(p.numel() for p in model.parameters()):,} parameters)")
        
        # ============================================================
        # Step 3: Training
        # ============================================================
        print("\n[Step 3/6] Training...")
        
        if args.skip_training:
            logger.info("Skipping training phase")
            train_history = {
                'train_loss': [1.0, 0.8, 0.6, 0.5, 0.4],
                'train_accuracy': [0.3, 0.5, 0.6, 0.7, 0.75],
                'val_accuracy': [0.25, 0.45, 0.55, 0.65, 0.7]
            }
            print("  ✓ Skipped (using dummy history)")
        else:
            logger.info("Training model")
            
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            
            train_history = {'train_loss': [], 'train_accuracy': [], 'val_accuracy': []}
            epochs = 3  # Quick training
            
            for epoch in range(epochs):
                model.train()
                total_loss = 0
                correct = 0
                total = 0
                
                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    _, predicted = outputs.max(1)
                    correct += predicted.eq(labels).sum().item()
                    total += labels.size(0)
                
                train_history['train_loss'].append(total_loss / len(train_loader))
                train_history['train_accuracy'].append(correct / total)
                
                # Quick validation
                model.eval()
                val_correct = 0
                val_total = 0
                with torch.no_grad():
                    for inputs, labels in test_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model(inputs)
                        _, predicted = outputs.max(1)
                        val_correct += predicted.eq(labels).sum().item()
                        val_total += labels.size(0)
                
                train_history['val_accuracy'].append(val_correct / val_total)
                
                logger.info(f"Epoch {epoch+1}: Loss={train_history['train_loss'][-1]:.4f}, "
                           f"Acc={train_history['train_accuracy'][-1]:.4f}")
            
            print(f"  ✓ Training complete ({epochs} epochs)")
        
        # Plot training curves
        plot_training_curves(
            train_history,
            os.path.join(output_dir, "training_curves"),
            formats=["png"]
        )
        print("  ✓ Training curves plot saved")
        
        # ============================================================
        # Step 4: VLM Encoding
        # ============================================================
        print("\n[Step 4/6] VLM Encoding...")
        
        if args.skip_vlm:
            logger.info("Using random concept embeddings")
            embedding_dim = 512
            concept_embeddings = {
                i: torch.randn(embedding_dim) for i in range(num_classes)
            }
            print("  ✓ Random embeddings generated")
        else:
            logger.info("Initializing CLIP encoder")
            try:
                vlm_encoder = VLMEncoder("openai/clip-vit-base-patch32", device)
                concept_embedding = ConceptEmbedding(vlm_encoder, alpha=0.7)
                
                for class_idx in range(min(num_classes, 3)):  # Only 3 for speed
                    class_name = class_names[class_idx]
                    descriptions = get_class_concept_descriptions("CIFAR10", class_idx, class_name)
                    
                    # Skip visual embedding for speed
                    concept_embedding.compute_text_embedding(class_idx, descriptions[:5])
                    
                concept_embeddings = concept_embedding.text_embeddings
                print(f"  ✓ CLIP embeddings computed for {len(concept_embeddings)} classes")
            except Exception as e:
                logger.warning(f"VLM failed: {e}, using random embeddings")
                concept_embeddings = {i: torch.randn(512) for i in range(num_classes)}
                print(f"  ⚠ CLIP failed, using random embeddings")
        
        # ============================================================
        # Step 5: CAV Training
        # ============================================================
        print("\n[Step 5/6] CAV Training...")
        
        layer_names = ["conv3", "conv4"]
        cav_trainer = CAVTrainer()
        
        model.eval()
        
        for layer_name in layer_names:
            try:
                extractor = ActivationExtractor(model, [layer_name])
                
                # Get some activations
                concept_acts = []
                random_acts = []
                
                for batch_idx, (inputs, labels) in enumerate(train_loader):
                    if batch_idx >= 2:  # Just 2 batches
                        break
                    
                    inputs = inputs.to(device)
                    with torch.no_grad():
                        _ = model(inputs)
                    
                    acts = extractor.get_activations()[layer_name]
                    acts_flat = acts.view(acts.size(0), -1).cpu().numpy()
                    
                    # Split by some criterion (here: first half = concept)
                    mid = len(acts_flat) // 2
                    concept_acts.append(acts_flat[:mid])
                    random_acts.append(acts_flat[mid:])
                    
                    extractor.clear()
                
                extractor.remove_hooks()
                
                concept_acts = np.vstack(concept_acts)
                random_acts = np.vstack(random_acts)
                
                cav, accuracy = cav_trainer.train_cav(concept_acts, random_acts, layer_name)
                logger.info(f"CAV trained for {layer_name}: accuracy={accuracy:.4f}")
                print(f"  ✓ {layer_name}: CAV accuracy = {accuracy:.4f}")
                
            except Exception as e:
                logger.warning(f"CAV training failed for {layer_name}: {e}")
                print(f"  ⚠ {layer_name}: Failed ({e})")
        
        # ============================================================
        # Step 6: Results Summary
        # ============================================================
        print("\n[Step 6/6] Results Summary...")
        
        # Create dummy before/after comparison
        acc_before = {i: 0.5 + np.random.rand() * 0.3 for i in range(num_classes)}
        acc_after = {i: v + np.random.rand() * 0.1 for i, v in acc_before.items()}
        acc_after[0] = acc_before[0] + 0.15  # Imbalanced class improvement
        
        plot_per_class_accuracy(
            acc_before, acc_after, class_names,
            os.path.join(output_dir, "per_class_accuracy"),
            imbalance_classes=[0],
            formats=["png"]
        )
        print("  ✓ Per-class accuracy plot saved")
        
        # Save results
        results = {
            "status": "success",
            "device": device,
            "num_classes": num_classes,
            "cav_layers": list(cav_trainer.cavs.keys()),
            "cav_accuracies": cav_trainer.accuracies,
            "train_history": train_history,
        }
        
        logger.save_results(results)
        print("  ✓ Results saved")
        
        # ============================================================
        # Done
        # ============================================================
        print("\n" + "=" * 70)
        print("DEBUG RUN COMPLETE ✓")
        print("=" * 70)
        print(f"\nOutputs saved to: {output_dir}")
        print("\nGenerated files:")
        for f in sorted(os.listdir(output_dir)):
            print(f"  - {f}")
        
        logger.finalize()
        
    finally:
        if cleanup and args.output_dir is None:
            print(f"\n[Cleanup] Temporary directory will be deleted: {output_dir}")
            # Uncomment to auto-cleanup:
            # shutil.rmtree(output_dir)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
