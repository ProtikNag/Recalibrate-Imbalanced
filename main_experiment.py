#!/usr/bin/env python3
"""
TCAV-Based Recalibration for Imbalanced Datasets - Enhanced Version

Features:
- Support for custom untrained CNN models
- Comprehensive result recording and logging
- Configurable class imbalance injection
- Detailed visualizations and comparisons

Experiments:
1. Retrain only with target class (alignment + classification from target class)
2. Retrain with full dataset (alignment for target, classification for all)

Usage:
    python main_experiment.py --experiment 1 --model_name custom_cnn --layer conv3 \
        --target_class zebra --concept concept1 --dataset_path ./dataset \
        --concept_path ./concept --imbalance_class zebra --imbalance_ratio 0.1
"""

import os
import copy
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from datetime import datetime
import json
from collections import defaultdict

from utils_imbalanced import (
    load_model,
    get_model_layers,
    get_base_model_image_size,
    train_cav,
    compute_tcav_score,
    evaluate_accuracy,
    evaluate_detailed,
    compute_avg_confidence,
    get_class_distribution,
    compute_confusion_matrix,
    get_misclassification_analysis
)
from dataloader_imbalanced import (
    ImbalancedDataset,
    ConceptDataset,
    SingleClassDataset,
    get_class_folders,
    create_imbalanced_dataset
)
from logger_system import ExperimentLogger
from visualizations import ResultVisualizer

# Device configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Deterministic settings
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
if DEVICE == 'cuda':
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

torch.use_deterministic_algorithms(True, warn_only=True)

# Global variables for hooks
activation = {}
output_shape = {}


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if DEVICE == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id):
    """Initialize worker with deterministic seed."""
    worker_seed = 42 + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_activation(layer_name):
    """Create a forward hook to capture layer activations."""
    def hook(model, input, output):
        activation[layer_name] = output
        output_shape[layer_name] = output.shape
    return hook


def compute_cav(model, concept_loader, random_loader, layer_name, 
                classifier_type='LinearSVC', logger=None):
    """Compute Concept Activation Vector (CAV) for a given layer."""
    concept_acts, random_acts = [], []
    model.eval()
    
    if logger:
        logger.log_info(f"Computing CAV for layer: {layer_name}")
    
    with torch.no_grad():
        for imgs in concept_loader:
            if isinstance(imgs, (list, tuple)):
                imgs = imgs[0]
            imgs = imgs.to(DEVICE)
            _ = model(imgs)
            concept_acts.append(activation[layer_name].view(imgs.size(0), -1).cpu().numpy())
        
        for imgs in random_loader:
            if isinstance(imgs, (list, tuple)):
                imgs = imgs[0]
            imgs = imgs.to(DEVICE)
            _ = model(imgs)
            random_acts.append(activation[layer_name].view(imgs.size(0), -1).cpu().numpy())
    
    concept_acts = np.vstack(concept_acts)
    random_acts = np.vstack(random_acts)
    
    if logger:
        logger.log_info(f"Concept activations shape: {concept_acts.shape}")
        logger.log_info(f"Random activations shape: {random_acts.shape}")
    
    cav = train_cav(concept_acts, random_acts, classifier_type=classifier_type)
    return torch.tensor(cav, dtype=torch.float32, device=DEVICE)


class Experiment1Runner:
    """
    Experiment 1: Retrain only with target class data.
    Both alignment and classification losses from target class only.
    """
    
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.model = None
        self.cav_vector = None
        self.layer_name = config.layer
        
    def setup(self, model, cav_vector):
        """Set up the experiment with model and CAV."""
        self.model = copy.deepcopy(model).to(DEVICE)
        self.cav_vector = cav_vector
        
        # Register forward hook
        self.model.get_submodule(self.layer_name).register_forward_hook(
            get_activation(self.layer_name)
        )
        
        # Freeze all layers except target layer
        trainable_params = 0
        total_params = 0
        for name, param in self.model.named_parameters():
            total_params += param.numel()
            if self.layer_name in name:
                param.requires_grad = True
                trainable_params += param.numel()
            else:
                param.requires_grad = False
        
        self.logger.log_info(f"Total parameters: {total_params:,}")
        self.logger.log_info(f"Trainable parameters: {trainable_params:,}")
        
        # Keep BatchNorm and Dropout in eval mode
        self.model.apply(
            lambda m: m.eval() if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.Dropout)) else None
        )
        
    def train(self, target_loader, lambda_align=0.5, epochs=10, lr=1e-3):
        """Train using only target class data."""
        lambda_cls = 1.0 - lambda_align
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()), 
            lr=lr
        )
        
        loss_history = {"total": [], "cls": [], "align": []}
        self.model.train()
        
        self.logger.log_info(f"[Experiment 1] Training with target class only")
        self.logger.log_info(f"Lambda Align: {lambda_align}, Lambda Cls: {lambda_cls}")
        
        for epoch in range(epochs):
            total_loss_epoch = cls_loss_epoch = align_loss_epoch = 0.0
            n_batches = 0
            
            for imgs, labels in target_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                
                outputs = self.model(imgs)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                cls_loss = nn.CrossEntropyLoss()(outputs, labels)
                
                f_l = activation[self.layer_name].view(imgs.size(0), -1)
                cosine_sim = F.cosine_similarity(f_l, self.cav_vector.unsqueeze(0), dim=1)
                align_loss = (1 - torch.mean(torch.abs(cosine_sim)))
                
                loss = lambda_align * align_loss + lambda_cls * cls_loss
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=7)
                optimizer.step()
                
                total_loss_epoch += loss.item()
                cls_loss_epoch += cls_loss.item()
                align_loss_epoch += align_loss.item()
                n_batches += 1
            
            avg_total = total_loss_epoch / max(n_batches, 1)
            avg_cls = cls_loss_epoch / max(n_batches, 1)
            avg_align = align_loss_epoch / max(n_batches, 1)
            
            loss_history["total"].append(avg_total)
            loss_history["cls"].append(avg_cls)
            loss_history["align"].append(avg_align)
            
            self.logger.log_epoch(epoch + 1, epochs, avg_total, avg_cls, avg_align)
        
        return loss_history


class Experiment2Runner:
    """
    Experiment 2: Retrain with full dataset.
    Alignment loss only for target class, classification loss for all classes.
    """
    
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.model = None
        self.cav_vector = None
        self.layer_name = config.layer
        self.target_class_idx = None
        
    def setup(self, model, cav_vector, target_class_idx):
        """Set up the experiment with model, CAV, and target class index."""
        self.model = copy.deepcopy(model).to(DEVICE)
        self.cav_vector = cav_vector
        self.target_class_idx = target_class_idx
        
        # Register forward hook
        self.model.get_submodule(self.layer_name).register_forward_hook(
            get_activation(self.layer_name)
        )
        
        # Freeze all layers except target layer
        trainable_params = 0
        total_params = 0
        for name, param in self.model.named_parameters():
            total_params += param.numel()
            if self.layer_name in name:
                param.requires_grad = True
                trainable_params += param.numel()
            else:
                param.requires_grad = False
        
        self.logger.log_info(f"Total parameters: {total_params:,}")
        self.logger.log_info(f"Trainable parameters: {trainable_params:,}")
        
        # Keep BatchNorm and Dropout in eval mode
        self.model.apply(
            lambda m: m.eval() if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.Dropout)) else None
        )
        
    def train(self, full_loader, lambda_align=0.5, epochs=10, lr=1e-3):
        """Train using full dataset with selective alignment loss."""
        lambda_cls = 1.0 - lambda_align
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()), 
            lr=lr
        )
        
        loss_history = {"total": [], "cls": [], "align": []}
        self.model.train()
        
        self.logger.log_info(f"[Experiment 2] Training with full dataset")
        self.logger.log_info(f"Target class index: {self.target_class_idx}")
        self.logger.log_info(f"Lambda Align: {lambda_align}, Lambda Cls: {lambda_cls}")
        
        for epoch in range(epochs):
            total_loss_epoch = cls_loss_epoch = align_loss_epoch = 0.0
            n_batches = 0
            target_samples_count = 0
            
            for imgs, labels in full_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                
                outputs = self.model(imgs)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                cls_loss = nn.CrossEntropyLoss()(outputs, labels)
                
                mask = (labels == self.target_class_idx)
                align_loss = torch.tensor(0.0, device=DEVICE)
                
                if mask.any():
                    f_l = activation[self.layer_name].view(imgs.size(0), -1)
                    f_l_target = f_l[mask]
                    cosine_sim = F.cosine_similarity(
                        f_l_target, 
                        self.cav_vector.unsqueeze(0), 
                        dim=1
                    )
                    align_loss = (1 - torch.mean(torch.abs(cosine_sim)))
                    target_samples_count += mask.sum().item()
                
                loss = lambda_align * align_loss + lambda_cls * cls_loss
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=7)
                optimizer.step()
                
                total_loss_epoch += loss.item()
                cls_loss_epoch += cls_loss.item()
                align_loss_epoch += align_loss.item()
                n_batches += 1
            
            avg_total = total_loss_epoch / max(n_batches, 1)
            avg_cls = cls_loss_epoch / max(n_batches, 1)
            avg_align = align_loss_epoch / max(n_batches, 1)
            
            loss_history["total"].append(avg_total)
            loss_history["cls"].append(avg_cls)
            loss_history["align"].append(avg_align)
            
            self.logger.log_epoch(epoch + 1, epochs, avg_total, avg_cls, avg_align,
                                 extra_info=f"Target samples: {target_samples_count}")
        
        return loss_history


def run_experiment(args):
    """Main experiment runner."""
    set_seed(args.seed)
    
    # Create results directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name = f"exp{args.experiment}_{args.model_name}_{args.target_class}_{timestamp}"
    results_dir = os.path.join(args.results_path, exp_name)
    os.makedirs(results_dir, exist_ok=True)
    
    # Initialize logger
    logger = ExperimentLogger(results_dir, exp_name)
    logger.log_header("TCAV-Based Recalibration Experiment")
    
    # Log configuration
    config_dict = vars(args)
    logger.log_config(config_dict)
    
    # Save configuration
    config_path = os.path.join(results_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    logger.log_section("Environment Setup")
    logger.log_info(f"Device: {DEVICE}")
    logger.log_info(f"PyTorch version: {torch.__version__}")
    logger.log_info(f"Results directory: {results_dir}")
    
    # Get image size for model
    image_size = get_base_model_image_size(args.model_name)
    logger.log_info(f"Image size: {image_size}x{image_size}")
    
    # Define transforms
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Load datasets
    logger.log_section("Dataset Loading")
    class_folders = get_class_folders(args.dataset_path)
    class_names = sorted(class_folders.keys())
    logger.log_info(f"Found classes: {class_names}")
    
    if args.target_class not in class_names:
        raise ValueError(f"Target class '{args.target_class}' not found. Available: {class_names}")
    
    target_class_idx = class_names.index(args.target_class)
    logger.log_info(f"Target class: {args.target_class} (index: {target_class_idx})")
    
    # Create imbalance ratios
    imbalance_ratios = {name: 1.0 for name in class_names}
    if args.imbalance_class and args.imbalance_ratio:
        if args.imbalance_class in class_names:
            imbalance_ratios[args.imbalance_class] = args.imbalance_ratio
            logger.log_info(f"Applying imbalance: {args.imbalance_class} -> {args.imbalance_ratio*100:.1f}%")
        else:
            logger.log_warning(f"Imbalance class '{args.imbalance_class}' not found")
    
    # Create datasets with imbalance
    full_dataset = create_imbalanced_dataset(
        args.dataset_path,
        transform=train_transform,
        class_names=class_names,
        imbalance_ratios=imbalance_ratios,
        is_train=True
    )
    
    val_dataset = create_imbalanced_dataset(
        args.dataset_path,
        transform=val_transform,
        class_names=class_names,
        imbalance_ratios={name: 1.0 for name in class_names},  # Full validation set
        is_train=False
    )
    
    # Log dataset statistics
    train_class_counts = full_dataset.get_class_counts()
    val_class_counts = val_dataset.get_class_counts()
    
    logger.log_info("Training set class distribution:")
    for name, count in train_class_counts.items():
        logger.log_info(f"  {name}: {count} images")
    
    logger.log_info("Validation set class distribution:")
    for name, count in val_class_counts.items():
        logger.log_info(f"  {name}: {count} images")
    
    # Create data loaders
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    
    full_loader = DataLoader(
        full_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        worker_init_fn=worker_init_fn,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Create target class only loader
    target_dataset = SingleClassDataset(
        os.path.join(args.dataset_path, args.target_class),
        transform=train_transform,
        class_idx=target_class_idx,
        sample_ratio=imbalance_ratios.get(args.target_class, 1.0)
    )
    
    target_loader = DataLoader(
        target_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        worker_init_fn=worker_init_fn,
        num_workers=0
    )
    
    logger.log_info(f"Full training dataset size: {len(full_dataset)}")
    logger.log_info(f"Target class dataset size: {len(target_dataset)}")
    logger.log_info(f"Validation dataset size: {len(val_dataset)}")
    
    # Load concept datasets
    logger.log_section("Concept Loading")
    concept_path = os.path.join(args.concept_path, args.target_class, args.concept)
    if not os.path.exists(concept_path):
        raise FileNotFoundError(f"Concept path not found: {concept_path}")
    
    concept_dataset = ConceptDataset(concept_path, transform=val_transform)
    concept_loader = DataLoader(concept_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    logger.log_info(f"Concept dataset size: {len(concept_dataset)}")
    
    # Random dataset
    random_classes = [c for c in class_names if c != args.target_class]
    random_class = random_classes[0] if random_classes else args.target_class
    random_path = os.path.join(args.dataset_path, random_class)
    
    random_dataset = ConceptDataset(random_path, transform=val_transform)
    random_loader = DataLoader(random_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    logger.log_info(f"Random dataset size: {len(random_dataset)} (from class: {random_class})")
    
    # Load model
    logger.log_section("Model Loading")
    model = load_model(
        args.model_name, 
        args.model_path, 
        num_classes=len(class_names),
        pretrained=args.pretrained,
        input_size=image_size
    )
    model.to(DEVICE)
    
    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.log_info(f"Model: {args.model_name}")
    logger.log_info(f"Pretrained: {args.pretrained}")
    logger.log_info(f"Total parameters: {total_params:,}")
    
    # Get available layers
    if args.layer == 'auto':
        available_layers = get_model_layers(model)
        logger.log_info(f"Available layers: {available_layers}")
        args.layer = available_layers[len(available_layers) * 2 // 3]
        logger.log_info(f"Auto-selected layer: {args.layer}")
    
    # Register hook
    model.get_submodule(args.layer).register_forward_hook(get_activation(args.layer))
    logger.log_info(f"Registered hook on layer: {args.layer}")
    
    # Compute CAV
    logger.log_section("CAV Computation")
    cav_vector = compute_cav(
        model, concept_loader, random_loader, args.layer,
        classifier_type=args.classifier_type, logger=logger
    )
    logger.log_info(f"CAV shape: {cav_vector.shape}")
    
    # Evaluate BEFORE recalibration
    logger.log_section("Evaluation BEFORE Recalibration")
    
    results_before = evaluate_detailed(model, val_loader, class_names, DEVICE)
    tcav_before = compute_tcav_score(
        model, args.layer, cav_vector, target_loader, target_class_idx, activation, DEVICE
    )
    
    # Log detailed results before
    logger.log_evaluation_results(results_before, "BEFORE", tcav_before)
    
    # Run experiment
    logger.log_section(f"Running Experiment {args.experiment}")
    
    if args.experiment == 1:
        runner = Experiment1Runner(args, logger)
        runner.setup(model, cav_vector)
        loss_history = runner.train(
            target_loader,
            lambda_align=args.lambda_align,
            epochs=args.epochs,
            lr=args.learning_rate
        )
        trained_model = runner.model
        
    elif args.experiment == 2:
        runner = Experiment2Runner(args, logger)
        runner.setup(model, cav_vector, target_class_idx)
        loss_history = runner.train(
            full_loader,
            lambda_align=args.lambda_align,
            epochs=args.epochs,
            lr=args.learning_rate
        )
        trained_model = runner.model
    else:
        raise ValueError(f"Unknown experiment: {args.experiment}")
    
    # Evaluate AFTER recalibration
    logger.log_section("Evaluation AFTER Recalibration")
    
    results_after = evaluate_detailed(trained_model, val_loader, class_names, DEVICE)
    tcav_after = compute_tcav_score(
        trained_model, args.layer, cav_vector, target_loader, target_class_idx, activation, DEVICE
    )
    
    # Log detailed results after
    logger.log_evaluation_results(results_after, "AFTER", tcav_after)
    
    # Log comparison
    logger.log_section("Results Comparison")
    logger.log_comparison(results_before, results_after, tcav_before, tcav_after, class_names)
    
    # Save all results
    logger.log_section("Saving Results")
    
    all_results = {
        "experiment": args.experiment,
        "model_name": args.model_name,
        "layer": args.layer,
        "target_class": args.target_class,
        "concept": args.concept,
        "imbalance_class": args.imbalance_class,
        "imbalance_ratio": args.imbalance_ratio,
        "lambda_align": args.lambda_align,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "train_class_counts": train_class_counts,
        "val_class_counts": val_class_counts,
        "results_before": results_before,
        "results_after": results_after,
        "tcav_before": tcav_before,
        "tcav_after": tcav_after,
        "loss_history": loss_history
    }
    
    # Save detailed results JSON
    results_json_path = os.path.join(results_dir, "detailed_results.json")
    with open(results_json_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.log_info(f"Saved detailed results to: {results_json_path}")
    
    # Generate visualizations
    logger.log_section("Generating Visualizations")
    visualizer = ResultVisualizer(results_dir)
    
    # Loss plots
    visualizer.plot_loss_curves(loss_history, args.epochs)
    logger.log_info("Generated loss curves")
    
    # Confusion matrices
    visualizer.plot_confusion_matrices(
        results_before['confusion_matrix'],
        results_after['confusion_matrix'],
        class_names
    )
    logger.log_info("Generated confusion matrices")
    
    # Per-class accuracy comparison
    visualizer.plot_per_class_comparison(
        results_before['per_class'],
        results_after['per_class'],
        class_names
    )
    logger.log_info("Generated per-class comparison")
    
    # Metrics comparison
    visualizer.plot_metrics_comparison(results_before, results_after, tcav_before, tcav_after)
    logger.log_info("Generated metrics comparison")
    
    # Class distribution
    visualizer.plot_class_distribution(train_class_counts, val_class_counts)
    logger.log_info("Generated class distribution plot")
    
    # Misclassification analysis
    visualizer.plot_misclassification_analysis(
        results_before['misclassification_matrix'],
        results_after['misclassification_matrix'],
        class_names
    )
    logger.log_info("Generated misclassification analysis")
    
    # Save model
    model_path = os.path.join(results_dir, f"model_exp{args.experiment}.pth")
    torch.save(trained_model.state_dict(), model_path)
    logger.log_info(f"Saved model to: {model_path}")
    
    # Generate summary report
    logger.log_section("Experiment Summary")
    logger.log_summary(all_results)
    
    # Close logger
    logger.close()
    
    print(f"\n{'='*60}")
    print("EXPERIMENT COMPLETED")
    print(f"Results saved to: {results_dir}")
    print(f"{'='*60}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="TCAV-Based Recalibration for Imbalanced Datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Experiment 1 with custom CNN and 10% imbalance
  python main_experiment.py --experiment 1 --model_name custom_cnn \\
      --layer conv3 --target_class zebra --concept concept1 \\
      --dataset_path ./dataset --concept_path ./concept \\
      --imbalance_class zebra --imbalance_ratio 0.1

  # Experiment 2 with pretrained VGG16 and 25% imbalance
  python main_experiment.py --experiment 2 --model_name vgg16 --pretrained \\
      --layer features.24 --target_class deer --concept concept1 \\
      --dataset_path ./dataset --concept_path ./concept \\
      --imbalance_class deer --imbalance_ratio 0.25
        """
    )
    
    # Experiment selection
    parser.add_argument("--experiment", type=int, required=True, choices=[1, 2],
                        help="Experiment type: 1=target class only, 2=full dataset with selective alignment")
    
    # Model configuration
    parser.add_argument("--model_name", type=str, default="custom_cnn",
                        choices=["custom_cnn", "custom_cnn_small", "custom_cnn_large",
                                "vgg16", "resnet50", "resnet18", "inception_v3", 
                                "mobilenet_v3_small", "mobilenet_v3_large"],
                        help="CNN model architecture")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to pre-trained model weights")
    parser.add_argument("--pretrained", action="store_true",
                        help="Use pretrained weights (only for standard models)")
    parser.add_argument("--layer", type=str, default="auto",
                        help="Layer name for recalibration (use 'auto' for automatic selection)")
    
    # Dataset configuration
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to dataset directory")
    parser.add_argument("--concept_path", type=str, required=True,
                        help="Path to concept directory")
    parser.add_argument("--target_class", type=str, required=True,
                        help="Name of the target class for alignment")
    parser.add_argument("--concept", type=str, required=True,
                        help="Concept folder name to use")
    
    # Imbalance configuration
    parser.add_argument("--imbalance_class", type=str, default=None,
                        help="Class to make imbalanced (optional)")
    parser.add_argument("--imbalance_ratio", type=float, default=None,
                        help="Ratio of data to keep for imbalanced class (0.05, 0.1, 0.2, 0.25)")
    
    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--lambda_align", type=float, default=0.5,
                        help="Weight for alignment loss (0-1)")
    
    # CAV configuration
    parser.add_argument("--classifier_type", type=str, default="LinearSVC",
                        choices=["LinearSVC", "SGDClassifier", "LogisticRegression"],
                        help="Linear classifier type for CAV training")
    
    # Other settings
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--results_path", type=str, default="./results",
                        help="Path to save results")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not 0 <= args.lambda_align <= 1:
        parser.error("lambda_align must be between 0 and 1")
    
    if args.imbalance_ratio is not None and not 0 < args.imbalance_ratio <= 1:
        parser.error("imbalance_ratio must be between 0 and 1")
    
    if args.imbalance_ratio is not None and args.imbalance_class is None:
        parser.error("imbalance_class must be specified when using imbalance_ratio")
    
    # Run experiment
    run_experiment(args)


if __name__ == "__main__":
    main()
