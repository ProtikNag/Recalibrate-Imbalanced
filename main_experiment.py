#!/usr/bin/env python3
"""
TCAV-Based Recalibration for Imbalanced Datasets - Enhanced Version

Workflow:
1. Load model (custom CNN or pretrained)
2. Train model on imbalanced dataset (creates bias)
3. Evaluate biased model (BEFORE recalibration)
4. Apply TCAV-based recalibration
5. Evaluate recalibrated model (AFTER recalibration)
6. Compare results and generate visualizations

Experiments:
1. Recalibrate only with target class (alignment + classification from target class)
2. Recalibrate with full dataset (alignment for target, classification for all)

Usage:
    python main_experiment.py --experiment 1 --model_name custom_cnn --layer features.17 \
        --target_class zebra --concept stripes --dataset_path ./dataset \
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


class InitialTrainer:
    """
    Initial training phase: Train the model on imbalanced data.
    This creates a biased model that we will later try to fix with recalibration.
    """
    
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        
    def train(self, model, train_loader, val_loader, epochs=10, lr=1e-3):
        """
        Train model from scratch on imbalanced data.
        
        Args:
            model: The model to train
            train_loader: Training data loader (imbalanced)
            val_loader: Validation data loader
            epochs: Number of training epochs
            lr: Learning rate
        
        Returns:
            Trained model and training history
        """
        model = model.to(DEVICE)
        model.train()
        
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        criterion = nn.CrossEntropyLoss()
        
        history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": []
        }
        
        best_val_acc = 0.0
        best_model_state = None
        
        self.logger.log_info(f"[Initial Training] Training model on imbalanced data")
        self.logger.log_info(f"Epochs: {epochs}, Learning Rate: {lr}")
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for imgs, labels in train_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                
                optimizer.zero_grad()
                outputs = model(imgs)
                
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                loss = criterion(outputs, labels)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            avg_train_loss = train_loss / len(train_loader)
            train_acc = train_correct / train_total
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                    outputs = model(imgs)
                    
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                    
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(outputs, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            avg_val_loss = val_loss / len(val_loader)
            val_acc = val_correct / val_total
            
            # Update scheduler
            scheduler.step(avg_val_loss)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = copy.deepcopy(model.state_dict())
            
            # Record history
            history["train_loss"].append(avg_train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(avg_val_loss)
            history["val_acc"].append(val_acc)
            
            self.logger.log_info(
                f"Epoch {epoch+1:3d}/{epochs} | "
                f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}"
            )
        
        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            self.logger.log_info(f"Loaded best model with validation accuracy: {best_val_acc:.4f}")
        
        return model, history


class Experiment1Runner:
    """
    Experiment 1: Recalibrate only with target class data.
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
        self.logger.log_info(f"Trainable parameters (recalibration): {trainable_params:,}")
        
        # Keep BatchNorm and Dropout in eval mode
        self.model.apply(
            lambda m: m.eval() if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.Dropout)) else None
        )
        
    def train(self, target_loader, lambda_align=0.5, epochs=10, lr=1e-3):
        """Recalibrate using only target class data."""
        lambda_cls = 1.0 - lambda_align
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()), 
            lr=lr
        )
        
        loss_history = {"total": [], "cls": [], "align": []}
        self.model.train()
        
        self.logger.log_info(f"[Experiment 1] Recalibrating with target class only")
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
    Experiment 2: Recalibrate with full dataset.
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
        self.logger.log_info(f"Trainable parameters (recalibration): {trainable_params:,}")
        
        # Keep BatchNorm and Dropout in eval mode
        self.model.apply(
            lambda m: m.eval() if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.Dropout)) else None
        )
        
    def train(self, full_loader, lambda_align=0.5, epochs=10, lr=1e-3):
        """Recalibrate using full dataset with selective alignment loss."""
        lambda_cls = 1.0 - lambda_align
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()), 
            lr=lr
        )
        
        loss_history = {"total": [], "cls": [], "align": []}
        self.model.train()
        
        self.logger.log_info(f"[Experiment 2] Recalibrating with full dataset")
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
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
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
    
    logger.log_info("Training set class distribution (with imbalance applied):")
    for name, count in train_class_counts.items():
        logger.log_info(f"  {name}: {count} images")
    
    logger.log_info("Validation set class distribution (balanced for fair evaluation):")
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
    
    # Random dataset (from a different class for CAV training)
    random_classes = [c for c in class_names if c != args.target_class]
    random_class = random_classes[0] if random_classes else args.target_class
    random_path = os.path.join(args.dataset_path, random_class)
    
    random_dataset = ConceptDataset(random_path, transform=val_transform)
    random_loader = DataLoader(random_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    logger.log_info(f"Random dataset size: {len(random_dataset)} (from class: {random_class})")
    
    # =========================================================================
    # PHASE 1: Load/Create Model
    # =========================================================================
    logger.log_section("Model Initialization")
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
    
    # =========================================================================
    # PHASE 2: Initial Training on Imbalanced Data (Creates Biased Model)
    # =========================================================================
    logger.log_section("Phase 1: Initial Training on Imbalanced Data")
    logger.log_info("Training model on imbalanced dataset to create bias...")
    logger.log_info("This simulates a real-world scenario where model is trained on biased data.")
    
    initial_trainer = InitialTrainer(args, logger)
    model, initial_train_history = initial_trainer.train(
        model=model,
        train_loader=full_loader,
        val_loader=val_loader,
        epochs=args.pretrain_epochs,
        lr=args.pretrain_lr
    )
    
    # Save initial training history
    initial_history_path = os.path.join(results_dir, "initial_training_history.json")
    with open(initial_history_path, 'w') as f:
        json.dump(initial_train_history, f, indent=2)
    logger.log_info(f"Saved initial training history to: {initial_history_path}")
    
    # Save the biased model
    biased_model_path = os.path.join(results_dir, "model_biased.pth")
    torch.save(model.state_dict(), biased_model_path)
    logger.log_info(f"Saved biased model to: {biased_model_path}")
    
    # =========================================================================
    # PHASE 3: Compute CAV
    # =========================================================================
    logger.log_section("Phase 2: CAV Computation")
    
    # Register hook for CAV computation
    model.get_submodule(args.layer).register_forward_hook(get_activation(args.layer))
    logger.log_info(f"Registered hook on layer: {args.layer}")
    
    cav_vector = compute_cav(
        model, concept_loader, random_loader, args.layer,
        classifier_type=args.classifier_type, logger=logger
    )
    logger.log_info(f"CAV shape: {cav_vector.shape}")
    
    # =========================================================================
    # PHASE 4: Evaluate BEFORE Recalibration (Biased Model)
    # =========================================================================
    logger.log_section("Phase 3: Evaluation BEFORE Recalibration")
    logger.log_info("Evaluating the biased model (trained on imbalanced data)...")
    
    results_before = evaluate_detailed(model, val_loader, class_names, DEVICE)
    tcav_before = compute_tcav_score(
        model, args.layer, cav_vector, target_loader, target_class_idx, activation, DEVICE
    )
    
    # Log detailed results before
    logger.log_evaluation_results(results_before, "BEFORE", tcav_before)
    
    # =========================================================================
    # PHASE 5: Apply TCAV-Based Recalibration
    # =========================================================================
    logger.log_section(f"Phase 4: Recalibration (Experiment {args.experiment})")
    
    if args.experiment == 1:
        runner = Experiment1Runner(args, logger)
        runner.setup(model, cav_vector)
        loss_history = runner.train(
            target_loader,
            lambda_align=args.lambda_align,
            epochs=args.recalib_epochs,
            lr=args.recalib_lr
        )
        trained_model = runner.model
        
    elif args.experiment == 2:
        runner = Experiment2Runner(args, logger)
        runner.setup(model, cav_vector, target_class_idx)
        loss_history = runner.train(
            full_loader,
            lambda_align=args.lambda_align,
            epochs=args.recalib_epochs,
            lr=args.recalib_lr
        )
        trained_model = runner.model
    else:
        raise ValueError(f"Unknown experiment: {args.experiment}")
    
    # =========================================================================
    # PHASE 6: Evaluate AFTER Recalibration
    # =========================================================================
    logger.log_section("Phase 5: Evaluation AFTER Recalibration")
    
    results_after = evaluate_detailed(trained_model, val_loader, class_names, DEVICE)
    tcav_after = compute_tcav_score(
        trained_model, args.layer, cav_vector, target_loader, target_class_idx, activation, DEVICE
    )
    
    # Log detailed results after
    logger.log_evaluation_results(results_after, "AFTER", tcav_after)
    
    # Log comparison
    logger.log_section("Results Comparison")
    logger.log_comparison(results_before, results_after, tcav_before, tcav_after, class_names)
    
    # =========================================================================
    # PHASE 7: Save All Results and Generate Visualizations
    # =========================================================================
    logger.log_section("Saving Results and Generating Visualizations")
    
    all_results = {
        "experiment": args.experiment,
        "model_name": args.model_name,
        "layer": args.layer,
        "target_class": args.target_class,
        "concept": args.concept,
        "imbalance_class": args.imbalance_class,
        "imbalance_ratio": args.imbalance_ratio,
        "lambda_align": args.lambda_align,
        "pretrain_epochs": args.pretrain_epochs,
        "recalib_epochs": args.recalib_epochs,
        "pretrain_lr": args.pretrain_lr,
        "recalib_lr": args.recalib_lr,
        "seed": args.seed,
        "train_class_counts": train_class_counts,
        "val_class_counts": val_class_counts,
        "initial_train_history": initial_train_history,
        "results_before": results_before,
        "results_after": results_after,
        "tcav_before": tcav_before,
        "tcav_after": tcav_after,
        "loss_history": loss_history,
        "epochs": args.recalib_epochs  # For backward compatibility with visualizations
    }
    
    # Save detailed results JSON
    results_json_path = os.path.join(results_dir, "detailed_results.json")
    with open(results_json_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.log_info(f"Saved detailed results to: {results_json_path}")
    
    # Generate visualizations
    visualizer = ResultVisualizer(results_dir)
    
    # Initial training loss/accuracy curves
    visualizer.plot_initial_training(initial_train_history, args.pretrain_epochs)
    logger.log_info("Generated initial training curves")
    
    # Recalibration loss plots
    visualizer.plot_loss_curves(loss_history, args.recalib_epochs)
    logger.log_info("Generated recalibration loss curves")
    
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
    
    # Save recalibrated model
    model_path = os.path.join(results_dir, f"model_recalibrated_exp{args.experiment}.pth")
    torch.save(trained_model.state_dict(), model_path)
    logger.log_info(f"Saved recalibrated model to: {model_path}")
    
    # Generate summary report
    logger.log_section("Experiment Summary")
    logger.log_summary(all_results)
    
    # Close logger
    logger.close()
    
    print(f"\n{'='*60}")
    print("EXPERIMENT COMPLETED")
    print(f"{'='*60}")
    print(f"Results saved to: {results_dir}")
    print(f"\nKey Results:")
    print(f"  Initial training epochs: {args.pretrain_epochs}")
    print(f"  Recalibration epochs: {args.recalib_epochs}")
    print(f"  Accuracy: {results_before['overall']['accuracy']:.4f} -> {results_after['overall']['accuracy']:.4f}")
    print(f"  TCAV Score: {tcav_before:.4f} -> {tcav_after:.4f}")
    print(f"  Target class ({args.target_class}) accuracy: "
          f"{results_before['per_class'][args.target_class]['accuracy']:.4f} -> "
          f"{results_after['per_class'][args.target_class]['accuracy']:.4f}")
    print(f"{'='*60}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="TCAV-Based Recalibration for Imbalanced Datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Workflow:
  1. Train model on imbalanced data (creates biased model)
  2. Evaluate biased model
  3. Apply TCAV-based recalibration  
  4. Evaluate recalibrated model
  5. Compare and visualize results

Examples:
  # Experiment 1: Recalibrate with target class only
  python main_experiment.py --experiment 1 --model_name custom_cnn \\
      --layer features.17 --target_class zebra --concept stripes \\
      --dataset_path ./dataset --concept_path ./concept \\
      --imbalance_class zebra --imbalance_ratio 0.1 \\
      --pretrain_epochs 30 --recalib_epochs 10

  # Experiment 2: Recalibrate with full dataset
  python main_experiment.py --experiment 2 --model_name custom_cnn \\
      --layer features.17 --target_class zebra --concept stripes \\
      --dataset_path ./dataset --concept_path ./concept \\
      --imbalance_class zebra --imbalance_ratio 0.1 \\
      --pretrain_epochs 30 --recalib_epochs 10
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
                        help="Path to pre-trained model weights (skip initial training if provided)")
    parser.add_argument("--pretrained", action="store_true",
                        help="Use ImageNet pretrained weights (only for standard models)")
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
    
    # Initial training hyperparameters
    parser.add_argument("--pretrain_epochs", type=int, default=30,
                        help="Number of epochs for initial training on imbalanced data")
    parser.add_argument("--pretrain_lr", type=float, default=1e-3,
                        help="Learning rate for initial training")
    
    # Recalibration hyperparameters
    parser.add_argument("--recalib_epochs", type=int, default=10,
                        help="Number of epochs for recalibration")
    parser.add_argument("--recalib_lr", type=float, default=1e-4,
                        help="Learning rate for recalibration")
    parser.add_argument("--lambda_align", type=float, default=0.5,
                        help="Weight for alignment loss (0-1)")
    
    # General training parameters
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for training")
    
    # CAV configuration
    parser.add_argument("--classifier_type", type=str, default="LinearSVC",
                        choices=["LinearSVC", "SGDClassifier", "LogisticRegression"],
                        help="Linear classifier type for CAV training")
    
    # Other settings
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--results_path", type=str, default="./results",
                        help="Path to save results")
    
    # Backward compatibility
    parser.add_argument("--epochs", type=int, default=None,
                        help="(Deprecated) Use --recalib_epochs instead")
    parser.add_argument("--learning_rate", type=float, default=None,
                        help="(Deprecated) Use --recalib_lr instead")
    
    args = parser.parse_args()
    
    # Handle deprecated arguments
    if args.epochs is not None:
        print("Warning: --epochs is deprecated. Use --recalib_epochs instead.")
        args.recalib_epochs = args.epochs
    if args.learning_rate is not None:
        print("Warning: --learning_rate is deprecated. Use --recalib_lr instead.")
        args.recalib_lr = args.learning_rate
    
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
