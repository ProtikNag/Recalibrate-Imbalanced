#!/usr/bin/env python3
"""
TCAV-Based Recalibration for Imbalanced Datasets - Caltech-101 Version

This version uses:
- Caltech-101 dataset with configurable classes
- Automatic concept extraction using DeepLabV3 segmentation
- Support for N classes (not limited to 3)
- Subject-based concepts (automatic segmentation)

Workflow:
1. Download/prepare Caltech-101 vehicle classes
2. Generate concepts using DeepLabV3 segmentation (if not exists)
3. Train model on imbalanced dataset (creates bias)
4. Evaluate biased model (BEFORE recalibration)
5. Apply TCAV-based recalibration
6. Evaluate recalibrated model (AFTER recalibration)
7. Compare results and generate visualizations

Usage:
    python main_experiment.py --experiment 3 \\
        --model_name custom_cnn \\
        --data_root ./data \\
        --concept_path ./concepts \\
        --classes "airplanes,Motorbikes,car_side,ferry,helicopter" \\
        --class_concept_map "airplanes:subject,Motorbikes:subject,car_side:subject,ferry:subject,helicopter:subject" \\
        --imbalance_class airplanes \\
        --imbalance_ratio 0.1 \\
        --pretrain_epochs 50 \\
        --recalib_epochs 20
"""

import os
import sys
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

from utils import (
    load_model,
    get_model_layers,
    get_suggested_layers,
    get_base_model_image_size,
    train_cav,
    compute_tcav_score,
    evaluate_detailed,
)
from dataloader_caltech import (
    Caltech101VehicleDataset,
    ConceptDataset,
    BackgroundDataset,
    SingleClassDataset,
    create_caltech_vehicle_dataset,
    create_concept_loaders,
    get_transforms,
    print_dataset_info,
    DEFAULT_VEHICLE_CLASSES,
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


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if DEVICE == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id: int):
    """Initialize worker with deterministic seed."""
    worker_seed = 42 + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_activation(layer_name: str):
    """Create a forward hook to capture layer activations."""
    def hook(model, input, output):
        activation[layer_name] = output
        output_shape[layer_name] = output.shape
    return hook


def compute_cav(model, concept_loader, random_loader, layer_name,
                classifier_type: str = 'LinearSVC', logger=None):
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


def parse_class_list(class_string: str) -> list:
    """Parse comma-separated class list."""
    if not class_string:
        return DEFAULT_VEHICLE_CLASSES
    return [c.strip() for c in class_string.split(',')]


def parse_class_concept_map(map_string: str) -> dict:
    """
    Parse class-concept mapping string.
    Format: "class1:concept1,class2:concept2,..."
    """
    if not map_string:
        return {}
    
    mapping = {}
    pairs = map_string.split(',')
    for pair in pairs:
        pair = pair.strip()
        if ':' in pair:
            class_name, concept_name = pair.split(':', 1)
            mapping[class_name.strip()] = concept_name.strip()
    
    return mapping


def check_and_generate_concepts(concept_path: str, data_root: str, 
                               class_names: list, logger) -> bool:
    """Check if concepts exist, generate if not."""
    background_path = os.path.join(concept_path, 'background')
    
    # Check if concepts already exist
    concepts_exist = os.path.isdir(background_path) and len(os.listdir(background_path)) > 0
    
    for class_name in class_names:
        subject_path = os.path.join(concept_path, class_name, 'subject')
        if not os.path.isdir(subject_path) or len(os.listdir(subject_path)) == 0:
            concepts_exist = False
            break
    
    if concepts_exist:
        logger.log_info("Concept images found, skipping generation")
        return True
    
    logger.log_info("Concept images not found, generating using DeepLabV3...")
    
    try:
        from concept_generator import ConceptGenerator
        
        # Get Caltech-101 image directory
        caltech_img_dir = os.path.join(data_root, 'caltech101', '101_ObjectCategories')
        
        if not os.path.isdir(caltech_img_dir):
            # Try to trigger download
            logger.log_info("Caltech-101 not found, downloading...")
            from torchvision.datasets import Caltech101
            Caltech101(root=data_root, download=True)
        
        generator = ConceptGenerator(device=DEVICE)
        stats = generator.generate_all_concepts(
            dataset_dir=caltech_img_dir,
            concept_output_dir=concept_path,
            class_names=class_names,
            max_images_per_class=200  # Limit for faster generation
        )
        
        logger.log_info("Concept generation complete:")
        for class_name, class_stats in stats.items():
            logger.log_info(f"  {class_name}: {class_stats['subjects_extracted']} subjects")
        
        return True
        
    except Exception as e:
        logger.log_error(f"Failed to generate concepts: {e}")
        logger.log_error("Please run concept_generator.py manually first")
        return False


class InitialTrainer:
    """
    Initial training phase: Train the model on imbalanced data.
    Creates a biased model that we will later try to fix with recalibration.
    """
    
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
    
    def train(self, model, train_loader, val_loader, epochs: int = 10, lr: float = 1e-4):
        """Train model from scratch on imbalanced data."""
        model = model.to(DEVICE)
        model.train()
        
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3,
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
            
            scheduler.step(avg_val_loss)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = copy.deepcopy(model.state_dict())
            
            history["train_loss"].append(avg_train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(avg_val_loss)
            history["val_acc"].append(val_acc)
            
            self.logger.log_info(
                f"Epoch {epoch + 1:3d}/{epochs} | "
                f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}"
            )
        
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            self.logger.log_info(f"Loaded best model with validation accuracy: {best_val_acc:.4f}")
        
        return model, history


class Experiment3Runner:
    """
    Experiment 3: Joint optimization with multiple concepts for different classes.
    
    Features:
    - Automatic layer selection based on TCAV scores
    - Different concept for each class
    - Ensures no layer is used for multiple classes
    - Joint optimization of all alignment losses
    """
    
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.model = None
        self.class_concept_map = {}
        self.class_layer_map = {}
        self.class_cav_map = {}
        self.class_idx_map = {}
        self.hooks = []
    
    def compute_tcav_scores_for_layers(self, model, layers, cav_vectors,
                                       target_loader, target_idx):
        """Compute TCAV scores for multiple layers."""
        scores = {}
        
        for layer_name in layers:
            if layer_name not in cav_vectors:
                continue
            
            cav = cav_vectors[layer_name]
            score = compute_tcav_score(
                model, layer_name, cav, target_loader, target_idx, activation, DEVICE
            )
            scores[layer_name] = score
        
        return scores
    
    def select_best_layers(self, model, class_names, class_loaders, class_indices,
                          concept_loaders, random_loader, available_layers,
                          classifier_type: str = 'LinearSVC'):
        """
        Select the best layer for each class based on TCAV scores.
        Ensures no layer is used for multiple classes.
        """
        self.logger.log_info("Selecting best layers for each class...")
        
        all_cavs = {}
        all_scores = {}
        
        # Register hooks for all layers
        for layer_name in available_layers:
            try:
                model.get_submodule(layer_name).register_forward_hook(
                    get_activation(layer_name)
                )
            except Exception as e:
                self.logger.log_warning(f"Could not register hook for layer {layer_name}: {e}")
        
        # Compute CAVs and TCAV scores for all class-layer combinations
        for class_name in class_names:
            if class_name not in concept_loaders:
                self.logger.log_warning(f"No concept loader for class {class_name}, skipping")
                continue
            
            self.logger.log_info(f"Computing CAVs for class: {class_name}")
            
            concept_loader = concept_loaders[class_name]
            target_loader = class_loaders[class_name]
            target_idx = class_indices[class_name]
            
            for layer_name in available_layers:
                try:
                    cav = compute_cav(
                        model, concept_loader, random_loader, layer_name,
                        classifier_type=classifier_type, logger=None
                    )
                    all_cavs[(class_name, layer_name)] = cav
                    
                    score = compute_tcav_score(
                        model, layer_name, cav, target_loader, target_idx,
                        activation, DEVICE
                    )
                    all_scores[(class_name, layer_name)] = score
                    
                    self.logger.log_debug(f"  Layer {layer_name}: TCAV score = {score:.4f}")
                except Exception as e:
                    self.logger.log_warning(f"  Error computing CAV for {class_name}/{layer_name}: {e}")
        
        # Select layers greedily - prioritize classes with lowest best scores
        used_layers = set()
        class_layer_map = {}
        class_cav_map = {}
        
        class_best_scores = {}
        for class_name in class_names:
            if class_name not in concept_loaders:
                continue
            scores = [all_scores.get((class_name, l), 1.0) for l in available_layers]
            class_best_scores[class_name] = min(scores) if scores else 1.0
        
        sorted_classes = sorted(class_best_scores.keys(), key=lambda c: class_best_scores[c])
        
        self.logger.log_info("\nLayer selection (prioritizing classes with lowest TCAV scores):")
        
        for class_name in sorted_classes:
            best_layer = None
            best_score = float('inf')
            
            for layer_name in available_layers:
                if layer_name in used_layers:
                    continue
                
                score = all_scores.get((class_name, layer_name), float('inf'))
                if score < best_score:
                    best_score = score
                    best_layer = layer_name
            
            if best_layer is not None:
                class_layer_map[class_name] = best_layer
                class_cav_map[class_name] = all_cavs[(class_name, best_layer)]
                used_layers.add(best_layer)
                
                self.logger.log_info(f"  {class_name}: {best_layer} (TCAV score: {best_score:.4f})")
            else:
                self.logger.log_warning(f"  {class_name}: No available layer")
        
        return class_layer_map, class_cav_map
    
    def setup(self, model, class_names, class_indices, class_layer_map, class_cav_map):
        """Set up the experiment with model and per-class configurations."""
        self.model = copy.deepcopy(model).to(DEVICE)
        self.class_idx_map = class_indices
        self.class_layer_map = class_layer_map
        self.class_cav_map = class_cav_map
        
        unique_layers = set(class_layer_map.values())
        
        self.logger.log_info(f"Setting up Experiment 3 with {len(unique_layers)} layers")
        
        for layer_name in unique_layers:
            hook = self.model.get_submodule(layer_name).register_forward_hook(
                get_activation(layer_name)
            )
            self.hooks.append(hook)
        
        trainable_params = 0
        total_params = 0
        for name, param in self.model.named_parameters():
            total_params += param.numel()
            is_trainable = any(layer_name in name for layer_name in unique_layers)
            if is_trainable:
                param.requires_grad = True
                trainable_params += param.numel()
            else:
                param.requires_grad = False
        
        self.logger.log_info(f"Total parameters: {total_params:,}")
        self.logger.log_info(f"Trainable parameters (recalibration): {trainable_params:,}")
        
        self.model.apply(
            lambda m: m.eval() if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.Dropout)) else None
        )
    
    def train(self, full_loader, lambda_align: float = 0.5, epochs: int = 10, lr: float = 1e-4):
        """Recalibrate using full dataset with joint optimization."""
        lambda_cls = 1.0 - lambda_align
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr
        )
        
        loss_history = {"total": [], "cls": [], "align": [], "per_class_align": {}}
        
        for class_name in self.class_layer_map.keys():
            loss_history["per_class_align"][class_name] = []
        
        self.model.train()
        
        self.logger.log_info(f"[Experiment 3] Joint optimization with multiple concepts")
        self.logger.log_info(f"Lambda Align: {lambda_align}, Lambda Cls: {lambda_cls}")
        self.logger.log_info(f"Class-Layer assignments: {self.class_layer_map}")
        
        for epoch in range(epochs):
            total_loss_epoch = cls_loss_epoch = align_loss_epoch = 0.0
            per_class_align_epoch = {c: 0.0 for c in self.class_layer_map.keys()}
            n_batches = 0
            class_sample_counts = {c: 0 for c in self.class_layer_map.keys()}
            
            for imgs, labels in full_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                
                outputs = self.model(imgs)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                cls_loss = nn.CrossEntropyLoss()(outputs, labels)
                
                total_align_loss = torch.tensor(0.0, device=DEVICE)
                num_align_terms = 0
                
                for class_name, layer_name in self.class_layer_map.items():
                    class_idx = self.class_idx_map[class_name]
                    cav_vector = self.class_cav_map[class_name]
                    
                    mask = (labels == class_idx)
                    
                    if mask.any():
                        f_l = activation[layer_name].view(imgs.size(0), -1)
                        f_l_target = f_l[mask]
                        cosine_sim = F.cosine_similarity(
                            f_l_target,
                            cav_vector.unsqueeze(0),
                            dim=1
                        )
                        class_align_loss = (1 - torch.mean(torch.abs(cosine_sim)))
                        total_align_loss = total_align_loss + class_align_loss
                        num_align_terms += 1
                        
                        per_class_align_epoch[class_name] += class_align_loss.item()
                        class_sample_counts[class_name] += mask.sum().item()
                
                if num_align_terms > 0:
                    avg_align_loss = total_align_loss / num_align_terms
                else:
                    avg_align_loss = torch.tensor(0.0, device=DEVICE)
                
                loss = lambda_align * avg_align_loss + lambda_cls * cls_loss
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=7)
                optimizer.step()
                
                total_loss_epoch += loss.item()
                cls_loss_epoch += cls_loss.item()
                align_loss_epoch += avg_align_loss.item()
                n_batches += 1
            
            avg_total = total_loss_epoch / max(n_batches, 1)
            avg_cls = cls_loss_epoch / max(n_batches, 1)
            avg_align = align_loss_epoch / max(n_batches, 1)
            
            loss_history["total"].append(avg_total)
            loss_history["cls"].append(avg_cls)
            loss_history["align"].append(avg_align)
            
            for class_name in self.class_layer_map.keys():
                count = class_sample_counts[class_name]
                if count > 0:
                    loss_history["per_class_align"][class_name].append(
                        per_class_align_epoch[class_name] / (n_batches or 1)
                    )
                else:
                    loss_history["per_class_align"][class_name].append(0.0)
            
            self.logger.log_epoch(epoch + 1, epochs, avg_total, avg_cls, avg_align)
        
        return loss_history


def run_experiment(args):
    """Run the complete experiment pipeline."""
    set_seed(args.seed)
    
    # Parse class-concept map and derive class names from it
    class_concept_map = parse_class_concept_map(args.class_concept_map)
    
    if not class_concept_map:
        raise ValueError("class_concept_map is required and must contain at least one class:concept pair")
    
    # Classes are derived from the class_concept_map keys
    class_names = list(class_concept_map.keys())
    num_classes = len(class_names)
    
    # Set up results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(
        args.results_path,
        f"exp{args.experiment}_{args.model_name}_{num_classes}classes_{timestamp}"
    )
    os.makedirs(results_dir, exist_ok=True)
    
    # Initialize logger
    logger = ExperimentLogger(results_dir, f"experiment_{args.experiment}")
    logger.log_header("TCAV-Based Recalibration - Caltech-101 Vehicles")
    
    # Log configuration
    config = vars(args)
    config['class_names'] = class_names
    config['num_classes'] = num_classes
    logger.log_config(config)
    
    # Save config
    with open(os.path.join(results_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2, default=str)
    
    # Check/generate concepts
    logger.log_section("Concept Generation")
    if not check_and_generate_concepts(args.concept_path, args.data_root, class_names, logger):
        logger.log_error("Concept generation failed. Please generate concepts manually.")
        return None
    
    # Get image size
    image_size = get_base_model_image_size(args.model_name)
    
    # Create datasets
    logger.log_section("Data Loading")
    
    train_dataset, val_dataset = create_caltech_vehicle_dataset(
        root=args.data_root,
        class_names=class_names,
        imbalance_class=args.imbalance_class,
        imbalance_ratio=args.imbalance_ratio,
        image_size=image_size,
        train_split=0.8,
        seed=args.seed,
        download=True
    )
    
    print_dataset_info(train_dataset, "Training Set")
    print_dataset_info(val_dataset, "Validation Set")
    
    train_class_counts = train_dataset.get_class_counts()
    val_class_counts = val_dataset.get_class_counts()
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        worker_init_fn=worker_init_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Create concept loaders
    concept_loaders, random_loader = create_concept_loaders(
        concept_base_path=args.concept_path,
        class_names=class_names,
        image_size=image_size,
        batch_size=args.batch_size
    )
    
    logger.log_info(f"Concept loaders created for: {list(concept_loaders.keys())}")
    
    # Load model
    logger.log_section("Model Setup")
    model = load_model(
        args.model_name,
        num_classes=num_classes,
        pretrained=args.pretrained,
        input_size=image_size
    )
    logger.log_info(f"Loaded model: {args.model_name} with {num_classes} classes")
    
    # Initial training
    logger.log_section("Initial Training (Creating Biased Model)")
    initial_trainer = InitialTrainer(args, logger)
    
    if args.model_path:
        logger.log_info(f"Loading pre-trained weights from: {args.model_path}")
        model.load_state_dict(torch.load(args.model_path))
        initial_history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    else:
        model, initial_history = initial_trainer.train(
            model, train_loader, val_loader,
            epochs=args.pretrain_epochs,
            lr=args.pretrain_lr
        )
    
    # Save biased model
    biased_model_path = os.path.join(results_dir, "model_biased.pth")
    torch.save(model.state_dict(), biased_model_path)
    logger.log_info(f"Saved biased model to: {biased_model_path}")
    
    # Evaluate before recalibration
    logger.log_section("Evaluation BEFORE Recalibration")
    results_before = evaluate_detailed(model, val_loader, class_names, DEVICE)
    
    # Initialize visualizer
    visualizer = ResultVisualizer(results_dir)
    
    # Plot initial training
    if initial_history['train_loss']:
        visualizer.plot_initial_training(initial_history, len(initial_history['train_loss']))
    
    # Get available layers
    available_layers = get_suggested_layers(args.model_name)
    if not available_layers:
        available_layers = get_model_layers(model, layer_types=(nn.Conv2d,))
        available_layers = available_layers[-min(10, len(available_layers)):]
    
    logger.log_info(f"Available layers for recalibration: {available_layers}")
    
    # Create per-class loaders
    class_loaders = {}
    for class_name in class_names:
        class_dataset = SingleClassDataset(train_dataset, class_name)
        class_loaders[class_name] = DataLoader(
            class_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2
        )
    
    class_indices = train_dataset.class_to_idx
    
    # Run Experiment 3
    logger.log_section("Running Experiment 3: Joint Multi-Class Optimization")
    
    runner = Experiment3Runner(args, logger)
    
    # Select best layers
    class_layer_map, class_cav_map = runner.select_best_layers(
        model, class_names, class_loaders, class_indices,
        concept_loaders, random_loader, available_layers,
        classifier_type=args.classifier_type
    )
    
    # Compute TCAV scores before recalibration
    tcav_scores_before = {}
    for class_name in class_layer_map.keys():
        layer = class_layer_map[class_name]
        cav = class_cav_map[class_name]
        score = compute_tcav_score(
            model, layer, cav, class_loaders[class_name],
            class_indices[class_name], activation, DEVICE
        )
        tcav_scores_before[class_name] = score
    
    tcav_before = np.mean(list(tcav_scores_before.values()))
    logger.log_evaluation_results(results_before, "BEFORE", tcav_before)
    
    # Setup and train
    runner.setup(model, class_names, class_indices, class_layer_map, class_cav_map)
    
    loss_history = runner.train(
        train_loader,
        lambda_align=args.lambda_align,
        epochs=args.recalib_epochs,
        lr=args.recalib_lr
    )
    
    trained_model = runner.model
    
    # Compute TCAV scores after recalibration
    tcav_scores_after = {}
    for class_name in class_layer_map.keys():
        layer = class_layer_map[class_name]
        cav = class_cav_map[class_name]
        score = compute_tcav_score(
            trained_model, layer, cav, class_loaders[class_name],
            class_indices[class_name], activation, DEVICE
        )
        tcav_scores_after[class_name] = score
    
    tcav_after = np.mean(list(tcav_scores_after.values()))
    
    # Evaluate after recalibration
    logger.log_section("Evaluation AFTER Recalibration")
    results_after = evaluate_detailed(trained_model, val_loader, class_names, DEVICE)
    logger.log_evaluation_results(results_after, "AFTER", tcav_after)
    
    # Compare results
    logger.log_section("Results Comparison")
    logger.log_comparison(results_before, results_after, tcav_before, tcav_after, class_names)
    
    # Prepare all results
    all_results = {
        'experiment': args.experiment,
        'model_name': args.model_name,
        'layer': 'multiple',
        'target_class': 'multiple',
        'concept': class_concept_map,
        'class_layer_map': class_layer_map,
        'lambda_align': args.lambda_align,
        'epochs': args.recalib_epochs,
        'imbalance_class': args.imbalance_class,
        'imbalance_ratio': args.imbalance_ratio,
        'results_before': results_before,
        'results_after': results_after,
        'tcav_before': tcav_before,
        'tcav_after': tcav_after,
        'tcav_scores_before': tcav_scores_before,
        'tcav_scores_after': tcav_scores_after,
        'loss_history': loss_history,
        'train_class_counts': train_class_counts,
        'val_class_counts': val_class_counts,
        'class_names': class_names,
        'num_classes': num_classes
    }
    
    # Save detailed results
    with open(os.path.join(results_dir, 'detailed_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # Generate visualizations
    logger.log_section("Generating Visualizations")
    
    visualizer.plot_loss_curves(loss_history, args.recalib_epochs)
    logger.log_info("Generated loss curves")
    
    visualizer.plot_confusion_matrices(
        results_before['confusion_matrix'],
        results_after['confusion_matrix'],
        class_names
    )
    logger.log_info("Generated confusion matrices")
    
    visualizer.plot_per_class_comparison(
        results_before['per_class'],
        results_after['per_class'],
        class_names
    )
    logger.log_info("Generated per-class comparison")
    
    visualizer.plot_accuracy_change(
        results_before['per_class'],
        results_after['per_class'],
        class_names
    )
    logger.log_info("Generated accuracy change plot")
    
    visualizer.plot_metrics_comparison(results_before, results_after, tcav_before, tcav_after)
    logger.log_info("Generated metrics comparison")
    
    visualizer.plot_class_distribution(train_class_counts, val_class_counts)
    logger.log_info("Generated class distribution plot")
    
    visualizer.plot_misclassification_analysis(
        results_before['misclassification_matrix'],
        results_after['misclassification_matrix'],
        class_names
    )
    logger.log_info("Generated misclassification analysis")
    
    visualizer.plot_experiment3_tcav_comparison(
        tcav_scores_before, tcav_scores_after, class_layer_map
    )
    logger.log_info("Generated Experiment 3 TCAV comparison")
    
    visualizer.create_summary_dashboard(all_results)
    logger.log_info("Generated summary dashboard")
    
    # Save recalibrated model
    model_path = os.path.join(results_dir, f"model_recalibrated_exp{args.experiment}.pth")
    torch.save(trained_model.state_dict(), model_path)
    logger.log_info(f"Saved recalibrated model to: {model_path}")
    
    # Generate summary report
    logger.log_section("Experiment Summary")
    logger.log_summary(all_results)
    
    # Close logger
    logger.close()
    
    print(f"\n{'=' * 60}")
    print("EXPERIMENT COMPLETED")
    print(f"{'=' * 60}")
    print(f"Results saved to: {results_dir}")
    print(f"\nKey Results:")
    print(f"  Number of classes: {num_classes}")
    print(f"  Initial training epochs: {args.pretrain_epochs}")
    print(f"  Recalibration epochs: {args.recalib_epochs}")
    print(f"  Accuracy: {results_before['overall']['accuracy']:.4f} -> {results_after['overall']['accuracy']:.4f}")
    print(f"  TCAV Score: {tcav_before:.4f} -> {tcav_after:.4f}")
    
    print(f"\n  Per-class results:")
    for class_name in class_names:
        print(f"    {class_name}: "
              f"{results_before['per_class'][class_name]['accuracy']:.4f} -> "
              f"{results_after['per_class'][class_name]['accuracy']:.4f}")
    
    print(f"\n  Class-layer assignments: {class_layer_map}")
    print(f"{'=' * 60}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="TCAV-Based Recalibration for Caltech-101 Vehicles",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with vehicle classes (classes derived from class_concept_map)
  python main_experiment.py \\
      --experiment 3 \\
      --model_name custom_cnn \\
      --dataset_path ./data \\
      --concept_path ./concepts \\
      --class_concept_map "airplanes:subject,Motorbikes:subject,car_side:subject,ferry:subject,helicopter:subject" \\
      --imbalance_class airplanes \\
      --imbalance_ratio 0.1 \\
      --pretrain_epochs 50 \\
      --recalib_epochs 20
        """
    )
    
    # Experiment selection
    parser.add_argument("--experiment", type=int, default=3, choices=[1, 2, 3],
                       help="Experiment type (default: 3)")
    
    # Model configuration
    parser.add_argument("--model_name", type=str, default="custom_cnn",
                       choices=["custom_cnn", "custom_cnn_small", "custom_cnn_large",
                               "vgg16", "resnet50", "resnet18", "inception_v3",
                               "mobilenet_v3_small", "mobilenet_v3_large"],
                       help="CNN model architecture")
    parser.add_argument("--model_path", type=str, default=None,
                       help="Path to pre-trained model weights")
    parser.add_argument("--pretrained", action="store_true",
                       help="Use ImageNet pretrained weights")
    
    # Dataset configuration
    parser.add_argument("--dataset_path", type=str, default="./data",
                       help="Root directory for Caltech-101 data")
    parser.add_argument("--concept_path", type=str, default="./concepts",
                       help="Path to concept directory")
    
    # Class-concept mapping (classes are derived from this)
    parser.add_argument("--class_concept_map", type=str, required=True,
                       help="Class-concept mapping: 'class1:concept1,class2:concept2,...' "
                            "Classes are automatically derived from this mapping.")
    
    # Imbalance configuration
    parser.add_argument("--imbalance_class", type=str, default=None,
                       help="Class to make imbalanced")
    parser.add_argument("--imbalance_ratio", type=float, default=None,
                       help="Ratio of data to keep for imbalanced class (0.05-1.0)")
    
    # Training hyperparameters
    parser.add_argument("--pretrain_epochs", type=int, default=30,
                       help="Number of epochs for initial training")
    parser.add_argument("--pretrain_lr", type=float, default=1e-3,
                       help="Learning rate for initial training")
    parser.add_argument("--recalib_epochs", type=int, default=10,
                       help="Number of epochs for recalibration")
    parser.add_argument("--recalib_lr", type=float, default=1e-4,
                       help="Learning rate for recalibration")
    parser.add_argument("--lambda_align", type=float, default=0.5,
                       help="Weight for alignment loss (0-1)")
    parser.add_argument("--batch_size", type=int, default=32,
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
    
    args = parser.parse_args()
    
    # Derive classes from class_concept_map
    args.data_root = args.dataset_path  # Alias for internal use
    
    # Validation
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
