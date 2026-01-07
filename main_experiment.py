"""
Main experiment runner for VL-CAV recalibration.

Orchestrates the complete pipeline:
1. Dataset loading with imbalance
2. Model training (biased)
3. VLM concept embedding generation
4. Bottleneck layer detection
5. VL-CAV based recalibration
6. Evaluation and visualization
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
import os
import random
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

from config import ExperimentConfig, parse_args, get_class_concept_descriptions
from models import create_model, get_conv_layer_names, ActivationExtractor
from models import freeze_layers, get_trainable_params, get_total_params
from dataloader import load_dataset, create_dataloaders, get_class_samples, compute_class_weights
from vlm_encoder import VLMEncoder, ConceptEmbedding
from vl_cav import CAVTrainer, SensitivityComputer, VLCAVRecalibrator, BottleneckDetector
from logger import ExperimentLogger
from visualizations import (
    plot_training_curves, plot_recalibration_losses, plot_confusion_matrix,
    plot_confusion_matrix_comparison, plot_class_distribution, plot_per_class_accuracy,
    plot_tcav_scores, plot_correlation_matrix, plot_experiment_summary
)


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_epoch(model: nn.Module, train_loader: DataLoader, 
                criterion: nn.Module, optimizer: optim.Optimizer,
                device: str) -> Tuple[float, float]:
    """
    Train for one epoch.
    
    Returns:
        (loss, accuracy)
    """
    model.train()
    total_loss = 0.0
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
    
    return total_loss / len(train_loader), correct / total


def evaluate(model: nn.Module, test_loader: DataLoader, 
             device: str, num_classes: int) -> Dict:
    """
    Evaluate model on test set.
    
    Returns:
        Dictionary with accuracy, per-class accuracy, predictions, labels
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Overall accuracy
    accuracy = (all_preds == all_labels).mean()
    
    # Per-class accuracy
    per_class_acc = {}
    for c in range(num_classes):
        mask = all_labels == c
        if mask.sum() > 0:
            per_class_acc[c] = (all_preds[mask] == all_labels[mask]).mean()
        else:
            per_class_acc[c] = 0.0
    
    return {
        'accuracy': accuracy,
        'per_class_accuracy': per_class_acc,
        'predictions': all_preds,
        'labels': all_labels
    }


def compute_tcav_scores(model: nn.Module, test_loader: DataLoader,
                        cav_trainer: CAVTrainer, layer_name: str,
                        num_classes: int, device: str) -> Dict[int, float]:
    """
    Compute TCAV scores for all classes at a specific layer.
    """
    sensitivity_computer = SensitivityComputer(model, device)
    
    # Collect all test data
    all_inputs = []
    all_labels = []
    for inputs, labels in test_loader:
        all_inputs.append(inputs)
        all_labels.append(labels)
    all_inputs = torch.cat(all_inputs)
    all_labels = torch.cat(all_labels)
    
    tcav_scores = {}
    cav = cav_trainer.get_cav(layer_name)
    
    if cav is None:
        return {c: 0.5 for c in range(num_classes)}
    
    for class_idx in range(num_classes):
        try:
            score = sensitivity_computer.compute_tcav_score(
                all_inputs, all_labels, layer_name, cav, class_idx
            )
            tcav_scores[class_idx] = score
        except Exception as e:
            tcav_scores[class_idx] = 0.5
    
    return tcav_scores


def run_experiment(config: ExperimentConfig):
    """
    Run the complete VL-CAV experiment.
    """
    # Initialize logger
    logger = ExperimentLogger(
        config.experiment_output_dir, 
        config.experiment_name
    )
    
    # Log configuration
    logger.log_config(config.to_dict())
    
    # Set seed
    set_seed(config.seed)
    logger.info(f"Random seed set to: {config.seed}")
    
    # Check device
    if config.device == "cuda" and not torch.cuda.is_available():
        config.device = "cpu"
        logger.warning("CUDA not available, using CPU")
    device = config.device
    logger.info(f"Using device: {device}")
    
    # ==================== Phase 1: Data Loading ====================
    logger.start_phase("Data Loading")
    
    train_dataset, test_dataset, dataset_info = load_dataset(
        dataset_name=config.dataset_name,
        root=config.dataset_path,
        image_size=config.image_size,
        imbalance_classes=config.imbalance_classes,
        imbalance_ratio=config.imbalance_ratio,
        seed=config.seed
    )
    
    config.num_classes = dataset_info['num_classes']
    class_names = dataset_info['class_names']
    
    logger.info(f"Dataset: {config.dataset_name}")
    logger.info(f"Number of classes: {config.num_classes}")
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Test samples: {len(test_dataset)}")
    logger.info(f"Imbalance factor: {dataset_info.get('imbalance_factor', 1.0):.2f}")
    
    if dataset_info.get('imbalanced'):
        logger.info(f"Imbalanced classes: {config.imbalance_classes}")
        logger.info(f"Imbalance ratio: {config.imbalance_ratio}")
    
    train_loader, test_loader = create_dataloaders(
        train_dataset, test_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers
    )
    
    # Plot class distribution
    if config.save_visualizations:
        plot_class_distribution(
            dataset_info['class_counts'],
            class_names,
            os.path.join(config.experiment_output_dir, "class_distribution"),
            imbalance_classes=config.imbalance_classes,
            formats=config.visualization_formats
        )
    
    logger.end_phase("Data Loading")
    
    # ==================== Phase 2: Model Training ====================
    logger.start_phase("Biased Model Training")
    
    model = create_model(
        config.model_name,
        config.num_classes,
        pretrained=config.pretrained
    ).to(device)
    
    logger.info(f"Model: {config.model_name}")
    logger.info(f"Total parameters: {get_total_params(model):,}")
    logger.info(f"Trainable parameters: {get_trainable_params(model):,}")
    
    # Get class weights for weighted loss
    class_weights = compute_class_weights(train_dataset).to(device)
    criterion = nn.CrossEntropyLoss()  # Use unweighted to simulate real imbalance
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate,
                           weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, config.pretrain_epochs)
    
    train_history = {'train_loss': [], 'train_accuracy': [], 
                     'val_loss': [], 'val_accuracy': []}
    
    for epoch in range(config.pretrain_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, 
                                            optimizer, device)
        
        # Evaluate
        eval_results = evaluate(model, test_loader, device, config.num_classes)
        val_acc = eval_results['accuracy']
        
        train_history['train_loss'].append(train_loss)
        train_history['train_accuracy'].append(train_acc)
        train_history['val_accuracy'].append(val_acc)
        
        scheduler.step()
        
        if (epoch + 1) % 5 == 0 or epoch == config.pretrain_epochs - 1:
            logger.info(f"Epoch {epoch+1}/{config.pretrain_epochs} - "
                       f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                       f"Val Acc: {val_acc:.4f}")
    
    # Evaluate before recalibration
    results_before = evaluate(model, test_loader, device, config.num_classes)
    logger.info(f"Accuracy before recalibration: {results_before['accuracy']:.4f}")
    logger.log_dict(results_before['per_class_accuracy'], "Per-class accuracy before")
    
    # Save biased model
    if config.save_models:
        logger.save_model(model, "model_biased")
    
    # Plot training curves
    if config.save_visualizations:
        plot_training_curves(
            train_history,
            os.path.join(config.experiment_output_dir, "training_curves"),
            title="Initial Model Training",
            formats=config.visualization_formats
        )
    
    logger.end_phase("Biased Model Training")
    
    # ==================== Phase 3: VLM Concept Embedding ====================
    logger.start_phase("VLM Concept Embedding")
    
    vlm_encoder = VLMEncoder(config.vlm_model, device)
    concept_embedding = ConceptEmbedding(vlm_encoder, alpha=config.alpha)
    
    logger.info(f"VLM model: {config.vlm_model}")
    logger.info(f"VLM embedding dimension: {vlm_encoder.get_embedding_dim()}")
    logger.info(f"Alpha (text/vision balance): {config.alpha}")
    
    # Generate concept embeddings for each class
    for class_idx in range(config.num_classes):
        class_name = class_names[class_idx] if class_idx < len(class_names) else f"class_{class_idx}"
        
        # Get text descriptions
        descriptions = get_class_concept_descriptions(
            config.dataset_name, class_idx, class_name
        )
        
        # Get visual examples
        images, _ = get_class_samples(train_dataset, class_idx, num_samples=50)
        
        # Compute unified embedding
        concept_embedding.compute_unified_embedding(class_idx, descriptions, images)
        
        logger.debug(f"Computed concept embedding for class {class_idx} ({class_name})")
    
    logger.info(f"Generated concept embeddings for {config.num_classes} classes")
    
    logger.end_phase("VLM Concept Embedding")
    
    # ==================== Phase 4: Bottleneck Detection ====================
    logger.start_phase("Bottleneck Detection")
    
    layer_names = get_conv_layer_names(model, config.model_name)
    logger.info(f"Analyzing {len(layer_names)} layers for bottleneck detection")
    
    # Train CAVs for each layer
    cav_trainer = CAVTrainer(classifier_type="sgd")
    
    # Get concept and random activations
    # Use samples from imbalanced classes if specified, else use all
    target_classes = config.imbalance_classes if config.imbalance_classes else list(range(config.num_classes))
    
    for layer_name in layer_names[:10]:  # Analyze first 10 layers for efficiency
        try:
            extractor = ActivationExtractor(model, [layer_name])
            
            # Collect concept activations (from target classes)
            concept_acts = []
            for class_idx in target_classes:
                images, _ = get_class_samples(train_dataset, class_idx, num_samples=30)
                with torch.no_grad():
                    _ = model(images.to(device))
                acts = extractor.get_activations()[layer_name]
                concept_acts.append(acts.view(acts.size(0), -1).cpu().numpy())
                extractor.clear()
            concept_acts = np.vstack(concept_acts)
            
            # Collect random activations (from other classes)
            random_acts = []
            other_classes = [c for c in range(config.num_classes) if c not in target_classes]
            if not other_classes:
                other_classes = list(range(config.num_classes))
            for class_idx in other_classes[:3]:
                images, _ = get_class_samples(train_dataset, class_idx, num_samples=30)
                with torch.no_grad():
                    _ = model(images.to(device))
                acts = extractor.get_activations()[layer_name]
                random_acts.append(acts.view(acts.size(0), -1).cpu().numpy())
                extractor.clear()
            random_acts = np.vstack(random_acts)
            
            extractor.remove_hooks()
            
            # Train CAV
            cav, accuracy = cav_trainer.train_cav(concept_acts, random_acts, layer_name)
            logger.debug(f"Layer {layer_name}: CAV accuracy = {accuracy:.4f}")
            
        except Exception as e:
            logger.warning(f"Failed to train CAV for layer {layer_name}: {e}")
    
    # Identify bottleneck layers
    bottleneck_detector = BottleneckDetector(model, device)
    
    # For simplicity, select top layers with highest CAV accuracy
    cav_accuracies = cav_trainer.accuracies
    if cav_accuracies:
        sorted_layers = sorted(cav_accuracies.items(), key=lambda x: x[1], reverse=True)
        bottleneck_layers = [layer for layer, acc in sorted_layers[:3] if acc > 0.6]
        
        if not bottleneck_layers and sorted_layers:
            bottleneck_layers = [sorted_layers[0][0]]
    else:
        # Default to last conv layer
        bottleneck_layers = [layer_names[-1]] if layer_names else []
    
    logger.info(f"Identified bottleneck layers: {bottleneck_layers}")
    
    # Compute TCAV scores before recalibration
    tcav_before = {}
    if bottleneck_layers:
        tcav_before = compute_tcav_scores(
            model, test_loader, cav_trainer, bottleneck_layers[0],
            config.num_classes, device
        )
        logger.log_dict(tcav_before, "TCAV scores before recalibration")
    
    logger.end_phase("Bottleneck Detection")
    
    # ==================== Phase 5: VL-CAV Recalibration ====================
    logger.start_phase("VL-CAV Recalibration")
    
    if not bottleneck_layers:
        logger.warning("No bottleneck layers found, skipping recalibration")
        recalib_history = {}
    else:
        recalibrator = VLCAVRecalibrator(
            model, vlm_encoder, concept_embedding,
            bottleneck_layers, device
        )
        
        recalib_history = recalibrator.recalibrate(
            train_loader, test_loader,
            epochs=config.recalib_epochs,
            lr=config.recalib_lr,
            lambda_cls=config.lambda_cls,
            lambda_align=config.lambda_align,
            target_classes=config.imbalance_classes if config.imbalance_classes else None,
            logger=logger
        )
        
        # Plot recalibration losses
        if config.save_visualizations:
            plot_recalibration_losses(
                recalib_history,
                os.path.join(config.experiment_output_dir, "recalibration_losses"),
                formats=config.visualization_formats
            )
    
    logger.end_phase("VL-CAV Recalibration")
    
    # ==================== Phase 6: Evaluation ====================
    logger.start_phase("Final Evaluation")
    
    # Evaluate after recalibration
    results_after = evaluate(model, test_loader, device, config.num_classes)
    logger.info(f"Accuracy after recalibration: {results_after['accuracy']:.4f}")
    logger.log_dict(results_after['per_class_accuracy'], "Per-class accuracy after")
    
    # Compute improvement
    improvement = results_after['accuracy'] - results_before['accuracy']
    logger.info(f"Overall improvement: {improvement:+.4f}")
    
    # Compute TCAV scores after recalibration
    tcav_after = {}
    if bottleneck_layers:
        tcav_after = compute_tcav_scores(
            model, test_loader, cav_trainer, bottleneck_layers[0],
            config.num_classes, device
        )
        logger.log_dict(tcav_after, "TCAV scores after recalibration")
    
    # Per-class improvement
    logger.log_separator()
    logger.info("Per-class improvements:")
    for c in range(config.num_classes):
        before = results_before['per_class_accuracy'].get(c, 0)
        after = results_after['per_class_accuracy'].get(c, 0)
        diff = after - before
        class_name = class_names[c] if c < len(class_names) else f"Class {c}"
        marker = "★" if c in config.imbalance_classes else ""
        logger.info(f"  {class_name} {marker}: {before:.4f} -> {after:.4f} ({diff:+.4f})")
    
    # Save recalibrated model
    if config.save_models:
        logger.save_model(model, "model_recalibrated")
    
    logger.end_phase("Final Evaluation")
    
    # ==================== Phase 7: Visualization ====================
    if config.save_visualizations:
        logger.start_phase("Visualization Generation")
        
        # Confusion matrices
        plot_confusion_matrix_comparison(
            results_before['labels'],
            results_before['predictions'],
            results_after['predictions'],
            class_names,
            os.path.join(config.experiment_output_dir, "confusion_matrices"),
            formats=config.visualization_formats
        )
        
        # Per-class accuracy comparison
        plot_per_class_accuracy(
            results_before['per_class_accuracy'],
            results_after['per_class_accuracy'],
            class_names,
            os.path.join(config.experiment_output_dir, "per_class_accuracy"),
            imbalance_classes=config.imbalance_classes,
            formats=config.visualization_formats
        )
        
        # TCAV scores
        if tcav_before and tcav_after:
            plot_tcav_scores(
                tcav_before, tcav_after, class_names,
                os.path.join(config.experiment_output_dir, "tcav_scores"),
                formats=config.visualization_formats
            )
        
        # Experiment summary
        summary_results = {
            'model_name': config.model_name,
            'dataset_name': config.dataset_name,
            'imbalance_ratio': config.imbalance_ratio,
            'alpha': config.alpha,
            'bottleneck_layers': bottleneck_layers,
            'accuracy_before': results_before['accuracy'],
            'accuracy_after': results_after['accuracy'],
            'per_class_acc_before': results_before['per_class_accuracy'],
            'per_class_acc_after': results_after['per_class_accuracy'],
            'tcav_before': tcav_before,
            'tcav_after': tcav_after,
            'train_history': recalib_history if recalib_history else train_history,
            'class_names': class_names,
        }
        
        if config.imbalance_classes:
            # Average accuracy for imbalanced classes
            imb_acc_before = np.mean([results_before['per_class_accuracy'][c] 
                                      for c in config.imbalance_classes])
            imb_acc_after = np.mean([results_after['per_class_accuracy'][c] 
                                     for c in config.imbalance_classes])
            summary_results['imbalance_class_acc_before'] = imb_acc_before
            summary_results['imbalance_class_acc_after'] = imb_acc_after
        
        plot_experiment_summary(
            summary_results,
            os.path.join(config.experiment_output_dir, "experiment_summary"),
            formats=config.visualization_formats
        )
        
        logger.end_phase("Visualization Generation")
    
    # ==================== Save Results ====================
    final_results = {
        'accuracy_before': results_before['accuracy'],
        'accuracy_after': results_after['accuracy'],
        'improvement': improvement,
        'per_class_accuracy_before': results_before['per_class_accuracy'],
        'per_class_accuracy_after': results_after['per_class_accuracy'],
        'tcav_before': tcav_before,
        'tcav_after': tcav_after,
        'bottleneck_layers': bottleneck_layers,
        'class_names': class_names,
        'config': config.to_dict()
    }
    
    logger.save_results(final_results)
    logger.finalize()
    
    return final_results


def main():
    """Main entry point."""
    config = parse_args()
    results = run_experiment(config)
    
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETED")
    print("="*60)
    print(f"Accuracy: {results['accuracy_before']:.4f} -> {results['accuracy_after']:.4f}")
    print(f"Improvement: {results['improvement']:+.4f}")
    print(f"Results saved to: {config.experiment_output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
