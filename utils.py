#!/usr/bin/env python3
"""
Utility functions for TCAV-Based Recalibration experiments.

Supports N-class classification with:
- Custom CNN models (untrained)
- Pretrained model loading with classifier adaptation
- Detailed evaluation metrics
- Confusion matrix computation
- Misclassification analysis
- CAV training and TCAV score computation
"""

import os
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from collections import Counter, defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# Import pre-trained models
from torchvision.models import (
    vgg16, VGG16_Weights,
    resnet50, ResNet50_Weights,
    resnet18, ResNet18_Weights,
    inception_v3, Inception_V3_Weights,
    mobilenet_v3_small, MobileNet_V3_Small_Weights,
    mobilenet_v3_large, MobileNet_V3_Large_Weights
)


# ============================================================================
# Custom CNN Models (Untrained) - Support for N classes
# ============================================================================

class CustomCNN(nn.Module):
    """
    Custom CNN model trained from scratch.
    A medium-sized architecture suitable for small to medium datasets.
    Supports any number of output classes.
    """
    
    def __init__(self, num_classes: int = 5, input_size: int = 224):
        super(CustomCNN, self).__init__()
        
        self.num_classes = num_classes
        self.input_size = input_size
        
        self.features = nn.Sequential(
            # Conv Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv Block 5
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # Calculate feature map size
        self.feature_size = self._get_feature_size(input_size)
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.feature_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _get_feature_size(self, input_size: int) -> int:
        """Calculate the size of flattened features."""
        with torch.no_grad():
            dummy = torch.zeros(1, 3, input_size, input_size)
            features = self.features(dummy)
            return features.view(1, -1).size(1)
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class CustomCNNSmall(nn.Module):
    """Smaller custom CNN for limited data scenarios."""
    
    def __init__(self, num_classes: int = 5, input_size: int = 224):
        super(CustomCNNSmall, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.feature_size = self._get_feature_size(input_size)
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.feature_size, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )
        
        self._initialize_weights()
    
    def _get_feature_size(self, input_size: int) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, 3, input_size, input_size)
            x = self.conv1(dummy)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv4(x)
            return x.view(1, -1).size(1)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class CustomCNNLarge(nn.Module):
    """Larger custom CNN for more complex datasets."""
    
    def __init__(self, num_classes: int = 5, input_size: int = 224):
        super(CustomCNNLarge, self).__init__()
        
        self.block1 = self._make_block(3, 64, 2)
        self.block2 = self._make_block(64, 128, 2)
        self.block3 = self._make_block(128, 256, 3)
        self.block4 = self._make_block(256, 512, 3)
        self.block5 = self._make_block(512, 512, 3)
        
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
        )
        
        self._initialize_weights()
    
    def _make_block(self, in_ch: int, out_ch: int, n_conv: int) -> nn.Sequential:
        layers = []
        layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ReLU(inplace=True))
        
        for _ in range(n_conv - 1):
            layers.append(nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU(inplace=True))
        
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# ============================================================================
# Model Loading
# ============================================================================

def load_model(model_name: str,
               num_classes: int = 5,
               pretrained: bool = False,
               input_size: int = 224) -> nn.Module:
    """
    Load a CNN model with specified number of output classes.
    
    Args:
        model_name: Name of the model architecture
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights (for standard models)
        input_size: Input image size
        
    Returns:
        PyTorch model
    """
    model_name = model_name.lower()
    
    # Custom models
    if model_name == 'custom_cnn':
        return CustomCNN(num_classes=num_classes, input_size=input_size)
    elif model_name == 'custom_cnn_small':
        return CustomCNNSmall(num_classes=num_classes, input_size=input_size)
    elif model_name == 'custom_cnn_large':
        return CustomCNNLarge(num_classes=num_classes, input_size=input_size)
    
    # Pretrained models
    elif model_name == 'vgg16':
        weights = VGG16_Weights.DEFAULT if pretrained else None
        model = vgg16(weights=weights)
        model.classifier[6] = nn.Linear(4096, num_classes)
        return model
    
    elif model_name == 'resnet50':
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        model = resnet50(weights=weights)
        model.fc = nn.Linear(2048, num_classes)
        return model
    
    elif model_name == 'resnet18':
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
        model.fc = nn.Linear(512, num_classes)
        return model
    
    elif model_name == 'inception_v3':
        weights = Inception_V3_Weights.DEFAULT if pretrained else None
        model = inception_v3(weights=weights, aux_logits=True)
        model.fc = nn.Linear(2048, num_classes)
        model.AuxLogits.fc = nn.Linear(768, num_classes)
        return model
    
    elif model_name == 'mobilenet_v3_small':
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = mobilenet_v3_small(weights=weights)
        model.classifier[3] = nn.Linear(1024, num_classes)
        return model
    
    elif model_name == 'mobilenet_v3_large':
        weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        model = mobilenet_v3_large(weights=weights)
        model.classifier[3] = nn.Linear(1280, num_classes)
        return model
    
    else:
        raise ValueError(f"Unknown model: {model_name}")


def get_model_layers(model: nn.Module,
                    layer_types: Tuple = (nn.Conv2d, nn.MaxPool2d)) -> List[str]:
    """
    Get names of all layers of specified types.
    
    Args:
        model: PyTorch model
        layer_types: Tuple of layer types to include
        
    Returns:
        List of layer names
    """
    layers = []
    for name, module in model.named_modules():
        if isinstance(module, layer_types):
            layers.append(name)
    return layers


def get_base_model_image_size(model_name: str) -> int:
    """Get the expected input size for a model."""
    if 'inception' in model_name.lower():
        return 299
    return 224


# ============================================================================
# Layer Suggestions
# ============================================================================

LAYER_SUGGESTIONS = {
    'custom_cnn': ['features.3', 'features.10', 'features.17', 'features.24', 'features.31'],
    'custom_cnn_small': ['conv1.0', 'conv2.0', 'conv3.0', 'conv4.0'],
    'custom_cnn_large': ['block1.0', 'block2.0', 'block3.0', 'block4.0', 'block5.0'],
    'vgg16': ['features.7', 'features.14', 'features.24', 'features.28'],
    'resnet50': ['layer1.2.conv3', 'layer2.3.conv3', 'layer3.5.conv3', 'layer4.2.conv3'],
    'resnet18': ['layer1.1.conv2', 'layer2.1.conv2', 'layer3.1.conv2', 'layer4.1.conv2'],
    'inception_v3': ['Mixed_5d.branch3x3dbl_3.conv', 'Mixed_6e.branch1x1.conv'],
    'mobilenet_v3_small': ['features.4.block.0.0', 'features.8.block.0.0'],
    'mobilenet_v3_large': ['features.7.block.0.0', 'features.12.block.0.0'],
}


def get_suggested_layers(model_name: str) -> List[str]:
    """Get suggested layers for a model."""
    return LAYER_SUGGESTIONS.get(model_name.lower(), [])


# ============================================================================
# CAV Training and TCAV Score Computation
# ============================================================================

def train_cav(concept_acts: np.ndarray,
              random_acts: np.ndarray,
              classifier_type: str = 'LinearSVC') -> np.ndarray:
    """
    Train a CAV (Concept Activation Vector) classifier.
    
    Args:
        concept_acts: Activations for concept images
        random_acts: Activations for random/background images
        classifier_type: Type of linear classifier
        
    Returns:
        CAV vector (normalized classifier weights)
    """
    # Prepare training data
    X = np.vstack([concept_acts, random_acts])
    y = np.array([1] * len(concept_acts) + [0] * len(random_acts))
    
    # Train classifier
    if classifier_type == 'LinearSVC':
        clf = LinearSVC(C=0.01, max_iter=5000, dual=False)
    elif classifier_type == 'SGDClassifier':
        clf = SGDClassifier(alpha=0.01, max_iter=5000, tol=1e-3)
    elif classifier_type == 'LogisticRegression':
        clf = LogisticRegression(C=0.01, max_iter=5000, solver='lbfgs')
    else:
        raise ValueError(f"Unknown classifier type: {classifier_type}")
    
    clf.fit(X, y)
    
    # Get CAV (normalized weight vector)
    cav = clf.coef_[0]
    cav = cav / (np.linalg.norm(cav) + 1e-7)
    
    return cav


def compute_tcav_score(model: nn.Module,
                       layer_name: str,
                       cav: torch.Tensor,
                       data_loader: torch.utils.data.DataLoader,
                       target_class_idx: int,
                       activation_dict: Dict,
                       device: str) -> float:
    """
    Compute TCAV score for a target class.
    
    Args:
        model: The model
        layer_name: Name of the layer to analyze
        cav: CAV vector
        data_loader: DataLoader for target class images
        target_class_idx: Index of the target class
        activation_dict: Dictionary to store activations (from hook)
        device: Device to run on
        
    Returns:
        TCAV score (0-1)
    """
    model.eval()
    positive_count = 0
    total_count = 0
    
    for batch in data_loader:
        if isinstance(batch, (list, tuple)):
            images = batch[0].to(device)
        else:
            images = batch.to(device)
        
        images.requires_grad = True
        
        outputs = model(images)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        
        # Get gradients w.r.t. target class
        target_scores = outputs[:, target_class_idx]
        
        for i in range(len(images)):
            model.zero_grad()
            target_scores[i].backward(retain_graph=True)
            
            # Get gradient of activations
            act = activation_dict[layer_name]
            grad = act.grad if act.grad is not None else torch.zeros_like(act)
            
            # Flatten and compute directional derivative
            grad_flat = grad[i].view(-1)
            
            # Handle dimension mismatch
            if grad_flat.shape[0] != cav.shape[0]:
                # Resize CAV to match gradient
                cav_resized = F.interpolate(
                    cav.unsqueeze(0).unsqueeze(0),
                    size=grad_flat.shape[0],
                    mode='linear',
                    align_corners=False
                ).squeeze()
            else:
                cav_resized = cav
            
            directional_deriv = torch.dot(grad_flat, cav_resized)
            
            if directional_deriv > 0:
                positive_count += 1
            total_count += 1
        
        # Clear gradients
        images.requires_grad = False
    
    return positive_count / max(total_count, 1)


# ============================================================================
# Evaluation Functions
# ============================================================================

def evaluate_detailed(model: nn.Module,
                      loader: torch.utils.data.DataLoader,
                      class_names: List[str],
                      device: str) -> Dict[str, Any]:
    """
    Comprehensive model evaluation with per-class metrics.
    
    Args:
        model: The model to evaluate
        loader: DataLoader for evaluation
        class_names: List of class names
        device: Device to run on
        
    Returns:
        Dictionary with:
        - overall: Overall accuracy, precision, recall, F1
        - per_class: Per-class metrics
        - confusion_matrix: Confusion matrix
        - misclassification_matrix: Which classes are confused
        - predictions: All predictions with confidences
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    all_confidences = []
    all_predictions_detail = []
    
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            probs = F.softmax(outputs, dim=1)
            confs, preds = torch.max(probs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_confidences.extend(confs.cpu().numpy())
            
            # Store detailed predictions
            for i in range(len(preds)):
                all_predictions_detail.append({
                    'true_label': labels[i].item(),
                    'pred_label': preds[i].item(),
                    'confidence': confs[i].item(),
                    'all_probs': probs[i].cpu().numpy().tolist()
                })
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_confidences = np.array(all_confidences)
    
    # Overall metrics
    overall = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, average='macro', zero_division=0),
        'recall': recall_score(all_labels, all_preds, average='macro', zero_division=0),
        'f1': f1_score(all_labels, all_preds, average='macro', zero_division=0),
        'avg_confidence': float(np.mean(all_confidences))
    }
    
    # Per-class metrics
    per_class = {}
    for idx, class_name in enumerate(class_names):
        mask_true = all_labels == idx
        
        tp = np.sum((all_preds == idx) & (all_labels == idx))
        fp = np.sum((all_preds == idx) & (all_labels != idx))
        fn = np.sum((all_preds != idx) & (all_labels == idx))
        tn = np.sum((all_preds != idx) & (all_labels != idx))
        
        total = np.sum(mask_true)
        correct = tp
        wrong = fn
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = correct / total if total > 0 else 0
        
        # Confidence analysis
        class_mask = all_labels == idx
        class_confidences = all_confidences[class_mask]
        avg_conf = float(np.mean(class_confidences)) if len(class_confidences) > 0 else 0
        
        correct_mask = (all_labels == idx) & (all_preds == idx)
        wrong_mask = (all_labels == idx) & (all_preds != idx)
        
        conf_when_correct = float(np.mean(all_confidences[correct_mask])) if np.sum(correct_mask) > 0 else 0
        conf_when_wrong = float(np.mean(all_confidences[wrong_mask])) if np.sum(wrong_mask) > 0 else 0
        
        per_class[class_name] = {
            'total': int(total),
            'correct': int(correct),
            'wrong': int(wrong),
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'true_positives': int(tp),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_negatives': int(tn),
            'avg_confidence': avg_conf,
            'conf_when_correct': conf_when_correct,
            'conf_when_wrong': conf_when_wrong
        }
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Misclassification analysis
    misclass_matrix = defaultdict(lambda: defaultdict(int))
    for true_label, pred_label in zip(all_labels, all_preds):
        if true_label != pred_label:
            true_name = class_names[true_label]
            pred_name = class_names[pred_label]
            misclass_matrix[true_name][pred_name] += 1
    
    misclass_matrix = {k: dict(v) for k, v in misclass_matrix.items()}
    
    # Top misclassifications
    misclass_pairs = []
    for true_class, pred_dict in misclass_matrix.items():
        for pred_class, count in pred_dict.items():
            misclass_pairs.append({
                'true_class': true_class,
                'predicted_as': pred_class,
                'count': count
            })
    misclass_pairs = sorted(misclass_pairs, key=lambda x: x['count'], reverse=True)
    
    return {
        'overall': overall,
        'per_class': per_class,
        'confusion_matrix': cm.tolist(),
        'misclassification_matrix': misclass_matrix,
        'top_misclassifications': misclass_pairs[:10],
        'predictions_detail': all_predictions_detail
    }


def compute_confusion_matrix(model: nn.Module,
                            loader: torch.utils.data.DataLoader,
                            class_names: List[str],
                            device: str) -> np.ndarray:
    """Compute confusion matrix."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return confusion_matrix(all_labels, all_preds)


def get_class_distribution(dataset) -> Dict[int, int]:
    """Get class distribution from dataset."""
    labels = []
    for i in range(len(dataset)):
        _, label = dataset[i]
        labels.append(label)
    return dict(Counter(labels))
