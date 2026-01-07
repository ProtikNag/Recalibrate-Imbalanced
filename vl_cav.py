"""
VL-CAV: VLM-Augmented Concept Activation Vectors.

Core implementation of:
- Traditional CAV computation
- VLM-augmented CAV computation
- Sensitivity score calculation
- TCAV score computation
- Layer recalibration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.model_selection import train_test_split
from scipy import stats

from models import ActivationExtractor, get_layer_output_dim
from vlm_encoder import VLMEncoder, ConceptEmbedding, CrossSpaceProjection


class CAVTrainer:
    """
    Traditional CAV (Concept Activation Vector) trainer.
    
    Trains a linear classifier to separate concept activations from random activations,
    then extracts the decision boundary normal as the CAV.
    """
    
    def __init__(self, classifier_type: str = "sgd"):
        """
        Initialize CAV trainer.
        
        Args:
            classifier_type: Type of classifier ("sgd" or "logistic")
        """
        self.classifier_type = classifier_type
        self.classifiers: Dict[str, object] = {}
        self.cavs: Dict[str, np.ndarray] = {}
        self.accuracies: Dict[str, float] = {}
        
    def train_cav(self, concept_activations: np.ndarray, 
                  random_activations: np.ndarray,
                  layer_name: str) -> Tuple[np.ndarray, float]:
        """
        Train a CAV for a specific layer.
        
        Args:
            concept_activations: Activations for concept examples [N_c, D]
            random_activations: Activations for random examples [N_r, D]
            layer_name: Name of the layer
            
        Returns:
            (CAV vector, classifier accuracy)
        """
        # Prepare data
        X = np.vstack([concept_activations, random_activations])
        y = np.hstack([np.ones(len(concept_activations)), 
                      np.zeros(len(random_activations))])
        
        # Split for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train classifier
        if self.classifier_type == "sgd":
            clf = SGDClassifier(
                loss='hinge', max_iter=1000, random_state=42, 
                class_weight='balanced'
            )
        else:
            clf = LogisticRegression(
                max_iter=1000, random_state=42, class_weight='balanced'
            )
            
        clf.fit(X_train, y_train)
        
        # Extract CAV (normal to decision boundary)
        cav = clf.coef_[0]
        cav = cav / np.linalg.norm(cav)  # Normalize
        
        # Compute accuracy
        accuracy = clf.score(X_val, y_val)
        
        # Store results
        self.classifiers[layer_name] = clf
        self.cavs[layer_name] = cav
        self.accuracies[layer_name] = accuracy
        
        return cav, accuracy
    
    def get_cav(self, layer_name: str) -> Optional[np.ndarray]:
        """Get trained CAV for a layer."""
        return self.cavs.get(layer_name, None)


class SensitivityComputer:
    """
    Computes sensitivity scores and TCAV scores.
    
    Sensitivity: S_{c,k}(x) = ∇_{a}f_k(x) · v_c
    TCAV: Fraction of class k samples with positive sensitivity
    """
    
    def __init__(self, model: nn.Module, device: str = "cuda"):
        """
        Initialize sensitivity computer.
        
        Args:
            model: The CNN model
            device: Device to run on
        """
        self.model = model
        self.device = device
        
    def compute_sensitivity(self, inputs: torch.Tensor, 
                            layer_name: str, 
                            cav: np.ndarray,
                            target_class: int) -> torch.Tensor:
        """
        Compute sensitivity scores for inputs.
        
        Args:
            inputs: Input images [N, C, H, W]
            layer_name: Layer to compute sensitivity at
            cav: Concept Activation Vector
            target_class: Class to compute sensitivity for
            
        Returns:
            Sensitivity scores [N]
        """
        self.model.eval()
        inputs = inputs.to(self.device)
        inputs.requires_grad_(True)
        
        # Get layer module
        layer_module = dict(self.model.named_modules())[layer_name]
        
        # Forward pass with gradient tracking
        activations = []
        def hook(module, input, output):
            activations.append(output)
        
        handle = layer_module.register_forward_hook(hook)
        outputs = self.model(inputs)
        handle.remove()
        
        # Get activations
        act = activations[0]
        
        # Compute gradient of class output w.r.t. activations
        class_outputs = outputs[:, target_class]
        
        # Compute gradients
        grads = torch.autograd.grad(
            class_outputs.sum(), act, create_graph=False
        )[0]
        
        # Flatten gradients
        grads_flat = grads.view(grads.size(0), -1)
        
        # Convert CAV to tensor
        cav_tensor = torch.tensor(cav, dtype=torch.float32, device=self.device)
        
        # If CAV dimension doesn't match, we need to handle it
        if grads_flat.size(1) != cav_tensor.size(0):
            # Average pool gradients to match CAV dimension
            if grads_flat.size(1) > cav_tensor.size(0):
                ratio = grads_flat.size(1) // cav_tensor.size(0)
                grads_flat = grads_flat.view(grads_flat.size(0), -1, ratio).mean(dim=-1)
            else:
                # Expand CAV
                ratio = cav_tensor.size(0) // grads_flat.size(1)
                cav_tensor = cav_tensor.view(-1, ratio).mean(dim=-1)
        
        # Compute directional derivative (sensitivity)
        sensitivity = torch.matmul(grads_flat, cav_tensor)
        
        return sensitivity.detach()
    
    def compute_tcav_score(self, inputs: torch.Tensor,
                           labels: torch.Tensor,
                           layer_name: str,
                           cav: np.ndarray,
                           target_class: int) -> float:
        """
        Compute TCAV score for a class.
        
        TCAV = |{x ∈ X_k : S_{c,k}(x) > 0}| / |X_k|
        
        Args:
            inputs: All inputs
            labels: All labels
            layer_name: Layer name
            cav: CAV vector
            target_class: Class to compute TCAV for
            
        Returns:
            TCAV score (0 to 1)
        """
        # Filter inputs for target class
        mask = labels == target_class
        class_inputs = inputs[mask]
        
        if len(class_inputs) == 0:
            return 0.0
        
        # Compute sensitivities
        sensitivities = self.compute_sensitivity(
            class_inputs, layer_name, cav, target_class
        )
        
        # TCAV score = fraction with positive sensitivity
        positive_count = (sensitivities > 0).sum().item()
        tcav_score = positive_count / len(sensitivities)
        
        return tcav_score


class VLCAVRecalibrator:
    """
    VL-CAV based layer recalibration.
    
    Fine-tunes bottleneck layers to align activations with VLM-based concept embeddings.
    """
    
    def __init__(self, model: nn.Module, 
                 vlm_encoder: VLMEncoder,
                 concept_embedding: ConceptEmbedding,
                 bottleneck_layers: List[str],
                 device: str = "cuda"):
        """
        Initialize recalibrator.
        
        Args:
            model: CNN model to recalibrate
            vlm_encoder: VLM encoder
            concept_embedding: Concept embedding manager
            bottleneck_layers: Layers to recalibrate
            device: Device to run on
        """
        self.model = model.to(device)
        self.vlm_encoder = vlm_encoder
        self.concept_embedding = concept_embedding
        self.bottleneck_layers = bottleneck_layers
        self.device = device
        
        # Create projection networks for each bottleneck layer
        self.projections: Dict[str, CrossSpaceProjection] = {}
        self._init_projections()
        
    def _init_projections(self):
        """Initialize projection networks for bottleneck layers."""
        vlm_dim = self.vlm_encoder.get_embedding_dim()
        
        for layer_name in self.bottleneck_layers:
            cnn_dim = get_layer_output_dim(
                self.model, layer_name, 
                input_size=(1, 3, 224, 224),
                device=self.device
            )
            projection = CrossSpaceProjection(cnn_dim, vlm_dim).to(self.device)
            self.projections[layer_name] = projection
            
    def compute_alignment_loss(self, activations: torch.Tensor,
                               labels: torch.Tensor,
                               layer_name: str) -> torch.Tensor:
        """
        Compute alignment loss for a batch.
        
        L_align = Σ_i (1 - cos(φ(a_i), s_{y_i}))
        
        Args:
            activations: Layer activations [N, D]
            labels: Class labels [N]
            layer_name: Layer name
            
        Returns:
            Alignment loss
        """
        projection = self.projections[layer_name]
        
        # Flatten activations
        act_flat = activations.view(activations.size(0), -1)
        
        # Project to VLM space
        projected = projection(act_flat)
        
        # Get concept embeddings for each sample's class
        loss = 0.0
        for i, label in enumerate(labels):
            label_idx = label.item()
            concept_emb = self.concept_embedding.get_concept_embedding(label_idx)
            
            if concept_emb is not None:
                # Cosine similarity loss
                cos_sim = F.cosine_similarity(
                    projected[i:i+1], 
                    concept_emb.unsqueeze(0).to(self.device), 
                    dim=-1
                )
                loss += (1 - cos_sim)
        
        return loss / len(labels)
    
    def recalibrate(self, train_loader: DataLoader,
                    val_loader: DataLoader,
                    epochs: int = 10,
                    lr: float = 1e-4,
                    lambda_cls: float = 0.4,
                    lambda_align: float = 0.6,
                    target_classes: Optional[List[int]] = None,
                    logger = None) -> Dict:
        """
        Perform VL-CAV based recalibration.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs
            lr: Learning rate
            lambda_cls: Weight for classification loss
            lambda_align: Weight for alignment loss
            target_classes: Classes to focus alignment on (None = all)
            logger: Optional logger
            
        Returns:
            Training history dict
        """
        # Freeze all layers except bottleneck layers
        for name, param in self.model.named_parameters():
            param.requires_grad = False
            for bn_layer in self.bottleneck_layers:
                if bn_layer in name:
                    param.requires_grad = True
                    break
        
        # Also train projection networks
        trainable_params = []
        for param in self.model.parameters():
            if param.requires_grad:
                trainable_params.append(param)
        for projection in self.projections.values():
            trainable_params.extend(projection.parameters())
        
        optimizer = torch.optim.Adam(trainable_params, lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        # Training history
        history = {
            'train_loss': [], 'train_cls_loss': [], 'train_align_loss': [],
            'val_loss': [], 'val_accuracy': []
        }
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            for projection in self.projections.values():
                projection.train()
            
            epoch_loss = 0.0
            epoch_cls_loss = 0.0
            epoch_align_loss = 0.0
            num_batches = 0
            
            for batch_idx, (inputs, labels) in enumerate(train_loader):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass with activation extraction
                extractor = ActivationExtractor(self.model, self.bottleneck_layers)
                outputs = self.model(inputs)
                activations = extractor.get_activations()
                extractor.remove_hooks()
                
                # Classification loss
                cls_loss = criterion(outputs, labels)
                
                # Alignment loss for bottleneck layers
                align_loss = 0.0
                for layer_name in self.bottleneck_layers:
                    if layer_name in activations:
                        act = activations[layer_name]
                        
                        # Filter for target classes if specified
                        if target_classes is not None:
                            mask = torch.zeros_like(labels, dtype=torch.bool)
                            for tc in target_classes:
                                mask |= (labels == tc)
                            if mask.sum() > 0:
                                align_loss += self.compute_alignment_loss(
                                    act[mask], labels[mask], layer_name
                                )
                        else:
                            align_loss += self.compute_alignment_loss(
                                act, labels, layer_name
                            )
                
                align_loss = align_loss / len(self.bottleneck_layers)
                
                # Combined loss
                total_loss = lambda_cls * cls_loss + lambda_align * align_loss
                
                total_loss.backward()
                optimizer.step()
                
                epoch_loss += total_loss.item()
                epoch_cls_loss += cls_loss.item()
                epoch_align_loss += align_loss.item() if isinstance(align_loss, torch.Tensor) else align_loss
                num_batches += 1
            
            # Record training metrics
            history['train_loss'].append(epoch_loss / num_batches)
            history['train_cls_loss'].append(epoch_cls_loss / num_batches)
            history['train_align_loss'].append(epoch_align_loss / num_batches)
            
            # Validation
            val_loss, val_acc = self._validate(val_loader, criterion)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_acc)
            
            if logger:
                logger.info(
                    f"Epoch {epoch+1}/{epochs} - "
                    f"Loss: {epoch_loss/num_batches:.4f} "
                    f"(cls: {epoch_cls_loss/num_batches:.4f}, "
                    f"align: {epoch_align_loss/num_batches:.4f}) - "
                    f"Val Acc: {val_acc:.4f}"
                )
        
        return history
    
    def _validate(self, val_loader: DataLoader, 
                  criterion: nn.Module) -> Tuple[float, float]:
        """Validate the model."""
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)
        
        return total_loss / len(val_loader), correct / total


class BottleneckDetector:
    """
    Detects bottleneck layers using sensitivity correlation analysis.
    
    Based on Algorithm 1 from the TCAV recalibration paper.
    """
    
    def __init__(self, model: nn.Module, device: str = "cuda"):
        """
        Initialize bottleneck detector.
        
        Args:
            model: The CNN model
            device: Device to run on
        """
        self.model = model
        self.device = device
        self.sensitivity_scores: Dict[str, Dict[int, np.ndarray]] = {}
        self.correlation_matrix: Optional[np.ndarray] = None
        
    def compute_layer_sensitivities(self, dataloader: DataLoader,
                                    layer_names: List[str],
                                    cav_trainer: CAVTrainer,
                                    num_classes: int) -> Dict[str, Dict[int, np.ndarray]]:
        """
        Compute sensitivity scores for all layers and classes.
        
        Args:
            dataloader: Data loader
            layer_names: List of layer names to analyze
            cav_trainer: Trained CAV trainer
            num_classes: Number of classes
            
        Returns:
            Dictionary mapping layer -> class -> sensitivity scores
        """
        sensitivity_computer = SensitivityComputer(self.model, self.device)
        
        # Collect all inputs and labels
        all_inputs = []
        all_labels = []
        for inputs, labels in dataloader:
            all_inputs.append(inputs)
            all_labels.append(labels)
        all_inputs = torch.cat(all_inputs)
        all_labels = torch.cat(all_labels)
        
        # Compute sensitivities for each layer and class
        self.sensitivity_scores = {}
        
        for layer_name in layer_names:
            self.sensitivity_scores[layer_name] = {}
            cav = cav_trainer.get_cav(layer_name)
            
            if cav is None:
                continue
            
            for class_idx in range(num_classes):
                mask = all_labels == class_idx
                if mask.sum() == 0:
                    continue
                
                class_inputs = all_inputs[mask]
                
                # Compute sensitivities in batches
                sensitivities = []
                batch_size = 32
                for i in range(0, len(class_inputs), batch_size):
                    batch = class_inputs[i:i+batch_size]
                    sens = sensitivity_computer.compute_sensitivity(
                        batch, layer_name, cav, class_idx
                    )
                    sensitivities.append(sens.cpu().numpy())
                
                self.sensitivity_scores[layer_name][class_idx] = np.concatenate(sensitivities)
        
        return self.sensitivity_scores
    
    def compute_correlation_matrix(self, layer_names: List[str],
                                   target_class: int) -> np.ndarray:
        """
        Compute correlation matrix between layers for a target class.
        
        Args:
            layer_names: List of layer names
            target_class: Class to analyze
            
        Returns:
            Correlation matrix [num_layers, num_layers]
        """
        n_layers = len(layer_names)
        corr_matrix = np.zeros((n_layers, n_layers))
        
        for i, layer_i in enumerate(layer_names):
            for j, layer_j in enumerate(layer_names):
                if i >= j:
                    continue
                
                if (layer_i in self.sensitivity_scores and 
                    layer_j in self.sensitivity_scores and
                    target_class in self.sensitivity_scores[layer_i] and
                    target_class in self.sensitivity_scores[layer_j]):
                    
                    sens_i = self.sensitivity_scores[layer_i][target_class]
                    sens_j = self.sensitivity_scores[layer_j][target_class]
                    
                    # Ensure same length
                    min_len = min(len(sens_i), len(sens_j))
                    if min_len > 1:
                        corr, _ = stats.pearsonr(sens_i[:min_len], sens_j[:min_len])
                        corr_matrix[i, j] = corr
                        corr_matrix[j, i] = corr
        
        np.fill_diagonal(corr_matrix, 1.0)
        self.correlation_matrix = corr_matrix
        
        return corr_matrix
    
    def identify_bottleneck_layers(self, layer_names: List[str],
                                   num_classes: int,
                                   correlation_threshold: float = 0.0) -> List[str]:
        """
        Identify bottleneck layers based on correlation analysis.
        
        A layer is a bottleneck if it has positive correlation with all
        subsequent layers for all target classes.
        
        Args:
            layer_names: List of layer names
            num_classes: Number of classes
            correlation_threshold: Minimum correlation required
            
        Returns:
            List of bottleneck layer names
        """
        bottleneck_layers = []
        
        for layer_idx, layer_name in enumerate(layer_names[:-1]):
            is_bottleneck = True
            
            for class_idx in range(num_classes):
                # Check correlation with all subsequent layers
                corr_matrix = self.compute_correlation_matrix(layer_names, class_idx)
                
                for j in range(layer_idx + 1, len(layer_names)):
                    if corr_matrix[layer_idx, j] <= correlation_threshold:
                        is_bottleneck = False
                        break
                
                if not is_bottleneck:
                    break
            
            if is_bottleneck:
                bottleneck_layers.append(layer_name)
        
        return bottleneck_layers
    
    def select_best_bottleneck(self, bottleneck_layers: List[str],
                               num_classes: int) -> str:
        """
        Select the best bottleneck layer based on average TCAV sensitivity.
        
        Args:
            bottleneck_layers: List of candidate bottleneck layers
            num_classes: Number of classes
            
        Returns:
            Name of the best bottleneck layer
        """
        if len(bottleneck_layers) == 0:
            raise ValueError("No bottleneck layers found")
        
        if len(bottleneck_layers) == 1:
            return bottleneck_layers[0]
        
        # Compute average sensitivity magnitude for each bottleneck
        avg_sensitivity = {}
        
        for layer_name in bottleneck_layers:
            if layer_name not in self.sensitivity_scores:
                continue
            
            total_sens = 0.0
            count = 0
            
            for class_idx in range(num_classes):
                if class_idx in self.sensitivity_scores[layer_name]:
                    sens = self.sensitivity_scores[layer_name][class_idx]
                    total_sens += np.abs(sens).mean()
                    count += 1
            
            if count > 0:
                avg_sensitivity[layer_name] = total_sens / count
        
        # Return layer with highest average sensitivity
        if avg_sensitivity:
            return max(avg_sensitivity, key=avg_sensitivity.get)
        else:
            return bottleneck_layers[-1]  # Default to deepest layer
