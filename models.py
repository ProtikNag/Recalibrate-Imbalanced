"""
CNN model definitions with layer extraction support for VL-CAV.

Provides various CNN architectures with methods to:
- Extract intermediate layer activations
- Get layer names for bottleneck detection
- Freeze/unfreeze specific layers
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Dict, List, Optional, Tuple, Callable
from collections import OrderedDict


class ActivationExtractor:
    """
    Utility class to extract activations from intermediate layers.
    
    Usage:
        extractor = ActivationExtractor(model, ['layer1', 'layer2'])
        output = model(x)
        activations = extractor.get_activations()
    """
    
    def __init__(self, model: nn.Module, layer_names: List[str]):
        """
        Args:
            model: PyTorch model
            layer_names: List of layer names to extract activations from
        """
        self.model = model
        self.layer_names = layer_names
        self.activations: Dict[str, torch.Tensor] = {}
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._register_hooks()
        
    def _register_hooks(self):
        """Register forward hooks on specified layers."""
        for name, module in self.model.named_modules():
            if name in self.layer_names:
                hook = module.register_forward_hook(self._get_hook(name))
                self.hooks.append(hook)
                
    def _get_hook(self, name: str) -> Callable:
        """Create hook function for a specific layer."""
        def hook(module, input, output):
            # Handle different output types
            if isinstance(output, torch.Tensor):
                self.activations[name] = output.detach()
            elif isinstance(output, tuple):
                self.activations[name] = output[0].detach()
        return hook
    
    def get_activations(self) -> Dict[str, torch.Tensor]:
        """Return collected activations."""
        return self.activations
    
    def clear(self):
        """Clear stored activations."""
        self.activations = {}
        
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


def get_conv_layer_names(model: nn.Module, model_name: str) -> List[str]:
    """
    Get names of convolutional layers suitable for CAV analysis.
    
    Args:
        model: The model instance
        model_name: Name of the model architecture
        
    Returns:
        List of layer names that can be used for CAV
    """
    layer_names = []
    
    if model_name.startswith("resnet"):
        # ResNet layers
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d) and 'downsample' not in name:
                layer_names.append(name)
            elif name in ['layer1', 'layer2', 'layer3', 'layer4']:
                layer_names.append(name)
                
    elif model_name.startswith("vgg"):
        # VGG layers
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                layer_names.append(name)
                
    elif model_name.startswith("mobilenet"):
        # MobileNet layers
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d) or 'block' in name.lower():
                if isinstance(module, nn.Sequential) or isinstance(module, nn.Conv2d):
                    layer_names.append(name)
                    
    elif model_name == "custom_cnn":
        # Custom CNN layers
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                layer_names.append(name)
    
    return layer_names


def get_layer_output_dim(model: nn.Module, layer_name: str, 
                         input_size: Tuple[int, int, int, int] = (1, 3, 224, 224),
                         device: str = "cpu") -> int:
    """
    Get the output dimension of a specific layer.
    
    Args:
        model: The model
        layer_name: Name of the layer
        input_size: Input tensor size (B, C, H, W)
        device: Device to run on
        
    Returns:
        Flattened output dimension of the layer
    """
    model = model.to(device)
    model.eval()
    
    extractor = ActivationExtractor(model, [layer_name])
    
    with torch.no_grad():
        x = torch.randn(input_size).to(device)
        _ = model(x)
        activations = extractor.get_activations()
        
    extractor.remove_hooks()
    
    if layer_name in activations:
        act = activations[layer_name]
        # Flatten all dimensions except batch
        return act.view(act.size(0), -1).size(1)
    else:
        raise ValueError(f"Layer {layer_name} not found in model")


class CustomCNN(nn.Module):
    """
    Custom CNN architecture for experiments.
    
    A medium-sized CNN with clear layer structure for CAV analysis.
    """
    
    def __init__(self, num_classes: int = 10, input_channels: int = 3):
        super().__init__()
        
        # Feature extraction layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.classifier(x)
        return x
    
    def get_layer_names(self) -> List[str]:
        """Get names of convolutional blocks."""
        return ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']


def create_model(model_name: str, num_classes: int, pretrained: bool = True,
                 input_channels: int = 3) -> nn.Module:
    """
    Create a CNN model with the specified architecture.
    
    Args:
        model_name: Name of the model architecture
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        input_channels: Number of input channels (3 for RGB)
        
    Returns:
        PyTorch model
    """
    weights = "IMAGENET1K_V1" if pretrained else None
    
    if model_name == "resnet18":
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
    elif model_name == "resnet34":
        model = models.resnet34(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
    elif model_name == "resnet50":
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
    elif model_name == "vgg16":
        model = models.vgg16(weights=weights)
        model.classifier[-1] = nn.Linear(4096, num_classes)
        
    elif model_name == "vgg19":
        model = models.vgg19(weights=weights)
        model.classifier[-1] = nn.Linear(4096, num_classes)
        
    elif model_name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        
    elif model_name == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        
    elif model_name == "custom_cnn":
        model = CustomCNN(num_classes=num_classes, input_channels=input_channels)
        
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model


def freeze_layers(model: nn.Module, layer_names: Optional[List[str]] = None,
                  freeze_all_except: Optional[List[str]] = None):
    """
    Freeze specific layers in the model.
    
    Args:
        model: The model
        layer_names: List of layer names to freeze (if provided)
        freeze_all_except: Freeze all layers except these (if provided)
    """
    if freeze_all_except is not None:
        # Freeze everything first
        for param in model.parameters():
            param.requires_grad = False
        # Unfreeze specified layers
        for name, module in model.named_modules():
            if name in freeze_all_except:
                for param in module.parameters():
                    param.requires_grad = True
                    
    elif layer_names is not None:
        # Freeze only specified layers
        for name, module in model.named_modules():
            if name in layer_names:
                for param in module.parameters():
                    param.requires_grad = False


def unfreeze_layers(model: nn.Module, layer_names: List[str]):
    """
    Unfreeze specific layers in the model.
    
    Args:
        model: The model
        layer_names: List of layer names to unfreeze
    """
    for name, module in model.named_modules():
        if name in layer_names:
            for param in module.parameters():
                param.requires_grad = True


def get_trainable_params(model: nn.Module) -> int:
    """Count trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_total_params(model: nn.Module) -> int:
    """Count total parameters in the model."""
    return sum(p.numel() for p in model.parameters())
