"""
Configuration management for VL-CAV experiments.

Handles experiment parameters, paths, and hyperparameters.
"""

import argparse
import json
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
from datetime import datetime


@dataclass
class ExperimentConfig:
    """Configuration for VL-CAV recalibration experiments."""
    
    # Experiment identification
    experiment_name: str = "vl_cav_experiment"
    experiment_id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    seed: int = 42
    
    # Model configuration
    model_name: str = "resnet18"
    pretrained: bool = True
    num_classes: int = 10
    
    # Dataset configuration
    dataset_name: str = "CIFAR10"
    dataset_path: str = "./data"
    image_size: int = 224
    
    # Imbalance configuration
    imbalance_classes: List[int] = field(default_factory=list)
    imbalance_ratio: float = 0.1  # Keep this fraction of samples for imbalanced classes
    
    # VLM configuration
    vlm_model: str = "openai/clip-vit-base-patch32"
    vlm_embedding_dim: int = 512
    alpha: float = 0.7  # Text-vision balance (higher = more text)
    num_concept_descriptions: int = 10  # Number of text descriptions per class
    
    # Training configuration
    batch_size: int = 32
    pretrain_epochs: int = 30
    recalib_epochs: int = 10
    learning_rate: float = 1e-3
    recalib_lr: float = 1e-4
    weight_decay: float = 1e-4
    
    # Recalibration configuration
    lambda_cls: float = 0.4
    lambda_align: float = 0.6
    projection_hidden_dim: int = 256
    
    # Bottleneck detection
    correlation_threshold: float = 0.0  # Positive correlation required
    
    # Output configuration
    output_dir: str = "./results"
    save_models: bool = True
    save_visualizations: bool = True
    visualization_formats: List[str] = field(default_factory=lambda: ["png", "svg"])
    
    # Device configuration
    device: str = "cuda"
    num_workers: int = 4
    
    def __post_init__(self):
        """Create output directory with experiment ID."""
        self.experiment_output_dir = os.path.join(
            self.output_dir,
            f"{self.experiment_name}_{self.experiment_id}"
        )
        
    def save(self, path: Optional[str] = None) -> str:
        """Save configuration to JSON file."""
        if path is None:
            os.makedirs(self.experiment_output_dir, exist_ok=True)
            path = os.path.join(self.experiment_output_dir, "config.json")
        
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
        return path
    
    @classmethod
    def load(cls, path: str) -> "ExperimentConfig":
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


def get_class_concept_descriptions(dataset_name: str, class_idx: int, class_name: str) -> List[str]:
    """
    Generate diverse textual descriptions for a class concept.
    
    These descriptions are used to create robust VLM-based concept embeddings.
    
    Args:
        dataset_name: Name of the dataset
        class_idx: Class index
        class_name: Human-readable class name
        
    Returns:
        List of textual descriptions for the concept
    """
    # Base templates for generating diverse descriptions
    templates = [
        f"A photo of a {class_name}",
        f"An image showing a {class_name}",
        f"A clear picture of a {class_name}",
        f"A {class_name} in the center of the image",
        f"The distinctive features of a {class_name}",
        f"Visual characteristics of a {class_name}",
        f"A typical {class_name}",
        f"An example of a {class_name}",
        f"The appearance of a {class_name}",
        f"A {class_name} with its defining attributes",
    ]
    
    # Dataset-specific additions
    if dataset_name.upper() == "CIFAR10":
        cifar10_attributes = {
            "airplane": ["with wings and fuselage", "flying in the sky", "on a runway"],
            "automobile": ["with four wheels", "on the road", "a car vehicle"],
            "bird": ["with feathers and beak", "with wings", "a flying animal"],
            "cat": ["with whiskers and fur", "a feline animal", "with pointed ears"],
            "deer": ["with antlers", "a forest animal", "with brown fur"],
            "dog": ["with fur and tail", "a canine animal", "a pet animal"],
            "frog": ["with green skin", "an amphibian", "with webbed feet"],
            "horse": ["with mane and hooves", "an equine animal", "a riding animal"],
            "ship": ["on water", "a sailing vessel", "a boat"],
            "truck": ["a large vehicle", "with cargo bed", "a commercial vehicle"],
        }
        if class_name.lower() in cifar10_attributes:
            for attr in cifar10_attributes[class_name.lower()]:
                templates.append(f"A {class_name} {attr}")
                
    elif dataset_name.upper() == "CIFAR100":
        # Add more specific descriptions for CIFAR-100 superclasses
        templates.extend([
            f"A detailed view of a {class_name}",
            f"The texture and shape of a {class_name}",
            f"Recognizable features of a {class_name}",
        ])
        
    return templates[:15]  # Return up to 15 descriptions


def parse_args() -> ExperimentConfig:
    """Parse command line arguments and return configuration."""
    parser = argparse.ArgumentParser(
        description="VL-CAV: VLM-Augmented Concept Activation Vectors for CNN Recalibration"
    )
    
    # Experiment settings
    parser.add_argument("--experiment_name", type=str, default="vl_cav_experiment",
                        help="Name for this experiment")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    # Model settings
    parser.add_argument("--model_name", type=str, default="resnet18",
                        choices=["resnet18", "resnet34", "resnet50", "vgg16", "vgg19",
                                "mobilenet_v3_small", "mobilenet_v3_large", "custom_cnn"],
                        help="CNN architecture to use")
    parser.add_argument("--pretrained", action="store_true",
                        help="Use ImageNet pretrained weights")
    
    # Dataset settings
    parser.add_argument("--dataset_name", type=str, default="CIFAR10",
                        choices=["CIFAR10", "CIFAR100", "STL10", "FashionMNIST"],
                        help="Dataset to use")
    parser.add_argument("--dataset_path", type=str, default="./data",
                        help="Path to store/load dataset")
    parser.add_argument("--image_size", type=int, default=224,
                        help="Input image size")
    
    # Imbalance settings
    parser.add_argument("--imbalance_classes", type=str, default="",
                        help="Comma-separated class indices to make imbalanced (e.g., '0,3,5')")
    parser.add_argument("--imbalance_ratio", type=float, default=0.1,
                        help="Fraction of samples to keep for imbalanced classes (0.0-1.0)")
    
    # VLM settings
    parser.add_argument("--vlm_model", type=str, default="openai/clip-vit-base-patch32",
                        help="VLM model for text/vision encoding")
    parser.add_argument("--alpha", type=float, default=0.7,
                        help="Text-vision balance (0=all vision, 1=all text)")
    parser.add_argument("--num_concept_descriptions", type=int, default=10,
                        help="Number of text descriptions per class concept")
    
    # Training settings
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--pretrain_epochs", type=int, default=30,
                        help="Epochs for initial biased model training")
    parser.add_argument("--recalib_epochs", type=int, default=10,
                        help="Epochs for recalibration")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                        help="Learning rate for pretraining")
    parser.add_argument("--recalib_lr", type=float, default=1e-4,
                        help="Learning rate for recalibration")
    
    # Recalibration settings
    parser.add_argument("--lambda_cls", type=float, default=0.4,
                        help="Weight for classification loss during recalibration")
    parser.add_argument("--lambda_align", type=float, default=0.6,
                        help="Weight for alignment loss during recalibration")
    
    # Output settings
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Directory for experiment outputs")
    parser.add_argument("--no_save_models", action="store_true",
                        help="Don't save model checkpoints")
    parser.add_argument("--no_visualizations", action="store_true",
                        help="Don't generate visualizations")
    
    # Device settings
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu", "mps"],
                        help="Device to use for training")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loading workers")
    
    args = parser.parse_args()
    
    # Parse imbalance classes
    imbalance_classes = []
    if args.imbalance_classes:
        imbalance_classes = [int(x.strip()) for x in args.imbalance_classes.split(",")]
    
    # Create config
    config = ExperimentConfig(
        experiment_name=args.experiment_name,
        seed=args.seed,
        model_name=args.model_name,
        pretrained=args.pretrained,
        dataset_name=args.dataset_name,
        dataset_path=args.dataset_path,
        image_size=args.image_size,
        imbalance_classes=imbalance_classes,
        imbalance_ratio=args.imbalance_ratio,
        vlm_model=args.vlm_model,
        alpha=args.alpha,
        num_concept_descriptions=args.num_concept_descriptions,
        batch_size=args.batch_size,
        pretrain_epochs=args.pretrain_epochs,
        recalib_epochs=args.recalib_epochs,
        learning_rate=args.learning_rate,
        recalib_lr=args.recalib_lr,
        lambda_cls=args.lambda_cls,
        lambda_align=args.lambda_align,
        output_dir=args.output_dir,
        save_models=not args.no_save_models,
        save_visualizations=not args.no_visualizations,
        device=args.device,
        num_workers=args.num_workers,
    )
    
    return config
