#!/usr/bin/env python3
"""
Data loaders for TCAV-Based Recalibration with Caltech-101 dataset.

Features:
- Automatic Caltech-101 download from torchvision
- Configurable class selection (support for N classes)
- Configurable imbalance injection
- Train/validation split handling
- Concept and background dataset support
"""

import os
import random
import shutil
from PIL import Image
from typing import Dict, List, Optional, Tuple
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import Caltech101

# Supported image extensions
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}


def is_image_file(filename: str) -> bool:
    """Check if a file is an image based on extension."""
    return os.path.splitext(filename.lower())[1] in IMAGE_EXTENSIONS


# Vehicle classes available in Caltech-101
CALTECH101_VEHICLE_CLASSES = [
    'airplanes',      # 800 images
    'Motorbikes',     # 798 images
    'car_side',       # 123 images  
    'Faces',          # For testing (435 images)
    'ferry',          # 67 images
    'helicopter',     # 88 images
    'schooner',       # 63 images (sailing ship)
]

# Default 5 vehicle classes
DEFAULT_VEHICLE_CLASSES = [
    'airplanes',
    'Motorbikes', 
    'car_side',
    'ferry',
    'helicopter'
]


class Caltech101VehicleDataset(Dataset):
    """
    Caltech-101 dataset filtered for vehicle classes with imbalance support.
    
    Args:
        root: Root directory for Caltech-101 data
        class_names: List of class names to include
        transform: Torchvision transforms to apply
        imbalance_ratios: Dict mapping class names to sampling ratios (0-1)
        is_train: Whether this is training data
        train_split: Fraction for training (default 0.8)
        seed: Random seed for reproducibility
        download: Whether to download if not present
    """
    
    def __init__(self,
                 root: str,
                 class_names: Optional[List[str]] = None,
                 transform: Optional[transforms.Compose] = None,
                 imbalance_ratios: Optional[Dict[str, float]] = None,
                 is_train: bool = True,
                 train_split: float = 0.8,
                 seed: int = 42,
                 download: bool = True):
        
        self.root = root
        self.transform = transform
        self.is_train = is_train
        self.train_split = train_split
        self.seed = seed
        
        # Download/load Caltech-101
        self.caltech = Caltech101(
            root=root,
            target_type='category',
            transform=None,  # We'll apply transforms ourselves
            download=download
        )
        
        # Get all category names
        all_categories = self.caltech.categories
        
        # Filter to requested classes
        if class_names is None:
            self.class_names = DEFAULT_VEHICLE_CLASSES
        else:
            self.class_names = class_names
        
        # Validate class names exist
        for name in self.class_names:
            if name not in all_categories:
                available = [c for c in all_categories if any(
                    v.lower() in c.lower() for v in ['air', 'car', 'motor', 'ferry', 'heli', 'boat', 'plane']
                )]
                raise ValueError(
                    f"Class '{name}' not found in Caltech-101. "
                    f"Vehicle-related classes: {available}"
                )
        
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        self.idx_to_class = {idx: name for name, idx in self.class_to_idx.items()}
        
        # Get category to Caltech index mapping
        self.category_to_caltech_idx = {cat: idx for idx, cat in enumerate(all_categories)}
        
        # Default: no imbalance
        if imbalance_ratios is None:
            self.imbalance_ratios = {name: 1.0 for name in self.class_names}
        else:
            self.imbalance_ratios = imbalance_ratios
            # Fill in missing classes with 1.0
            for name in self.class_names:
                if name not in self.imbalance_ratios:
                    self.imbalance_ratios[name] = 1.0
        
        # Load and filter samples
        self.samples = []
        self.original_counts = {}
        self.actual_counts = {}
        self._load_samples()
    
    def _load_samples(self):
        """Load and filter samples with train/val split and imbalance."""
        random.seed(self.seed)
        
        # Group Caltech indices by our target classes
        class_indices = {name: [] for name in self.class_names}
        
        for idx in range(len(self.caltech)):
            _, caltech_label = self.caltech[idx]
            caltech_class_name = self.caltech.categories[caltech_label]
            
            if caltech_class_name in self.class_names:
                class_indices[caltech_class_name].append(idx)
        
        # Process each class
        for class_name in self.class_names:
            indices = class_indices[class_name]
            random.shuffle(indices)
            
            # Train/val split
            split_idx = int(len(indices) * self.train_split)
            
            if self.is_train:
                selected_indices = indices[:split_idx]
            else:
                selected_indices = indices[split_idx:]
            
            self.original_counts[class_name] = len(selected_indices)
            
            # Apply imbalance ratio
            ratio = self.imbalance_ratios.get(class_name, 1.0)
            n_keep = max(1, int(len(selected_indices) * ratio))
            selected_indices = selected_indices[:n_keep]
            
            self.actual_counts[class_name] = len(selected_indices)
            
            # Add to samples
            our_label = self.class_to_idx[class_name]
            for caltech_idx in selected_indices:
                self.samples.append((caltech_idx, our_label))
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        caltech_idx, label = self.samples[idx]
        
        # Get image from Caltech dataset
        image, _ = self.caltech[caltech_idx]
        
        # Ensure RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_counts(self) -> Dict[str, int]:
        """Get current class counts."""
        return self.actual_counts.copy()
    
    def get_original_counts(self) -> Dict[str, int]:
        """Get original (before imbalance) class counts."""
        return self.original_counts.copy()
    
    def get_imbalance_info(self) -> Dict[str, Dict]:
        """Get detailed imbalance information."""
        info = {}
        for class_name in self.class_names:
            orig = self.original_counts.get(class_name, 0)
            actual = self.actual_counts.get(class_name, 0)
            ratio = self.imbalance_ratios.get(class_name, 1.0)
            info[class_name] = {
                'original': orig,
                'actual': actual,
                'ratio': ratio,
                'kept_percent': (actual / orig * 100) if orig > 0 else 0
            }
        return info


class ConceptDataset(Dataset):
    """Dataset for loading concept images (subject extractions)."""
    
    def __init__(self, concept_path: str, transform: Optional[transforms.Compose] = None):
        self.concept_path = concept_path
        self.transform = transform
        self.image_paths = []
        self._load_images()
    
    def _load_images(self):
        if not os.path.isdir(self.concept_path):
            raise FileNotFoundError(f"Concept directory not found: {self.concept_path}")
        
        for img_file in sorted(os.listdir(self.concept_path)):
            if is_image_file(img_file):
                self.image_paths.append(os.path.join(self.concept_path, img_file))
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in: {self.concept_path}")
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image


class BackgroundDataset(Dataset):
    """Dataset for loading background/random images."""
    
    def __init__(self, background_path: str, transform: Optional[transforms.Compose] = None):
        self.background_path = background_path
        self.transform = transform
        self.image_paths = []
        self._load_images()
    
    def _load_images(self):
        if not os.path.isdir(self.background_path):
            raise FileNotFoundError(f"Background directory not found: {self.background_path}")
        
        for img_file in sorted(os.listdir(self.background_path)):
            if is_image_file(img_file):
                self.image_paths.append(os.path.join(self.background_path, img_file))
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in: {self.background_path}")
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image


class SingleClassDataset(Dataset):
    """
    Dataset for loading images from a single class.
    Used for class-specific CAV computation.
    """
    
    def __init__(self,
                 base_dataset: Caltech101VehicleDataset,
                 class_name: str,
                 transform: Optional[transforms.Compose] = None):
        """
        Args:
            base_dataset: The full Caltech101VehicleDataset
            class_name: Name of the class to extract
            transform: Optional transform override
        """
        self.base_dataset = base_dataset
        self.class_name = class_name
        self.transform = transform or base_dataset.transform
        
        # Get indices for this class
        target_label = base_dataset.class_to_idx[class_name]
        self.indices = [
            i for i, (_, label) in enumerate(base_dataset.samples)
            if label == target_label
        ]
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        base_idx = self.indices[idx]
        return self.base_dataset[base_idx]


def get_transforms(image_size: int = 224, is_train: bool = True) -> transforms.Compose:
    """
    Get standard transforms for training/validation.
    
    Args:
        image_size: Target image size
        is_train: Whether to apply training augmentations
        
    Returns:
        Composed transforms
    """
    if is_train:
        return transforms.Compose([
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])


def create_caltech_vehicle_dataset(
    root: str,
    class_names: Optional[List[str]] = None,
    imbalance_class: Optional[str] = None,
    imbalance_ratio: Optional[float] = None,
    image_size: int = 224,
    train_split: float = 0.8,
    seed: int = 42,
    download: bool = True
) -> Tuple[Caltech101VehicleDataset, Caltech101VehicleDataset]:
    """
    Create train and validation datasets for Caltech-101 vehicles.
    
    Args:
        root: Root directory for data
        class_names: List of class names (default: 5 vehicle classes)
        imbalance_class: Class to make imbalanced
        imbalance_ratio: Ratio for imbalanced class (0-1)
        image_size: Target image size
        train_split: Train/val split ratio
        seed: Random seed
        download: Whether to download if not present
        
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    # Set up imbalance ratios
    imbalance_ratios = None
    if imbalance_class and imbalance_ratio:
        imbalance_ratios = {imbalance_class: imbalance_ratio}
    
    # Create datasets
    train_dataset = Caltech101VehicleDataset(
        root=root,
        class_names=class_names,
        transform=get_transforms(image_size, is_train=True),
        imbalance_ratios=imbalance_ratios,
        is_train=True,
        train_split=train_split,
        seed=seed,
        download=download
    )
    
    val_dataset = Caltech101VehicleDataset(
        root=root,
        class_names=class_names,
        transform=get_transforms(image_size, is_train=False),
        imbalance_ratios=imbalance_ratios,
        is_train=False,
        train_split=train_split,
        seed=seed,
        download=False  # Already downloaded
    )
    
    return train_dataset, val_dataset


def create_concept_loaders(
    concept_base_path: str,
    class_names: List[str],
    image_size: int = 224,
    batch_size: int = 16
) -> Tuple[Dict[str, DataLoader], DataLoader]:
    """
    Create concept and background data loaders.
    
    Args:
        concept_base_path: Base path for concept directories
        class_names: List of class names
        image_size: Target image size
        batch_size: Batch size for loaders
        
    Returns:
        Tuple of (concept_loaders dict, background_loader)
    """
    transform = get_transforms(image_size, is_train=False)
    
    concept_loaders = {}
    for class_name in class_names:
        concept_path = os.path.join(concept_base_path, class_name, 'subject')
        
        if os.path.isdir(concept_path) and len(os.listdir(concept_path)) > 0:
            concept_dataset = ConceptDataset(concept_path, transform)
            concept_loaders[class_name] = DataLoader(
                concept_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=2
            )
        else:
            print(f"Warning: No concept images found for {class_name}")
    
    # Background loader
    background_path = os.path.join(concept_base_path, 'background')
    if not os.path.isdir(background_path):
        raise FileNotFoundError(
            f"Background directory not found: {background_path}\n"
            "Run concept_generator.py first to generate concepts."
        )
    
    background_dataset = BackgroundDataset(background_path, transform)
    background_loader = DataLoader(
        background_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    
    return concept_loaders, background_loader


def print_dataset_info(dataset: Caltech101VehicleDataset, name: str = "Dataset"):
    """Print detailed dataset information."""
    print(f"\n{'=' * 60}")
    print(f"{name}")
    print(f"{'=' * 60}")
    print(f"Total samples: {len(dataset)}")
    print(f"Number of classes: {len(dataset.class_names)}")
    print(f"\nClass distribution:")
    
    total = sum(dataset.actual_counts.values())
    for class_name in dataset.class_names:
        count = dataset.actual_counts.get(class_name, 0)
        orig = dataset.original_counts.get(class_name, 0)
        ratio = dataset.imbalance_ratios.get(class_name, 1.0)
        pct = (count / total * 100) if total > 0 else 0
        bar = '█' * int(pct / 2)
        
        if ratio < 1.0:
            print(f"  {class_name:15s}: {count:4d}/{orig:4d} ({ratio*100:5.1f}%) {bar} [IMBALANCED]")
        else:
            print(f"  {class_name:15s}: {count:4d} ({pct:5.1f}%) {bar}")
    
    print(f"{'=' * 60}")


# Imbalance presets
IMBALANCE_PRESETS = {
    'extreme': 0.05,   # 5% of data
    'severe': 0.10,    # 10% of data
    'moderate': 0.20,  # 20% of data
    'mild': 0.25,      # 25% of data
    'light': 0.50,     # 50% of data
    'balanced': 1.0    # 100% of data
}


def get_imbalance_ratio(preset_name: str) -> float:
    """Get imbalance ratio from preset name."""
    return IMBALANCE_PRESETS.get(preset_name.lower(), 1.0)


if __name__ == "__main__":
    # Test the dataloader
    print("Testing Caltech-101 Vehicle Dataloader...")
    
    train_ds, val_ds = create_caltech_vehicle_dataset(
        root='./data',
        class_names=DEFAULT_VEHICLE_CLASSES,
        imbalance_class='airplanes',
        imbalance_ratio=0.1,
        download=True
    )
    
    print_dataset_info(train_ds, "Training Set")
    print_dataset_info(val_ds, "Validation Set")
    
    # Test loading a sample
    img, label = train_ds[0]
    print(f"\nSample image shape: {img.shape}")
    print(f"Sample label: {label} ({train_ds.idx_to_class[label]})")
