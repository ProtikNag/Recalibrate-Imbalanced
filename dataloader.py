"""
Dataset loading utilities with class imbalance support.

Supports multiple PyTorch datasets:
- CIFAR-10, CIFAR-100
- STL-10
- FashionMNIST

Provides functionality to:
- Create artificially imbalanced datasets
- Get class-specific data subsets
- Compute dataset statistics
"""

import torch
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from torchvision import datasets, transforms
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from collections import Counter


def get_dataset_info(dataset_name: str) -> Dict:
    """
    Get information about a dataset.
    
    Args:
        dataset_name: Name of the dataset
        
    Returns:
        Dictionary with dataset info (num_classes, class_names, etc.)
    """
    info = {
        "CIFAR10": {
            "num_classes": 10,
            "class_names": ["airplane", "automobile", "bird", "cat", "deer",
                           "dog", "frog", "horse", "ship", "truck"],
            "image_size": 32,
            "channels": 3,
            "mean": (0.4914, 0.4822, 0.4465),
            "std": (0.2470, 0.2435, 0.2616)
        },
        "CIFAR100": {
            "num_classes": 100,
            "class_names": None,  # Too many to list
            "image_size": 32,
            "channels": 3,
            "mean": (0.5071, 0.4867, 0.4408),
            "std": (0.2675, 0.2565, 0.2761)
        },
        "STL10": {
            "num_classes": 10,
            "class_names": ["airplane", "bird", "car", "cat", "deer",
                           "dog", "horse", "monkey", "ship", "truck"],
            "image_size": 96,
            "channels": 3,
            "mean": (0.4467, 0.4398, 0.4066),
            "std": (0.2603, 0.2566, 0.2713)
        },
        "FashionMNIST": {
            "num_classes": 10,
            "class_names": ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"],
            "image_size": 28,
            "channels": 1,
            "mean": (0.2860,),
            "std": (0.3530,)
        }
    }
    
    if dataset_name not in info:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(info.keys())}")
    
    return info[dataset_name]


def get_transforms(dataset_name: str, image_size: int = 224, 
                   augment: bool = True) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Get train and test transforms for a dataset.
    
    Args:
        dataset_name: Name of the dataset
        image_size: Target image size
        augment: Whether to apply data augmentation for training
        
    Returns:
        (train_transform, test_transform)
    """
    info = get_dataset_info(dataset_name)
    mean = info["mean"]
    std = info["std"]
    channels = info["channels"]
    
    # Common test transform
    test_transform_list = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]
    
    # Add grayscale to RGB conversion for single-channel datasets
    if channels == 1:
        test_transform_list.insert(0, transforms.Grayscale(num_output_channels=3))
        # Adjust normalization for 3 channels
        mean = mean * 3
        std = std * 3
        test_transform_list[-1] = transforms.Normalize(mean, std)
    
    test_transform = transforms.Compose(test_transform_list)
    
    # Training transform with augmentation
    if augment:
        train_transform_list = [
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]
        if channels == 1:
            train_transform_list.insert(0, transforms.Grayscale(num_output_channels=3))
    else:
        train_transform_list = test_transform_list.copy()
        
    train_transform = transforms.Compose(train_transform_list)
    
    return train_transform, test_transform


class ImbalancedDataset(Dataset):
    """
    Wrapper dataset that creates class imbalance.
    
    Reduces samples from specified classes by a given ratio.
    """
    
    def __init__(self, dataset: Dataset, imbalance_classes: List[int],
                 imbalance_ratio: float, seed: int = 42):
        """
        Create imbalanced dataset.
        
        Args:
            dataset: Base dataset
            imbalance_classes: Class indices to make imbalanced
            imbalance_ratio: Fraction of samples to keep (0.0-1.0)
            seed: Random seed for reproducibility
        """
        self.dataset = dataset
        self.imbalance_classes = imbalance_classes
        self.imbalance_ratio = imbalance_ratio
        
        # Get all targets
        if hasattr(dataset, 'targets'):
            targets = np.array(dataset.targets)
        elif hasattr(dataset, 'labels'):
            targets = np.array(dataset.labels)
        else:
            # Extract targets manually
            targets = np.array([dataset[i][1] for i in range(len(dataset))])
        
        self.original_targets = targets
        
        # Create subset indices
        np.random.seed(seed)
        self.indices = []
        self.class_counts = {}
        
        for class_idx in range(max(targets) + 1):
            class_indices = np.where(targets == class_idx)[0]
            
            if class_idx in imbalance_classes:
                # Reduce this class
                num_keep = max(1, int(len(class_indices) * imbalance_ratio))
                selected = np.random.choice(class_indices, num_keep, replace=False)
            else:
                selected = class_indices
                
            self.indices.extend(selected.tolist())
            self.class_counts[class_idx] = len(selected)
        
        self.indices = np.array(self.indices)
        np.random.shuffle(self.indices)
        
        # Update targets for selected indices
        self.targets = targets[self.indices]
        
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        original_idx = self.indices[idx]
        return self.dataset[original_idx]
    
    def get_class_counts(self) -> Dict[int, int]:
        """Get number of samples per class."""
        return self.class_counts
    
    def get_class_indices(self, class_idx: int) -> np.ndarray:
        """Get indices of samples belonging to a specific class."""
        return np.where(self.targets == class_idx)[0]
    
    def get_imbalance_factor(self) -> float:
        """Compute imbalance factor (max_count / min_count)."""
        counts = list(self.class_counts.values())
        return max(counts) / max(min(counts), 1)


def load_dataset(dataset_name: str, root: str = "./data", 
                 image_size: int = 224, augment: bool = True,
                 imbalance_classes: Optional[List[int]] = None,
                 imbalance_ratio: float = 0.1,
                 seed: int = 42) -> Tuple[Dataset, Dataset, Dict]:
    """
    Load a dataset with optional imbalance.
    
    Args:
        dataset_name: Name of the dataset
        root: Root directory for data
        image_size: Target image size
        augment: Whether to apply data augmentation
        imbalance_classes: Class indices to make imbalanced (None = balanced)
        imbalance_ratio: Fraction of samples to keep for imbalanced classes
        seed: Random seed
        
    Returns:
        (train_dataset, test_dataset, dataset_info)
    """
    train_transform, test_transform = get_transforms(dataset_name, image_size, augment)
    info = get_dataset_info(dataset_name)
    
    # Load base datasets
    if dataset_name == "CIFAR10":
        train_dataset = datasets.CIFAR10(root=root, train=True, download=True, 
                                         transform=train_transform)
        test_dataset = datasets.CIFAR10(root=root, train=False, download=True,
                                        transform=test_transform)
        
    elif dataset_name == "CIFAR100":
        train_dataset = datasets.CIFAR100(root=root, train=True, download=True,
                                          transform=train_transform)
        test_dataset = datasets.CIFAR100(root=root, train=False, download=True,
                                         transform=test_transform)
        # Get class names
        info["class_names"] = train_dataset.classes
        
    elif dataset_name == "STL10":
        train_dataset = datasets.STL10(root=root, split='train', download=True,
                                       transform=train_transform)
        test_dataset = datasets.STL10(root=root, split='test', download=True,
                                      transform=test_transform)
        
    elif dataset_name == "FashionMNIST":
        train_dataset = datasets.FashionMNIST(root=root, train=True, download=True,
                                              transform=train_transform)
        test_dataset = datasets.FashionMNIST(root=root, train=False, download=True,
                                             transform=test_transform)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Apply imbalance if specified
    if imbalance_classes and len(imbalance_classes) > 0:
        train_dataset = ImbalancedDataset(
            train_dataset, imbalance_classes, imbalance_ratio, seed
        )
        info["class_counts"] = train_dataset.get_class_counts()
        info["imbalance_factor"] = train_dataset.get_imbalance_factor()
        info["imbalanced"] = True
    else:
        # Compute class counts for balanced dataset
        if hasattr(train_dataset, 'targets'):
            targets = train_dataset.targets
        elif hasattr(train_dataset, 'labels'):
            targets = train_dataset.labels
        else:
            targets = [train_dataset[i][1] for i in range(len(train_dataset))]
        info["class_counts"] = dict(Counter(targets))
        info["imbalance_factor"] = 1.0
        info["imbalanced"] = False
    
    return train_dataset, test_dataset, info


def create_dataloaders(train_dataset: Dataset, test_dataset: Dataset,
                       batch_size: int = 32, num_workers: int = 4,
                       use_weighted_sampler: bool = False) -> Tuple[DataLoader, DataLoader]:
    """
    Create data loaders for training and testing.
    
    Args:
        train_dataset: Training dataset
        test_dataset: Test dataset
        batch_size: Batch size
        num_workers: Number of data loading workers
        use_weighted_sampler: Whether to use weighted sampling to balance classes
        
    Returns:
        (train_loader, test_loader)
    """
    if use_weighted_sampler and isinstance(train_dataset, ImbalancedDataset):
        # Create weighted sampler to balance classes
        class_counts = train_dataset.get_class_counts()
        weights = []
        for idx in range(len(train_dataset)):
            label = train_dataset.targets[idx]
            weights.append(1.0 / class_counts[label])
        
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, sampler=sampler,
            num_workers=num_workers, pin_memory=True
        )
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True
        )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, test_loader


def get_class_samples(dataset: Dataset, class_idx: int, 
                      num_samples: Optional[int] = None) -> Tuple[torch.Tensor, List[int]]:
    """
    Get samples from a specific class.
    
    Args:
        dataset: The dataset
        class_idx: Class index to get samples from
        num_samples: Maximum number of samples (None = all)
        
    Returns:
        (images_tensor, indices)
    """
    # Get targets
    if hasattr(dataset, 'targets'):
        targets = np.array(dataset.targets)
    elif hasattr(dataset, 'labels'):
        targets = np.array(dataset.labels)
    elif isinstance(dataset, ImbalancedDataset):
        targets = dataset.targets
    else:
        targets = np.array([dataset[i][1] for i in range(len(dataset))])
    
    # Find indices for the class
    class_indices = np.where(targets == class_idx)[0]
    
    if num_samples is not None and num_samples < len(class_indices):
        class_indices = np.random.choice(class_indices, num_samples, replace=False)
    
    # Get images
    images = []
    for idx in class_indices:
        img, _ = dataset[idx]
        images.append(img)
    
    return torch.stack(images), class_indices.tolist()


def compute_class_weights(dataset: Dataset) -> torch.Tensor:
    """
    Compute class weights for weighted loss function.
    
    Args:
        dataset: The dataset
        
    Returns:
        Tensor of class weights
    """
    if isinstance(dataset, ImbalancedDataset):
        class_counts = dataset.get_class_counts()
    else:
        if hasattr(dataset, 'targets'):
            targets = dataset.targets
        else:
            targets = [dataset[i][1] for i in range(len(dataset))]
        class_counts = dict(Counter(targets))
    
    num_classes = len(class_counts)
    total_samples = sum(class_counts.values())
    
    weights = []
    for i in range(num_classes):
        count = class_counts.get(i, 1)
        weight = total_samples / (num_classes * count)
        weights.append(weight)
    
    return torch.tensor(weights, dtype=torch.float32)
