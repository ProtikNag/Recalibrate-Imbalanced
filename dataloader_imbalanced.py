"""
Enhanced data loaders for TCAV-Based Recalibration with imbalanced datasets.

Features:
- Configurable imbalance injection
- Support for multiple imbalance ratios (5%, 10%, 20%, 25%, etc.)
- Train/validation split handling
"""

import os
import random
from PIL import Image
from torch.utils.data import Dataset
from collections import Counter

# Supported image extensions
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}


def is_image_file(filename):
    """Check if a file is an image based on extension."""
    return os.path.splitext(filename.lower())[1] in IMAGE_EXTENSIONS


def get_class_folders(dataset_path):
    """Get class folders from dataset directory."""
    class_folders = {}

    for item in sorted(os.listdir(dataset_path)):
        item_path = os.path.join(dataset_path, item)
        if os.path.isdir(item_path):
            has_images = any(is_image_file(f) for f in os.listdir(item_path))
            if has_images:
                class_folders[item] = item_path

    return class_folders


class ImbalancedDataset(Dataset):
    """
    Dataset with configurable class imbalance.
    
    Args:
        dataset_path: Path to dataset root directory
        transform: Torchvision transforms to apply
        class_names: List of class names
        imbalance_ratios: Dict mapping class names to sampling ratios (0-1)
        is_train: Whether this is training data
        train_split: Fraction for training (default 0.8)
        seed: Random seed for reproducibility
    """

    def __init__(self, dataset_path, transform=None, class_names=None,
                 imbalance_ratios=None, is_train=True, train_split=0.8, seed=42):
        self.dataset_path = dataset_path
        self.transform = transform
        self.is_train = is_train
        self.train_split = train_split
        self.seed = seed

        # Get class names
        class_folders = get_class_folders(dataset_path)
        if class_names is None:
            self.class_names = sorted(class_folders.keys())
        else:
            self.class_names = class_names

        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}

        # Default: no imbalance
        if imbalance_ratios is None:
            self.imbalance_ratios = {name: 1.0 for name in self.class_names}
        else:
            self.imbalance_ratios = imbalance_ratios

        # Load samples
        self.samples = []
        self.original_counts = {}
        self.actual_counts = {}
        self._load_samples()

    def _load_samples(self):
        """Load image paths and labels with imbalance applied."""
        random.seed(self.seed)

        for class_name in self.class_names:
            class_path = os.path.join(self.dataset_path, class_name)

            if not os.path.isdir(class_path):
                continue

            class_idx = self.class_to_idx[class_name]

            # Get all image files
            all_images = sorted([
                f for f in os.listdir(class_path)
                if is_image_file(f)
            ])

            # Shuffle deterministically
            random.shuffle(all_images)

            # Split into train/val
            split_idx = int(len(all_images) * self.train_split)

            if self.is_train:
                selected_images = all_images[:split_idx]
            else:
                selected_images = all_images[split_idx:]

            self.original_counts[class_name] = len(selected_images)

            # Apply imbalance ratio
            ratio = self.imbalance_ratios.get(class_name, 1.0)
            n_keep = max(1, int(len(selected_images) * ratio))
            selected_images = selected_images[:n_keep]

            self.actual_counts[class_name] = len(selected_images)

            for img_file in selected_images:
                img_path = os.path.join(class_path, img_file)
                self.samples.append((img_path, class_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label

    def get_class_counts(self):
        """Get current class counts."""
        return self.actual_counts.copy()

    def get_original_counts(self):
        """Get original (before imbalance) class counts."""
        return self.original_counts.copy()

    def get_imbalance_info(self):
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


class SingleClassDataset(Dataset):
    """
    Dataset for loading images from a single class directory.
    
    Args:
        class_path: Path to class directory
        transform: Torchvision transforms
        class_idx: Class index to assign
        sample_ratio: Ratio of samples to use (for imbalance)
        seed: Random seed
    """

    def __init__(self, class_path, transform=None, class_idx=0,
                 sample_ratio=1.0, seed=42):
        self.class_path = class_path
        self.transform = transform
        self.class_idx = class_idx
        self.sample_ratio = sample_ratio
        self.seed = seed

        self.image_paths = []
        self._load_images()

    def _load_images(self):
        """Load image paths with optional subsampling."""
        if not os.path.isdir(self.class_path):
            raise FileNotFoundError(f"Directory not found: {self.class_path}")

        all_images = sorted([
            os.path.join(self.class_path, f)
            for f in os.listdir(self.class_path)
            if is_image_file(f)
        ])

        if len(all_images) == 0:
            raise ValueError(f"No images found in: {self.class_path}")

        # Apply sampling ratio
        if self.sample_ratio < 1.0:
            random.seed(self.seed)
            n_keep = max(1, int(len(all_images) * self.sample_ratio))
            random.shuffle(all_images)
            all_images = all_images[:n_keep]

        self.image_paths = all_images

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, self.class_idx


class ConceptDataset(Dataset):
    """Dataset for loading concept images (no labels)."""

    def __init__(self, concept_path, transform=None):
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

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image


def create_imbalanced_dataset(dataset_path, transform, class_names,
                              imbalance_ratios=None, is_train=True,
                              train_split=0.8, seed=42):
    """
    Create a dataset with configurable imbalance.
    
    Args:
        dataset_path: Path to dataset
        transform: Transforms to apply
        class_names: List of class names
        imbalance_ratios: Dict mapping class names to ratios (0-1)
                         e.g., {'zebra': 0.1} keeps only 10% of zebra images
        is_train: Training or validation set
        train_split: Train/val split ratio
        seed: Random seed
    
    Returns:
        ImbalancedDataset instance
    """
    if imbalance_ratios is None:
        imbalance_ratios = {name: 1.0 for name in class_names}

    return ImbalancedDataset(
        dataset_path=dataset_path,
        transform=transform,
        class_names=class_names,
        imbalance_ratios=imbalance_ratios,
        is_train=is_train,
        train_split=train_split,
        seed=seed
    )


def get_dataset_info(dataset_path):
    """Get information about a dataset."""
    info = {
        'path': dataset_path,
        'classes': {},
        'total_images': 0
    }

    class_folders = get_class_folders(dataset_path)

    for class_name, class_path in class_folders.items():
        n_images = len([f for f in os.listdir(class_path) if is_image_file(f)])
        info['classes'][class_name] = n_images
        info['total_images'] += n_images

    info['num_classes'] = len(info['classes'])

    return info


def print_dataset_info(dataset_path, imbalance_ratios=None):
    """Print detailed dataset information."""
    info = get_dataset_info(dataset_path)

    print(f"\nDataset: {info['path']}")
    print(f"{'=' * 60}")
    print(f"Number of classes: {info['num_classes']}")
    print(f"Total images: {info['total_images']}")
    print(f"\nClass distribution:")

    for class_name, count in sorted(info['classes'].items()):
        percentage = (count / info['total_images']) * 100

        # Show imbalance effect if provided
        if imbalance_ratios and class_name in imbalance_ratios:
            ratio = imbalance_ratios[class_name]
            effective = int(count * ratio)
            bar = '█' * int(percentage * ratio / 2)
            print(f"  {class_name:15s}: {effective:5d}/{count:5d} ({ratio * 100:5.1f}%) {bar}")
        else:
            bar = '█' * int(percentage / 2)
            print(f"  {class_name:15s}: {count:5d} ({percentage:5.1f}%) {bar}")

    print(f"{'=' * 60}")


# Predefined imbalance configurations
IMBALANCE_PRESETS = {
    'extreme': 0.05,  # 5% of data
    'severe': 0.10,  # 10% of data
    'moderate': 0.20,  # 20% of data
    'mild': 0.25,  # 25% of data
    'light': 0.50,  # 50% of data
    'balanced': 1.0  # 100% of data
}


def get_imbalance_ratio(preset_name):
    """Get imbalance ratio from preset name."""
    return IMBALANCE_PRESETS.get(preset_name.lower(), 1.0)
