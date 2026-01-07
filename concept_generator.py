#!/usr/bin/env python3
"""
Automatic Concept Generator using DeepLabV3 Segmentation.

This module automatically extracts:
- Subject/Foreground: Used as concept images for each class
- Background: Used as random/negative samples for CAV training

The segmentation model identifies the main object in each image,
creating a cleaner concept representation than manual selection.
"""

import os
import numpy as np
from PIL import Image
from typing import Tuple, Optional, Dict, List
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights


class ConceptGenerator:
    """
    Generates concept images using DeepLabV3 segmentation.
    
    Extracts subject (foreground) and background from images,
    creating clean concept representations for TCAV analysis.
    """
    
    # PASCAL VOC class indices for vehicles and related objects
    VEHICLE_CLASSES = {
        'aeroplane': 1,
        'bicycle': 2,
        'bus': 6,
        'car': 7,
        'motorbike': 14,
        'train': 19,
        'boat': 4,  # for ferry
    }
    
    # General object class (for non-vehicle objects)
    PERSON_CLASS = 15
    
    def __init__(self, device: Optional[str] = None, min_subject_ratio: float = 0.05):
        """
        Initialize the concept generator.
        
        Args:
            device: Device to run segmentation on ('cuda' or 'cpu')
            min_subject_ratio: Minimum ratio of subject pixels to consider valid
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.min_subject_ratio = min_subject_ratio
        
        # Load DeepLabV3 model
        print(f"Loading DeepLabV3 model on {self.device}...")
        self.model = deeplabv3_resnet101(
            weights=DeepLabV3_ResNet101_Weights.DEFAULT
        ).to(self.device)
        self.model.eval()
        
        # Preprocessing transform
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        print("ConceptGenerator initialized successfully")
    
    def segment_image(self, image: Image.Image) -> np.ndarray:
        """
        Segment an image using DeepLabV3.
        
        Args:
            image: PIL Image to segment
            
        Returns:
            Segmentation mask as numpy array (H, W) with class indices
        """
        # Ensure RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        original_size = image.size  # (W, H)
        
        # Preprocess
        input_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        
        # Run inference
        with torch.no_grad():
            output = self.model(input_tensor)['out'][0]
            
        # Get class predictions
        output = F.interpolate(
            output.unsqueeze(0),
            size=(original_size[1], original_size[0]),  # (H, W)
            mode='bilinear',
            align_corners=False
        ).squeeze(0)
        
        mask = output.argmax(0).cpu().numpy()
        
        return mask
    
    def extract_subject_mask(self, mask: np.ndarray, 
                            target_classes: Optional[List[int]] = None) -> np.ndarray:
        """
        Extract subject (foreground) mask from segmentation.
        
        Args:
            mask: Segmentation mask from segment_image()
            target_classes: Specific class indices to consider as subject
                          If None, all non-background classes are subject
                          
        Returns:
            Binary mask where 1 = subject, 0 = background
        """
        if target_classes is not None:
            # Only consider specified classes as subject
            subject_mask = np.isin(mask, target_classes).astype(np.uint8)
        else:
            # All non-background (class 0) pixels are subject
            subject_mask = (mask > 0).astype(np.uint8)
        
        return subject_mask
    
    def extract_subject_image(self, image: Image.Image, 
                             mask: np.ndarray,
                             padding: int = 10) -> Optional[Image.Image]:
        """
        Extract the subject region from an image using its mask.
        
        Args:
            image: Original PIL Image
            mask: Binary subject mask
            padding: Padding around the bounding box
            
        Returns:
            Cropped subject image or None if subject too small
        """
        # Find bounding box of subject
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        
        if not rows.any() or not cols.any():
            return None
        
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        
        # Add padding
        h, w = mask.shape
        rmin = max(0, rmin - padding)
        rmax = min(h, rmax + padding)
        cmin = max(0, cmin - padding)
        cmax = min(w, cmax + padding)
        
        # Check if subject is large enough
        subject_ratio = mask.sum() / (h * w)
        if subject_ratio < self.min_subject_ratio:
            return None
        
        # Crop
        image_array = np.array(image)
        subject_crop = image_array[rmin:rmax, cmin:cmax]
        
        return Image.fromarray(subject_crop)
    
    def extract_background_image(self, image: Image.Image,
                                mask: np.ndarray,
                                min_size: Tuple[int, int] = (32, 32)) -> Optional[Image.Image]:
        """
        Extract background region from an image.
        
        Args:
            image: Original PIL Image
            mask: Binary subject mask (1 = subject, 0 = background)
            min_size: Minimum size for background crop
            
        Returns:
            Background image or None if not enough background
        """
        # Invert mask to get background
        bg_mask = (1 - mask).astype(np.uint8)
        
        # Find largest contiguous background region
        # For simplicity, we'll just mask out the subject and return the full image
        image_array = np.array(image)
        
        # Create background by setting subject pixels to mean color
        bg_image = image_array.copy()
        mean_color = image_array.mean(axis=(0, 1)).astype(np.uint8)
        
        # Expand mask to 3 channels
        mask_3d = np.stack([mask] * 3, axis=-1)
        bg_image = np.where(mask_3d, mean_color, bg_image)
        
        return Image.fromarray(bg_image)
    
    def process_image(self, image_path: str,
                     target_classes: Optional[List[int]] = None
                     ) -> Dict[str, Optional[Image.Image]]:
        """
        Process a single image to extract subject and background.
        
        Args:
            image_path: Path to the image file
            target_classes: Specific segmentation classes to consider as subject
            
        Returns:
            Dictionary with 'subject' and 'background' images (or None)
        """
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return {'subject': None, 'background': None}
        
        # Segment
        seg_mask = self.segment_image(image)
        
        # Extract subject mask
        subject_mask = self.extract_subject_mask(seg_mask, target_classes)
        
        # Extract images
        subject_img = self.extract_subject_image(image, subject_mask)
        background_img = self.extract_background_image(image, subject_mask)
        
        return {
            'subject': subject_img,
            'background': background_img,
            'mask': subject_mask
        }
    
    def generate_concepts_for_class(self,
                                   class_image_dir: str,
                                   output_concept_dir: str,
                                   output_background_dir: str,
                                   class_name: str,
                                   max_images: Optional[int] = None,
                                   target_seg_classes: Optional[List[int]] = None
                                   ) -> Dict[str, int]:
        """
        Generate concept images for an entire class directory.
        
        Args:
            class_image_dir: Directory containing class images
            output_concept_dir: Directory to save concept (subject) images
            output_background_dir: Directory to save background images
            class_name: Name of the class (for logging)
            max_images: Maximum number of images to process
            target_seg_classes: Specific segmentation classes to look for
            
        Returns:
            Statistics dictionary with counts
        """
        os.makedirs(output_concept_dir, exist_ok=True)
        os.makedirs(output_background_dir, exist_ok=True)
        
        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
        image_files = [
            f for f in os.listdir(class_image_dir)
            if os.path.splitext(f.lower())[1] in image_extensions
        ]
        
        if max_images:
            image_files = image_files[:max_images]
        
        stats = {
            'total': len(image_files),
            'subjects_extracted': 0,
            'backgrounds_extracted': 0,
            'failed': 0
        }
        
        print(f"\nProcessing {class_name}: {len(image_files)} images")
        
        for img_file in tqdm(image_files, desc=f"  {class_name}"):
            img_path = os.path.join(class_image_dir, img_file)
            
            result = self.process_image(img_path, target_seg_classes)
            
            base_name = os.path.splitext(img_file)[0]
            
            # Save subject
            if result['subject'] is not None:
                subject_path = os.path.join(output_concept_dir, f"{base_name}_subject.png")
                result['subject'].save(subject_path)
                stats['subjects_extracted'] += 1
            
            # Save background
            if result['background'] is not None:
                bg_path = os.path.join(output_background_dir, f"{base_name}_bg.png")
                result['background'].save(bg_path)
                stats['backgrounds_extracted'] += 1
            
            if result['subject'] is None and result['background'] is None:
                stats['failed'] += 1
        
        return stats
    
    def generate_all_concepts(self,
                             dataset_dir: str,
                             concept_output_dir: str,
                             class_names: List[str],
                             max_images_per_class: Optional[int] = None
                             ) -> Dict[str, Dict[str, int]]:
        """
        Generate concepts for all classes in a dataset.
        
        Args:
            dataset_dir: Root directory of the dataset (with class subdirectories)
            concept_output_dir: Root directory for concept outputs
            class_names: List of class names to process
            max_images_per_class: Maximum images per class
            
        Returns:
            Statistics for each class
        """
        all_stats = {}
        
        # Create shared background directory
        background_dir = os.path.join(concept_output_dir, 'background')
        os.makedirs(background_dir, exist_ok=True)
        
        for class_name in class_names:
            class_dir = os.path.join(dataset_dir, class_name)
            
            if not os.path.isdir(class_dir):
                print(f"Warning: Class directory not found: {class_dir}")
                continue
            
            # Output directories
            concept_dir = os.path.join(concept_output_dir, class_name, 'subject')
            
            # Determine target segmentation classes
            target_seg = None
            lower_name = class_name.lower()
            if 'plane' in lower_name or 'air' in lower_name:
                target_seg = [self.VEHICLE_CLASSES['aeroplane']]
            elif 'bike' in lower_name or 'motor' in lower_name:
                target_seg = [self.VEHICLE_CLASSES['motorbike'], self.VEHICLE_CLASSES['bicycle']]
            elif 'car' in lower_name:
                target_seg = [self.VEHICLE_CLASSES['car']]
            elif 'bus' in lower_name:
                target_seg = [self.VEHICLE_CLASSES['bus']]
            elif 'ferry' in lower_name or 'boat' in lower_name:
                target_seg = [self.VEHICLE_CLASSES['boat']]
            elif 'train' in lower_name:
                target_seg = [self.VEHICLE_CLASSES['train']]
            elif 'copter' in lower_name or 'heli' in lower_name:
                # Helicopters might be classified as aeroplanes
                target_seg = [self.VEHICLE_CLASSES['aeroplane']]
            
            stats = self.generate_concepts_for_class(
                class_image_dir=class_dir,
                output_concept_dir=concept_dir,
                output_background_dir=background_dir,
                class_name=class_name,
                max_images=max_images_per_class,
                target_seg_classes=target_seg
            )
            
            all_stats[class_name] = stats
            print(f"  {class_name}: {stats['subjects_extracted']}/{stats['total']} subjects extracted")
        
        return all_stats


def generate_concepts_cli():
    """Command-line interface for concept generation."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate concept images using DeepLabV3 segmentation"
    )
    parser.add_argument("--dataset_path", type=str, required=True,
                       help="Path to dataset directory")
    parser.add_argument("--output_path", type=str, required=True,
                       help="Path to output concept directory")
    parser.add_argument("--classes", type=str, required=True,
                       help="Comma-separated list of class names")
    parser.add_argument("--max_images", type=int, default=None,
                       help="Maximum images per class")
    parser.add_argument("--device", type=str, default=None,
                       help="Device (cuda/cpu)")
    
    args = parser.parse_args()
    
    class_names = [c.strip() for c in args.classes.split(',')]
    
    generator = ConceptGenerator(device=args.device)
    stats = generator.generate_all_concepts(
        dataset_dir=args.dataset_path,
        concept_output_dir=args.output_path,
        class_names=class_names,
        max_images_per_class=args.max_images
    )
    
    print("\n" + "=" * 50)
    print("CONCEPT GENERATION COMPLETE")
    print("=" * 50)
    for class_name, class_stats in stats.items():
        print(f"{class_name}:")
        print(f"  Subjects: {class_stats['subjects_extracted']}/{class_stats['total']}")
        print(f"  Backgrounds: {class_stats['backgrounds_extracted']}")


if __name__ == "__main__":
    generate_concepts_cli()
