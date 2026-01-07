"""
VLM Encoder module for VL-CAV.

Provides CLIP-based encoding for:
- Text descriptions -> embeddings
- Images -> embeddings
- Unified multimodal concept embeddings

This enables VLM-augmented concept activation vectors.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

try:
    from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("Warning: transformers not installed. VLM features will be limited.")


class VLMEncoder:
    """
    Vision-Language Model encoder using CLIP.
    
    Provides methods to encode text and images into a shared embedding space.
    """
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32",
                 device: str = "cuda"):
        """
        Initialize VLM encoder.
        
        Args:
            model_name: HuggingFace model name for CLIP
            device: Device to run model on
        """
        self.model_name = model_name
        self.device = device
        self.embedding_dim = 512  # Default CLIP embedding dimension
        
        if HAS_TRANSFORMERS:
            self.model = CLIPModel.from_pretrained(model_name).to(device)
            self.processor = CLIPProcessor.from_pretrained(model_name)
            self.model.eval()
            
            # Get actual embedding dimension
            with torch.no_grad():
                dummy_text = self.processor(text=["test"], return_tensors="pt", padding=True)
                dummy_text = {k: v.to(device) for k, v in dummy_text.items() if k != "pixel_values"}
                dummy_out = self.model.get_text_features(**dummy_text)
                self.embedding_dim = dummy_out.shape[-1]
        else:
            self.model = None
            self.processor = None
            
    @torch.no_grad()
    def encode_text(self, texts: List[str], normalize: bool = True) -> torch.Tensor:
        """
        Encode text descriptions into embeddings.
        
        Args:
            texts: List of text descriptions
            normalize: Whether to L2-normalize embeddings
            
        Returns:
            Text embeddings [N, embedding_dim]
        """
        if not HAS_TRANSFORMERS or self.model is None:
            # Return dummy embeddings if CLIP not available
            return torch.randn(len(texts), self.embedding_dim).to(self.device)
        
        inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items() if k != "pixel_values"}
        
        text_features = self.model.get_text_features(**inputs)
        
        if normalize:
            text_features = F.normalize(text_features, p=2, dim=-1)
            
        return text_features
    
    @torch.no_grad()
    def encode_images(self, images: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """
        Encode images into embeddings.
        
        Args:
            images: Image tensor [N, C, H, W] in range [0, 1]
            normalize: Whether to L2-normalize embeddings
            
        Returns:
            Image embeddings [N, embedding_dim]
        """
        if not HAS_TRANSFORMERS or self.model is None:
            return torch.randn(images.size(0), self.embedding_dim).to(self.device)
        
        # CLIP expects specific preprocessing
        # The processor handles normalization
        images = images.to(self.device)
        
        # Process images (resize and normalize for CLIP)
        processed = self.processor(images=images, return_tensors="pt", do_rescale=False)
        pixel_values = processed["pixel_values"].to(self.device)
        
        image_features = self.model.get_image_features(pixel_values=pixel_values)
        
        if normalize:
            image_features = F.normalize(image_features, p=2, dim=-1)
            
        return image_features
    
    @torch.no_grad()
    def encode_images_from_pil(self, images: List, normalize: bool = True) -> torch.Tensor:
        """
        Encode PIL images into embeddings.
        
        Args:
            images: List of PIL images
            normalize: Whether to L2-normalize embeddings
            
        Returns:
            Image embeddings [N, embedding_dim]
        """
        if not HAS_TRANSFORMERS or self.model is None:
            return torch.randn(len(images), self.embedding_dim).to(self.device)
        
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        image_features = self.model.get_image_features(**inputs)
        
        if normalize:
            image_features = F.normalize(image_features, p=2, dim=-1)
            
        return image_features
    
    def get_embedding_dim(self) -> int:
        """Return the embedding dimension."""
        return self.embedding_dim


class ConceptEmbedding:
    """
    Manages concept embeddings for VL-CAV.
    
    Creates unified concept embeddings by combining:
    - Text descriptions (from VLM text encoder)
    - Visual examples (from VLM vision encoder)
    """
    
    def __init__(self, vlm_encoder: VLMEncoder, alpha: float = 0.7):
        """
        Initialize concept embedding manager.
        
        Args:
            vlm_encoder: VLM encoder instance
            alpha: Text-vision balance (0=all vision, 1=all text)
        """
        self.vlm_encoder = vlm_encoder
        self.alpha = alpha
        self.concept_embeddings: Dict[int, torch.Tensor] = {}
        self.text_embeddings: Dict[int, torch.Tensor] = {}
        self.visual_embeddings: Dict[int, torch.Tensor] = {}
        
    def compute_text_embedding(self, class_idx: int, descriptions: List[str]) -> torch.Tensor:
        """
        Compute text-based concept embedding.
        
        Args:
            class_idx: Class index
            descriptions: List of text descriptions
            
        Returns:
            Mean text embedding
        """
        text_emb = self.vlm_encoder.encode_text(descriptions)
        mean_emb = text_emb.mean(dim=0)
        mean_emb = F.normalize(mean_emb, p=2, dim=0)
        self.text_embeddings[class_idx] = mean_emb
        return mean_emb
    
    def compute_visual_embedding(self, class_idx: int, images: torch.Tensor) -> torch.Tensor:
        """
        Compute vision-based concept embedding.
        
        Args:
            class_idx: Class index
            images: Tensor of images [N, C, H, W]
            
        Returns:
            Mean visual embedding
        """
        visual_emb = self.vlm_encoder.encode_images(images)
        mean_emb = visual_emb.mean(dim=0)
        mean_emb = F.normalize(mean_emb, p=2, dim=0)
        self.visual_embeddings[class_idx] = mean_emb
        return mean_emb
    
    def compute_unified_embedding(self, class_idx: int, descriptions: List[str],
                                   images: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute unified multimodal concept embedding.
        
        Combines text and visual embeddings using alpha weighting:
        s_unified = alpha * s_text + (1-alpha) * s_visual
        
        When images are not available (or very few), relies more on text (alpha -> 1).
        
        Args:
            class_idx: Class index
            descriptions: List of text descriptions
            images: Optional tensor of images [N, C, H, W]
            
        Returns:
            Unified concept embedding
        """
        # Compute text embedding
        text_emb = self.compute_text_embedding(class_idx, descriptions)
        
        if images is not None and images.size(0) > 0:
            # Compute visual embedding
            visual_emb = self.compute_visual_embedding(class_idx, images)
            
            # Adapt alpha based on number of visual examples
            # Fewer examples -> rely more on text
            num_images = images.size(0)
            adaptive_alpha = self.alpha
            if num_images < 10:
                adaptive_alpha = min(0.9, self.alpha + 0.2)
            elif num_images < 5:
                adaptive_alpha = 0.95
                
            # Combine embeddings
            unified_emb = adaptive_alpha * text_emb + (1 - adaptive_alpha) * visual_emb
        else:
            # No visual examples, use text only
            unified_emb = text_emb
            
        # Normalize
        unified_emb = F.normalize(unified_emb, p=2, dim=0)
        self.concept_embeddings[class_idx] = unified_emb
        
        return unified_emb
    
    def get_concept_embedding(self, class_idx: int) -> Optional[torch.Tensor]:
        """Get precomputed concept embedding for a class."""
        return self.concept_embeddings.get(class_idx, None)
    
    def get_all_embeddings(self) -> Dict[int, torch.Tensor]:
        """Get all concept embeddings."""
        return self.concept_embeddings
    
    def get_embedding_matrix(self, class_indices: List[int]) -> torch.Tensor:
        """
        Get embeddings for specified classes as a matrix.
        
        Args:
            class_indices: List of class indices
            
        Returns:
            Embedding matrix [num_classes, embedding_dim]
        """
        embeddings = []
        for idx in class_indices:
            if idx in self.concept_embeddings:
                embeddings.append(self.concept_embeddings[idx])
            else:
                raise ValueError(f"Concept embedding not found for class {idx}")
        return torch.stack(embeddings)


class CrossSpaceProjection(nn.Module):
    """
    Learns projection from CNN activation space to VLM embedding space.
    
    φ: R^d_cnn -> R^d_vlm
    
    This enables computing VL-CAVs that align CNN activations with VLM concepts.
    """
    
    def __init__(self, cnn_dim: int, vlm_dim: int, hidden_dim: int = 256):
        """
        Initialize cross-space projection.
        
        Args:
            cnn_dim: Dimension of flattened CNN activations
            vlm_dim: Dimension of VLM embeddings
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        
        self.projection = nn.Sequential(
            nn.Linear(cnn_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, vlm_dim)
        )
        
        # Initialize weights
        for m in self.projection:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)
                
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project CNN activations to VLM space.
        
        Args:
            x: CNN activations [batch, cnn_dim]
            
        Returns:
            Projected embeddings [batch, vlm_dim]
        """
        projected = self.projection(x)
        return F.normalize(projected, p=2, dim=-1)


class VLCAVComputer:
    """
    Computes VLM-Augmented Concept Activation Vectors (VL-CAVs).
    
    Uses unified concept embeddings from VLM space and projects them
    back to CNN activation space to serve as concept directions.
    """
    
    def __init__(self, vlm_encoder: VLMEncoder, concept_embedding: ConceptEmbedding,
                 cnn_dim: int, device: str = "cuda"):
        """
        Initialize VL-CAV computer.
        
        Args:
            vlm_encoder: VLM encoder instance
            concept_embedding: Concept embedding manager
            cnn_dim: Dimension of flattened CNN activations
            device: Device to run on
        """
        self.vlm_encoder = vlm_encoder
        self.concept_embedding = concept_embedding
        self.device = device
        
        # Create projection network
        vlm_dim = vlm_encoder.get_embedding_dim()
        self.projection = CrossSpaceProjection(cnn_dim, vlm_dim).to(device)
        
        # Store VL-CAVs
        self.vl_cavs: Dict[int, torch.Tensor] = {}
        
    def train_projection(self, cnn_activations: torch.Tensor, 
                         vlm_embeddings: torch.Tensor,
                         epochs: int = 100, lr: float = 1e-3) -> float:
        """
        Train the cross-space projection using paired data.
        
        Objective: min ||φ(a_cnn) - s_vlm||^2
        
        Args:
            cnn_activations: CNN activations [N, cnn_dim]
            vlm_embeddings: Corresponding VLM embeddings [N, vlm_dim]
            epochs: Training epochs
            lr: Learning rate
            
        Returns:
            Final loss value
        """
        self.projection.train()
        optimizer = torch.optim.Adam(self.projection.parameters(), lr=lr)
        
        cnn_activations = cnn_activations.to(self.device)
        vlm_embeddings = vlm_embeddings.to(self.device)
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            projected = self.projection(cnn_activations)
            loss = F.mse_loss(projected, vlm_embeddings)
            
            loss.backward()
            optimizer.step()
            
        self.projection.eval()
        return loss.item()
    
    def compute_vl_cav(self, class_idx: int) -> torch.Tensor:
        """
        Compute VL-CAV for a class.
        
        The VL-CAV is the concept embedding in VLM space,
        which will be used for alignment in that space.
        
        Args:
            class_idx: Class index
            
        Returns:
            VL-CAV vector
        """
        concept_emb = self.concept_embedding.get_concept_embedding(class_idx)
        if concept_emb is None:
            raise ValueError(f"Concept embedding not found for class {class_idx}")
        
        self.vl_cavs[class_idx] = concept_emb
        return concept_emb
    
    def get_vl_cav(self, class_idx: int) -> Optional[torch.Tensor]:
        """Get precomputed VL-CAV for a class."""
        return self.vl_cavs.get(class_idx, None)
    
    def project_activations(self, activations: torch.Tensor) -> torch.Tensor:
        """
        Project CNN activations to VLM space.
        
        Args:
            activations: CNN activations [batch, cnn_dim]
            
        Returns:
            Projected embeddings [batch, vlm_dim]
        """
        self.projection.eval()
        with torch.no_grad():
            return self.projection(activations.to(self.device))
    
    def compute_alignment_score(self, activations: torch.Tensor, 
                                class_idx: int) -> torch.Tensor:
        """
        Compute alignment between activations and class concept.
        
        Args:
            activations: CNN activations [batch, cnn_dim]
            class_idx: Class index
            
        Returns:
            Cosine similarity scores [batch]
        """
        projected = self.project_activations(activations)
        vl_cav = self.get_vl_cav(class_idx)
        
        if vl_cav is None:
            vl_cav = self.compute_vl_cav(class_idx)
            
        # Cosine similarity
        return F.cosine_similarity(projected, vl_cav.unsqueeze(0), dim=-1)
