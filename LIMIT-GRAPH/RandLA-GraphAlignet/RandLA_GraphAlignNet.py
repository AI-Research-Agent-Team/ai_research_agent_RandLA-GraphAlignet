# -*- coding: utf-8 -*-
"""
RandLA-GraphAlignNet: Architecture Overview
Extends RandLA-Net to produce graph-aligned embeddings from 3D point clouds
Enables multilingual spatial annotation and reasoning through semantic graph alignment
"""

import sys
import os
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
import math

# Import dependencies
from dependencies import *

@dataclass
class RandLAConfig:
    """Configuration for RandLA-Net components"""
    num_layers: int = 4
    num_points: int = 4096
    num_classes: int = 13
    sub_sampling_ratio: List[int] = None
    decimation: int = 4
    k_n: int = 16
    feature_dim: int = 8
    dropout_prob: float = 0.5
    
    def __post_init__(self):
        if self.sub_sampling_ratio is None:
            self.sub_sampling_ratio = [4, 4, 4, 4]

@dataclass
class GraphAlignConfig:
    """Configuration for graph alignment components"""
    embedding_dim: int = 256
    graph_hidden_dim: int = 128
    num_attention_heads: int = 8
    alignment_layers: int = 3
    temperature: float = 0.1
    margin: float = 0.5
    lambda_align: float = 1.0
    lambda_semantic: float = 0.5
    lambda_spatial: float = 0.3

class LocalSpatialEncoding(nn.Module):
    """Local Spatial Encoding (LSE) module from RandLA-Net"""
    
    def __init__(self, feature_dim, k_n=16):
        super(LocalSpatialEncoding, self).__init__()
        self.k_n = k_n
        self.feature_dim = feature_dim
        
        # MLP for encoding relative positions
        self.mlp = nn.Sequential(
            nn.Linear(10, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, feature_dim)
        )
        
    def forward(self, points, features):
        """
        Args:
            points: (B, N, 3) point coordinates
            features: (B, N, C) point features
        Returns:
            encoded_features: (B, N, feature_dim) encoded features
        """
        batch_size, num_points, _ = points.shape
        
        # Find k-nearest neighbors
        distances = torch.cdist(points, points)  # (B, N, N)
        _, knn_indices = torch.topk(distances, self.k_n, dim=-1, largest=False)  # (B, N, k_n)
        
        # Gather neighbor points and features
        knn_points = torch.gather(points.unsqueeze(2).expand(-1, -1, self.k_n, -1), 
                                 1, knn_indices.unsqueeze(-1).expand(-1, -1, -1, 3))  # (B, N, k_n, 3)
        knn_features = torch.gather(features.unsqueeze(2).expand(-1, -1, self.k_n, -1),
                                   1, knn_indices.unsqueeze(-1).expand(-1, -1, -1, features.shape[-1]))  # (B, N, k_n, C)
        
        # Compute relative positions and distances
        center_points = points.unsqueeze(2)  # (B, N, 1, 3)
        relative_pos = knn_points - center_points  # (B, N, k_n, 3)
        relative_dist = torch.norm(relative_pos, dim=-1, keepdim=True)  # (B, N, k_n, 1)
        
        # Compute relative features
        center_features = features.unsqueeze(2)  # (B, N, 1, C)
        relative_features = knn_features - center_features  # (B, N, k_n, C)
        
        # Concatenate all relative information
        relative_info = torch.cat([
            relative_pos,  # (B, N, k_n, 3)
            relative_dist,  # (B, N, k_n, 1)
            relative_features  # (B, N, k_n, C)
        ], dim=-1)  # (B, N, k_n, 3+1+C)
        
        # Pad or truncate to expected size (10 features)
        if relative_info.shape[-1] > 10:
            relative_info = relative_info[..., :10]
        elif relative_info.shape[-1] < 10:
            padding = torch.zeros(*relative_info.shape[:-1], 10 - relative_info.shape[-1], 
                                device=relative_info.device)
            relative_info = torch.cat([relative_info, padding], dim=-1)
        
        # Apply MLP to encode spatial relationships
        encoded = self.mlp(relative_info)  # (B, N, k_n, feature_dim)
        
        # Aggregate over neighbors (max pooling)
        encoded_features = torch.max(encoded, dim=2)[0]  # (B, N, feature_dim)
        
        return encoded_features

class AttentivePooling(nn.Module):
    """Attentive pooling module for RandLA-Net"""
    
    def __init__(self, feature_dim):
        super(AttentivePooling, self).__init__()
        self.feature_dim = feature_dim
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, features):
        """
        Args:
            features: (B, N, C) input features
        Returns:
            pooled_features: (B, C) pooled features
        """
        # Compute attention weights
        attention_weights = self.attention(features)  # (B, N, 1)
        
        # Apply attention weights
        weighted_features = features * attention_weights  # (B, N, C)
        
        # Sum over points
        pooled_features = torch.sum(weighted_features, dim=1)  # (B, C)
        
        return pooled_features

class RandLAEncoder(nn.Module):
    """RandLA-Net encoder for point cloud feature extraction"""
    
    def __init__(self, config: RandLAConfig):
        super(RandLAEncoder, self).__init__()
        self.config = config
        
        # Initial feature projection
        self.input_projection = nn.Linear(3 + config.feature_dim, 32)
        
        # Local spatial encoding layers
        self.lse_layers = nn.ModuleList()
        self.mlp_layers = nn.ModuleList()
        
        current_dim = 32
        for i in range(config.num_layers):
            # Local spatial encoding
            self.lse_layers.append(LocalSpatialEncoding(current_dim, config.k_n))
            
            # MLP for feature transformation
            next_dim = current_dim * 2
            self.mlp_layers.append(nn.Sequential(
                nn.Linear(current_dim, next_dim),
                nn.BatchNorm1d(next_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout_prob)
            ))
            current_dim = next_dim
        
        # Final feature dimension
        self.final_dim = current_dim
        
        # Attentive pooling for global features
        self.global_pooling = AttentivePooling(current_dim)
        
    def forward(self, points, features=None):
        """
        Args:
            points: (B, N, 3) point coordinates
            features: (B, N, F) point features (optional)
        Returns:
            point_features: (B, N, final_dim) point-wise features
            global_features: (B, final_dim) global features
        """
        batch_size, num_points, _ = points.shape
        
        # Compute basic features if not provided
        if features is None:
            features = compute_point_features(points.cpu().numpy())
            features = torch.from_numpy(features).to(points.device).float()
        
        # Concatenate points and features
        x = torch.cat([points, features], dim=-1)  # (B, N, 3+F)
        
        # Initial projection
        x = self.input_projection(x)  # (B, N, 32)
        
        # Apply LSE and MLP layers
        for lse, mlp in zip(self.lse_layers, self.mlp_layers):
            # Local spatial encoding
            spatial_features = lse(points, x)  # (B, N, current_dim)
            
            # MLP transformation (need to handle batch norm)
            x_reshaped = spatial_features.view(-1, spatial_features.shape[-1])  # (B*N, current_dim)
            x_transformed = mlp(x_reshaped)  # (B*N, next_dim)
            x = x_transformed.view(batch_size, num_points, -1)  # (B, N, next_dim)
        
        # Global features via attentive pooling
        global_features = self.global_pooling(x)  # (B, final_dim)
        
        return x, global_features

class RandLA_GraphAlignNet(nn.Module):
    """
    RandLA-GraphAlignNet: Main architecture that extends RandLA-Net
    for semantic graph alignment and multilingual spatial reasoning
    """
    
    def __init__(self, 
                 randla_config: RandLAConfig = None,
                 graph_config: GraphAlignConfig = None,
                 num_classes: int = 13,
                 graph_vocab: Dict[str, int] = None):
        super(RandLA_GraphAlignNet, self).__init__()
        
        # Use default configs if not provided
        self.randla_config = randla_config or RandLAConfig()
        self.graph_config = graph_config or GraphAlignConfig()
        self.num_classes = num_classes
        self.graph_vocab = graph_vocab or {}
        
        # RandLA-Net encoder
        self.encoder = RandLAEncoder(self.randla_config)
        
        # Graph semantic projector (will be imported from separate module)
        from GraphSemanticProjector import GraphSemanticProjector
        self.graph_projector = GraphSemanticProjector(
            input_dim=self.encoder.final_dim,
            config=self.graph_config,
            graph_vocab=self.graph_vocab
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.graph_config.embedding_dim, self.graph_config.embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.randla_config.dropout_prob),
            nn.Linear(self.graph_config.embedding_dim // 2, num_classes)
        )
        
        # Multilingual spatial reasoning head
        self.spatial_reasoner = nn.Sequential(
            nn.Linear(self.graph_config.embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, len(SUPPORTED_LANGUAGES) * 8)  # 8 spatial relations per language
        )
        
    def forward(self, points, features=None, graph_data=None, language='en', training=False):
        """
        Forward pass through RandLA-GraphAlignNet
        
        Args:
            points: (B, N, 3) point coordinates
            features: (B, N, F) point features (optional)
            graph_data: Graph data for alignment (optional)
            language: Target language for spatial reasoning
            training: Whether in training mode
            
        Returns:
            Dictionary containing:
                - logits: (B, num_classes) classification logits
                - point_features: (B, N, final_dim) point-wise features
                - global_features: (B, final_dim) global features
                - graph_embeddings: (B, embedding_dim) graph-aligned embeddings
                - spatial_relations: (B, 8) spatial relation predictions
        """
        
        # Extract features using RandLA-Net encoder
        point_features, global_features = self.encoder(points, features)
        
        # Project to graph-aligned embedding space
        graph_embeddings, alignment_loss = self.graph_projector(
            global_features, 
            graph_data=graph_data,
            training=training
        )
        
        # Classification
        logits = self.classifier(graph_embeddings)
        
        # Multilingual spatial reasoning
        spatial_logits = self.spatial_reasoner(graph_embeddings)
        
        # Reshape spatial logits for the target language
        lang_idx = list(SUPPORTED_LANGUAGES.keys()).index(language) if language in SUPPORTED_LANGUAGES else 0
        spatial_relations = spatial_logits.view(-1, len(SUPPORTED_LANGUAGES), 8)[:, lang_idx, :]
        
        results = {
            'logits': logits,
            'point_features': point_features,
            'global_features': global_features,
            'graph_embeddings': graph_embeddings,
            'spatial_relations': spatial_relations,
            'alignment_loss': alignment_loss
        }
        
        return results
    
    def compute_loss(self, predictions, targets, spatial_targets=None, alpha=1.0, beta=0.5):
        """
        Compute total loss including classification, alignment, and spatial reasoning
        
        Args:
            predictions: Model predictions dictionary
            targets: (B,) classification targets
            spatial_targets: (B, 8) spatial relation targets (optional)
            alpha: Weight for classification loss
            beta: Weight for spatial reasoning loss
            
        Returns:
            Dictionary containing individual and total losses
        """
        
        # Classification loss
        classification_loss = F.cross_entropy(predictions['logits'], targets)
        
        # Alignment loss (from graph projector)
        alignment_loss = predictions['alignment_loss']
        
        # Spatial reasoning loss
        spatial_loss = torch.tensor(0.0, device=predictions['logits'].device)
        if spatial_targets is not None:
            spatial_loss = F.mse_loss(predictions['spatial_relations'], spatial_targets)
        
        # Total loss
        total_loss = (alpha * classification_loss + 
                     self.graph_config.lambda_align * alignment_loss +
                     beta * spatial_loss)
        
        return {
            'total_loss': total_loss,
            'classification_loss': classification_loss,
            'alignment_loss': alignment_loss,
            'spatial_loss': spatial_loss
        }
    
    def extract_multilingual_features(self, points, features=None, languages=['en', 'es', 'ar', 'id']):
        """
        Extract features for multiple languages simultaneously
        
        Args:
            points: (B, N, 3) point coordinates
            features: (B, N, F) point features (optional)
            languages: List of target languages
            
        Returns:
            Dictionary mapping languages to their respective features
        """
        
        # Get base features
        point_features, global_features = self.encoder(points, features)
        graph_embeddings, _ = self.graph_projector(global_features, training=False)
        
        multilingual_features = {}
        
        for lang in languages:
            if lang in SUPPORTED_LANGUAGES:
                # Get spatial relations for this language
                spatial_logits = self.spatial_reasoner(graph_embeddings)
                lang_idx = list(SUPPORTED_LANGUAGES.keys()).index(lang)
                spatial_relations = spatial_logits.view(-1, len(SUPPORTED_LANGUAGES), 8)[:, lang_idx, :]
                
                multilingual_features[lang] = {
                    'embeddings': graph_embeddings,
                    'spatial_relations': spatial_relations,
                    'spatial_terms': SUPPORTED_LANGUAGES[lang]['spatial_terms']
                }
        
        return multilingual_features
    
    def get_model_info(self):
        """Get detailed model information"""
        
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': 'RandLA-GraphAlignNet',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'randla_config': self.randla_config,
            'graph_config': self.graph_config,
            'num_classes': self.num_classes,
            'supported_languages': list(SUPPORTED_LANGUAGES.keys()),
            'encoder_output_dim': self.encoder.final_dim,
            'graph_embedding_dim': self.graph_config.embedding_dim
        }

def create_model(num_classes=13, 
                randla_config=None, 
                graph_config=None,
                graph_vocab=None):
    """
    Factory function to create RandLA-GraphAlignNet model
    
    Args:
        num_classes: Number of classification classes
        randla_config: RandLA-Net configuration
        graph_config: Graph alignment configuration
        graph_vocab: Graph vocabulary for alignment
        
    Returns:
        RandLA_GraphAlignNet model instance
    """
    
    model = RandLA_GraphAlignNet(
        randla_config=randla_config,
        graph_config=graph_config,
        num_classes=num_classes,
        graph_vocab=graph_vocab
    )
    
    return model

def demo_model():
    """Demonstrate the RandLA-GraphAlignNet model"""
    
    print("🚀 RandLA-GraphAlignNet Demo")
    print("=" * 40)
    
    # Create model
    model = create_model(num_classes=13)
    model_info = model.get_model_info()
    
    print(f"📊 Model Info:")
    print(f"  - Total parameters: {model_info['total_parameters']:,}")
    print(f"  - Trainable parameters: {model_info['trainable_parameters']:,}")
    print(f"  - Encoder output dim: {model_info['encoder_output_dim']}")
    print(f"  - Graph embedding dim: {model_info['graph_embedding_dim']}")
    print(f"  - Supported languages: {len(model_info['supported_languages'])}")
    
    # Create dummy input
    batch_size = 2
    num_points = 1024
    points = torch.randn(batch_size, num_points, 3)
    
    print(f"\n🔍 Forward Pass Demo:")
    print(f"  - Input shape: {points.shape}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        results = model(points, language='en')
    
    print(f"  - Classification logits: {results['logits'].shape}")
    print(f"  - Point features: {results['point_features'].shape}")
    print(f"  - Global features: {results['global_features'].shape}")
    print(f"  - Graph embeddings: {results['graph_embeddings'].shape}")
    print(f"  - Spatial relations: {results['spatial_relations'].shape}")
    
    # Multilingual features
    multilingual_features = model.extract_multilingual_features(points, languages=['en', 'es', 'ar'])
    print(f"\n🌍 Multilingual Features:")
    for lang, features in multilingual_features.items():
        print(f"  - {SUPPORTED_LANGUAGES[lang]['name']}: {features['embeddings'].shape}")
    
    print("\n✅ Demo completed successfully!")
    return model

if __name__ == "__main__":
    demo_model()
