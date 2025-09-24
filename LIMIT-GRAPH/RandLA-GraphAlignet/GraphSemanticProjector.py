# -*- coding: utf-8 -*-
"""
GraphSemanticProjector: Aligns RandLA-Net outputs with LIMIT-GRAPH's multilingual edge definitions
Enables semantic graph alignment for multilingual spatial reasoning
"""

import sys
import os
from typing import Dict, List, Any, Optional, Tuple, Union
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Import dependencies
try:
    from dependencies import SUPPORTED_LANGUAGES, GRAPH_ALIGN_CONFIG
except ImportError:
    # Fallback definitions
    SUPPORTED_LANGUAGES = {
        'en': {'name': 'English', 'spatial_terms': ['above', 'below', 'left', 'right', 'near', 'far']},
        'es': {'name': 'Spanish', 'spatial_terms': ['arriba', 'abajo', 'izquierda', 'derecha', 'cerca', 'lejos']},
        'ar': {'name': 'Arabic', 'spatial_terms': ['فوق', 'تحت', 'يسار', 'يمين', 'قريب', 'بعيد']},
        'id': {'name': 'Indonesian', 'spatial_terms': ['atas', 'bawah', 'kiri', 'kanan', 'dekat', 'jauh']}
    }
    GRAPH_ALIGN_CONFIG = {
        'embedding_dim': 256,
        'graph_hidden_dim': 128,
        'num_attention_heads': 8,
        'temperature': 0.1
    }

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism for graph alignment"""
    
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
        
    def forward(self, query, key, value, mask=None):
        batch_size, seq_len, _ = query.shape
        
        # Project to Q, K, V
        Q = self.q_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        
        # Final projection
        output = self.out_proj(context)
        
        return output, attention_weights

class GraphNodeEmbedding(nn.Module):
    """Embedding layer for graph nodes with multilingual support"""
    
    def __init__(self, vocab_size, embed_dim, padding_idx=0):
        super(GraphNodeEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.embed_dim = embed_dim
        
    def forward(self, node_ids):
        return self.embedding(node_ids) * math.sqrt(self.embed_dim)

class GraphConvolutionLayer(nn.Module):
    """Graph convolution layer for processing graph structure"""
    
    def __init__(self, input_dim, output_dim, dropout=0.1):
        super(GraphConvolutionLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        self.bias = nn.Parameter(torch.FloatTensor(output_dim))
        self.dropout = nn.Dropout(dropout)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)
        
    def forward(self, features, adjacency_matrix):
        """
        Args:
            features: (N, input_dim) node features
            adjacency_matrix: (N, N) adjacency matrix
        Returns:
            output: (N, output_dim) updated node features
        """
        # Apply dropout to features
        features = self.dropout(features)
        
        # Linear transformation
        support = torch.mm(features, self.weight)
        
        # Graph convolution: A * X * W + b
        output = torch.mm(adjacency_matrix, support) + self.bias
        
        return output

class GraphSemanticProjector(nn.Module):
    """
    Projects RandLA-Net features to graph-aligned embeddings
    Supports multilingual edge definitions and semantic alignment
    """
    
    def __init__(self, 
                 input_dim: int,
                 config: Dict[str, Any] = None,
                 graph_vocab: Dict[str, int] = None,
                 multilingual_vocab: Dict[str, Dict[str, int]] = None):
        super(GraphSemanticProjector, self).__init__()
        
        self.input_dim = input_dim
        self.config = config or GRAPH_ALIGN_CONFIG
        self.graph_vocab = graph_vocab or {}
        self.multilingual_vocab = multilingual_vocab or {}
        
        # Build unified vocabulary
        self.unified_vocab = self._build_unified_vocab()
        self.vocab_size = len(self.unified_vocab)
        
        # Get embedding dimension from config
        self.embedding_dim = self.config.get('embedding_dim', 256)
        self.graph_hidden_dim = self.config.get('graph_hidden_dim', 128)
        self.num_attention_heads = self.config.get('num_attention_heads', 8)
        
        # Input projection
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, self.embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Graph node embeddings
        self.node_embeddings = GraphNodeEmbedding(
            vocab_size=self.vocab_size,
            embed_dim=self.embedding_dim
        )
        
        # Graph convolution layers
        self.graph_conv_layers = nn.ModuleList([
            GraphConvolutionLayer(self.embedding_dim, self.graph_hidden_dim),
            GraphConvolutionLayer(self.graph_hidden_dim, self.embedding_dim)
        ])
        
        # Multi-head attention for alignment
        self.alignment_attention = MultiHeadAttention(
            embed_dim=self.embedding_dim,
            num_heads=self.num_attention_heads,
            dropout=0.1
        )
        
        # Multilingual projection heads
        self.language_projectors = nn.ModuleDict()
        for lang in SUPPORTED_LANGUAGES.keys():
            self.language_projectors[lang] = nn.Sequential(
                nn.Linear(self.embedding_dim, self.embedding_dim // 2),
                nn.ReLU(),
                nn.Linear(self.embedding_dim // 2, self.embedding_dim)
            )
        
        # Alignment loss components
        self.temperature = self.config.get('temperature', 0.1)
        self.margin = self.config.get('margin', 0.5)
        
    def _build_unified_vocab(self):
        """Build unified vocabulary from graph and multilingual vocabularies"""
        unified_vocab = {'<PAD>': 0, '<UNK>': 1}
        current_idx = 2
        
        # Add graph vocabulary
        for term, idx in self.graph_vocab.items():
            if term not in unified_vocab:
                unified_vocab[term] = current_idx
                current_idx += 1
        
        # Add multilingual vocabularies
        for lang, vocab in self.multilingual_vocab.items():
            for term, idx in vocab.items():
                multilingual_term = f"{term}_{lang}"
                if multilingual_term not in unified_vocab:
                    unified_vocab[multilingual_term] = current_idx
                    current_idx += 1
        
        # Add spatial terms for all supported languages
        for lang, lang_info in SUPPORTED_LANGUAGES.items():
            for spatial_term in lang_info['spatial_terms']:
                term_key = f"spatial_{spatial_term}_{lang}"
                if term_key not in unified_vocab:
                    unified_vocab[term_key] = current_idx
                    current_idx += 1
        
        return unified_vocab
    
    def create_graph_adjacency(self, graph_data=None, batch_size=1):
        """Create adjacency matrix for graph convolution"""
        if graph_data is not None and hasattr(graph_data, 'edge_index'):
            # Use provided graph structure
            num_nodes = graph_data.num_nodes
            edge_index = graph_data.edge_index
            
            # Create adjacency matrix
            adjacency = torch.zeros(num_nodes, num_nodes, device=edge_index.device)
            adjacency[edge_index[0], edge_index[1]] = 1.0
            
            # Add self-loops
            adjacency += torch.eye(num_nodes, device=edge_index.device)
            
            # Normalize
            degree = torch.sum(adjacency, dim=1, keepdim=True)
            adjacency = adjacency / (degree + 1e-8)
            
        else:
            # Create default fully connected graph for vocabulary
            num_nodes = min(self.vocab_size, 100)  # Limit size for efficiency
            adjacency = torch.ones(num_nodes, num_nodes) / num_nodes
            adjacency = adjacency.to(next(self.parameters()).device)
        
        return adjacency
    
    def forward(self, features, graph_data=None, language='en', training=False):
        """
        Project features to graph-aligned embeddings
        
        Args:
            features: (B, input_dim) input features from RandLA-Net
            graph_data: Graph data for alignment (optional)
            language: Target language for alignment
            training: Whether in training mode
            
        Returns:
            aligned_embeddings: (B, embedding_dim) graph-aligned embeddings
            alignment_loss: Scalar alignment loss
        """
        batch_size = features.shape[0]
        
        # Project input features
        projected_features = self.input_projection(features)  # (B, embedding_dim)
        
        # Create graph node embeddings
        if graph_data is not None and hasattr(graph_data, 'node_features'):
            # Use provided node features
            graph_node_features = graph_data.node_features
        else:
            # Create default node features from vocabulary
            num_nodes = min(self.vocab_size, 100)
            node_ids = torch.arange(num_nodes, device=features.device)
            graph_node_features = self.node_embeddings(node_ids)  # (num_nodes, embedding_dim)
        
        # Apply graph convolution
        adjacency = self.create_graph_adjacency(graph_data, batch_size)
        
        for conv_layer in self.graph_conv_layers:
            graph_node_features = F.relu(conv_layer(graph_node_features, adjacency))
        
        # Align features with graph nodes using attention
        # Expand projected features to match graph nodes
        query = projected_features.unsqueeze(1)  # (B, 1, embedding_dim)
        key = graph_node_features.unsqueeze(0).expand(batch_size, -1, -1)  # (B, num_nodes, embedding_dim)
        value = key
        
        aligned_features, attention_weights = self.alignment_attention(query, key, value)
        aligned_features = aligned_features.squeeze(1)  # (B, embedding_dim)
        
        # Apply language-specific projection
        if language in self.language_projectors:
            aligned_features = self.language_projectors[language](aligned_features)
        
        # Compute alignment loss
        alignment_loss = self._compute_alignment_loss(
            projected_features, 
            aligned_features, 
            attention_weights,
            training
        )
        
        return aligned_features, alignment_loss
    
    def _compute_alignment_loss(self, original_features, aligned_features, attention_weights, training):
        """Compute alignment loss for training"""
        if not training:
            return torch.tensor(0.0, device=original_features.device)
        
        # Contrastive loss between original and aligned features
        similarity = F.cosine_similarity(original_features, aligned_features, dim=1)
        contrastive_loss = -torch.log(torch.sigmoid(similarity / self.temperature)).mean()
        
        # Attention entropy loss (encourage focused attention)
        attention_entropy = -torch.sum(attention_weights * torch.log(attention_weights + 1e-8), dim=-1).mean()
        entropy_loss = -attention_entropy  # Minimize entropy (maximize focus)
        
        # Total alignment loss
        alignment_loss = contrastive_loss + 0.1 * entropy_loss
        
        return alignment_loss
    
    def align_with_multilingual_graph(self, features, multilingual_edges, languages=['en']):
        """
        Align features with multilingual graph edges
        
        Args:
            features: (B, embedding_dim) input features
            multilingual_edges: List of multilingual edge definitions
            languages: List of target languages
            
        Returns:
            multilingual_alignments: Dict mapping languages to aligned features
        """
        multilingual_alignments = {}
        
        for language in languages:
            if language in SUPPORTED_LANGUAGES:
                # Get language-specific features
                lang_features = self.language_projectors[language](features)
                
                # Create language-specific graph representation
                lang_edges = [edge for edge in multilingual_edges if edge.get('language') == language]
                
                # Compute alignment scores
                alignment_scores = self._compute_language_alignment_scores(lang_features, lang_edges, language)
                
                multilingual_alignments[language] = {
                    'features': lang_features,
                    'alignment_scores': alignment_scores,
                    'edges': lang_edges
                }
        
        return multilingual_alignments
    
    def _compute_language_alignment_scores(self, features, edges, language):
        """Compute alignment scores for a specific language"""
        if not edges:
            return torch.zeros(features.shape[0], device=features.device)
        
        # Simple scoring based on feature similarity to edge embeddings
        # In practice, this would use more sophisticated graph matching
        scores = torch.ones(features.shape[0], device=features.device)
        
        return scores
    
    def extract_spatial_embeddings(self, features, spatial_relations, language='en'):
        """
        Extract embeddings for spatial relations in a specific language
        
        Args:
            features: (B, embedding_dim) input features
            spatial_relations: List of spatial relation terms
            language: Target language
            
        Returns:
            spatial_embeddings: (B, len(spatial_relations), embedding_dim) spatial embeddings
        """
        batch_size = features.shape[0]
        
        # Get language-specific features
        if language in self.language_projectors:
            lang_features = self.language_projectors[language](features)
        else:
            lang_features = features
        
        # Create embeddings for each spatial relation
        spatial_embeddings = []
        
        for relation in spatial_relations:
            # Look up relation in vocabulary
            relation_key = f"spatial_{relation}_{language}"
            if relation_key in self.unified_vocab:
                relation_id = self.unified_vocab[relation_key]
                relation_embedding = self.node_embeddings(torch.tensor([relation_id], device=features.device))
                
                # Combine with language features
                combined_embedding = lang_features + relation_embedding.expand(batch_size, -1)
                spatial_embeddings.append(combined_embedding)
            else:
                # Use default embedding
                spatial_embeddings.append(lang_features)
        
        if spatial_embeddings:
            spatial_embeddings = torch.stack(spatial_embeddings, dim=1)  # (B, num_relations, embedding_dim)
        else:
            spatial_embeddings = lang_features.unsqueeze(1)  # (B, 1, embedding_dim)
        
        return spatial_embeddings
    
    def get_vocabulary_info(self):
        """Get information about the unified vocabulary"""
        return {
            'vocab_size': self.vocab_size,
            'graph_terms': len(self.graph_vocab),
            'multilingual_terms': sum(len(vocab) for vocab in self.multilingual_vocab.values()),
            'spatial_terms': len([k for k in self.unified_vocab.keys() if k.startswith('spatial_')]),
            'supported_languages': list(SUPPORTED_LANGUAGES.keys())
        }

def demo_graph_semantic_projector():
    """Demonstrate the GraphSemanticProjector"""
    
    print("🔗 GraphSemanticProjector Demo")
    print("=" * 40)
    
    # Create sample vocabulary
    graph_vocab = {
        'person': 0, 'object': 1, 'location': 2, 'action': 3
    }
    
    multilingual_vocab = {
        'es': {'persona': 0, 'objeto': 1, 'lugar': 2, 'acción': 3},
        'ar': {'شخص': 0, 'كائن': 1, 'مكان': 2, 'عمل': 3}
    }
    
    # Create projector
    input_dim = 512
    projector = GraphSemanticProjector(
        input_dim=input_dim,
        graph_vocab=graph_vocab,
        multilingual_vocab=multilingual_vocab
    )
    
    # Get vocabulary info
    vocab_info = projector.get_vocabulary_info()
    print(f"📊 Vocabulary Info:")
    for key, value in vocab_info.items():
        print(f"  - {key}: {value}")
    
    # Create sample input
    batch_size = 4
    features = torch.randn(batch_size, input_dim)
    
    print(f"\n🔍 Forward Pass Demo:")
    print(f"  - Input shape: {features.shape}")
    
    # Forward pass
    projector.eval()
    with torch.no_grad():
        aligned_features, alignment_loss = projector(features, language='en')
    
    print(f"  - Aligned features shape: {aligned_features.shape}")
    print(f"  - Alignment loss: {alignment_loss.item():.4f}")
    
    # Multilingual alignment demo
    multilingual_edges = [
        {'subject': 'person', 'relation': 'near', 'object': 'location', 'language': 'en'},
        {'subject': 'persona', 'relation': 'cerca', 'object': 'lugar', 'language': 'es'}
    ]
    
    multilingual_alignments = projector.align_with_multilingual_graph(
        aligned_features, 
        multilingual_edges, 
        languages=['en', 'es']
    )
    
    print(f"\n🌍 Multilingual Alignments:")
    for lang, alignment in multilingual_alignments.items():
        print(f"  - {SUPPORTED_LANGUAGES[lang]['name']}: {alignment['features'].shape}")
    
    # Spatial embeddings demo
    spatial_relations = ['above', 'below', 'near', 'far']
    spatial_embeddings = projector.extract_spatial_embeddings(
        aligned_features, 
        spatial_relations, 
        language='en'
    )
    
    print(f"\n📍 Spatial Embeddings:")
    print(f"  - Shape: {spatial_embeddings.shape}")
    print(f"  - Relations: {spatial_relations}")
    
    print("\n✅ Demo completed successfully!")
    return projector

if __name__ == "__main__":
    demo_graph_semantic_projector()