# -*- coding: utf-8 -*-
"""
Dependencies & Setup for RandLA-GraphAlignNet
Extends RandLA-Net for semantic graph alignment and multilingual spatial reasoning
"""

import sys
import os
from typing import Dict, List, Any, Optional, Tuple, Union
import warnings

# Core dependencies
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Point cloud processing
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    warnings.warn("Open3D not available. Point cloud visualization will be limited.")

# Graph processing
import networkx as nx
try:
    from torch_geometric.data import Data, Batch
    from torch_geometric.nn import GCNConv, GATConv, MessagePassing
    from torch_geometric.utils import add_self_loops, degree
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    warnings.warn("PyTorch Geometric not available. Graph operations will be limited.")

# Scientific computing
import scipy.spatial
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score, f1_score
from sklearn.manifold import TSNE
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
try:
    import plotly.graph_objects as go
    import plotly.express as px
    import dash
    from dash import dcc, html, Input, Output
    DASH_AVAILABLE = True
except ImportError:
    DASH_AVAILABLE = False
    warnings.warn("Dash not available. Interactive visualization will be limited.")

# RDF and semantic web (from our previous implementation)
try:
    from rdflib import Graph, URIRef, Literal, Namespace
    from rdflib.namespace import RDF, RDFS, OWL, SKOS
    RDF_AVAILABLE = True
except ImportError:
    RDF_AVAILABLE = False
    warnings.warn("RDFLib not available. RDF operations will be limited.")

# Add project paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

# Import LIMIT-Graph components
try:
    from extensions.LIMIT_GRAPH.rdf import create_full_rdf_system
    from extensions.LIMIT_GRAPH.agents.entity_linker import EntityLinker
    from extensions.LIMIT_GRAPH.agents.graph_reasoner import GraphReasoner
    LIMIT_GRAPH_AVAILABLE = True
except ImportError:
    LIMIT_GRAPH_AVAILABLE = False
    warnings.warn("LIMIT-Graph components not available. Some features will be limited.")

# Multilingual support
import re
from collections import defaultdict
import unicodedata

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔧 Using device: {DEVICE}")

# Supported languages for multilingual spatial reasoning
SUPPORTED_LANGUAGES = {
    'en': {'name': 'English', 'rtl': False, 'spatial_terms': ['above', 'below', 'left', 'right', 'near', 'far', 'inside', 'outside']},
    'es': {'name': 'Spanish', 'rtl': False, 'spatial_terms': ['arriba', 'abajo', 'izquierda', 'derecha', 'cerca', 'lejos', 'dentro', 'fuera']},
    'ar': {'name': 'Arabic', 'rtl': True, 'spatial_terms': ['فوق', 'تحت', 'يسار', 'يمين', 'قريب', 'بعيد', 'داخل', 'خارج']},
    'id': {'name': 'Indonesian', 'rtl': False, 'spatial_terms': ['atas', 'bawah', 'kiri', 'kanan', 'dekat', 'jauh', 'dalam', 'luar']},
    'zh': {'name': 'Chinese', 'rtl': False, 'spatial_terms': ['上', '下', '左', '右', '近', '远', '内', '外']},
    'hi': {'name': 'Hindi', 'rtl': False, 'spatial_terms': ['ऊपर', 'नीचे', 'बाएं', 'दाएं', 'पास', 'दूर', 'अंदर', 'बाहर']}
}

# RandLA-Net specific configurations
RANDLA_CONFIG = {
    'num_layers': 4,
    'num_points': 4096,
    'num_classes': 13,  # Standard semantic segmentation classes
    'sub_sampling_ratio': [4, 4, 4, 4],
    'decimation': 4,
    'k_n': 16,  # Number of neighbors for local spatial encoding
    'feature_dim': 8,
    'dropout_prob': 0.5
}

# Graph alignment configurations
GRAPH_ALIGN_CONFIG = {
    'embedding_dim': 256,
    'graph_hidden_dim': 128,
    'num_attention_heads': 8,
    'alignment_layers': 3,
    'temperature': 0.1,
    'margin': 0.5,
    'lambda_align': 1.0,
    'lambda_semantic': 0.5,
    'lambda_spatial': 0.3
}

def check_dependencies():
    """Check if all required dependencies are available"""
    
    dependencies_status = {
        'torch': torch.__version__,
        'numpy': np.__version__,
        'networkx': nx.__version__,
        'pandas': pd.__version__,
        'open3d': o3d.__version__ if OPEN3D_AVAILABLE else 'Not available',
        'torch_geometric': 'Available' if TORCH_GEOMETRIC_AVAILABLE else 'Not available',
        'dash': dash.__version__ if DASH_AVAILABLE else 'Not available',
        'rdflib': 'Available' if RDF_AVAILABLE else 'Not available',
        'limit_graph': 'Available' if LIMIT_GRAPH_AVAILABLE else 'Not available'
    }
    
    print("📋 Dependency Status:")
    for dep, version in dependencies_status.items():
        status = "✅" if version != 'Not available' else "❌"
        print(f"  {status} {dep}: {version}")
    
    return dependencies_status

def setup_environment():
    """Setup the environment for RandLA-GraphAlignNet"""
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Configure matplotlib for better plots
    plt.style.use('default')
    try:
        sns.set_palette("husl")
    except:
        pass  # Seaborn might not be available
    
    # Create output directories
    output_dirs = [
        'output/randla_graph_align',
        'output/randla_graph_align/models',
        'output/randla_graph_align/visualizations',
        'output/randla_graph_align/evaluations',
        'output/randla_graph_align/multilingual_annotations'
    ]
    
    for dir_path in output_dirs:
        os.makedirs(dir_path, exist_ok=True)
    
    print("🔧 Environment setup complete")
    return output_dirs

def get_device_info():
    """Get detailed device information"""
    
    device_info = {
        'device': str(DEVICE),
        'cuda_available': torch.cuda.is_available(),
        'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
    }
    
    if torch.cuda.is_available():
        device_info['cuda_device_name'] = torch.cuda.get_device_name(0)
        device_info['cuda_memory'] = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    return device_info

# Utility functions for point cloud processing
def normalize_point_cloud(points):
    """Normalize point cloud to unit sphere"""
    centroid = np.mean(points, axis=0)
    points_centered = points - centroid
    max_distance = np.max(np.linalg.norm(points_centered, axis=1))
    points_normalized = points_centered / max_distance
    return points_normalized, centroid, max_distance

def compute_point_features(points, normals=None):
    """Compute basic geometric features for points"""
    features = []
    
    # Height feature (z-coordinate)
    features.append(points[:, 2:3])
    
    # Distance from origin
    distances = np.linalg.norm(points, axis=1, keepdims=True)
    features.append(distances)
    
    # If normals are available, add them
    if normals is not None:
        features.append(normals)
    
    # Local density (requires k-NN computation)
    if len(points) > 10:
        nbrs = NearestNeighbors(n_neighbors=min(10, len(points)), algorithm='ball_tree').fit(points)
        distances, indices = nbrs.kneighbors(points)
        local_density = 1.0 / (np.mean(distances[:, 1:], axis=1, keepdims=True) + 1e-8)
        features.append(local_density)
    
    return np.concatenate(features, axis=1)

# Language processing utilities
def detect_spatial_language(text):
    """Detect language based on spatial terms"""
    text_lower = text.lower()
    
    language_scores = {}
    for lang_code, lang_info in SUPPORTED_LANGUAGES.items():
        score = sum(1 for term in lang_info['spatial_terms'] if term in text_lower)
        if score > 0:
            language_scores[lang_code] = score / len(lang_info['spatial_terms'])
    
    if language_scores:
        return max(language_scores.items(), key=lambda x: x[1])[0]
    return 'en'  # Default to English

def extract_spatial_relations(text, language='en'):
    """Extract spatial relations from text"""
    if language not in SUPPORTED_LANGUAGES:
        language = 'en'
    
    spatial_terms = SUPPORTED_LANGUAGES[language]['spatial_terms']
    relations = []
    
    # Simple pattern matching for spatial relations
    words = text.lower().split()
    for i, word in enumerate(words):
        if word in spatial_terms:
            # Try to find subject and object
            subject = words[i-1] if i > 0 else None
            object_word = words[i+1] if i < len(words)-1 else None
            
            if subject and object_word:
                relations.append({
                    'subject': subject,
                    'relation': word,
                    'object': object_word,
                    'language': language
                })
    
    return relations

# Installation instructions
INSTALLATION_INSTRUCTIONS = """
# RandLA-GraphAlignNet Installation Instructions

## 1. Create conda environment
conda create -n randla_graph python=3.8
conda activate randla_graph

## 2. Install PyTorch (choose appropriate version for your system)
# For CUDA 11.1
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html

# For CPU only
pip install torch==1.9.0+cpu torchvision==0.10.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

## 3. Install PyTorch Geometric
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.9.0+cu111.html

## 4. Install other dependencies
pip install numpy pandas scikit-learn networkx matplotlib seaborn
pip install open3d plotly dash
pip install rdflib sparqlwrapper

## 5. Install optional dependencies
pip install neo4j  # For Neo4j visualization
pip install tensorboard  # For training visualization

## 6. Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch_geometric; print('PyTorch Geometric: OK')"
python -c "import open3d; print(f'Open3D: {open3d.__version__}')"
"""

# Initialize on import
if __name__ == "__main__":
    print("🚀 RandLA-GraphAlignNet Dependencies")
    print("=" * 50)
    
    # Check dependencies
    deps = check_dependencies()
    
    # Setup environment
    output_dirs = setup_environment()
    
    # Show device info
    device_info = get_device_info()
    print(f"\n💻 Device Info:")
    for key, value in device_info.items():
        print(f"  {key}: {value}")
    
    # Show configuration
    print(f"\n⚙️ Configuration:")
    print(f"  RandLA-Net layers: {RANDLA_CONFIG['num_layers']}")
    print(f"  Graph embedding dim: {GRAPH_ALIGN_CONFIG['embedding_dim']}")
    print(f"  Supported languages: {len(SUPPORTED_LANGUAGES)}")
    
    print("\n✅ Dependencies setup complete!")
    
    # Show installation instructions if dependencies are missing
    missing_deps = [k for k, v in deps.items() if v == 'Not available']
    if missing_deps:
        print(f"\n⚠️ Missing dependencies: {missing_deps}")
        print("\n📋 Installation Instructions:")
        print(INSTALLATION_INSTRUCTIONS)
else:
    # Silent setup when imported
    try:
        setup_environment()
    except:
        pass  # Ignore errors during import
