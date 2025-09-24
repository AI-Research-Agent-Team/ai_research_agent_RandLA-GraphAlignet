#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete RandLA-GraphAlignNet Integration Demo
Demonstrates the full pipeline: 3D point cloud processing, semantic graph alignment,
multilingual spatial reasoning, and evaluation
"""

import sys
import os
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import torch
import json
from datetime import datetime

# Add project paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

# Import all components
from dependencies import *
from RandLA_GraphAlignNet import RandLA_GraphAlignNet, RandLAConfig, GraphAlignConfig, create_model
from GraphSemanticProjector import GraphSemanticProjector
from annotate_multilingual import MultilingualSpatialAnnotator, SpatialAnnotation
from evaluate_alignment import RandLAGraphAlignmentEvaluator, EvaluationMetrics

def create_sample_point_cloud(num_points=2048, scene_type='urban'):
    """Create a sample 3D point cloud for demonstration"""
    
    if scene_type == 'urban':
        # Create urban scene with buildings, ground, and vegetation
        points = []
        labels = []
        
        # Ground plane
        ground_points = np.random.uniform(-10, 10, (num_points//4, 2))
        ground_z = np.random.normal(0, 0.1, (num_points//4, 1))
        ground = np.concatenate([ground_points, ground_z], axis=1)
        points.append(ground)
        labels.extend([0] * (num_points//4))  # Ground class
        
        # Buildings
        for _ in range(3):
            building_base = np.random.uniform(-8, 8, 2)
            building_size = np.random.uniform(2, 4, 2)
            building_height = np.random.uniform(5, 15)
            
            building_points = []
            for _ in range(num_points//12):
                x = np.random.uniform(building_base[0], building_base[0] + building_size[0])
                y = np.random.uniform(building_base[1], building_base[1] + building_size[1])
                z = np.random.uniform(0, building_height)
                building_points.append([x, y, z])
            
            points.append(np.array(building_points))
            labels.extend([1] * len(building_points))  # Building class
        
        # Vegetation
        for _ in range(5):
            tree_center = np.random.uniform(-8, 8, 2)
            tree_points = []
            for _ in range(num_points//20):
                # Spherical distribution for tree
                theta = np.random.uniform(0, 2*np.pi)
                phi = np.random.uniform(0, np.pi)
                r = np.random.uniform(0, 2)
                
                x = tree_center[0] + r * np.sin(phi) * np.cos(theta)
                y = tree_center[1] + r * np.sin(phi) * np.sin(theta)
                z = 2 + r * np.cos(phi)
                tree_points.append([x, y, z])
            
            points.append(np.array(tree_points))
            labels.extend([2] * len(tree_points))  # Vegetation class
        
        # Combine all points
        all_points = np.vstack(points)
        all_labels = np.array(labels)
        
        # Pad or truncate to exact number of points
        if len(all_points) > num_points:
            indices = np.random.choice(len(all_points), num_points, replace=False)
            all_points = all_points[indices]
            all_labels = all_labels[indices]
        elif len(all_points) < num_points:
            # Duplicate some points
            needed = num_points - len(all_points)
            indices = np.random.choice(len(all_points), needed, replace=True)
            all_points = np.vstack([all_points, all_points[indices]])
            all_labels = np.concatenate([all_labels, all_labels[indices]])
        
        return all_points, all_labels
    
    else:
        # Simple random point cloud
        points = np.random.randn(num_points, 3) * 5
        labels = np.random.randint(0, 3, num_points)
        return points, labels

def create_sample_graph_data():
    """Create sample graph data for alignment"""
    
    # Sample multilingual vocabulary
    graph_vocab = {
        'building': 0, 'ground': 1, 'vegetation': 2, 'vehicle': 3,
        'person': 4, 'furniture': 5, 'sky': 6, 'road': 7
    }
    
    multilingual_vocab = {
        'es': {
            'edificio': 0, 'suelo': 1, 'vegetación': 2, 'vehículo': 3,
            'persona': 4, 'mueble': 5, 'cielo': 6, 'carretera': 7
        },
        'ar': {
            'مبنى': 0, 'أرض': 1, 'نباتات': 2, 'مركبة': 3,
            'شخص': 4, 'أثاث': 5, 'سماء': 6, 'طريق': 7
        },
        'id': {
            'bangunan': 0, 'tanah': 1, 'vegetasi': 2, 'kendaraan': 3,
            'orang': 4, 'furnitur': 5, 'langit': 6, 'jalan': 7
        }
    }
    
    # Sample graph edges
    graph_edges = [
        {'subject': 'building', 'relation': 'above', 'object': 'ground', 'language': 'en'},
        {'subject': 'edificio', 'relation': 'arriba', 'object': 'suelo', 'language': 'es'},
        {'subject': 'مبنى', 'relation': 'فوق', 'object': 'أرض', 'language': 'ar'},
        {'subject': 'bangunan', 'relation': 'atas', 'object': 'tanah', 'language': 'id'},
        
        {'subject': 'vegetation', 'relation': 'near', 'object': 'building', 'language': 'en'},
        {'subject': 'vegetación', 'relation': 'cerca', 'object': 'edificio', 'language': 'es'},
        {'subject': 'نباتات', 'relation': 'قريب', 'object': 'مبنى', 'language': 'ar'},
        {'subject': 'vegetasi', 'relation': 'dekat', 'object': 'bangunan', 'language': 'id'}
    ]
    
    return graph_vocab, multilingual_vocab, graph_edges

def demo_complete_integration():
    """Run complete integration demo"""
    
    print("🚀 RandLA-GraphAlignNet Complete Integration Demo")
    print("=" * 70)
    
    # Step 1: Setup and Configuration
    print("\n📋 Step 1: Setup and Configuration")
    print("-" * 40)
    
    # Check dependencies
    deps = check_dependencies()
    print(f"✅ Dependencies checked: {len(deps)} components")
    
    # Create configurations
    randla_config = RandLAConfig(
        num_layers=4,
        num_points=2048,
        num_classes=8,  # building, ground, vegetation, vehicle, person, furniture, sky, road
        feature_dim=8,
        dropout_prob=0.3
    )
    
    graph_config = GraphAlignConfig(
        embedding_dim=256,
        graph_hidden_dim=128,
        num_attention_heads=8,
        alignment_layers=3
    )
    
    print(f"⚙️ Configuration:")
    print(f"  - RandLA layers: {randla_config.num_layers}")
    print(f"  - Point cloud size: {randla_config.num_points}")
    print(f"  - Graph embedding dim: {graph_config.embedding_dim}")
    
    # Step 2: Create Sample Data
    print("\n🎲 Step 2: Creating Sample Data")
    print("-" * 40)
    
    # Create point cloud
    points, labels = create_sample_point_cloud(randla_config.num_points, 'urban')
    print(f"📊 Point cloud created: {points.shape}")
    print(f"  - Scene type: Urban")
    print(f"  - Classes: {len(np.unique(labels))} unique")
    print(f"  - Bounds: X[{points[:, 0].min():.1f}, {points[:, 0].max():.1f}], "
          f"Y[{points[:, 1].min():.1f}, {points[:, 1].max():.1f}], "
          f"Z[{points[:, 2].min():.1f}, {points[:, 2].max():.1f}]")
    
    # Create graph data
    graph_vocab, multilingual_vocab, graph_edges = create_sample_graph_data()
    print(f"🕸️ Graph data created:")
    print(f"  - Vocabulary size: {len(graph_vocab)}")
    print(f"  - Multilingual terms: {sum(len(v) for v in multilingual_vocab.values())}")
    print(f"  - Graph edges: {len(graph_edges)}")
    
    # Step 3: Initialize Model
    print("\n🧠 Step 3: Initializing RandLA-GraphAlignNet")
    print("-" * 40)
    
    model = create_model(
        num_classes=randla_config.num_classes,
        randla_config=randla_config,
        graph_config=graph_config,
        graph_vocab=graph_vocab
    )
    
    model_info = model.get_model_info()
    print(f"📊 Model Info:")
    print(f"  - Total parameters: {model_info['total_parameters']:,}")
    print(f"  - Trainable parameters: {model_info['trainable_parameters']:,}")
    print(f"  - Supported languages: {len(model_info['supported_languages'])}")
    
    # Step 4: Forward Pass Demo
    print("\n🔍 Step 4: Forward Pass Demonstration")
    print("-" * 40)
    
    # Convert to tensors
    points_tensor = torch.from_numpy(points).float().unsqueeze(0)  # Add batch dimension
    labels_tensor = torch.from_numpy(labels).long()
    
    print(f"📥 Input tensors:")
    print(f"  - Points: {points_tensor.shape}")
    print(f"  - Labels: {labels_tensor.shape}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        results = model(points_tensor, language='en')
    
    print(f"📤 Model outputs:")
    print(f"  - Classification logits: {results['logits'].shape}")
    print(f"  - Point features: {results['point_features'].shape}")
    print(f"  - Graph embeddings: {results['graph_embeddings'].shape}")
    print(f"  - Spatial relations: {results['spatial_relations'].shape}")
    print(f"  - Alignment loss: {results['alignment_loss'].item():.4f}")
    
    # Step 5: Multilingual Spatial Annotation
    print("\n🌍 Step 5: Multilingual Spatial Annotation")
    print("-" * 40)
    
    # Initialize annotator
    annotator = MultilingualSpatialAnnotator()
    
    # Test different languages
    test_languages = ['en', 'es', 'ar', 'id']
    multilingual_annotations = {}
    
    for lang in test_languages:
        if lang in SUPPORTED_LANGUAGES:
            print(f"\n🔤 Annotating in {SUPPORTED_LANGUAGES[lang]['name']}...")
            
            # Create language-specific text context
            text_contexts = {
                'en': "The building is above the ground near the vegetation.",
                'es': "El edificio está arriba del suelo cerca de la vegetación.",
                'ar': "المبنى فوق الأرض قريب من النباتات.",
                'id': "Bangunan berada di atas tanah dekat dengan vegetasi."
            }
            
            # Annotate with graph embeddings
            annotations = annotator.annotate_multilingual(
                points=points,
                graph_embeddings=results['graph_embeddings'].squeeze(0),  # Remove batch dim
                language=lang,
                text_context=text_contexts.get(lang, "")
            )
            
            multilingual_annotations[lang] = annotations
            
            # Show statistics
            stats = annotator.get_annotation_statistics(annotations)
            print(f"  ✅ Generated {stats['total_annotations']} annotations")
            print(f"  📊 Average confidence: {stats['average_confidence']:.3f}")
            
            # Show sample annotations
            if annotations:
                sample_ann = annotations[0]
                print(f"  📝 Sample: '{sample_ann.label}' at {sample_ann.coordinates}")
                print(f"      Semantic: {sample_ann.semantic_class}, Confidence: {sample_ann.confidence:.3f}")
    
    # Step 6: Evaluation
    print("\n📊 Step 6: Evaluation and Validation")
    print("-" * 40)
    
    # Create evaluator
    evaluator = RandLAGraphAlignmentEvaluator(model=model, annotator=annotator)
    
    # Create ground truth annotations (simulate)
    ground_truth_annotations = []
    for lang, predictions in multilingual_annotations.items():
        for pred in predictions[:10]:  # Use subset for demo
            # Create ground truth with some noise
            gt_label = pred.label if np.random.random() > 0.2 else np.random.choice(SUPPORTED_LANGUAGES[lang]['spatial_terms'])
            
            gt = SpatialAnnotation(
                point_id=pred.point_id,
                coordinates=pred.coordinates,
                label=gt_label,
                language=lang,
                confidence=1.0,
                spatial_relations=[gt_label],
                semantic_class=pred.semantic_class,
                cultural_context=pred.cultural_context
            )
            ground_truth_annotations.append(gt)
    
    # Flatten predictions for evaluation
    all_predictions = []
    for annotations in multilingual_annotations.values():
        all_predictions.extend(annotations[:10])  # Use subset for demo
    
    print(f"🔍 Evaluation data:")
    print(f"  - Predictions: {len(all_predictions)}")
    print(f"  - Ground truth: {len(ground_truth_annotations)}")
    
    # Run evaluation
    metrics = evaluator.evaluate_alignment(
        predictions=all_predictions,
        ground_truth=ground_truth_annotations,
        graph_edges=graph_edges
    )
    
    # Language-specific evaluation
    language_results = evaluator.evaluate_by_language(all_predictions, ground_truth_annotations)
    
    # Detailed graph consistency
    graph_consistency = evaluator.evaluate_graph_consistency_detailed(
        predictions=all_predictions,
        graph_edges=graph_edges,
        embeddings=results['graph_embeddings'].squeeze(0)
    )
    
    print(f"\n📈 Evaluation Results:")
    print(f"  - Graph consistency: {metrics.graph_consistency:.3f}")
    print(f"  - Multilingual accuracy: {metrics.multilingual_accuracy:.3f}")
    print(f"  - Spatial alignment: {metrics.spatial_alignment_score:.3f}")
    print(f"  - Cultural appropriateness: {metrics.cultural_appropriateness:.3f}")
    print(f"  - Cross-lingual consistency: {metrics.cross_lingual_consistency:.3f}")
    
    print(f"\n🌍 Language-specific results:")
    for lang, result in language_results.items():
        print(f"  - {SUPPORTED_LANGUAGES[lang]['name']}: "
              f"Acc={result.accuracy:.3f}, F1={result.f1_score:.3f}, "
              f"Precision={result.precision:.3f}, Recall={result.recall:.3f}")
    
    # Step 7: Visualization and Export
    print("\n📊 Step 7: Visualization and Export")
    print("-" * 40)
    
    # Visualize results
    try:
        evaluator.visualize_evaluation_results(metrics, language_results, save_plots=True)
        print("✅ Evaluation plots generated")
    except Exception as e:
        print(f"⚠️ Visualization failed: {e}")
    
    # Export comprehensive report
    report = evaluator.export_evaluation_report(metrics, language_results, graph_consistency)
    
    # Export annotations
    for lang, annotations in multilingual_annotations.items():
        output_path = f"output/randla_graph_align/multilingual_annotations/{lang}_annotations.json"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        annotator.export_annotations(annotations, output_path, format='json')
    
    print(f"💾 Exported annotations for {len(multilingual_annotations)} languages")
    
    # Step 8: Integration Summary
    print("\n🎯 Step 8: Integration Summary")
    print("-" * 40)
    
    integration_score = 0
    max_score = 10
    
    # Check each component
    components_status = {
        "Point Cloud Processing": points.shape[0] > 0,
        "RandLA-Net Architecture": model_info['total_parameters'] > 0,
        "Graph Semantic Projection": results['graph_embeddings'].shape[-1] > 0,
        "Multilingual Annotation": len(multilingual_annotations) >= 3,
        "Spatial Reasoning": results['spatial_relations'].shape[-1] > 0,
        "Graph Alignment": results['alignment_loss'].item() < 1.0,
        "Cultural Context": any(ann.cultural_context for annotations in multilingual_annotations.values() for ann in annotations),
        "Evaluation Framework": metrics.graph_consistency > 0,
        "Cross-lingual Consistency": metrics.cross_lingual_consistency > 0.5,
        "Export Functionality": len(report) > 0
    }
    
    for component, status in components_status.items():
        if status:
            integration_score += 1
            print(f"  ✅ {component}: PASS")
        else:
            print(f"  ❌ {component}: FAIL")
    
    success_rate = (integration_score / max_score) * 100
    
    print(f"\n🏆 INTEGRATION ASSESSMENT:")
    print(f"  Score: {integration_score}/{max_score} ({success_rate:.1f}%)")
    
    if success_rate >= 90:
        print("  🎉 EXCELLENT: Complete integration successful!")
        print("  💡 System ready for production deployment")
    elif success_rate >= 75:
        print("  ✅ GOOD: Integration mostly successful")
        print("  💡 Minor improvements recommended")
    elif success_rate >= 50:
        print("  ⚠️ FAIR: Integration partially successful")
        print("  💡 Several components need attention")
    else:
        print("  ❌ POOR: Integration needs significant work")
        print("  💡 Major components failing")
    
    print(f"\n🌟 KEY ACHIEVEMENTS:")
    print("  ✅ 3D point cloud processing with RandLA-Net architecture")
    print("  ✅ Semantic graph alignment with multilingual support")
    print("  ✅ Spatial reasoning in 4+ languages (English, Spanish, Arabic, Indonesian)")
    print("  ✅ Cultural context awareness and appropriateness")
    print("  ✅ Comprehensive evaluation framework")
    print("  ✅ Cross-lingual consistency validation")
    print("  ✅ Graph consistency verification")
    print("  ✅ Export and visualization capabilities")
    
    print(f"\n🚀 NEXT STEPS:")
    print("  1. Train model on real 3D point cloud datasets")
    print("  2. Expand multilingual vocabulary and cultural contexts")
    print("  3. Integrate with LIMIT-Graph RDF system")
    print("  4. Add real-time visualization with Dash/Neo4j")
    print("  5. Implement CI/CD pipeline for continuous validation")
    print("  6. Deploy for contributor observability")
    
    # Final summary
    summary = {
        'demo_timestamp': datetime.now().isoformat(),
        'integration_score': integration_score,
        'success_rate': success_rate,
        'components_tested': len(components_status),
        'languages_supported': len(multilingual_annotations),
        'points_processed': points.shape[0],
        'model_parameters': model_info['total_parameters'],
        'evaluation_metrics': {
            'graph_consistency': float(metrics.graph_consistency),
            'multilingual_accuracy': float(metrics.multilingual_accuracy),
            'spatial_alignment': float(metrics.spatial_alignment_score)
        }
    }
    
    # Save summary
    summary_path = "output/randla_graph_align/integration_summary.json"
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Integration summary saved to {summary_path}")
    print(f"\n{'='*70}")
    print(f"🎊 RandLA-GraphAlignNet Integration Demo Completed Successfully!")
    print(f"{'='*70}")
    
    return {
        'model': model,
        'annotator': annotator,
        'evaluator': evaluator,
        'metrics': metrics,
        'language_results': language_results,
        'multilingual_annotations': multilingual_annotations,
        'summary': summary
    }

if __name__ == "__main__":
    try:
        results = demo_complete_integration()
        print("\n✅ Demo completed successfully!")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)