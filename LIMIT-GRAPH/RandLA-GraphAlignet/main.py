#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RandLA-GraphAlignNet Main Entry Point
Standalone application for multilingual spatial reasoning with Neo4j integration
"""

import sys
import os
import argparse
import json
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add current directory to path for standalone execution
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import all components
from dependencies import check_dependencies, setup_environment, SUPPORTED_LANGUAGES
from RandLA_GraphAlignNet import create_model, RandLAConfig, GraphAlignConfig
from GraphSemanticProjector import GraphSemanticProjector
from annotate_multilingual import MultilingualSpatialAnnotator, SpatialAnnotation
from evaluate_alignment import RandLAGraphAlignmentEvaluator
from neo4j_integration import Neo4jSpatialVisualizer

import numpy as np
import torch

def create_sample_data(num_points: int = 2048, scene_type: str = 'urban'):
    """Create sample 3D point cloud data"""
    
    if scene_type == 'urban':
        # Create realistic urban scene
        points = []
        labels = []
        
        # Ground plane
        ground_size = num_points // 4
        ground_x = np.random.uniform(-20, 20, ground_size)
        ground_y = np.random.uniform(-20, 20, ground_size)
        ground_z = np.random.normal(0, 0.2, ground_size)
        ground_points = np.column_stack([ground_x, ground_y, ground_z])
        points.append(ground_points)
        labels.extend([0] * ground_size)  # Ground class
        
        # Buildings
        building_size = num_points // 3
        for _ in range(3):
            # Random building location and size
            bx, by = np.random.uniform(-15, 15, 2)
            bw, bh = np.random.uniform(3, 8, 2)
            height = np.random.uniform(8, 25)
            
            # Generate building points
            n_points = building_size // 3
            building_x = np.random.uniform(bx, bx + bw, n_points)
            building_y = np.random.uniform(by, by + bh, n_points)
            building_z = np.random.uniform(0, height, n_points)
            building_points = np.column_stack([building_x, building_y, building_z])
            points.append(building_points)
            labels.extend([1] * n_points)  # Building class
        
        # Vegetation
        veg_size = num_points - sum(len(p) for p in points)
        for _ in range(5):
            # Random tree location
            tx, ty = np.random.uniform(-18, 18, 2)
            tree_height = np.random.uniform(3, 12)
            tree_radius = np.random.uniform(1, 3)
            
            # Generate tree points (spherical distribution)
            n_points = max(1, veg_size // 5)
            theta = np.random.uniform(0, 2*np.pi, n_points)
            phi = np.random.uniform(0, np.pi, n_points)
            r = np.random.uniform(0, tree_radius, n_points)
            
            tree_x = tx + r * np.sin(phi) * np.cos(theta)
            tree_y = ty + r * np.sin(phi) * np.sin(theta)
            tree_z = tree_height/2 + r * np.cos(phi)
            tree_points = np.column_stack([tree_x, tree_y, tree_z])
            points.append(tree_points)
            labels.extend([2] * n_points)  # Vegetation class
        
        # Combine all points
        all_points = np.vstack(points)
        all_labels = np.array(labels)
        
        # Ensure exact number of points
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
        points = np.random.randn(num_points, 3) * 10
        labels = np.random.randint(0, 3, num_points)
        return points, labels

def run_multilingual_processing(model, annotator, points, embeddings, languages):
    """Run multilingual spatial annotation processing"""
    
    print(f"\n🌍 Processing {len(languages)} languages...")
    
    multilingual_annotations = {}
    
    for lang in languages:
        if lang in SUPPORTED_LANGUAGES:
            print(f"  🔤 Processing {SUPPORTED_LANGUAGES[lang]['name']}...")
            
            # Create language-specific context
            contexts = {
                'en': "The building is above the ground near the vegetation.",
                'es': "El edificio está arriba del suelo cerca de la vegetación.",
                'ar': "المبنى فوق الأرض قريب من النباتات.",
                'id': "Bangunan berada di atas tanah dekat dengan vegetasi.",
                'zh': "建筑物在地面上方靠近植被。",
                'hi': "इमारत जमीन के ऊपर वनस्पति के पास है।"
            }
            
            annotations = annotator.annotate_multilingual(
                points=points,
                graph_embeddings=embeddings,
                language=lang,
                text_context=contexts.get(lang, "")
            )
            
            multilingual_annotations[lang] = annotations
            
            # Show statistics
            stats = annotator.get_annotation_statistics(annotations)
            print(f"    ✅ {stats['total_annotations']} annotations, "
                  f"confidence: {stats['average_confidence']:.3f}")
    
    return multilingual_annotations

def run_evaluation(evaluator, multilingual_annotations, graph_edges):
    """Run comprehensive evaluation"""
    
    print(f"\n📊 Running evaluation...")
    
    # Flatten all annotations
    all_predictions = []
    for annotations in multilingual_annotations.values():
        all_predictions.extend(annotations)
    
    # Create ground truth (simulate with some noise)
    ground_truth = []
    for pred in all_predictions:
        # 80% chance of correct label
        if np.random.random() > 0.2:
            gt_label = pred.label
        else:
            # Random incorrect label
            spatial_terms = SUPPORTED_LANGUAGES[pred.language]['spatial_terms']
            gt_label = np.random.choice(spatial_terms)
        
        gt = SpatialAnnotation(
            point_id=pred.point_id,
            coordinates=pred.coordinates,
            label=gt_label,
            language=pred.language,
            confidence=1.0,
            spatial_relations=[gt_label],
            semantic_class=pred.semantic_class,
            cultural_context=pred.cultural_context
        )
        ground_truth.append(gt)
    
    # Run evaluation
    metrics = evaluator.evaluate_alignment(all_predictions, ground_truth, graph_edges)
    language_results = evaluator.evaluate_by_language(all_predictions, ground_truth)
    
    print(f"  📈 Results:")
    print(f"    - Graph consistency: {metrics.graph_consistency:.3f}")
    print(f"    - Multilingual accuracy: {metrics.multilingual_accuracy:.3f}")
    print(f"    - Spatial alignment: {metrics.spatial_alignment_score:.3f}")
    print(f"    - Cultural appropriateness: {metrics.cultural_appropriateness:.3f}")
    
    return metrics, language_results, all_predictions, ground_truth

def run_neo4j_integration(multilingual_annotations, graph_edges, session_id):
    """Run Neo4j integration and visualization"""
    
    print(f"\n🔗 Neo4j Integration...")
    
    try:
        # Initialize Neo4j visualizer
        visualizer = Neo4jSpatialVisualizer()
        
        # Create session
        visualizer.create_point_cloud_session(
            session_id,
            metadata={
                "application": "RandLA-GraphAlignNet",
                "languages": list(multilingual_annotations.keys()),
                "timestamp": datetime.now().isoformat()
            }
        )
        
        # Store all annotations
        all_annotations = []
        for annotations in multilingual_annotations.values():
            all_annotations.extend(annotations)
        
        stored_count = visualizer.store_spatial_annotations(all_annotations, session_id)
        print(f"  ✅ Stored {stored_count} annotations in Neo4j")
        
        # Create relationships
        rel_count = visualizer.create_spatial_relationships(graph_edges, session_id)
        print(f"  ✅ Created {rel_count} spatial relationships")
        
        # Get statistics
        stats = visualizer.get_spatial_graph_statistics()
        print(f"  📊 Neo4j Statistics:")
        print(f"    - Total annotations: {stats['confidence_statistics'].get('total_annotations', 0)}")
        print(f"    - Languages: {list(stats['language_distribution'].keys())}")
        
        # Generate visualization
        viz_url = visualizer.visualize_spatial_graph(session_id)
        
        # Export queries
        visualizer.export_cypher_queries(f"neo4j_queries_{session_id}.cypher")
        
        visualizer.close()
        
        return viz_url, stats
        
    except Exception as e:
        print(f"  ⚠️ Neo4j integration failed: {e}")
        print(f"     Make sure Neo4j is running at bolt://localhost:7687")
        return None, None

def main():
    """Main entry point for RandLA-GraphAlignNet"""
    
    parser = argparse.ArgumentParser(description="RandLA-GraphAlignNet: Multilingual Spatial Reasoning for 3D Point Clouds")
    parser.add_argument("--mode", choices=["demo", "process", "evaluate", "neo4j"], default="demo",
                       help="Operation mode")
    parser.add_argument("--languages", nargs="+", default=["en", "es", "ar", "id"],
                       help="Languages to process")
    parser.add_argument("--num-points", type=int, default=2048,
                       help="Number of points in point cloud")
    parser.add_argument("--scene-type", choices=["urban", "random"], default="urban",
                       help="Type of scene to generate")
    parser.add_argument("--output-dir", default="output",
                       help="Output directory for results")
    parser.add_argument("--neo4j-uri", default="bolt://localhost:7687",
                       help="Neo4j connection URI")
    parser.add_argument("--neo4j-user", default="neo4j",
                       help="Neo4j username")
    parser.add_argument("--neo4j-password", default="_KDITp62FstVa-ImKe1fdScIlkdOhrRUSdJ1uFNcook",
                       help="Neo4j password")
    parser.add_argument("--session-id", default=None,
                       help="Session ID for Neo4j (auto-generated if not provided)")
    
    args = parser.parse_args()
    
    print("🚀 RandLA-GraphAlignNet Standalone Application")
    print("=" * 60)
    
    # Check dependencies
    print("\n📋 Checking dependencies...")
    deps = check_dependencies()
    missing_deps = [k for k, v in deps.items() if v == 'Not available']
    if missing_deps:
        print(f"❌ Missing dependencies: {missing_deps}")
        print("   Please install missing dependencies and try again")
        return 1
    
    print("✅ All dependencies available")
    
    # Setup environment
    setup_environment()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate session ID if not provided
    if args.session_id is None:
        args.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print(f"\n⚙️ Configuration:")
    print(f"  - Mode: {args.mode}")
    print(f"  - Languages: {args.languages}")
    print(f"  - Points: {args.num_points}")
    print(f"  - Scene: {args.scene_type}")
    print(f"  - Session ID: {args.session_id}")
    
    # Initialize components
    print(f"\n🧠 Initializing model...")
    
    randla_config = RandLAConfig(
        num_layers=4,
        num_points=args.num_points,
        num_classes=8,
        feature_dim=8
    )
    
    graph_config = GraphAlignConfig(
        embedding_dim=256,
        graph_hidden_dim=128,
        num_attention_heads=8
    )
    
    model = create_model(
        num_classes=randla_config.num_classes,
        randla_config=randla_config,
        graph_config=graph_config
    )
    
    annotator = MultilingualSpatialAnnotator()
    evaluator = RandLAGraphAlignmentEvaluator(model=model, annotator=annotator)
    
    print(f"✅ Model initialized: {model.get_model_info()['total_parameters']:,} parameters")
    
    # Create sample data
    print(f"\n🎲 Creating sample data...")
    points, labels = create_sample_data(args.num_points, args.scene_type)
    
    # Forward pass
    points_tensor = torch.from_numpy(points).float().unsqueeze(0)
    model.eval()
    with torch.no_grad():
        results = model(points_tensor, language=args.languages[0])
    
    embeddings = results['graph_embeddings'].squeeze(0)
    
    print(f"✅ Generated embeddings: {embeddings.shape}")
    
    # Create graph edges for relationships
    graph_edges = []
    for lang in args.languages:
        if lang in SUPPORTED_LANGUAGES:
            spatial_terms = SUPPORTED_LANGUAGES[lang]['spatial_terms']
            # Add some sample relationships
            if len(spatial_terms) >= 4:
                graph_edges.extend([
                    {'subject': spatial_terms[0], 'relation': 'opposite_of', 'object': spatial_terms[1], 'language': lang},
                    {'subject': spatial_terms[2], 'relation': 'opposite_of', 'object': spatial_terms[3], 'language': lang},
                ])
    
    # Run processing based on mode
    if args.mode in ["demo", "process"]:
        # Multilingual processing
        multilingual_annotations = run_multilingual_processing(
            model, annotator, points, embeddings, args.languages
        )
        
        # Save annotations
        for lang, annotations in multilingual_annotations.items():
            output_file = os.path.join(args.output_dir, f"{args.session_id}_{lang}_annotations.json")
            annotator.export_annotations(annotations, output_file, format='json')
        
        print(f"\n💾 Annotations saved to {args.output_dir}")
    
    if args.mode in ["demo", "evaluate"]:
        # Evaluation
        if 'multilingual_annotations' not in locals():
            multilingual_annotations = run_multilingual_processing(
                model, annotator, points, embeddings, args.languages
            )
        
        metrics, language_results, predictions, ground_truth = run_evaluation(
            evaluator, multilingual_annotations, graph_edges
        )
        
        # Save evaluation results
        try:
            evaluator.visualize_evaluation_results(metrics, language_results, save_plots=True)
            report = evaluator.export_evaluation_report(metrics, language_results)
            
            # Move files to output directory
            import shutil
            for file in ["evaluation_results.png", "evaluation_report.json"]:
                if os.path.exists(file):
                    shutil.move(file, os.path.join(args.output_dir, f"{args.session_id}_{file}"))
            
        except Exception as e:
            print(f"⚠️ Visualization failed: {e}")
    
    if args.mode in ["demo", "neo4j"]:
        # Neo4j integration
        if 'multilingual_annotations' not in locals():
            multilingual_annotations = run_multilingual_processing(
                model, annotator, points, embeddings, args.languages
            )
        
        viz_url, neo4j_stats = run_neo4j_integration(
            multilingual_annotations, graph_edges, args.session_id
        )
        
        if viz_url:
            print(f"\n🎨 Neo4j Visualization: {viz_url}")
    
    # Final summary
    print(f"\n🎯 Summary:")
    print(f"  - Session ID: {args.session_id}")
    print(f"  - Languages processed: {len(args.languages)}")
    print(f"  - Points processed: {args.num_points}")
    print(f"  - Output directory: {args.output_dir}")
    
    if 'metrics' in locals():
        print(f"  - Graph consistency: {metrics.graph_consistency:.3f}")
        print(f"  - Multilingual accuracy: {metrics.multilingual_accuracy:.3f}")
    
    if viz_url:
        print(f"  - Neo4j visualization: Available")
    
    print(f"\n✅ RandLA-GraphAlignNet processing completed successfully!")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())