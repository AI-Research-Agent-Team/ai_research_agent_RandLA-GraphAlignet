# -*- coding: utf-8 -*-
"""
Evaluation Harness for RandLA-GraphAlignNet
Validates graph consistency and multilingual label accuracy
"""

import sys
import os
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
import json
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Import dependencies
try:
    from dependencies import SUPPORTED_LANGUAGES, setup_environment
    from annotate_multilingual import MultilingualSpatialAnnotator, SpatialAnnotation
    from GraphSemanticProjector import GraphSemanticProjector
except ImportError:
    # Fallback definitions
    SUPPORTED_LANGUAGES = {
        'en': {'name': 'English', 'spatial_terms': ['above', 'below', 'left', 'right', 'near', 'far']},
        'es': {'name': 'Spanish', 'spatial_terms': ['arriba', 'abajo', 'izquierda', 'derecha', 'cerca', 'lejos']},
        'ar': {'name': 'Arabic', 'spatial_terms': ['فوق', 'تحت', 'يسار', 'يمين', 'قريب', 'بعيد']},
        'id': {'name': 'Indonesian', 'spatial_terms': ['atas', 'bawah', 'kiri', 'kanan', 'dekat', 'jauh']}
    }

@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics"""
    graph_consistency: float
    multilingual_accuracy: float
    spatial_alignment_score: float
    cultural_appropriateness: float
    confidence_calibration: float
    cross_lingual_consistency: float

@dataclass
class LanguageEvaluationResult:
    """Evaluation results for a specific language"""
    language: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: np.ndarray
    spatial_term_accuracy: Dict[str, float]
    confidence_distribution: List[float]

@dataclass
class GraphConsistencyResult:
    """Graph consistency evaluation results"""
    overall_consistency: float
    edge_alignment_score: float
    node_embedding_quality: float
    multilingual_edge_consistency: float
    semantic_coherence: float

class RandLAGraphAlignmentEvaluator:
    """
    Comprehensive evaluation harness for RandLA-GraphAlignNet
    Validates graph consistency and multilingual label accuracy
    """
    
    def __init__(self, 
                 model=None,
                 annotator: MultilingualSpatialAnnotator = None,
                 output_dir: str = "output/randla_graph_align/evaluations"):
        
        self.model = model
        self.annotator = annotator or MultilingualSpatialAnnotator()
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Evaluation thresholds
        self.thresholds = {
            'high_quality': 0.8,
            'medium_quality': 0.6,
            'low_quality': 0.4
        }
        
        # Supported evaluation metrics
        self.metrics = [
            'accuracy', 'precision', 'recall', 'f1_score',
            'graph_consistency', 'multilingual_accuracy',
            'spatial_alignment', 'cultural_appropriateness'
        ]
        
        print(f"📊 RandLA-GraphAlignment Evaluator initialized")
        print(f"   Output directory: {output_dir}")
    
    def evaluate_alignment(self, 
                          predictions: List[SpatialAnnotation],
                          ground_truth: List[SpatialAnnotation],
                          graph_edges: List[Dict[str, Any]] = None) -> EvaluationMetrics:
        """
        Main evaluation function for alignment quality
        
        Args:
            predictions: List of predicted spatial annotations
            ground_truth: List of ground truth annotations
            graph_edges: Optional graph edge information
            
        Returns:
            EvaluationMetrics object with comprehensive results
        """
        
        print("🔍 Evaluating alignment quality...")
        
        # Compute graph consistency
        graph_consistency = self.compute_graph_consistency(predictions, graph_edges)
        
        # Compute multilingual accuracy
        multilingual_accuracy = self.compute_multilingual_accuracy(predictions, ground_truth)
        
        # Compute spatial alignment score
        spatial_alignment = self.compute_spatial_alignment_score(predictions, ground_truth)
        
        # Compute cultural appropriateness
        cultural_appropriateness = self.compute_cultural_appropriateness(predictions, ground_truth)
        
        # Compute confidence calibration
        confidence_calibration = self.compute_confidence_calibration(predictions, ground_truth)
        
        # Compute cross-lingual consistency
        cross_lingual_consistency = self.compute_cross_lingual_consistency(predictions)
        
        metrics = EvaluationMetrics(
            graph_consistency=graph_consistency,
            multilingual_accuracy=multilingual_accuracy,
            spatial_alignment_score=spatial_alignment,
            cultural_appropriateness=cultural_appropriateness,
            confidence_calibration=confidence_calibration,
            cross_lingual_consistency=cross_lingual_consistency
        )
        
        print(f"✅ Evaluation completed:")
        print(f"   Graph consistency: {graph_consistency:.3f}")
        print(f"   Multilingual accuracy: {multilingual_accuracy:.3f}")
        print(f"   Spatial alignment: {spatial_alignment:.3f}")
        
        return metrics
    
    def compute_graph_consistency(self, 
                                predictions: List[SpatialAnnotation],
                                graph_edges: List[Dict[str, Any]] = None) -> float:
        """
        Compute graph consistency score
        
        Args:
            predictions: List of predicted annotations
            graph_edges: Graph edge information
            
        Returns:
            Graph consistency score (0-1)
        """
        
        if not predictions:
            return 0.0
        
        if not graph_edges:
            # Use spatial relations from predictions
            consistency_scores = []
            
            for pred in predictions:
                # Check if spatial relations are internally consistent
                if pred.spatial_relations:
                    # Simple consistency check: no contradictory relations
                    contradictory_pairs = [
                        ('above', 'below'), ('left', 'right'), 
                        ('near', 'far'), ('inside', 'outside')
                    ]
                    
                    relations_set = set(pred.spatial_relations)
                    is_consistent = True
                    
                    for pair in contradictory_pairs:
                        if pair[0] in relations_set and pair[1] in relations_set:
                            is_consistent = False
                            break
                    
                    consistency_scores.append(1.0 if is_consistent else 0.0)
                else:
                    consistency_scores.append(0.5)  # Neutral score for no relations
            
            return np.mean(consistency_scores) if consistency_scores else 0.0
        
        else:
            # Use provided graph edges for consistency checking
            edge_consistency_scores = []
            
            for edge in graph_edges:
                # Find predictions that match this edge
                matching_preds = [
                    pred for pred in predictions 
                    if pred.label == edge.get('relation', '')
                ]
                
                if matching_preds:
                    # Check if predictions are consistent with edge
                    avg_confidence = np.mean([pred.confidence for pred in matching_preds])
                    edge_consistency_scores.append(avg_confidence)
                else:
                    edge_consistency_scores.append(0.0)
            
            return np.mean(edge_consistency_scores) if edge_consistency_scores else 0.0
    
    def compute_multilingual_accuracy(self,
                                    predictions: List[SpatialAnnotation],
                                    ground_truth: List[SpatialAnnotation]) -> float:
        """
        Compute multilingual label accuracy
        
        Args:
            predictions: Predicted annotations
            ground_truth: Ground truth annotations
            
        Returns:
            Multilingual accuracy score (0-1)
        """
        
        if not predictions or not ground_truth:
            return 0.0
        
        # Group by language
        lang_accuracies = {}
        
        for lang in SUPPORTED_LANGUAGES.keys():
            pred_lang = [p for p in predictions if p.language == lang]
            gt_lang = [g for g in ground_truth if g.language == lang]
            
            if pred_lang and gt_lang:
                # Match predictions to ground truth by point_id
                matched_pairs = []
                
                for pred in pred_lang:
                    matching_gt = [g for g in gt_lang if g.point_id == pred.point_id]
                    if matching_gt:
                        matched_pairs.append((pred, matching_gt[0]))
                
                if matched_pairs:
                    # Compute accuracy for this language
                    correct = sum(1 for pred, gt in matched_pairs if pred.label == gt.label)
                    lang_accuracies[lang] = correct / len(matched_pairs)
        
        # Return average accuracy across languages
        return np.mean(list(lang_accuracies.values())) if lang_accuracies else 0.0
    
    def compute_spatial_alignment_score(self,
                                      predictions: List[SpatialAnnotation],
                                      ground_truth: List[SpatialAnnotation]) -> float:
        """
        Compute spatial alignment score based on coordinate consistency
        
        Args:
            predictions: Predicted annotations
            ground_truth: Ground truth annotations
            
        Returns:
            Spatial alignment score (0-1)
        """
        
        if not predictions or not ground_truth:
            return 0.0
        
        alignment_scores = []
        
        for pred in predictions:
            # Find closest ground truth annotation
            distances = []
            for gt in ground_truth:
                if gt.language == pred.language:
                    # Compute 3D distance
                    pred_coords = np.array(pred.coordinates)
                    gt_coords = np.array(gt.coordinates)
                    distance = np.linalg.norm(pred_coords - gt_coords)
                    distances.append((distance, gt))
            
            if distances:
                # Get closest ground truth
                closest_distance, closest_gt = min(distances, key=lambda x: x[0])
                
                # Compute alignment score based on distance and label match
                distance_score = np.exp(-closest_distance)  # Exponential decay
                label_match = 1.0 if pred.label == closest_gt.label else 0.0
                
                alignment_score = 0.7 * distance_score + 0.3 * label_match
                alignment_scores.append(alignment_score)
        
        return np.mean(alignment_scores) if alignment_scores else 0.0
    
    def compute_cultural_appropriateness(self,
                                       predictions: List[SpatialAnnotation],
                                       ground_truth: List[SpatialAnnotation]) -> float:
        """
        Compute cultural appropriateness score
        
        Args:
            predictions: Predicted annotations
            ground_truth: Ground truth annotations
            
        Returns:
            Cultural appropriateness score (0-1)
        """
        
        if not predictions:
            return 0.0
        
        cultural_scores = []
        
        for pred in predictions:
            # Check if cultural context is appropriate for language
            expected_contexts = {
                'en': ['western', 'individualistic'],
                'es': ['latin_american', 'iberian', 'collectivistic'],
                'ar': ['middle_eastern', 'islamic', 'hierarchical'],
                'id': ['southeast_asian', 'collectivistic', 'harmony_oriented']
            }
            
            if pred.language in expected_contexts:
                expected = expected_contexts[pred.language]
                if pred.cultural_context and any(ctx in pred.cultural_context for ctx in expected):
                    cultural_scores.append(1.0)
                else:
                    cultural_scores.append(0.5)
            else:
                cultural_scores.append(0.5)  # Neutral for unknown languages
        
        return np.mean(cultural_scores) if cultural_scores else 0.0
    
    def compute_confidence_calibration(self,
                                     predictions: List[SpatialAnnotation],
                                     ground_truth: List[SpatialAnnotation]) -> float:
        """
        Compute confidence calibration score
        
        Args:
            predictions: Predicted annotations
            ground_truth: Ground truth annotations
            
        Returns:
            Confidence calibration score (0-1)
        """
        
        if not predictions or not ground_truth:
            return 0.0
        
        # Match predictions to ground truth
        matched_pairs = []
        for pred in predictions:
            matching_gt = [g for g in ground_truth 
                          if g.point_id == pred.point_id and g.language == pred.language]
            if matching_gt:
                matched_pairs.append((pred, matching_gt[0]))
        
        if not matched_pairs:
            return 0.0
        
        # Compute calibration: high confidence should correlate with correctness
        calibration_scores = []
        
        for pred, gt in matched_pairs:
            is_correct = pred.label == gt.label
            confidence = pred.confidence
            
            # Good calibration: high confidence for correct predictions, low for incorrect
            if is_correct:
                calibration_score = confidence
            else:
                calibration_score = 1.0 - confidence
            
            calibration_scores.append(calibration_score)
        
        return np.mean(calibration_scores)
    
    def compute_cross_lingual_consistency(self, predictions: List[SpatialAnnotation]) -> float:
        """
        Compute cross-lingual consistency score
        
        Args:
            predictions: Predicted annotations
            
        Returns:
            Cross-lingual consistency score (0-1)
        """
        
        if not predictions:
            return 0.0
        
        # Group predictions by point_id
        point_groups = {}
        for pred in predictions:
            if pred.point_id not in point_groups:
                point_groups[pred.point_id] = []
            point_groups[pred.point_id].append(pred)
        
        consistency_scores = []
        
        for point_id, point_preds in point_groups.items():
            if len(point_preds) > 1:
                # Check consistency across languages for the same point
                semantic_classes = [pred.semantic_class for pred in point_preds]
                confidences = [pred.confidence for pred in point_preds]
                
                # Semantic consistency: same semantic class across languages
                semantic_consistency = len(set(semantic_classes)) == 1
                
                # Confidence consistency: similar confidence levels
                confidence_std = np.std(confidences)
                confidence_consistency = confidence_std < 0.2  # Threshold for consistency
                
                # Combined consistency score
                consistency_score = 0.7 * semantic_consistency + 0.3 * confidence_consistency
                consistency_scores.append(consistency_score)
        
        return np.mean(consistency_scores) if consistency_scores else 1.0
    
    def evaluate_by_language(self, 
                           predictions: List[SpatialAnnotation],
                           ground_truth: List[SpatialAnnotation]) -> Dict[str, LanguageEvaluationResult]:
        """
        Evaluate performance for each language separately
        
        Args:
            predictions: Predicted annotations
            ground_truth: Ground truth annotations
            
        Returns:
            Dictionary mapping languages to evaluation results
        """
        
        results = {}
        
        for lang in SUPPORTED_LANGUAGES.keys():
            pred_lang = [p for p in predictions if p.language == lang]
            gt_lang = [g for g in ground_truth if g.language == lang]
            
            if not pred_lang or not gt_lang:
                continue
            
            # Match predictions to ground truth
            matched_pairs = []
            for pred in pred_lang:
                matching_gt = [g for g in gt_lang if g.point_id == pred.point_id]
                if matching_gt:
                    matched_pairs.append((pred, matching_gt[0]))
            
            if not matched_pairs:
                continue
            
            # Extract labels for evaluation
            pred_labels = [pred.label for pred, _ in matched_pairs]
            true_labels = [gt.label for _, gt in matched_pairs]
            
            # Compute metrics
            accuracy = accuracy_score(true_labels, pred_labels)
            precision, recall, f1, _ = precision_recall_fscore_support(
                true_labels, pred_labels, average='weighted', zero_division=0
            )
            
            # Confusion matrix
            unique_labels = list(set(true_labels + pred_labels))
            cm = confusion_matrix(true_labels, pred_labels, labels=unique_labels)
            
            # Spatial term accuracy
            spatial_terms = SUPPORTED_LANGUAGES[lang]['spatial_terms']
            spatial_term_accuracy = {}
            
            for term in spatial_terms:
                term_true = [1 if label == term else 0 for label in true_labels]
                term_pred = [1 if label == term else 0 for label in pred_labels]
                
                if sum(term_true) > 0:  # Only compute if term appears in ground truth
                    term_accuracy = accuracy_score(term_true, term_pred)
                    spatial_term_accuracy[term] = term_accuracy
            
            # Confidence distribution
            confidences = [pred.confidence for pred, _ in matched_pairs]
            
            result = LanguageEvaluationResult(
                language=lang,
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1,
                confusion_matrix=cm,
                spatial_term_accuracy=spatial_term_accuracy,
                confidence_distribution=confidences
            )
            
            results[lang] = result
        
        return results
    
    def evaluate_graph_consistency_detailed(self, 
                                          predictions: List[SpatialAnnotation],
                                          graph_edges: List[Dict[str, Any]] = None,
                                          embeddings: torch.Tensor = None) -> GraphConsistencyResult:
        """
        Detailed graph consistency evaluation
        
        Args:
            predictions: Predicted annotations
            graph_edges: Graph edge information
            embeddings: Node embeddings for quality assessment
            
        Returns:
            GraphConsistencyResult with detailed metrics
        """
        
        # Overall consistency (from main method)
        overall_consistency = self.compute_graph_consistency(predictions, graph_edges)
        
        # Edge alignment score
        edge_alignment_score = self._compute_edge_alignment_score(predictions, graph_edges)
        
        # Node embedding quality
        node_embedding_quality = self._compute_node_embedding_quality(embeddings) if embeddings is not None else 0.0
        
        # Multilingual edge consistency
        multilingual_edge_consistency = self._compute_multilingual_edge_consistency(predictions, graph_edges)
        
        # Semantic coherence
        semantic_coherence = self._compute_semantic_coherence(predictions)
        
        return GraphConsistencyResult(
            overall_consistency=overall_consistency,
            edge_alignment_score=edge_alignment_score,
            node_embedding_quality=node_embedding_quality,
            multilingual_edge_consistency=multilingual_edge_consistency,
            semantic_coherence=semantic_coherence
        )
    
    def _compute_edge_alignment_score(self, predictions, graph_edges):
        """Compute edge alignment score"""
        if not graph_edges:
            return 0.0
        
        alignment_scores = []
        for edge in graph_edges:
            # Find predictions that could align with this edge
            matching_preds = [
                pred for pred in predictions
                if pred.label == edge.get('relation', '') or 
                   pred.semantic_class == edge.get('object_type', '')
            ]
            
            if matching_preds:
                avg_confidence = np.mean([pred.confidence for pred in matching_preds])
                alignment_scores.append(avg_confidence)
            else:
                alignment_scores.append(0.0)
        
        return np.mean(alignment_scores) if alignment_scores else 0.0
    
    def _compute_node_embedding_quality(self, embeddings):
        """Compute node embedding quality"""
        if embeddings is None or len(embeddings) == 0:
            return 0.0
        
        # Compute embedding statistics
        embedding_norms = torch.norm(embeddings, dim=-1)
        norm_std = torch.std(embedding_norms).item()
        
        # Good embeddings should have consistent norms
        quality_score = max(0.0, 1.0 - norm_std)
        
        return quality_score
    
    def _compute_multilingual_edge_consistency(self, predictions, graph_edges):
        """Compute multilingual edge consistency"""
        if not graph_edges:
            return 1.0
        
        # Group predictions by language
        lang_groups = {}
        for pred in predictions:
            if pred.language not in lang_groups:
                lang_groups[pred.language] = []
            lang_groups[pred.language].append(pred)
        
        if len(lang_groups) < 2:
            return 1.0  # Perfect consistency if only one language
        
        # Check consistency across languages
        consistency_scores = []
        
        for edge in graph_edges:
            lang_predictions = {}
            for lang, preds in lang_groups.items():
                matching_preds = [
                    pred for pred in preds
                    if pred.label == edge.get('relation', '')
                ]
                if matching_preds:
                    lang_predictions[lang] = matching_preds
            
            if len(lang_predictions) > 1:
                # Check if predictions are consistent across languages
                confidences = []
                for lang_preds in lang_predictions.values():
                    avg_conf = np.mean([pred.confidence for pred in lang_preds])
                    confidences.append(avg_conf)
                
                # Consistency is high if confidences are similar
                consistency = 1.0 - np.std(confidences)
                consistency_scores.append(max(0.0, consistency))
        
        return np.mean(consistency_scores) if consistency_scores else 1.0
    
    def _compute_semantic_coherence(self, predictions):
        """Compute semantic coherence score"""
        if not predictions:
            return 0.0
        
        # Group by semantic class
        semantic_groups = {}
        for pred in predictions:
            if pred.semantic_class not in semantic_groups:
                semantic_groups[pred.semantic_class] = []
            semantic_groups[pred.semantic_class].append(pred)
        
        coherence_scores = []
        
        for semantic_class, preds in semantic_groups.items():
            if len(preds) > 1:
                # Check if spatial relations are coherent within semantic class
                spatial_relations = []
                for pred in preds:
                    spatial_relations.extend(pred.spatial_relations)
                
                if spatial_relations:
                    # Coherence is high if similar spatial relations are used
                    relation_counts = {}
                    for rel in spatial_relations:
                        relation_counts[rel] = relation_counts.get(rel, 0) + 1
                    
                    # Normalized entropy as coherence measure
                    total = sum(relation_counts.values())
                    entropy = -sum((count/total) * np.log(count/total) 
                                 for count in relation_counts.values())
                    max_entropy = np.log(len(relation_counts))
                    
                    coherence = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 1.0
                    coherence_scores.append(coherence)
        
        return np.mean(coherence_scores) if coherence_scores else 1.0
    
    def visualize_evaluation_results(self, 
                                   metrics: EvaluationMetrics,
                                   language_results: Dict[str, LanguageEvaluationResult] = None,
                                   save_plots: bool = True):
        """
        Visualize evaluation results
        
        Args:
            metrics: Overall evaluation metrics
            language_results: Per-language evaluation results
            save_plots: Whether to save plots to files
        """
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('RandLA-GraphAlignNet Evaluation Results', fontsize=16)
        
        # Overall metrics radar chart
        ax = axes[0, 0]
        metrics_names = ['Graph\nConsistency', 'Multilingual\nAccuracy', 'Spatial\nAlignment', 
                        'Cultural\nAppropriateness', 'Confidence\nCalibration', 'Cross-lingual\nConsistency']
        metrics_values = [metrics.graph_consistency, metrics.multilingual_accuracy, 
                         metrics.spatial_alignment_score, metrics.cultural_appropriateness,
                         metrics.confidence_calibration, metrics.cross_lingual_consistency]
        
        angles = np.linspace(0, 2*np.pi, len(metrics_names), endpoint=False)
        metrics_values += metrics_values[:1]  # Complete the circle
        angles = np.concatenate((angles, [angles[0]]))
        
        ax.plot(angles, metrics_values, 'o-', linewidth=2, label='Performance')
        ax.fill(angles, metrics_values, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics_names)
        ax.set_ylim(0, 1)
        ax.set_title('Overall Performance Metrics')
        ax.grid(True)
        
        # Language-wise accuracy comparison
        if language_results:
            ax = axes[0, 1]
            languages = list(language_results.keys())
            accuracies = [result.accuracy for result in language_results.values()]
            
            bars = ax.bar(languages, accuracies, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(languages)])
            ax.set_title('Accuracy by Language')
            ax.set_ylabel('Accuracy')
            ax.set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, acc in zip(bars, accuracies):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                       f'{acc:.3f}', ha='center', va='bottom')
        
        # Confidence distribution
        if language_results:
            ax = axes[0, 2]
            for lang, result in language_results.items():
                ax.hist(result.confidence_distribution, alpha=0.6, label=SUPPORTED_LANGUAGES[lang]['name'], bins=20)
            ax.set_title('Confidence Distribution by Language')
            ax.set_xlabel('Confidence')
            ax.set_ylabel('Frequency')
            ax.legend()
        
        # F1 scores by language
        if language_results:
            ax = axes[1, 0]
            languages = list(language_results.keys())
            f1_scores = [result.f1_score for result in language_results.values()]
            
            bars = ax.bar(languages, f1_scores, color=['#9467bd', '#8c564b', '#e377c2', '#7f7f7f'][:len(languages)])
            ax.set_title('F1 Score by Language')
            ax.set_ylabel('F1 Score')
            ax.set_ylim(0, 1)
            
            for bar, f1 in zip(bars, f1_scores):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                       f'{f1:.3f}', ha='center', va='bottom')
        
        # Confusion matrix for first language
        if language_results:
            first_lang = list(language_results.keys())[0]
            result = language_results[first_lang]
            
            ax = axes[1, 1]
            sns.heatmap(result.confusion_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title(f'Confusion Matrix - {SUPPORTED_LANGUAGES[first_lang]["name"]}')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
        
        # Metrics comparison
        ax = axes[1, 2]
        metric_names = ['Graph\nConsistency', 'Multilingual\nAccuracy', 'Spatial\nAlignment']
        metric_values = [metrics.graph_consistency, metrics.multilingual_accuracy, metrics.spatial_alignment_score]
        
        bars = ax.bar(metric_names, metric_values, color=['#ff9999', '#66b3ff', '#99ff99'])
        ax.set_title('Key Performance Metrics')
        ax.set_ylabel('Score')
        ax.set_ylim(0, 1)
        
        for bar, val in zip(bars, metric_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                   f'{val:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_plots:
            plot_path = os.path.join(self.output_dir, 'evaluation_results.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"📊 Evaluation plots saved to {plot_path}")
        
        plt.show()
    
    def export_evaluation_report(self, 
                               metrics: EvaluationMetrics,
                               language_results: Dict[str, LanguageEvaluationResult] = None,
                               graph_consistency: GraphConsistencyResult = None):
        """
        Export comprehensive evaluation report
        
        Args:
            metrics: Overall evaluation metrics
            language_results: Per-language results
            graph_consistency: Detailed graph consistency results
        """
        
        report = {
            'evaluation_timestamp': str(torch.datetime.now()),
            'overall_metrics': {
                'graph_consistency': float(metrics.graph_consistency),
                'multilingual_accuracy': float(metrics.multilingual_accuracy),
                'spatial_alignment_score': float(metrics.spatial_alignment_score),
                'cultural_appropriateness': float(metrics.cultural_appropriateness),
                'confidence_calibration': float(metrics.confidence_calibration),
                'cross_lingual_consistency': float(metrics.cross_lingual_consistency)
            }
        }
        
        # Add language-specific results
        if language_results:
            report['language_results'] = {}
            for lang, result in language_results.items():
                report['language_results'][lang] = {
                    'language_name': SUPPORTED_LANGUAGES[lang]['name'],
                    'accuracy': float(result.accuracy),
                    'precision': float(result.precision),
                    'recall': float(result.recall),
                    'f1_score': float(result.f1_score),
                    'spatial_term_accuracy': {k: float(v) for k, v in result.spatial_term_accuracy.items()},
                    'avg_confidence': float(np.mean(result.confidence_distribution)),
                    'confidence_std': float(np.std(result.confidence_distribution))
                }
        
        # Add detailed graph consistency results
        if graph_consistency:
            report['graph_consistency_detailed'] = {
                'overall_consistency': float(graph_consistency.overall_consistency),
                'edge_alignment_score': float(graph_consistency.edge_alignment_score),
                'node_embedding_quality': float(graph_consistency.node_embedding_quality),
                'multilingual_edge_consistency': float(graph_consistency.multilingual_edge_consistency),
                'semantic_coherence': float(graph_consistency.semantic_coherence)
            }
        
        # Save report
        report_path = os.path.join(self.output_dir, 'evaluation_report.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"📄 Evaluation report saved to {report_path}")
        
        return report

def demo_evaluation_harness():
    """Demonstrate the evaluation harness"""
    
    print("📊 RandLA-GraphAlignNet Evaluation Harness Demo")
    print("=" * 60)
    
    # Create evaluator
    evaluator = RandLAGraphAlignmentEvaluator()
    
    # Create sample predictions and ground truth
    num_samples = 50
    languages = ['en', 'es', 'ar', 'id']
    
    predictions = []
    ground_truth = []
    
    for i in range(num_samples):
        lang = languages[i % len(languages)]
        spatial_terms = SUPPORTED_LANGUAGES[lang]['spatial_terms']
        
        # Create prediction
        pred = SpatialAnnotation(
            point_id=i,
            coordinates=(np.random.randn(), np.random.randn(), np.random.randn()),
            label=np.random.choice(spatial_terms),
            language=lang,
            confidence=np.random.uniform(0.5, 1.0),
            spatial_relations=[np.random.choice(spatial_terms)],
            semantic_class=np.random.choice(['building', 'vegetation', 'ground', 'vehicle']),
            cultural_context=f"{lang}_context"
        )
        predictions.append(pred)
        
        # Create ground truth (with some noise)
        gt_label = pred.label if np.random.random() > 0.2 else np.random.choice(spatial_terms)
        gt = SpatialAnnotation(
            point_id=i,
            coordinates=pred.coordinates,
            label=gt_label,
            language=lang,
            confidence=1.0,
            spatial_relations=[gt_label],
            semantic_class=pred.semantic_class,
            cultural_context=pred.cultural_context
        )
        ground_truth.append(gt)
    
    print(f"📊 Sample Data:")
    print(f"  - Predictions: {len(predictions)}")
    print(f"  - Ground truth: {len(ground_truth)}")
    print(f"  - Languages: {len(languages)}")
    
    # Run evaluation
    print(f"\n🔍 Running evaluation...")
    
    # Overall evaluation
    metrics = evaluator.evaluate_alignment(predictions, ground_truth)
    
    # Language-specific evaluation
    language_results = evaluator.evaluate_by_language(predictions, ground_truth)
    
    # Detailed graph consistency
    sample_embeddings = torch.randn(num_samples, 256)
    graph_consistency = evaluator.evaluate_graph_consistency_detailed(
        predictions, embeddings=sample_embeddings
    )
    
    print(f"\n📈 Results Summary:")
    print(f"  - Graph consistency: {metrics.graph_consistency:.3f}")
    print(f"  - Multilingual accuracy: {metrics.multilingual_accuracy:.3f}")
    print(f"  - Spatial alignment: {metrics.spatial_alignment_score:.3f}")
    print(f"  - Cultural appropriateness: {metrics.cultural_appropriateness:.3f}")
    
    print(f"\n🌍 Language Results:")
    for lang, result in language_results.items():
        print(f"  - {SUPPORTED_LANGUAGES[lang]['name']}: "
              f"Acc={result.accuracy:.3f}, F1={result.f1_score:.3f}")
    
    # Visualize results
    evaluator.visualize_evaluation_results(metrics, language_results, save_plots=True)
    
    # Export report
    report = evaluator.export_evaluation_report(metrics, language_results, graph_consistency)
    
    print(f"\n✅ Demo completed successfully!")
    return evaluator, metrics, language_results

if __name__ == "__main__":
    demo_evaluation_harness()
