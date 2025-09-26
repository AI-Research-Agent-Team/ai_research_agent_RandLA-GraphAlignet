# -*- coding: utf-8 -*-
"""
Multilingual Spatial Annotation Hook for RandLA-GraphAlignNet
Supports Indonesian, Spanish, Arabic, and other languages for spatial reasoning
"""

import sys
import os
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
import json
import numpy as np
import torch
import torch.nn.functional as F

# Import dependencies
try:
    from dependencies import SUPPORTED_LANGUAGES, detect_spatial_language, extract_spatial_relations
    from GraphSemanticProjector import GraphSemanticProjector
except ImportError:
    # Fallback definitions
    SUPPORTED_LANGUAGES = {
        'en': {'name': 'English', 'spatial_terms': ['above', 'below', 'left', 'right', 'near', 'far', 'inside', 'outside']},
        'es': {'name': 'Spanish', 'spatial_terms': ['arriba', 'abajo', 'izquierda', 'derecha', 'cerca', 'lejos', 'dentro', 'fuera']},
        'ar': {'name': 'Arabic', 'spatial_terms': ['فوق', 'تحت', 'يسار', 'يمين', 'قريب', 'بعيد', 'داخل', 'خارج']},
        'id': {'name': 'Indonesian', 'spatial_terms': ['atas', 'bawah', 'kiri', 'kanan', 'dekat', 'jauh', 'dalam', 'luar']},
        'zh': {'name': 'Chinese', 'spatial_terms': ['上', '下', '左', '右', '近', '远', '内', '外']},
        'hi': {'name': 'Hindi', 'spatial_terms': ['ऊपर', 'नीचे', 'बाएं', 'दाएं', 'पास', 'दूर', 'अंदर', 'बाहर']}
    }


@dataclass
class SpatialAnnotation:
    """Represents a spatial annotation with multilingual support"""
    point_id: int
    coordinates: Tuple[float, float, float]
    label: str
    language: str
    confidence: float
    spatial_relations: List[str]
    semantic_class: str
    cultural_context: Optional[str] = None

@dataclass
class MultilingualCorpus:
    """Represents a multilingual corpus for spatial annotation"""
    language: str
    texts: List[str]
    spatial_terms: List[str]
    cultural_markers: List[str]
    annotation_guidelines: Dict[str, Any]

class MultilingualSpatialAnnotator:
    """
    Multilingual spatial annotation system for point clouds
    Supports Indonesian, Spanish, Arabic, and other languages
    """
    
    def __init__(self, 
                 graph_projector: GraphSemanticProjector = None,
                 supported_languages: List[str] = None):
        
        self.graph_projector = graph_projector
        self.supported_languages = supported_languages or list(SUPPORTED_LANGUAGES.keys())
        
        # Language-specific annotation models
        self.language_models = {}
        self._initialize_language_models()
        
        # Spatial relation vocabularies
        self.spatial_vocabularies = self._build_spatial_vocabularies()
        
        # Cultural context mappings
        self.cultural_contexts = self._build_cultural_contexts()
        
        # Annotation confidence thresholds
        self.confidence_thresholds = {
            'high': 0.8,
            'medium': 0.6,
            'low': 0.4
        }
        
        print(f"🌍 Multilingual Spatial Annotator initialized for {len(self.supported_languages)} languages")
    
    def _initialize_language_models(self):
        """Initialize language-specific models"""
        for lang in self.supported_languages:
            if lang in SUPPORTED_LANGUAGES:
                self.language_models[lang] = {
                    'spatial_classifier': self._create_spatial_classifier(lang),
                    'semantic_mapper': self._create_semantic_mapper(lang),
                    'confidence_estimator': self._create_confidence_estimator(lang)
                }
    
    def _create_spatial_classifier(self, language):
        """Create spatial relation classifier for a language"""
        num_spatial_terms = len(SUPPORTED_LANGUAGES[language]['spatial_terms'])
        
        classifier = torch.nn.Sequential(
            torch.nn.Linear(256, 128),  # Assuming 256-dim embeddings
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(128, num_spatial_terms),
            torch.nn.Softmax(dim=-1)
        )
        
        return classifier
    
    def _create_semantic_mapper(self, language):
        """Create semantic class mapper for a language"""
        # Standard semantic classes for point cloud segmentation
        semantic_classes = ['ground', 'building', 'vegetation', 'vehicle', 'person', 'furniture', 'other']
        
        mapper = torch.nn.Sequential(
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, len(semantic_classes)),
            torch.nn.Softmax(dim=-1)
        )
        
        return mapper
    
    def _create_confidence_estimator(self, language):
        """Create confidence estimator for a language"""
        estimator = torch.nn.Sequential(
            torch.nn.Linear(256, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
            torch.nn.Sigmoid()
        )
        
        return estimator
    
    def _build_spatial_vocabularies(self):
        """Build spatial vocabularies for all supported languages"""
        vocabularies = {}
        
        for lang, lang_info in SUPPORTED_LANGUAGES.items():
            if lang in self.supported_languages:
                vocabularies[lang] = {
                    'spatial_terms': lang_info['spatial_terms'],
                    'term_to_id': {term: i for i, term in enumerate(lang_info['spatial_terms'])},
                    'id_to_term': {i: term for i, term in enumerate(lang_info['spatial_terms'])}
                }
        
        return vocabularies
    
    def _build_cultural_contexts(self):
        """Build cultural context mappings"""
        contexts = {
            'en': ['western', 'individualistic', 'direct_communication'],
            'es': ['latin_american', 'iberian', 'collectivistic', 'high_context'],
            'ar': ['middle_eastern', 'islamic', 'high_context', 'hierarchical'],
            'id': ['southeast_asian', 'collectivistic', 'harmony_oriented', 'indirect'],
            'zh': ['east_asian', 'confucian', 'collectivistic', 'hierarchical'],
            'hi': ['south_asian', 'hierarchical', 'collectivistic', 'context_dependent']
        }
        
        return contexts
    
    def annotate_multilingual(self, 
                            points: np.ndarray, 
                            graph_embeddings: torch.Tensor, 
                            language: str = 'en',
                            text_context: str = None) -> List[SpatialAnnotation]:
        """
        Annotate point cloud with multilingual spatial labels
        
        Args:
            points: (N, 3) point coordinates
            graph_embeddings: (N, embedding_dim) graph-aligned embeddings
            language: Target language for annotations
            text_context: Optional text context for better annotation
            
        Returns:
            List of SpatialAnnotation objects
        """
        
        if language not in self.supported_languages:
            print(f"⚠️ Language {language} not supported. Using English as fallback.")
            language = 'en'
        
        annotations = []
        
        # Get language-specific models
        lang_models = self.language_models.get(language, self.language_models['en'])
        
        # Process each point
        for i, (point, embedding) in enumerate(zip(points, graph_embeddings)):
            
            # Classify spatial relations
            spatial_probs = lang_models['spatial_classifier'](embedding.unsqueeze(0))
            spatial_class_id = torch.argmax(spatial_probs, dim=-1).item()
            spatial_confidence = torch.max(spatial_probs).item()
            
            # Get spatial term
            spatial_term = self.spatial_vocabularies[language]['id_to_term'][spatial_class_id]
            
            # Classify semantic class
            semantic_probs = lang_models['semantic_mapper'](embedding.unsqueeze(0))
            semantic_class_id = torch.argmax(semantic_probs, dim=-1).item()
            semantic_classes = ['ground', 'building', 'vegetation', 'vehicle', 'person', 'furniture', 'other']
            semantic_class = semantic_classes[semantic_class_id]
            
            # Estimate overall confidence
            overall_confidence = lang_models['confidence_estimator'](embedding.unsqueeze(0)).item()
            
            # Extract spatial relations from text context if available
            spatial_relations = []
            if text_context:
                extracted_relations = extract_spatial_relations(text_context, language)
                spatial_relations = [rel['relation'] for rel in extracted_relations]
            
            # Create annotation
            annotation = SpatialAnnotation(
                point_id=i,
                coordinates=(float(point[0]), float(point[1]), float(point[2])),
                label=spatial_term,
                language=language,
                confidence=overall_confidence,
                spatial_relations=spatial_relations,
                semantic_class=semantic_class,
                cultural_context=self.cultural_contexts.get(language, ['unknown'])[0]
            )
            
            annotations.append(annotation)
        
        return annotations
    
    def multilingual_lookup(self, 
                          embedding: torch.Tensor, 
                          language: str,
                          similarity_threshold: float = 0.7) -> Dict[str, Any]:
        """
        Perform multilingual lookup for a single embedding
        
        Args:
            embedding: (embedding_dim,) single embedding vector
            language: Target language
            similarity_threshold: Minimum similarity for confident lookup
            
        Returns:
            Dictionary with lookup results
        """
        
        if language not in self.supported_languages:
            language = 'en'
        
        # Get language-specific models
        lang_models = self.language_models.get(language, self.language_models['en'])
        
        # Classify spatial relation
        spatial_probs = lang_models['spatial_classifier'](embedding.unsqueeze(0))
        spatial_class_id = torch.argmax(spatial_probs, dim=-1).item()
        spatial_confidence = torch.max(spatial_probs).item()
        
        # Get spatial term
        spatial_term = self.spatial_vocabularies[language]['id_to_term'][spatial_class_id]
        
        # Classify semantic class
        semantic_probs = lang_models['semantic_mapper'](embedding.unsqueeze(0))
        semantic_class_id = torch.argmax(semantic_probs, dim=-1).item()
        semantic_classes = ['ground', 'building', 'vegetation', 'vehicle', 'person', 'furniture', 'other']
        semantic_class = semantic_classes[semantic_class_id]
        
        # Overall confidence
        overall_confidence = lang_models['confidence_estimator'](embedding.unsqueeze(0)).item()
        
        # Determine confidence level
        if overall_confidence >= self.confidence_thresholds['high']:
            confidence_level = 'high'
        elif overall_confidence >= self.confidence_thresholds['medium']:
            confidence_level = 'medium'
        else:
            confidence_level = 'low'
        
        return {
            'spatial_term': spatial_term,
            'semantic_class': semantic_class,
            'confidence': overall_confidence,
            'confidence_level': confidence_level,
            'language': language,
            'cultural_context': self.cultural_contexts.get(language, ['unknown'])[0],
            'is_confident': overall_confidence >= similarity_threshold
        }
    
    def create_multilingual_corpus(self, 
                                 language: str,
                                 domain: str = 'spatial') -> MultilingualCorpus:
        """
        Create a multilingual corpus for a specific language and domain
        
        Args:
            language: Target language
            domain: Domain for corpus (e.g., 'spatial', 'urban', 'indoor')
            
        Returns:
            MultilingualCorpus object
        """
        
        if language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Language {language} not supported")
        
        lang_info = SUPPORTED_LANGUAGES[language]
        
        # Generate sample texts based on spatial terms
        sample_texts = []
        for spatial_term in lang_info['spatial_terms']:
            if language == 'en':
                sample_texts.extend([
                    f"The object is {spatial_term} the reference point.",
                    f"Move {spatial_term} to reach the target.",
                    f"The building is located {spatial_term}."
                ])
            elif language == 'es':
                sample_texts.extend([
                    f"El objeto está {spatial_term} del punto de referencia.",
                    f"Muévete {spatial_term} para alcanzar el objetivo.",
                    f"El edificio está ubicado {spatial_term}."
                ])
            elif language == 'ar':
                sample_texts.extend([
                    f"الكائن {spatial_term} نقطة المرجع.",
                    f"تحرك {spatial_term} للوصول إلى الهدف.",
                    f"المبنى يقع {spatial_term}."
                ])
            elif language == 'id':
                sample_texts.extend([
                    f"Objek berada {spatial_term} titik referensi.",
                    f"Bergerak {spatial_term} untuk mencapai target.",
                    f"Bangunan terletak {spatial_term}."
                ])
        
        # Cultural markers
        cultural_markers = self.cultural_contexts.get(language, [])
        
        # Annotation guidelines
        annotation_guidelines = {
            'spatial_relations': lang_info['spatial_terms'],
            'confidence_threshold': 0.6,
            'cultural_sensitivity': True,
            'context_awareness': True,
            'domain_specific': domain == 'spatial'
        }
        
        corpus = MultilingualCorpus(
            language=language,
            texts=sample_texts,
            spatial_terms=lang_info['spatial_terms'],
            cultural_markers=cultural_markers,
            annotation_guidelines=annotation_guidelines
        )
        
        return corpus
    
    def batch_annotate_multilingual(self,
                                   points_batch: List[np.ndarray],
                                   embeddings_batch: List[torch.Tensor],
                                   languages: List[str],
                                   text_contexts: List[str] = None) -> Dict[str, List[SpatialAnnotation]]:
        """
        Batch annotate multiple point clouds in different languages
        
        Args:
            points_batch: List of point arrays
            embeddings_batch: List of embedding tensors
            languages: List of target languages
            text_contexts: Optional list of text contexts
            
        Returns:
            Dictionary mapping languages to annotation lists
        """
        
        results = {}
        
        for i, (points, embeddings, language) in enumerate(zip(points_batch, embeddings_batch, languages)):
            text_context = text_contexts[i] if text_contexts else None
            
            annotations = self.annotate_multilingual(
                points=points,
                graph_embeddings=embeddings,
                language=language,
                text_context=text_context
            )
            
            if language not in results:
                results[language] = []
            results[language].extend(annotations)
        
        return results
    
    def export_annotations(self, 
                          annotations: List[SpatialAnnotation],
                          output_path: str,
                          format: str = 'json'):
        """
        Export annotations to file
        
        Args:
            annotations: List of annotations to export
            output_path: Output file path
            format: Export format ('json', 'csv', 'xml')
        """
        
        if format == 'json':
            annotation_data = []
            for ann in annotations:
                annotation_data.append({
                    'point_id': ann.point_id,
                    'coordinates': ann.coordinates,
                    'label': ann.label,
                    'language': ann.language,
                    'confidence': ann.confidence,
                    'spatial_relations': ann.spatial_relations,
                    'semantic_class': ann.semantic_class,
                    'cultural_context': ann.cultural_context
                })
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(annotation_data, f, indent=2, ensure_ascii=False)
        
        elif format == 'csv':
            import csv
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['point_id', 'x', 'y', 'z', 'label', 'language', 
                               'confidence', 'semantic_class', 'cultural_context'])
                
                for ann in annotations:
                    writer.writerow([
                        ann.point_id, ann.coordinates[0], ann.coordinates[1], ann.coordinates[2],
                        ann.label, ann.language, ann.confidence, ann.semantic_class, ann.cultural_context
                    ])
        
        print(f"✅ Exported {len(annotations)} annotations to {output_path}")
    
    def get_annotation_statistics(self, annotations: List[SpatialAnnotation]) -> Dict[str, Any]:
        """Get statistics about annotations"""
        
        if not annotations:
            return {}
        
        # Language distribution
        language_counts = {}
        for ann in annotations:
            language_counts[ann.language] = language_counts.get(ann.language, 0) + 1
        
        # Confidence distribution
        confidences = [ann.confidence for ann in annotations]
        avg_confidence = np.mean(confidences)
        
        # Semantic class distribution
        semantic_counts = {}
        for ann in annotations:
            semantic_counts[ann.semantic_class] = semantic_counts.get(ann.semantic_class, 0) + 1
        
        # Spatial relation distribution
        spatial_counts = {}
        for ann in annotations:
            spatial_counts[ann.label] = spatial_counts.get(ann.label, 0) + 1
        
        return {
            'total_annotations': len(annotations),
            'language_distribution': language_counts,
            'average_confidence': avg_confidence,
            'confidence_range': (min(confidences), max(confidences)),
            'semantic_class_distribution': semantic_counts,
            'spatial_relation_distribution': spatial_counts,
            'supported_languages': self.supported_languages
        }

def demo_multilingual_annotation():
    """Demonstrate multilingual spatial annotation"""
    
    print("🌍 Multilingual Spatial Annotation Demo")
    print("=" * 50)
    
    # Create annotator
    annotator = MultilingualSpatialAnnotator()
    
    # Create sample point cloud
    num_points = 100
    points = np.random.randn(num_points, 3)
    embeddings = torch.randn(num_points, 256)
    
    print(f"📊 Sample Data:")
    print(f"  - Points: {points.shape}")
    print(f"  - Embeddings: {embeddings.shape}")
    print(f"  - Supported languages: {len(annotator.supported_languages)}")
    
    # Test different languages
    test_languages = ['en', 'es', 'ar', 'id']
    all_annotations = {}
    
    for lang in test_languages:
        if lang in annotator.supported_languages:
            print(f"\n🔍 Annotating in {SUPPORTED_LANGUAGES[lang]['name']}...")
            
            # Create sample text context
            if lang == 'en':
                text_context = "The building is above the ground near the trees."
            elif lang == 'es':
                text_context = "El edificio está arriba del suelo cerca de los árboles."
            elif lang == 'ar':
                text_context = "المبنى فوق الأرض قريب من الأشجار."
            elif lang == 'id':
                text_context = "Bangunan berada di atas tanah dekat dengan pohon-pohon."
            else:
                text_context = None
            
            annotations = annotator.annotate_multilingual(
                points=points,
                graph_embeddings=embeddings,
                language=lang,
                text_context=text_context
            )
            
            all_annotations[lang] = annotations
            
            # Show sample annotations
            print(f"  ✅ Generated {len(annotations)} annotations")
            if annotations:
                sample_ann = annotations[0]
                print(f"  📝 Sample: '{sample_ann.label}' (confidence: {sample_ann.confidence:.3f})")
    
    # Get statistics
    print(f"\n📈 Annotation Statistics:")
    for lang, annotations in all_annotations.items():
        stats = annotator.get_annotation_statistics(annotations)
        print(f"  - {SUPPORTED_LANGUAGES[lang]['name']}: {stats['total_annotations']} annotations, "
              f"avg confidence: {stats['average_confidence']:.3f}")
    
    # Test multilingual lookup
    print(f"\n🔍 Multilingual Lookup Demo:")
    sample_embedding = embeddings[0]
    
    for lang in test_languages[:3]:  # Test first 3 languages
        if lang in annotator.supported_languages:
            lookup_result = annotator.multilingual_lookup(sample_embedding, lang)
            print(f"  - {SUPPORTED_LANGUAGES[lang]['name']}: '{lookup_result['spatial_term']}' "
                  f"({lookup_result['confidence_level']} confidence)")
    
    # Create multilingual corpus
    print(f"\n📚 Multilingual Corpus Demo:")
    for lang in ['en', 'es', 'id']:
        if lang in annotator.supported_languages:
            corpus = annotator.create_multilingual_corpus(lang, domain='spatial')
            print(f"  - {SUPPORTED_LANGUAGES[lang]['name']}: {len(corpus.texts)} texts, "
                  f"{len(corpus.spatial_terms)} spatial terms")
    
    # Export annotations
    output_path = "output/randla_graph_align/multilingual_annotations/demo_annotations.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if all_annotations:
        # Export English annotations as example
        annotator.export_annotations(all_annotations['en'], output_path, format='json')
    
    print(f"\n✅ Demo completed successfully!")
    return annotator, all_annotations

if __name__ == "__main__":
    demo_multilingual_annotation()
