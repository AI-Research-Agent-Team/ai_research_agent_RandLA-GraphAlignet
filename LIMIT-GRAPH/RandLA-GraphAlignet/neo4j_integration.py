# -*- coding: utf-8 -*-
"""
Neo4j Integration for RandLA-GraphAlignNet
Provides real-time visualization and graph database storage for spatial annotations
"""

import sys
import os
from typing import Dict, List, Any, Optional, Tuple
import json
import numpy as np
from datetime import datetime
import logging

# Neo4j driver
try:
    from neo4j import GraphDatabase, basic_auth
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    print("⚠️ Neo4j driver not available. Install with: pip install neo4j")

# Import local components
from annotate_multilingual import SpatialAnnotation, MultilingualSpatialAnnotator
from dependencies import SUPPORTED_LANGUAGES

class Neo4jSpatialVisualizer:
    """
    Neo4j integration for RandLA-GraphAlignNet spatial annotations
    Provides real-time visualization and persistent storage
    """
    
    def __init__(self, 
                 uri: str = "bolt://localhost:7687",
                 username: str = "neo4j", 
                 password: str = "_KDITp62FstVa-ImKe1fdScIlkdOhrRUSdJ1uFNcook"):
        
        if not NEO4J_AVAILABLE:
            raise ImportError("Neo4j driver not available. Install with: pip install neo4j")
        
        self.uri = uri
        self.username = username
        self.password = password
        
        # Initialize driver
        try:
            self.driver = GraphDatabase.driver(
                uri, 
                auth=basic_auth(username, password),
                encrypted=False  # For local development
            )
            
            # Test connection
            with self.driver.session() as session:
                result = session.run("RETURN 1 as test")
                test_value = result.single()["test"]
                if test_value == 1:
                    print(f"✅ Connected to Neo4j at {uri}")
                else:
                    raise Exception("Connection test failed")
                    
        except Exception as e:
            print(f"❌ Failed to connect to Neo4j: {e}")
            print(f"   Make sure Neo4j is running at {uri}")
            print(f"   Username: {username}")
            print(f"   Check your Neo4j configuration")
            raise
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize schema
        self._initialize_schema()
        
    def _initialize_schema(self):
        """Initialize Neo4j schema for spatial annotations"""
        
        with self.driver.session() as session:
            # Create constraints and indexes
            constraints_and_indexes = [
                # Unique constraints
                "CREATE CONSTRAINT spatial_annotation_id IF NOT EXISTS FOR (n:SpatialAnnotation) REQUIRE n.id IS UNIQUE",
                "CREATE CONSTRAINT point_cloud_id IF NOT EXISTS FOR (n:PointCloud) REQUIRE n.id IS UNIQUE",
                "CREATE CONSTRAINT language_code IF NOT EXISTS FOR (n:Language) REQUIRE n.code IS UNIQUE",
                
                # Indexes for performance
                "CREATE INDEX spatial_annotation_language IF NOT EXISTS FOR (n:SpatialAnnotation) ON (n.language)",
                "CREATE INDEX spatial_annotation_confidence IF NOT EXISTS FOR (n:SpatialAnnotation) ON (n.confidence)",
                "CREATE INDEX spatial_annotation_semantic_class IF NOT EXISTS FOR (n:SpatialAnnotation) ON (n.semantic_class)",
                "CREATE INDEX point_coordinates IF NOT EXISTS FOR (n:SpatialAnnotation) ON (n.x, n.y, n.z)",
            ]
            
            for constraint in constraints_and_indexes:
                try:
                    session.run(constraint)
                except Exception as e:
                    # Constraint might already exist
                    self.logger.debug(f"Constraint/Index creation: {e}")
        
        # Create language nodes
        self._create_language_nodes()
        
        print("🔧 Neo4j schema initialized")
    
    def _create_language_nodes(self):
        """Create language nodes in Neo4j"""
        
        with self.driver.session() as session:
            for lang_code, lang_info in SUPPORTED_LANGUAGES.items():
                session.run("""
                    MERGE (l:Language {code: $code})
                    SET l.name = $name,
                        l.rtl = $rtl,
                        l.spatial_terms = $spatial_terms,
                        l.updated_at = datetime()
                """, 
                code=lang_code,
                name=lang_info['name'],
                rtl=lang_info.get('rtl', False),
                spatial_terms=lang_info['spatial_terms']
                )
    
    def create_point_cloud_session(self, 
                                  session_id: str,
                                  metadata: Dict[str, Any] = None) -> str:
        """
        Create a new point cloud session in Neo4j
        
        Args:
            session_id: Unique session identifier
            metadata: Optional metadata for the session
            
        Returns:
            Created session ID
        """
        
        with self.driver.session() as session:
            result = session.run("""
                CREATE (pc:PointCloud {
                    id: $session_id,
                    created_at: datetime(),
                    metadata: $metadata
                })
                RETURN pc.id as id
            """, 
            session_id=session_id,
            metadata=json.dumps(metadata or {})
            )
            
            created_id = result.single()["id"]
            self.logger.info(f"Created point cloud session: {created_id}")
            return created_id
    
    def store_spatial_annotations(self, 
                                 annotations: List[SpatialAnnotation],
                                 session_id: str = None) -> int:
        """
        Store spatial annotations in Neo4j
        
        Args:
            annotations: List of spatial annotations
            session_id: Optional session ID to associate annotations with
            
        Returns:
            Number of annotations stored
        """
        
        if not annotations:
            return 0
        
        # Create session if not provided
        if session_id is None:
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.create_point_cloud_session(session_id)
        
        stored_count = 0
        
        with self.driver.session() as session:
            for ann in annotations:
                try:
                    # Create spatial annotation node
                    session.run("""
                        CREATE (sa:SpatialAnnotation {
                            id: $id,
                            point_id: $point_id,
                            x: $x,
                            y: $y,
                            z: $z,
                            label: $label,
                            language: $language,
                            confidence: $confidence,
                            semantic_class: $semantic_class,
                            cultural_context: $cultural_context,
                            spatial_relations: $spatial_relations,
                            created_at: datetime()
                        })
                    """,
                    id=f"{session_id}_{ann.point_id}",
                    point_id=ann.point_id,
                    x=float(ann.coordinates[0]),
                    y=float(ann.coordinates[1]),
                    z=float(ann.coordinates[2]),
                    label=ann.label,
                    language=ann.language,
                    confidence=float(ann.confidence),
                    semantic_class=ann.semantic_class,
                    cultural_context=ann.cultural_context,
                    spatial_relations=ann.spatial_relations
                    )
                    
                    # Link to point cloud session
                    session.run("""
                        MATCH (pc:PointCloud {id: $session_id})
                        MATCH (sa:SpatialAnnotation {id: $annotation_id})
                        CREATE (pc)-[:CONTAINS]->(sa)
                    """,
                    session_id=session_id,
                    annotation_id=f"{session_id}_{ann.point_id}"
                    )
                    
                    # Link to language
                    session.run("""
                        MATCH (l:Language {code: $language})
                        MATCH (sa:SpatialAnnotation {id: $annotation_id})
                        CREATE (sa)-[:IN_LANGUAGE]->(l)
                    """,
                    language=ann.language,
                    annotation_id=f"{session_id}_{ann.point_id}"
                    )
                    
                    stored_count += 1
                    
                except Exception as e:
                    self.logger.error(f"Failed to store annotation {ann.point_id}: {e}")
        
        self.logger.info(f"Stored {stored_count} spatial annotations in session {session_id}")
        return stored_count
    
    def create_spatial_relationships(self, 
                                   graph_edges: List[Dict[str, Any]],
                                   session_id: str = None) -> int:
        """
        Create spatial relationships between annotations
        
        Args:
            graph_edges: List of graph edge definitions
            session_id: Optional session ID to filter annotations
            
        Returns:
            Number of relationships created
        """
        
        relationships_created = 0
        
        with self.driver.session() as session:
            for edge in graph_edges:
                try:
                    # Create relationship between spatial annotations
                    query = """
                    MATCH (source:SpatialAnnotation)
                    MATCH (target:SpatialAnnotation)
                    WHERE source.label = $subject 
                      AND target.label = $object
                      AND source.language = $language
                      AND target.language = $language
                    """
                    
                    if session_id:
                        query += " AND source.id STARTS WITH $session_id AND target.id STARTS WITH $session_id"
                    
                    query += """
                    CREATE (source)-[r:SPATIAL_RELATION {
                        type: $relation,
                        language: $language,
                        created_at: datetime()
                    }]->(target)
                    RETURN count(r) as created
                    """
                    
                    result = session.run(query,
                        subject=edge['subject'],
                        object=edge['object'],
                        relation=edge['relation'],
                        language=edge['language'],
                        session_id=session_id
                    )
                    
                    created = result.single()["created"]
                    relationships_created += created
                    
                except Exception as e:
                    self.logger.error(f"Failed to create relationship for edge {edge}: {e}")
        
        self.logger.info(f"Created {relationships_created} spatial relationships")
        return relationships_created
    
    def query_spatial_annotations(self, 
                                 language: str = None,
                                 semantic_class: str = None,
                                 min_confidence: float = 0.0,
                                 session_id: str = None) -> List[Dict[str, Any]]:
        """
        Query spatial annotations from Neo4j
        
        Args:
            language: Filter by language
            semantic_class: Filter by semantic class
            min_confidence: Minimum confidence threshold
            session_id: Filter by session ID
            
        Returns:
            List of annotation dictionaries
        """
        
        query = "MATCH (sa:SpatialAnnotation)"
        conditions = []
        params = {}
        
        if language:
            conditions.append("sa.language = $language")
            params["language"] = language
        
        if semantic_class:
            conditions.append("sa.semantic_class = $semantic_class")
            params["semantic_class"] = semantic_class
        
        if min_confidence > 0:
            conditions.append("sa.confidence >= $min_confidence")
            params["min_confidence"] = min_confidence
        
        if session_id:
            conditions.append("sa.id STARTS WITH $session_id")
            params["session_id"] = session_id
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += """
        RETURN sa.id as id, sa.point_id as point_id,
               sa.x as x, sa.y as y, sa.z as z,
               sa.label as label, sa.language as language,
               sa.confidence as confidence, sa.semantic_class as semantic_class,
               sa.cultural_context as cultural_context,
               sa.spatial_relations as spatial_relations
        ORDER BY sa.confidence DESC
        """
        
        results = []
        with self.driver.session() as session:
            result = session.run(query, params)
            for record in result:
                results.append(dict(record))
        
        return results
    
    def get_spatial_graph_statistics(self) -> Dict[str, Any]:
        """Get statistics about the spatial graph in Neo4j"""
        
        with self.driver.session() as session:
            # Count nodes by type
            node_counts = session.run("""
                MATCH (n)
                RETURN labels(n)[0] as label, count(n) as count
            """).data()
            
            # Count relationships by type
            rel_counts = session.run("""
                MATCH ()-[r]->()
                RETURN type(r) as type, count(r) as count
            """).data()
            
            # Language distribution
            lang_dist = session.run("""
                MATCH (sa:SpatialAnnotation)
                RETURN sa.language as language, count(sa) as count
                ORDER BY count DESC
            """).data()
            
            # Confidence statistics
            confidence_stats = session.run("""
                MATCH (sa:SpatialAnnotation)
                RETURN avg(sa.confidence) as avg_confidence,
                       min(sa.confidence) as min_confidence,
                       max(sa.confidence) as max_confidence,
                       count(sa) as total_annotations
            """).single()
            
            return {
                "node_counts": {item["label"]: item["count"] for item in node_counts},
                "relationship_counts": {item["type"]: item["count"] for item in rel_counts},
                "language_distribution": {item["language"]: item["count"] for item in lang_dist},
                "confidence_statistics": dict(confidence_stats) if confidence_stats else {}
            }
    
    def export_cypher_queries(self, output_file: str = "neo4j_queries.cypher"):
        """Export useful Cypher queries for manual exploration"""
        
        queries = {
            "Basic Queries": [
                "// Count all spatial annotations",
                "MATCH (sa:SpatialAnnotation) RETURN count(sa);",
                "",
                "// Get annotations by language",
                "MATCH (sa:SpatialAnnotation) WHERE sa.language = 'en' RETURN sa LIMIT 10;",
                "",
                "// Find high-confidence annotations",
                "MATCH (sa:SpatialAnnotation) WHERE sa.confidence > 0.8 RETURN sa ORDER BY sa.confidence DESC;",
            ],
            
            "Spatial Queries": [
                "// Find annotations near a point (within distance)",
                "MATCH (sa:SpatialAnnotation)",
                "WHERE sqrt((sa.x - 0)^2 + (sa.y - 0)^2 + (sa.z - 0)^2) < 5",
                "RETURN sa;",
                "",
                "// Find spatial relationships",
                "MATCH (source:SpatialAnnotation)-[r:SPATIAL_RELATION]->(target:SpatialAnnotation)",
                "RETURN source.label, r.type, target.label, r.language;",
            ],
            
            "Multilingual Queries": [
                "// Compare same concept across languages",
                "MATCH (sa1:SpatialAnnotation), (sa2:SpatialAnnotation)",
                "WHERE sa1.semantic_class = sa2.semantic_class",
                "  AND sa1.language <> sa2.language",
                "RETURN sa1.language, sa1.label, sa2.language, sa2.label, sa1.semantic_class;",
                "",
                "// Language statistics",
                "MATCH (sa:SpatialAnnotation)",
                "RETURN sa.language, count(sa) as count, avg(sa.confidence) as avg_confidence",
                "ORDER BY count DESC;",
            ],
            
            "Analysis Queries": [
                "// Find most common spatial relations by language",
                "MATCH (sa:SpatialAnnotation)",
                "RETURN sa.language, sa.label, count(sa) as frequency",
                "ORDER BY sa.language, frequency DESC;",
                "",
                "// Cultural context analysis",
                "MATCH (sa:SpatialAnnotation)",
                "WHERE sa.cultural_context IS NOT NULL",
                "RETURN sa.cultural_context, count(sa) as count",
                "ORDER BY count DESC;",
            ]
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("// RandLA-GraphAlignNet Neo4j Queries\n")
            f.write(f"// Generated on {datetime.now().isoformat()}\n\n")
            
            for category, query_list in queries.items():
                f.write(f"// ========== {category} ==========\n")
                for line in query_list:
                    f.write(f"{line}\n")
                f.write("\n")
        
        print(f"📄 Cypher queries exported to {output_file}")
    
    def visualize_spatial_graph(self, session_id: str = None) -> str:
        """
        Generate Neo4j Browser visualization URL
        
        Args:
            session_id: Optional session ID to focus visualization
            
        Returns:
            Neo4j Browser URL for visualization
        """
        
        base_url = self.uri.replace("bolt://", "http://").replace(":7687", ":7474")
        
        if session_id:
            query = f"MATCH (pc:PointCloud {{id: '{session_id}'}})-[:CONTAINS]->(sa:SpatialAnnotation) RETURN pc, sa"
        else:
            query = "MATCH (sa:SpatialAnnotation)-[r:SPATIAL_RELATION]->(target) RETURN sa, r, target LIMIT 50"
        
        # URL encode the query
        import urllib.parse
        encoded_query = urllib.parse.quote(query)
        
        viz_url = f"{base_url}/browser/?cmd=play&arg={encoded_query}"
        
        print(f"🎨 Neo4j Browser visualization: {viz_url}")
        return viz_url
    
    def close(self):
        """Close Neo4j connection"""
        if hasattr(self, 'driver'):
            self.driver.close()
            print("🔌 Neo4j connection closed")

def demo_neo4j_integration():
    """Demonstrate Neo4j integration with RandLA-GraphAlignNet"""
    
    print("🚀 Neo4j Integration Demo for RandLA-GraphAlignNet")
    print("=" * 60)
    
    try:
        # Initialize Neo4j visualizer
        visualizer = Neo4jSpatialVisualizer()
        
        # Create sample annotations
        from annotate_multilingual import MultilingualSpatialAnnotator
        
        annotator = MultilingualSpatialAnnotator()
        
        # Create sample point cloud
        num_points = 50
        points = np.random.randn(num_points, 3) * 5
        
        # Create sample embeddings
        import torch
        embeddings = torch.randn(num_points, 256)
        
        # Generate multilingual annotations
        languages = ['en', 'es', 'ar', 'id']
        all_annotations = []
        
        for lang in languages:
            annotations = annotator.annotate_multilingual(
                points=points,
                graph_embeddings=embeddings,
                language=lang
            )
            all_annotations.extend(annotations[:10])  # Limit for demo
        
        print(f"📊 Generated {len(all_annotations)} multilingual annotations")
        
        # Create session and store annotations
        session_id = f"demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        visualizer.create_point_cloud_session(
            session_id, 
            metadata={"demo": True, "languages": languages, "points": num_points}
        )
        
        stored_count = visualizer.store_spatial_annotations(all_annotations, session_id)
        print(f"✅ Stored {stored_count} annotations in Neo4j")
        
        # Create spatial relationships
        graph_edges = [
            {'subject': 'above', 'relation': 'opposite_of', 'object': 'below', 'language': 'en'},
            {'subject': 'arriba', 'relation': 'opposite_of', 'object': 'abajo', 'language': 'es'},
            {'subject': 'فوق', 'relation': 'opposite_of', 'object': 'تحت', 'language': 'ar'},
            {'subject': 'atas', 'relation': 'opposite_of', 'object': 'bawah', 'language': 'id'},
        ]
        
        rel_count = visualizer.create_spatial_relationships(graph_edges, session_id)
        print(f"✅ Created {rel_count} spatial relationships")
        
        # Query annotations
        print(f"\n🔍 Querying annotations:")
        
        for lang in languages[:2]:  # Test first 2 languages
            results = visualizer.query_spatial_annotations(
                language=lang, 
                min_confidence=0.5,
                session_id=session_id
            )
            print(f"  - {SUPPORTED_LANGUAGES[lang]['name']}: {len(results)} annotations")
        
        # Get statistics
        stats = visualizer.get_spatial_graph_statistics()
        print(f"\n📈 Neo4j Graph Statistics:")
        print(f"  - Nodes: {stats['node_counts']}")
        print(f"  - Relationships: {stats['relationship_counts']}")
        print(f"  - Languages: {list(stats['language_distribution'].keys())}")
        
        if stats['confidence_statistics']:
            conf_stats = stats['confidence_statistics']
            print(f"  - Avg confidence: {conf_stats.get('avg_confidence', 0):.3f}")
            print(f"  - Total annotations: {conf_stats.get('total_annotations', 0)}")
        
        # Export Cypher queries
        visualizer.export_cypher_queries("demo_neo4j_queries.cypher")
        
        # Generate visualization URL
        viz_url = visualizer.visualize_spatial_graph(session_id)
        
        print(f"\n🎨 Visualization:")
        print(f"  - Neo4j Browser: {viz_url}")
        print(f"  - Session ID: {session_id}")
        
        print(f"\n💡 Next Steps:")
        print(f"  1. Open Neo4j Browser at http://localhost:7474")
        print(f"  2. Run query: MATCH (pc:PointCloud {{id: '{session_id}'}})-[:CONTAINS]->(sa) RETURN pc, sa")
        print(f"  3. Explore spatial relationships and multilingual annotations")
        print(f"  4. Use exported Cypher queries for analysis")
        
        # Close connection
        visualizer.close()
        
        return session_id, stats
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        print(f"\n🔧 Troubleshooting:")
        print(f"  1. Make sure Neo4j is running: docker run -p 7474:7474 -p 7687:7687 neo4j")
        print(f"  2. Check Neo4j credentials and connection")
        print(f"  3. Verify Neo4j driver installation: pip install neo4j")
        raise

if __name__ == "__main__":
    demo_neo4j_integration()