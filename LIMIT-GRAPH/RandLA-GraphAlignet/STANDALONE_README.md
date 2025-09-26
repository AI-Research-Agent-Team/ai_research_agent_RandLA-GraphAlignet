# RandLA-GraphAlignNet Standalone

> **Standalone deployment of RandLA-GraphAlignNet with Neo4j integration for multilingual spatial reasoning on 3D point clouds.**

## 🚀 Quick Start

### Option 1: Docker Compose (Recommended)

```bash
# Clone repository
git clone https://github.com/AI-Research-Agent-Team/ai_research_agent_RandLA-GraphAlignet.git
cd RandLA-GraphAlignNet

# Start all services
docker-compose up -d

# View logs
docker-compose logs -f randla-graphalignnet

# Access services
# - Neo4j Browser: http://localhost:7474
# - Dash Dashboard: http://localhost:8051
```

### Option 2: Local Installation

```bash
# Clone repository
git clone https://github.com/your-username/RandLA-GraphAlignNet.git
cd RandLA-GraphAlignNet

# Install dependencies
pip install -r requirements.txt

# Start Neo4j (using Docker)
docker run -d \
  --name neo4j-randla \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/_KDITp62FstVa-ImKe1fdScIlkdOhrRUSdJ1uFNcook \
  neo4j:4.4

# Run RandLA-GraphAlignNet
python main.py --mode demo --languages en es ar id
```

### Option 3: Python Package Installation

```bash
# Install from PyPI (when published)
pip install randla-graphalignnet

# Or install from source
pip install git+https://github.com/your-username/RandLA-GraphAlignNet.git

# Run
randla-graphalign --mode demo --languages en es ar id
```

## 📋 Prerequisites

### System Requirements
- Python 3.8+
- 4GB+ RAM
- CUDA-compatible GPU (optional, for acceleration)

### Neo4j Setup
- Neo4j 4.4+ running on `bolt://localhost:7687`
- Username: `neo4j`
- Password: `_KDITp62FstVa-ImKe1fdScIlkdOhrRUSdJ1uFNcook`

## 🔧 Configuration

### Environment Variables

```bash
# Neo4j Configuration
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="_KDITp62FstVa-ImKe1fdScIlkdOhrRUSdJ1uFNcook"

# Application Configuration
export RANDLA_OUTPUT_DIR="./output"
export RANDLA_LOG_LEVEL="INFO"
```

### Neo4j Configuration

Create or update Neo4j configuration:

```bash
# neo4j.conf additions
dbms.memory.heap.initial_size=512m
dbms.memory.heap.max_size=2G
dbms.memory.pagecache.size=1G
dbms.connector.bolt.listen_address=0.0.0.0:7687
dbms.connector.http.listen_address=0.0.0.0:7474
```

## 🎮 Usage

### Command Line Interface

```bash
# Basic demo with multilingual processing
python main.py --mode demo --languages en es ar id

# Process specific point cloud
python main.py --mode process --num-points 4096 --scene-type urban

# Run evaluation only
python main.py --mode evaluate --languages en es

# Neo4j integration only
python main.py --mode neo4j --session-id my_session

# Custom configuration
python main.py \
  --mode demo \
  --languages en es ar id zh hi \
  --num-points 2048 \
  --scene-type urban \
  --output-dir ./my_output \
  --session-id custom_session_001
```

### Python API

```python
from main import main
import sys

# Programmatic execution
sys.argv = [
    'main.py', 
    '--mode', 'demo', 
    '--languages', 'en', 'es', 'ar', 'id',
    '--num-points', '2048'
]

result = main()
print(f"Execution result: {result}")
```

### Advanced Usage

```python
# Direct component usage
from RandLA_GraphAlignNet import create_model
from annotate_multilingual import MultilingualSpatialAnnotator
from neo4j_integration import Neo4jSpatialVisualizer

# Initialize components
model = create_model(num_classes=8)
annotator = MultilingualSpatialAnnotator()
visualizer = Neo4jSpatialVisualizer()

# Process point cloud
import numpy as np
import torch

points = np.random.randn(2048, 3)
points_tensor = torch.from_numpy(points).float().unsqueeze(0)

# Get embeddings
with torch.no_grad():
    results = model(points_tensor, language='en')

embeddings = results['graph_embeddings'].squeeze(0)

# Generate annotations
annotations = annotator.annotate_multilingual(
    points=points,
    graph_embeddings=embeddings,
    language='es'
)

# Store in Neo4j
session_id = "custom_session"
visualizer.create_point_cloud_session(session_id)
visualizer.store_spatial_annotations(annotations, session_id)
```

## 🔗 Neo4j Integration

### Connection Details
- **URI**: `bolt://localhost:7687`
- **Username**: `neo4j`
- **Password**: `_KDITp62FstVa-ImKe1fdScIlkdOhrRUSdJ1uFNcook`
- **Browser**: http://localhost:7474

### Data Model

```cypher
// Node Types
(:SpatialAnnotation {id, point_id, x, y, z, label, language, confidence, semantic_class})
(:PointCloud {id, created_at, metadata})
(:Language {code, name, rtl, spatial_terms})

// Relationship Types
(:PointCloud)-[:CONTAINS]->(:SpatialAnnotation)
(:SpatialAnnotation)-[:IN_LANGUAGE]->(:Language)
(:SpatialAnnotation)-[:SPATIAL_RELATION {type, language}]->(:SpatialAnnotation)
```

### Sample Queries

```cypher
// View all spatial annotations
MATCH (sa:SpatialAnnotation) 
RETURN sa 
LIMIT 25;

// Find multilingual spatial relations
MATCH (sa1:SpatialAnnotation)-[r:SPATIAL_RELATION]->(sa2:SpatialAnnotation)
RETURN sa1.language, sa1.label, r.type, sa2.label, sa2.language;

// Language distribution
MATCH (sa:SpatialAnnotation)
RETURN sa.language, count(sa) as count, avg(sa.confidence) as avg_confidence
ORDER BY count DESC;

// High-confidence annotations by semantic class
MATCH (sa:SpatialAnnotation)
WHERE sa.confidence > 0.8
RETURN sa.semantic_class, count(sa) as count
ORDER BY count DESC;

// Spatial proximity analysis
MATCH (sa1:SpatialAnnotation), (sa2:SpatialAnnotation)
WHERE sa1 <> sa2 
  AND sqrt((sa1.x - sa2.x)^2 + (sa1.y - sa2.y)^2 + (sa1.z - sa2.z)^2) < 5
RETURN sa1, sa2, sqrt((sa1.x - sa2.x)^2 + (sa1.y - sa2.y)^2 + (sa1.z - sa2.z)^2) as distance
ORDER BY distance;
```

### Visualization

Access Neo4j Browser at http://localhost:7474 and run:

```cypher
// Visualize spatial relationships
MATCH (sa:SpatialAnnotation)-[r:SPATIAL_RELATION]->(target)
RETURN sa, r, target
LIMIT 50;

// Visualize by language
MATCH (sa:SpatialAnnotation)-[:IN_LANGUAGE]->(l:Language)
WHERE l.code IN ['en', 'es', 'ar', 'id']
RETURN sa, l;

// 3D point visualization (conceptual)
MATCH (sa:SpatialAnnotation)
RETURN sa.x as x, sa.y as y, sa.z as z, sa.label as label, sa.language as language
ORDER BY sa.confidence DESC
LIMIT 100;
```

## 📊 Monitoring & Observability

### Application Logs

```bash
# View real-time logs
tail -f output/randla_*.log

# Docker logs
docker-compose logs -f randla-graphalignnet
```

### Neo4j Monitoring

```cypher
// Database statistics
CALL db.stats.retrieve('GRAPH COUNTS');

// Memory usage
CALL dbms.queryJmx('org.neo4j:instance=kernel#0,name=Memory Pools') 
YIELD attributes 
RETURN attributes.Name, attributes.Usage;

// Active queries
CALL dbms.listQueries();
```

### Performance Metrics

```python
# Get system statistics
from main import main
from neo4j_integration import Neo4jSpatialVisualizer

# Run processing and get metrics
result = main()

# Get Neo4j statistics
visualizer = Neo4jSpatialVisualizer()
stats = visualizer.get_spatial_graph_statistics()
print(f"Neo4j Statistics: {stats}")
```

## 🐛 Troubleshooting

### Common Issues

#### Neo4j Connection Failed
```bash
# Check Neo4j status
docker ps | grep neo4j

# Restart Neo4j
docker restart neo4j-randla

# Check logs
docker logs neo4j-randla

# Test connection
python -c "
from neo4j import GraphDatabase
driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', '_KDITp62FstVa-ImKe1fdScIlkdOhrRUSdJ1uFNcook'))
with driver.session() as session:
    result = session.run('RETURN 1')
    print('✅ Neo4j connection successful')
driver.close()
"
```

#### Dependency Issues
```bash
# Check dependencies
python dependencies.py

# Reinstall PyTorch
pip uninstall torch torchvision torch-geometric
pip install torch==1.9.0+cpu torchvision==0.10.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
pip install torch-geometric -f https://data.pyg.org/whl/torch-1.9.0+cpu.html

# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

#### Memory Issues
```bash
# Reduce point cloud size
python main.py --mode demo --num-points 512 --languages en

# Monitor memory usage
python -c "
import psutil
print(f'Memory usage: {psutil.virtual_memory().percent}%')
print(f'Available memory: {psutil.virtual_memory().available / 1024**3:.1f} GB')
"
```

#### Permission Issues
```bash
# Fix output directory permissions
chmod -R 755 output/

# Docker permission issues
sudo chown -R $USER:$USER output/
```

### Debug Mode

```bash
# Enable debug logging
export RANDLA_LOG_LEVEL=DEBUG
python main.py --mode demo --languages en

# Verbose output
python main.py --mode demo --languages en --verbose

# Component testing
python -c "
from dependencies import check_dependencies
deps = check_dependencies()
for name, status in deps.items():
    print(f'{name}: {status}')
"
```

## 🔄 CI/CD Integration

### GitHub Actions

The repository includes a comprehensive CI/CD pipeline:

```yaml
# .github/workflows/ci.yml
- Component testing
- Neo4j integration testing  
- Docker container testing
- Security scanning
- Multi-platform testing (Python 3.8, 3.9, 3.10)
```

### Local Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Test individual components
python RandLA_GraphAlignNet.py
python GraphSemanticProjector.py
python annotate_multilingual.py
python evaluate_alignment.py
python neo4j_integration.py

# Integration test
python demo_complete_integration.py
```

## 📦 Deployment Options

### Docker Deployment

```bash
# Build and run
docker build -t randla-graphalignnet .
docker run -p 8050:8050 randla-graphalignnet

# With Neo4j
docker-compose up -d
```

### Kubernetes Deployment

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: randla-graphalignnet
spec:
  replicas: 1
  selector:
    matchLabels:
      app: randla-graphalignnet
  template:
    metadata:
      labels:
        app: randla-graphalignnet
    spec:
      containers:
      - name: randla-graphalignnet
        image: randla-graphalignnet:latest
        ports:
        - containerPort: 8050
        env:
        - name: NEO4J_URI
          value: "bolt://neo4j-service:7687"
```

### Cloud Deployment

```bash
# AWS ECS
aws ecs create-cluster --cluster-name randla-cluster

# Google Cloud Run
gcloud run deploy randla-graphalignnet --image gcr.io/project/randla-graphalignnet

# Azure Container Instances
az container create --resource-group myResourceGroup --name randla-graphalignnet --image randla-graphalignnet
```

## 📞 Support

### Getting Help

- 📧 **Email**: support@randla-graphalign.com
- 🐛 **Issues**: [GitHub Issues](https://github.com/your-username/RandLA-GraphAlignNet/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/your-username/RandLA-GraphAlignNet/discussions)

### Community

- 🌟 **Star the repository** if you find it useful
- 🍴 **Fork and contribute** to improve the system
- 📢 **Share your use cases** and success stories

---

**Built for standalone deployment with ❤️**

*Enabling multilingual spatial reasoning anywhere, anytime.*