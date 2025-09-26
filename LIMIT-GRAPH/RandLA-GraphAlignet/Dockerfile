# RandLA-GraphAlignNet Docker Image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==1.9.0+cpu torchvision==0.10.0+cpu -f https://download.pytorch.org/whl/torch_stable.html && \
    pip install --no-cache-dir torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.9.0+cpu.html && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create output directory
RUN mkdir -p output

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose ports for potential web interfaces
EXPOSE 8050 7474 7687

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from dependencies import check_dependencies; deps = check_dependencies(); exit(0 if all(v != 'Not available' for v in deps.values()) else 1)"

# Default command
CMD ["python", "main.py", "--mode", "demo", "--languages", "en", "es", "ar", "id"]

# Labels
LABEL maintainer="AI Research Agent Team"
LABEL version="1.0.0"
LABEL description="Multilingual Spatial Reasoning for 3D Point Clouds with Neo4j Integration"