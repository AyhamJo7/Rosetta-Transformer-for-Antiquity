# Docker Files

This directory contains Docker configurations for the Rosetta Transformer project.

## Files

- **Dockerfile.cpu**: Docker image for CPU-based deployment
- **Dockerfile.cuda**: Docker image for GPU-accelerated deployment with CUDA support

## Usage

### Build CPU Image

```bash
docker build -f docker/Dockerfile.cpu -t rosetta-transformer:cpu .
```

### Build CUDA Image

```bash
docker build -f docker/Dockerfile.cuda -t rosetta-transformer:cuda .
```

### Run Container

```bash
# CPU version
docker run -it rosetta-transformer:cpu

# GPU version (requires nvidia-docker)
docker run --gpus all -it rosetta-transformer:cuda
```

## Requirements

- Docker 20.10+
- For GPU support: NVIDIA Docker runtime (nvidia-docker2)
