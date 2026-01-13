#!/bin/bash
# Build and push Docker images to a registry
#
# Usage:
#   ./build-push-images.sh <registry>
#
# Example:
#   ./build-push-images.sh ghcr.io/purestorage-openconnect/fraud-detection-demo
#   ./build-push-images.sh your-registry.example.com/fraud-demo

set -e

REGISTRY="${1:-ghcr.io/purestorage-openconnect/fraud-detection-demo}"
TAG="${2:-latest}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
V2_DIR="$(dirname "$SCRIPT_DIR")"

echo "Building images for registry: $REGISTRY"
echo "Tag: $TAG"
echo ""

# Build dashboard
echo "=== Building dashboard ==="
docker build -t "$REGISTRY/dashboard:$TAG" "$V2_DIR/dashboard"

# Build worker-cpu
echo "=== Building worker-cpu ==="
docker build -t "$REGISTRY/worker-cpu:$TAG" "$V2_DIR/worker-cpu"

# Build worker-gpu (if it exists and is different from worker-cpu)
if [ -d "$V2_DIR/worker-gpu" ] && [ -f "$V2_DIR/worker-gpu/Dockerfile" ]; then
    echo "=== Building worker-gpu ==="
    docker build -t "$REGISTRY/worker-gpu:$TAG" "$V2_DIR/worker-gpu"
else
    echo "=== worker-gpu directory not found, using worker-cpu image ==="
    docker tag "$REGISTRY/worker-cpu:$TAG" "$REGISTRY/worker-gpu:$TAG"
fi

echo ""
echo "=== Built images ==="
docker images | grep "$REGISTRY"

echo ""
echo "=== Pushing images ==="
docker push "$REGISTRY/dashboard:$TAG"
docker push "$REGISTRY/worker-cpu:$TAG"
docker push "$REGISTRY/worker-gpu:$TAG"

echo ""
echo "Done! Images pushed to $REGISTRY"
