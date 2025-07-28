#!/bin/bash
# Docker build script for NIDS

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
ENVIRONMENT=${ENVIRONMENT:-development}
BUILD_ARGS=""
PUSH_IMAGES=${PUSH_IMAGES:-false}
REGISTRY=${REGISTRY:-""}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -p|--push)
            PUSH_IMAGES=true
            shift
            ;;
        -r|--registry)
            REGISTRY="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  -e, --environment    Environment (development|production) [default: development]"
            echo "  -p, --push          Push images to registry"
            echo "  -r, --registry      Docker registry URL"
            echo "  -h, --help          Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo -e "${GREEN}Building NIDS Docker images for ${ENVIRONMENT} environment${NC}"

# Set build target based on environment
if [ "$ENVIRONMENT" = "production" ]; then
    BUILD_TARGET="production"
else
    BUILD_TARGET="development"
fi

# Build base image
echo -e "${YELLOW}Building base image...${NC}"
docker build --target base -t nids:base .

# Build specific service images
echo -e "${YELLOW}Building API service image...${NC}"
docker build --target api -t nids:api .

echo -e "${YELLOW}Building dashboard service image...${NC}"
docker build --target dashboard -t nids:dashboard .

echo -e "${YELLOW}Building training service image...${NC}"
docker build --target training -t nids:training .

echo -e "${YELLOW}Building capture service image...${NC}"
docker build --target capture -t nids:capture .

# Tag images with registry if provided
if [ -n "$REGISTRY" ]; then
    echo -e "${YELLOW}Tagging images with registry ${REGISTRY}...${NC}"
    docker tag nids:api ${REGISTRY}/nids:api
    docker tag nids:dashboard ${REGISTRY}/nids:dashboard
    docker tag nids:training ${REGISTRY}/nids:training
    docker tag nids:capture ${REGISTRY}/nids:capture
fi

# Push images if requested
if [ "$PUSH_IMAGES" = true ] && [ -n "$REGISTRY" ]; then
    echo -e "${YELLOW}Pushing images to registry...${NC}"
    docker push ${REGISTRY}/nids:api
    docker push ${REGISTRY}/nids:dashboard
    docker push ${REGISTRY}/nids:training
    docker push ${REGISTRY}/nids:capture
fi

echo -e "${GREEN}Docker build completed successfully!${NC}"

# Show image sizes
echo -e "${YELLOW}Image sizes:${NC}"
docker images | grep nids