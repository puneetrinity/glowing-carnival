#!/bin/bash
# RunPod Serverless Deployment Script
# Deploy Qwen 2.5 7B Career Model with vLLM

set -e

echo "🚀 RunPod Serverless Deployment Script"
echo "========================================"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Configuration
IMAGE_NAME="qwen2-7b-career-vllm"
REGISTRY="registry.hub.docker.com"  # Change to your registry
DOCKER_USERNAME="${DOCKER_USERNAME:-your-username}"  # Set via env or replace
ENDPOINT_NAME="qwen-career-serverless"

# Step 1: Check prerequisites
echo -e "\n${YELLOW}Step 1: Checking prerequisites...${NC}"

if ! command -v docker &> /dev/null; then
    echo -e "${RED}✗ Docker not found${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Docker found${NC}"

if ! command -v runpodctl &> /dev/null; then
    echo -e "${YELLOW}runpodctl not found. Installing...${NC}"
    wget -qO /tmp/runpodctl https://github.com/runpod/runpodctl/releases/latest/download/runpodctl-linux-amd64
    chmod +x /tmp/runpodctl
    sudo mv /tmp/runpodctl /usr/local/bin/runpodctl
    echo -e "${GREEN}✓ runpodctl installed${NC}"
else
    echo -e "${GREEN}✓ runpodctl found${NC}"
fi

# Step 2: Login to RunPod
echo -e "\n${YELLOW}Step 2: RunPod Authentication${NC}"
if [ -z "$RUNPOD_API_KEY" ]; then
    echo -e "${YELLOW}RUNPOD_API_KEY not set${NC}"
    echo "Get your API key from: https://runpod.io/console/user/settings"
    read -p "Enter RunPod API Key: " RUNPOD_API_KEY
fi

runpodctl config --apiKey "$RUNPOD_API_KEY"
echo -e "${GREEN}✓ Authenticated with RunPod${NC}"

# Step 3: Build Docker image
echo -e "\n${YELLOW}Step 3: Building Docker image...${NC}"
echo "This will take 5-10 minutes (downloading model from HuggingFace)"

docker build -t ${IMAGE_NAME}:latest \
    --build-arg HF_MODEL=Puneetrinity/qwen2-7b-career \
    .

echo -e "${GREEN}✓ Docker image built: ${IMAGE_NAME}:latest${NC}"

# Step 4: Tag and push to registry
echo -e "\n${YELLOW}Step 4: Pushing to Docker registry...${NC}"

# Login to Docker Hub (or your registry)
echo "Login to Docker Hub:"
docker login

# Tag image
docker tag ${IMAGE_NAME}:latest ${DOCKER_USERNAME}/${IMAGE_NAME}:latest
docker tag ${IMAGE_NAME}:latest ${DOCKER_USERNAME}/${IMAGE_NAME}:$(date +%Y%m%d-%H%M%S)

# Push
docker push ${DOCKER_USERNAME}/${IMAGE_NAME}:latest
docker push ${DOCKER_USERNAME}/${IMAGE_NAME}:$(date +%Y%m%d-%H%M%S)

echo -e "${GREEN}✓ Image pushed to ${DOCKER_USERNAME}/${IMAGE_NAME}:latest${NC}"

# Step 5: Create serverless endpoint
echo -e "\n${YELLOW}Step 5: Creating RunPod Serverless Endpoint...${NC}"

FULL_IMAGE="${DOCKER_USERNAME}/${IMAGE_NAME}:latest"

echo "Creating endpoint: ${ENDPOINT_NAME}"
echo "Image: ${FULL_IMAGE}"

# Create endpoint using runpodctl
runpodctl create endpoint \
    --name "${ENDPOINT_NAME}" \
    --image "${FULL_IMAGE}" \
    --gpu "NVIDIA RTX A5000" \
    --idle-timeout 5 \
    --execution-timeout 180 \
    --min-workers 0 \
    --max-workers 3 \
    --gpu-count 1

echo -e "${GREEN}✓ Endpoint created${NC}"

# Step 6: Get endpoint details
echo -e "\n${YELLOW}Step 6: Getting endpoint details...${NC}"

ENDPOINT_ID=$(runpodctl get endpoint | grep "${ENDPOINT_NAME}" | awk '{print $1}')

if [ -z "$ENDPOINT_ID" ]; then
    echo -e "${RED}✗ Could not find endpoint${NC}"
    echo "Please check RunPod console: https://runpod.io/console/serverless"
    exit 1
fi

echo -e "${GREEN}✓ Endpoint ID: ${ENDPOINT_ID}${NC}"

# Step 7: Test endpoint
echo -e "\n${YELLOW}Step 7: Testing endpoint...${NC}"
echo "Waiting 30 seconds for cold start..."
sleep 30

# Get API key for requests
ENDPOINT_URL="https://api.runpod.ai/v2/${ENDPOINT_ID}/runsync"

echo "Testing with sample request..."
curl -X POST "${ENDPOINT_URL}" \
    -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
    -H "Content-Type: application/json" \
    -d '{
        "input": {
            "prompt": "What career advice would you give to a software engineer?",
            "sampling_params": {
                "max_tokens": 100,
                "temperature": 0.7,
                "stream": true
            }
        }
    }'

echo -e "\n${GREEN}✓ Test complete${NC}"

# Final summary
echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}✓ Deployment Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Endpoint Details:"
echo "  Name: ${ENDPOINT_NAME}"
echo "  ID: ${ENDPOINT_ID}"
echo "  URL: ${ENDPOINT_URL}"
echo ""
echo "Usage:"
echo "  curl -X POST '${ENDPOINT_URL}' \\"
echo "    -H 'Authorization: Bearer \$RUNPOD_API_KEY' \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"input\": {\"prompt\": \"Your prompt\", \"sampling_params\": {\"max_tokens\": 150}}}'"
echo ""
echo "Dashboard: https://runpod.io/console/serverless/${ENDPOINT_ID}"
echo ""
echo -e "${YELLOW}Important Notes:${NC}"
echo "  - First request (cold start) will take ~30-60s"
echo "  - Subsequent requests: <2s TTFT"
echo "  - Endpoint scales to 0 when idle (saves money)"
echo "  - Min workers: 0, Max workers: 3"
echo "  - GPU: NVIDIA RTX A5000 (24GB)"
echo ""
echo "Next steps:"
echo "  1. Run benchmark: python ../benchmark_llm.py"
echo "  2. Monitor: https://runpod.io/console/serverless/${ENDPOINT_ID}"
echo "  3. View logs: runpodctl get logs ${ENDPOINT_ID}"
