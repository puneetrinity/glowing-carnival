#!/bin/bash
# Direct RunPod Serverless Deployment using runpodctl
# Uses pre-built vLLM image with model pulled from HuggingFace at runtime

set -e

echo "🚀 RunPod Serverless Deployment (runpodctl method)"
echo "=================================================="

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Configuration
ENDPOINT_NAME="${ENDPOINT_NAME:-qwen-career-serverless}"
MODEL_NAME="${MODEL_NAME:-Puneetrinity/qwen2-7b-career}"
GPU_TYPE="${GPU_TYPE:-NVIDIA RTX A5000}"
MIN_WORKERS="${MIN_WORKERS:-0}"
MAX_WORKERS="${MAX_WORKERS:-3}"
IDLE_TIMEOUT="${IDLE_TIMEOUT:-5}"
EXECUTION_TIMEOUT="${EXECUTION_TIMEOUT:-180}"

# Check prerequisites
echo -e "\n${YELLOW}Step 1: Checking prerequisites...${NC}"

if ! command -v runpodctl &> /dev/null; then
    echo -e "${YELLOW}Installing runpodctl...${NC}"
    wget -qO /tmp/runpodctl https://github.com/runpod/runpodctl/releases/latest/download/runpodctl-linux-amd64
    chmod +x /tmp/runpodctl
    sudo mv /tmp/runpodctl /usr/local/bin/runpodctl
    echo -e "${GREEN}✓ runpodctl installed${NC}"
else
    echo -e "${GREEN}✓ runpodctl found${NC}"
fi

# Authenticate
echo -e "\n${YELLOW}Step 2: Authenticating with RunPod...${NC}"

if [ -z "$RUNPOD_API_KEY" ]; then
    echo -e "${YELLOW}RUNPOD_API_KEY not set${NC}"
    echo "Get your API key from: https://runpod.io/console/user/settings"
    read -p "Enter RunPod API Key: " RUNPOD_API_KEY
fi

runpodctl config --apiKey "$RUNPOD_API_KEY"
echo -e "${GREEN}✓ Authenticated${NC}"

# Create project directory on RunPod Network Volume (if needed)
echo -e "\n${YELLOW}Step 3: Preparing handler files...${NC}"

# Create a temporary directory for handler bundle
TEMP_DIR=$(mktemp -d)
echo "Using temp directory: $TEMP_DIR"

# Copy handler and V3 validation
cp handler.py "$TEMP_DIR/"
cp career_guidance_v3.py "$TEMP_DIR/"

# Create requirements.txt
cat > "$TEMP_DIR/requirements.txt" << 'EOF'
vllm==0.6.4.post1
huggingface-hub
runpod
ray==2.9.0
EOF

# Create start.py wrapper
cat > "$TEMP_DIR/start.py" << 'STARTPY'
"""
RunPod Serverless Startup Script
Downloads model and starts vLLM server, then runs handler
"""
import os
import sys
from huggingface_hub import snapshot_download

MODEL_NAME = os.getenv("MODEL_NAME", "Puneetrinity/qwen2-7b-career")
MODEL_PATH = os.getenv("MODEL_PATH", "/runpod-volume/models/qwen2-7b-career")

print(f"Downloading model: {MODEL_NAME}")
snapshot_download(
    repo_id=MODEL_NAME,
    local_dir=MODEL_PATH,
    ignore_patterns=["*.gguf", "*.bin"]
)
print(f"Model downloaded to: {MODEL_PATH}")

# Import and run handler
sys.path.insert(0, os.path.dirname(__file__))
from handler import handler
import runpod

print("Starting RunPod handler...")
runpod.serverless.start({"handler": handler})
STARTPY

echo -e "${GREEN}✓ Handler files prepared${NC}"

# Upload to RunPod (using project create)
echo -e "\n${YELLOW}Step 4: Creating RunPod project...${NC}"

cd "$TEMP_DIR"

# Initialize runpod project
cat > runpod.toml << TOML
[project]
name = "$ENDPOINT_NAME"
base_image = "runpod/pytorch:2.1.1-py3.10-cuda12.1.1-devel-ubuntu22.04"

[build]
python_version = "3.10"

[[build.env]]
MODEL_NAME = "$MODEL_NAME"
MODEL_PATH = "/runpod-volume/models/qwen2-7b-career"
MAX_MODEL_LEN = "4096"
GPU_MEMORY_UTILIZATION = "0.90"
MAX_NUM_SEQS = "8"
TOML

# Create Dockerfile for runpod project
cat > Dockerfile << 'DOCKERFILE'
FROM runpod/pytorch:2.1.1-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy handler files
COPY start.py handler.py career_guidance_v3.py ./

# Create model directory
RUN mkdir -p /runpod-volume/models

CMD ["python", "-u", "start.py"]
DOCKERFILE

echo -e "${GREEN}✓ Project configured${NC}"

# Build and deploy using runpodctl
echo -e "\n${YELLOW}Step 5: Building and deploying...${NC}"

# Build the project
echo "Building Docker image..."
runpodctl project build

# Deploy the project
echo "Deploying to RunPod..."
runpodctl project deploy \
    --name "$ENDPOINT_NAME" \
    --gpu "$GPU_TYPE" \
    --min-workers "$MIN_WORKERS" \
    --max-workers "$MAX_WORKERS" \
    --idle-timeout "$IDLE_TIMEOUT" \
    --execution-timeout "$EXECUTION_TIMEOUT"

# Get endpoint ID
ENDPOINT_ID=$(runpodctl get endpoint | grep "$ENDPOINT_NAME" | awk '{print $1}')

if [ -z "$ENDPOINT_ID" ]; then
    echo -e "${RED}✗ Could not find endpoint${NC}"
    echo "Check RunPod console: https://runpod.io/console/serverless"
    exit 1
fi

echo -e "${GREEN}✓ Deployment complete!${NC}"
echo -e "${GREEN}Endpoint ID: ${ENDPOINT_ID}${NC}"

# Cleanup
rm -rf "$TEMP_DIR"

# Test endpoint
ENDPOINT_URL="https://api.runpod.ai/v2/${ENDPOINT_ID}/runsync"

echo -e "\n${YELLOW}Step 6: Testing endpoint...${NC}"
echo "Waiting 30 seconds for cold start..."
sleep 30

echo "Sending test request..."
RESPONSE=$(curl -s -X POST "${ENDPOINT_URL}" \
    -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
    -H "Content-Type: application/json" \
    -d '{
        "input": {
            "prompt": "What skills should I learn for backend development?",
            "sampling_params": {
                "max_tokens": 100,
                "temperature": 0.7
            },
            "enable_validation": true
        }
    }')

echo "Response:"
echo "$RESPONSE" | python3 -m json.tool

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
echo "    -d '{\"input\": {\"prompt\": \"Your question\", \"enable_validation\": true}}'"
echo ""
echo "Dashboard: https://runpod.io/console/serverless/${ENDPOINT_ID}"
echo ""
echo -e "${YELLOW}Next Steps:${NC}"
echo "  1. Test with career questions (87% validation)"
echo "  2. Verify salary queries are blocked"
echo "  3. Run benchmark: cd /home/ews/llm && python benchmark_llm.py"
echo "  4. Monitor: runpodctl get logs ${ENDPOINT_ID}"
