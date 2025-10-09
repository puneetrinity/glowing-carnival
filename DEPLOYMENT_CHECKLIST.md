# RunPod Serverless Deployment Checklist
## Qwen 2.5 7B Career Model with vLLM

---

## ✅ Pre-Deployment Checklist

### Prerequisites
- [ ] Docker installed (`docker --version`)
- [ ] RunPod API key from https://runpod.io/console/user/settings
- [ ] Docker Hub account (or private registry)
- [ ] Sufficient credits in RunPod account (~$10 for testing)
- [ ] Model accessible: https://huggingface.co/Puneetrinity/qwen2-7b-career

### Environment Setup
- [ ] Set `RUNPOD_API_KEY` environment variable
- [ ] Set `DOCKER_USERNAME` environment variable (your Docker Hub username)
- [ ] Login to Docker Hub: `docker login`
- [ ] Verify HuggingFace model is public or you have access token

---

## 📦 Deployment Steps

### 1. Review Configuration Files

- [ ] **Dockerfile** - Model download and vLLM installation
  ```bash
  # Verify model path
  grep "ARG HF_MODEL" Dockerfile
  # Should show: Puneetrinity/qwen2-7b-career
  ```

- [ ] **handler.py** - RunPod serverless handler with streaming
  ```bash
  # Check handler is present
  ls -lh handler.py
  ```

- [ ] **deploy.sh** - Automated deployment script
  ```bash
  chmod +x deploy.sh
  ```

### 2. Build Docker Image (Local Test)

```bash
# Test build locally first
docker build -t qwen2-7b-career-vllm:test \
    --build-arg HF_MODEL=Puneetrinity/qwen2-7b-career \
    .

# This takes ~10-15 minutes (downloads ~15GB model)
# Monitor progress
```

**Expected output:**
- ✓ Base image pulled
- ✓ vLLM installed
- ✓ Model downloaded from HuggingFace
- ✓ Handler code copied
- ✓ Image built successfully

**Troubleshooting:**
- If model download fails → Check HuggingFace status
- If OOM during build → Increase Docker memory limit to 16GB+
- If vLLM install fails → Check CUDA compatibility

### 3. Test Locally (Optional but Recommended)

```bash
# Run container locally
docker run --gpus all -p 8000:8000 \
    -e RUNPOD_API_KEY=test \
    qwen2-7b-career-vllm:test

# In another terminal, test handler
python -c "
import requests, json
job = {
    'input': {
        'prompt': 'What is data science?',
        'sampling_params': {'max_tokens': 50, 'stream': False}
    }
}
result = requests.post('http://localhost:8000', json=job)
print(json.dumps(result.json(), indent=2))
"
```

**Expected:**
- ✓ vLLM engine initializes
- ✓ Model loads into GPU memory
- ✓ Handler responds with generated text
- ✓ Response time <5s after warm-up

### 4. Push to Docker Registry

```bash
# Set your Docker Hub username
export DOCKER_USERNAME="your-dockerhub-username"

# Tag image
docker tag qwen2-7b-career-vllm:test \
    ${DOCKER_USERNAME}/qwen2-7b-career-vllm:latest

# Push (this takes ~10-15 min, uploading ~15GB)
docker push ${DOCKER_USERNAME}/qwen2-7b-career-vllm:latest
```

**Monitor:**
- Upload progress in terminal
- Verify on Docker Hub: https://hub.docker.com/r/${DOCKER_USERNAME}/qwen2-7b-career-vllm

### 5. Deploy to RunPod Serverless

**Option A: Automated (Recommended)**

```bash
# Set API key
export RUNPOD_API_KEY="your-runpod-api-key"
export DOCKER_USERNAME="your-dockerhub-username"

# Run deployment script
./deploy.sh
```

**Option B: Manual**

```bash
# Install runpodctl
wget https://github.com/runpod/runpodctl/releases/latest/download/runpodctl-linux-amd64
chmod +x runpodctl-linux-amd64
sudo mv runpodctl-linux-amd64 /usr/local/bin/runpodctl

# Configure
runpodctl config --apiKey $RUNPOD_API_KEY

# Create endpoint
runpodctl create endpoint \
    --name "qwen-career-serverless" \
    --image "${DOCKER_USERNAME}/qwen2-7b-career-vllm:latest" \
    --gpu "NVIDIA RTX A5000" \
    --idle-timeout 5 \
    --execution-timeout 180 \
    --min-workers 0 \
    --max-workers 3 \
    --gpu-count 1
```

**Expected:**
- ✓ Endpoint created
- ✓ Endpoint ID returned (e.g., `abc123def456`)
- ✓ Workers scale to 0 (idle)

### 6. Test Serverless Endpoint

```bash
# Get endpoint ID from deploy.sh output or RunPod console
ENDPOINT_ID="your-endpoint-id"
ENDPOINT_URL="https://api.runpod.ai/v2/${ENDPOINT_ID}/runsync"

# Test request
curl -X POST "${ENDPOINT_URL}" \
    -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
    -H "Content-Type: application/json" \
    -d '{
        "input": {
            "prompt": "What career advice would you give to a software engineer transitioning to machine learning?",
            "sampling_params": {
                "max_tokens": 150,
                "temperature": 0.7,
                "stream": true
            }
        }
    }'
```

**Expected Response:**
```json
{
  "delayTime": 35000,  // Cold start ~30-60s first time
  "executionTime": 2500,
  "id": "...",
  "output": {
    "choices": [{
      "text": "For a software engineer...",
      "tokens": [...]
    }],
    "usage": {"input": 15, "output": 150}
  },
  "status": "COMPLETED"
}
```

---

## 🧪 Post-Deployment Testing

### Test Cold Start
- [ ] First request completes in <60s
- [ ] Response is coherent and relevant
- [ ] No errors in response

### Test Warm Performance
```bash
# Send 3 requests quickly
for i in {1..3}; do
    curl -X POST "${ENDPOINT_URL}" \
        -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
        -H "Content-Type: application/json" \
        -d '{"input": {"prompt": "Test '$i'", "sampling_params": {"max_tokens": 50}}}' &
done
wait
```

- [ ] Subsequent requests complete in <5s
- [ ] All requests succeed
- [ ] Output quality consistent

### Test Streaming
- [ ] `stream: true` returns token-by-token
- [ ] Tokens array populated
- [ ] Final response includes full text

### Test Concurrency
```bash
# Run benchmark
cd ..
python benchmark_llm.py
# Update RUNPOD_ENDPOINT with your new endpoint
```

**Target Metrics:**
- TTFT P50: <1s (warm)
- Throughput: >100 tokens/sec
- Success rate: >99%

---

## 📊 Monitoring & Optimization

### Check RunPod Dashboard
- [ ] View endpoint: https://runpod.io/console/serverless
- [ ] Monitor requests/sec
- [ ] Check error rate
- [ ] View GPU utilization
- [ ] Monitor costs

### View Logs
```bash
# Get recent logs
runpodctl get logs ${ENDPOINT_ID} --tail 100

# Stream logs
runpodctl get logs ${ENDPOINT_ID} --follow
```

### Optimize Performance

**If TTFT too high (>2s warm):**
- Reduce `MAX_NUM_BATCHED_TOKENS` in Dockerfile
- Reduce `MAX_NUM_SEQS` to 4
- Enable chunked prefill (already enabled)

**If throughput too low (<50 t/s):**
- Increase `GPU_MEMORY_UTILIZATION` to 0.95
- Increase `MAX_NUM_SEQS` to 16
- Use quantized model (AWQ/GPTQ)

**If OOM errors:**
- Reduce `MAX_MODEL_LEN` to 2048
- Reduce `GPU_MEMORY_UTILIZATION` to 0.85
- Quantize model
- Upgrade to A100

---

## 💰 Cost Management

### Current Configuration
- GPU: NVIDIA RTX A5000 (24GB)
- Cost: ~$0.89/hr when running
- Min workers: 0 (scales to zero)
- Max workers: 3

### Expected Costs
**Development/Testing:**
- Cold starts: ~$0.01 per cold start (1 min)
- Warm requests: ~$0.001 per request (2s)
- Idle: $0/hr (scales to 0)

**Production (1000 req/day):**
- Daily: ~$5-10 depending on usage pattern
- Monthly: ~$150-300

### Cost Optimization
- [ ] Set aggressive `idle-timeout` (currently 5 min)
- [ ] Use `min-workers: 0` (already set)
- [ ] Set `max-workers` based on peak load (currently 3)
- [ ] Monitor utilization and adjust

---

## 🔒 Security Checklist

- [ ] API key stored securely (not in code)
- [ ] Docker image doesn't contain secrets
- [ ] Model license allows commercial use (check Puneetrinity/qwen2-7b-career)
- [ ] Rate limiting configured (RunPod default)
- [ ] HTTPS only (RunPod default)

---

## 🚨 Troubleshooting

### Issue: Build fails with "Model not found"
**Solution:**
- Verify model exists: https://huggingface.co/Puneetrinity/qwen2-7b-career
- Check if model is private (requires HF token)
- If private, add `--build-arg HF_TOKEN=xxx` to docker build

### Issue: Cold start timeout
**Solution:**
- Increase `execution-timeout` to 300s
- Optimize model loading (use smaller model or quantized)
- Keep 1 worker warm (`min-workers: 1`)

### Issue: OOM during inference
**Solution:**
- Reduce `max_model_len` to 2048
- Reduce `gpu_memory_utilization` to 0.85
- Quantize model
- Use larger GPU (A100 40GB)

### Issue: Requests fail intermittently
**Solution:**
- Check RunPod status page
- View logs: `runpodctl get logs ${ENDPOINT_ID}`
- Verify network/firewall
- Add retry logic client-side

---

## ✅ Final Verification

Before going to production:

- [ ] Automated deployment runs end-to-end
- [ ] Endpoint responds within SLA (<2s warm, <60s cold)
- [ ] Streaming works correctly
- [ ] Concurrent requests handled (test with 8+ simultaneous)
- [ ] Error handling works (test invalid inputs)
- [ ] Logs show no errors
- [ ] Cost per 1k tokens is acceptable
- [ ] Monitoring/alerts configured
- [ ] Documentation updated with endpoint details

---

## 📚 Next Steps

1. **Run Full Benchmark:**
   ```bash
   cd ..
   python benchmark_llm.py
   # Compare RunPod vs Groq results
   ```

2. **Integrate into Application:**
   - Use endpoint URL in your app
   - Add retry logic + exponential backoff
   - Implement client-side timeout (120s)
   - Cache responses where appropriate

3. **Production Hardening:**
   - Set up monitoring (Grafana/Prometheus)
   - Configure autoscaling rules
   - Add request queuing
   - Implement circuit breaker pattern

4. **Optimize Further:**
   - Quantize model (AWQ) for 2-3x speedup
   - Profile GPU usage
   - A/B test different parameters
   - Consider multi-GPU if needed

---

## 📞 Support Resources

- RunPod Docs: https://docs.runpod.io/serverless/overview
- vLLM Docs: https://docs.vllm.ai/
- Model: https://huggingface.co/Puneetrinity/qwen2-7b-career
- RunPod Discord: https://discord.gg/runpod
- Issues: File issue in your repo

---

**Deployment Date:** _________________
**Deployed By:** _________________
**Endpoint ID:** _________________
**Status:** ⬜ Dev | ⬜ Staging | ⬜ Production
