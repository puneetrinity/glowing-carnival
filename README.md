# RunPod Serverless Deployment - Qwen 2.5 7B Career Model

Production-ready serverless deployment with vLLM, streaming, and autoscaling.

## 🚀 Quick Deploy

```bash
# 1. Set credentials
export RUNPOD_API_KEY="your-runpod-api-key"
export DOCKER_USERNAME="your-dockerhub-username"

# 2. Deploy everything
./deploy.sh

# Takes ~20-30 minutes total:
# - Build: 10-15 min (downloads model)
# - Push: 10-15 min (uploads ~15GB)
# - Deploy: 1-2 min
```

## 📁 Files

- `Dockerfile` - Container with vLLM + Qwen 2.5 7B model
- `handler.py` - RunPod serverless handler with delta streaming + V3 validation
- `career_guidance_v3.py` - V3 validation (87% career accuracy, intent blocking)
- `deploy.sh` - Automated deployment script
- `runpod.toml` - RunPod configuration
- `DEPLOYMENT_CHECKLIST.md` - Complete deployment guide
- `STREAMING_API.md` - Delta streaming documentation

## 🎯 Features

✅ **Delta Streaming** - Efficient token-by-token delivery (only new text, ~40% bandwidth savings)
✅ **V3 Validation** - 87% career guidance accuracy, blocks low-quality intents (salary/market)
✅ **Auto-scaling** - Scales to 0 when idle, up to 3 workers under load
✅ **Optimized vLLM** - Low TTFT, high throughput (vLLM 0.6.4.post1)
✅ **Production-ready** - Error handling, timeouts, regeneration on validation failure
✅ **Cost-efficient** - Pay only when running

> 📖 **Streaming Details:** See [STREAMING_API.md](STREAMING_API.md) for delta streaming semantics and client examples

## 📊 Expected Performance

| Metric | Target | Notes |
|--------|--------|-------|
| Cold Start | <60s | First request (model loading) |
| TTFT (warm) | <1s | Time to first token |
| Throughput | >100 t/s | Tokens per second |
| Per-token latency | ~10ms | Streaming interval |
| Cost | ~$0.001/request | When warm |

## 🧪 Test Your Deployment

```bash
# Get endpoint ID from deploy.sh output
ENDPOINT_ID="your-endpoint-id"
ENDPOINT_URL="https://api.runpod.ai/v2/${ENDPOINT_ID}/runsync"

# Test request
curl -X POST "${ENDPOINT_URL}" \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "prompt": "What career path should I take as a software engineer?",
      "sampling_params": {
        "max_tokens": 150,
        "temperature": 0.7,
        "stream": true
      }
    }
  }'
```

## 📈 Monitor & Manage

```bash
# View dashboard
open https://runpod.io/console/serverless/${ENDPOINT_ID}

# View logs
runpodctl get logs ${ENDPOINT_ID} --tail 100

# Update endpoint
runpodctl update endpoint ${ENDPOINT_ID} --max-workers 5
```

## 🔧 Configuration

Edit `Dockerfile` to tune performance:

```dockerfile
# Low latency (interactive chat)
ENV MAX_NUM_SEQS=4
ENV MAX_NUM_BATCHED_TOKENS=1024

# High throughput (batch API)
ENV MAX_NUM_SEQS=16
ENV MAX_NUM_BATCHED_TOKENS=4096
```

Then rebuild and redeploy.

## 💰 Cost Optimization

**Current setup:**
- GPU: RTX A5000 (24GB) @ ~$0.89/hr
- Min workers: 0 (scales to zero when idle)
- Idle timeout: 5 minutes

**For production:**
```bash
# Keep 1 worker warm (eliminates cold starts)
runpodctl update endpoint ${ENDPOINT_ID} --min-workers 1

# Or use spot instances (50-70% cheaper)
runpodctl update endpoint ${ENDPOINT_ID} --gpu-type "RTX A5000 Spot"
```

## 🐛 Troubleshooting

### Build fails
```bash
# Check Docker logs
docker build --progress=plain -t test .
```

### Deployment fails
```bash
# Verify runpodctl is configured
runpodctl config

# Check RunPod status
curl https://status.runpod.io
```

### Endpoint errors
```bash
# View logs
runpodctl get logs ${ENDPOINT_ID} --follow

# Check recent errors
runpodctl get logs ${ENDPOINT_ID} | grep -i error
```

## 📚 Documentation

- Full checklist: See `DEPLOYMENT_CHECKLIST.md`
- RunPod Docs: https://docs.runpod.io/serverless
- vLLM Docs: https://docs.vllm.ai
- Model: https://huggingface.co/Puneetrinity/qwen2-7b-career

## 🔄 Updates

To update the model or code:

```bash
# 1. Make changes to Dockerfile or handler.py
# 2. Rebuild
docker build -t ${DOCKER_USERNAME}/qwen2-7b-career-vllm:v2 .

# 3. Push
docker push ${DOCKER_USERNAME}/qwen2-7b-career-vllm:v2

# 4. Update endpoint
runpodctl update endpoint ${ENDPOINT_ID} \
    --image ${DOCKER_USERNAME}/qwen2-7b-career-vllm:v2
```

## ✅ Quick Checklist

Before deploying:
- [ ] Set `RUNPOD_API_KEY`
- [ ] Set `DOCKER_USERNAME`
- [ ] Login to Docker Hub (`docker login`)
- [ ] Review `DEPLOYMENT_CHECKLIST.md`

After deploying:
- [ ] Test cold start (<60s)
- [ ] Test warm request (<2s)
- [ ] Test streaming works
- [ ] Run benchmark
- [ ] Monitor for 24h
- [ ] Set up alerts

---

**Next:** See `DEPLOYMENT_CHECKLIST.md` for complete step-by-step guide.
