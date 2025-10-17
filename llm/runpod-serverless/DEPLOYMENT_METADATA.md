# Deployment Metadata – v1.0-production

## Service
- **Name:** chat-service
- **Environment:** production
- **Scaling:** workersMin=0, workersMax=5, scalerValue=1, idleTimeout=180s

## Model/runtime
- **Engine:** vLLM
- **max_num_batched_tokens:** fixed per vLLM constraint
- **Token policy:** prompt ≤ 1024 chars; max_tokens ∈ {64, 256}

## Performance (from soak test, 2000 req @ c=2)
- **Success:** 100% (2000/2000)
- **P50:** 3,449 ms
- **P95:** 6,929 ms
- **P99:** 7,355 ms
- **Avg:** 4,177 ms
- **Notes:** Queueing drives P95; execution ~2.7s. Cold starts possible at workersMin=0.

## Capacity assumptions
- ~35 req/min/worker observed
- workersMax=5 → ~175 req/min peak
- Daily theoretical: ~252k req/day (flat)

## SLOs
- **Availability:** ≥99%
- **Latency SLO:** P95 ≤ 8,000 ms

## Runbooks (abridged)
- **High P95, normal exec_ms** → raise workersMin (1–2) and/or workersMax
- **High exec_ms** → inspect token counts, GPU utilization, model config
- **Queue growth** → increase workersMax, cap client concurrency ≤3
- **Errors spike** → pause scale-up, drain, rollback to last tag

## Tag & release
```bash
git tag -a v1.0-production -m "Prod: 2k soak pass (P50 3.45s, P95 6.93s, 100% success); config workersMin=0/Max=5; rpm/worker ~35"
git push --tags
```
