# Capacity & DAU Planning

This section helps estimate how many daily active users (DAU) and peak request rates your current config can support.

## Assumptions (tweak as needed)
- Observed throughput: **~35 requests/min per worker**
- Config: `workersMin=0`, `workersMax=5` → **max throughput ~175 req/min**
- Latency SLO: **P95 ≤ 8000 ms**
- Typical chat usage: **r = 2–8 requests/user/day** (default **r=3**)
- Peak-to-average multiplier (**k**): commonly **8–12×** (default **k=10**)

## Quick Math
- **Daily capacity (flat-out 24h):** `175 req/min × 60 × 24 = 252,000 req/day`
- **DAU estimate:** `DAU ≈ 252,000 / r`
  - Example: `r = 3` → **~84,000 DAU** (theoretical, assumes perfectly flat traffic)
- **Peak planning:** real traffic peaks. With `k = 10`,
  - Average budget per minute: `175 req/min`
  - Peak budget per minute: **~175 req/min** (your current cap)
  - If `avg_req_per_min × k > 175`, raise `workersMax` or `workersMin`.

### Handy Table (defaults r=3, k=10)
| Target DAU | Avg req/day | Avg req/min | 10× Peak req/min | Fits current cap (175)? |
|------------|-------------|-------------|-------------------|--------------------------|
| 5,000      | 15,000      | 10.4        | 104               | ✅ Yes                   |
| 10,000     | 30,000      | 20.8        | 208               | ⚠️ Almost (raise max)    |
| 25,000     | 75,000      | 52.1        | 521               | ❌ No                    |

## Tuning Knobs
- **Lower P95 / smoother tails:** set `workersMin = 1–2` to avoid cold starts.
- **Higher peak:** increase `workersMax` (e.g., 6–8) and consider a client cap ≤ 3 concurrent.
- **Cost guard:** alert if utilization > 80% for 5 minutes.

## One-Liner Calculator (Python)
```python
def plan(dau, r=3, k=10, per_worker_rpm=35, workers_max=5):
    avg_rpm = dau * r / (24*60)
    peak_rpm = avg_rpm * k
    cap_rpm  = per_worker_rpm * workers_max
    return {"avg_rpm": avg_rpm, "peak_rpm": peak_rpm, "cap_rpm": cap_rpm, "fits": peak_rpm <= cap_rpm}
```
