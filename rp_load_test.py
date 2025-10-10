#!/usr/bin/env python3
import os, time, math, argparse, asyncio, statistics, csv
import httpx
from datetime import datetime, timezone

API_KEY = os.getenv("RUNPOD_API_KEY")
ENDPOINT_ID = os.getenv("RUNPOD_ENDPOINT_ID")
BASE = f"https://api.runpod.ai/v2/{ENDPOINT_ID}"

def pct(arr, p):
    if not arr: return None
    arr = sorted(arr)
    k = max(0, min(len(arr)-1, int(math.ceil(p/100*len(arr))-1)))
    return arr[k]

def build_payload(prompt: str):
    # Adjust to your handler's schema
    return {"input": {"prompt": prompt, "sampling_params": {"max_tokens": 150, "temperature": 0.2}, "enable_validation": True}}

async def call_runsync(client: httpx.AsyncClient, payload, timeout_s: int):
    t0 = time.time()
    r = await client.post(f"{BASE}/runsync",
                          headers={"Authorization": f"Bearer {API_KEY}","Content-Type":"application/json"},
                          json=payload, timeout=timeout_s)
    t1 = time.time()
    ok = (r.status_code == 200)
    out = None
    try:
        out = r.json()
    except Exception:
        pass
    tokens = -1
    delay_ms = None
    exec_ms = None
    # Optional: extract tokens from your handler response if available
    try:
        tokens = out.get("output", {}).get("usage", {}).get("output", -1)
        # RunPod runsync metadata
        delay_ms = out.get("delayTime")
        exec_ms = out.get("executionTime")
    except Exception:
        pass
    return {
        "ok": ok,
        "status": r.status_code,
        "latency_ms": (t1 - t0) * 1000,
        "delay_ms": delay_ms,
        "exec_ms": exec_ms,
        "tokens_generated": tokens,
        "body": out,
    }

async def call_run_async(client: httpx.AsyncClient, payload, timeout_s: int, poll_interval=0.25):
    r = await client.post(f"{BASE}/run",
                          headers={"Authorization": f"Bearer {API_KEY}","Content-Type":"application/json"},
                          json=payload, timeout=timeout_s)
    if r.status_code != 200:
        return {"ok": False, "status": r.status_code, "latency_ms": None, "tokens_generated": -1, "body": None}
    jid = r.json().get("id")
    t0 = time.time()
    while True:
        s = await client.get(f"{BASE}/status/{jid}",
                             headers={"Authorization": f"Bearer {API_KEY}"},
                             timeout=timeout_s)
        if s.status_code != 200:
            return {"ok": False, "status": s.status_code, "latency_ms": None, "tokens_generated": -1, "body": None}
        js = s.json()
        st = js.get("status")
        if st in ("COMPLETED","FAILED","CANCELLED"):
            ok = st == "COMPLETED"
            t1 = time.time()
            tokens = -1
            delay_ms = None
            exec_ms = None
            try:
                tokens = js.get("output", {}).get("usage", {}).get("output", -1)
                # Some status responses include these fields after completion
                delay_ms = js.get("delayTime")
                exec_ms = js.get("executionTime")
            except Exception:
                pass
            return {
                "ok": ok,
                "status": 200 if ok else 500,
                "latency_ms": (t1 - t0) * 1000,
                "delay_ms": delay_ms,
                "exec_ms": exec_ms,
                "tokens_generated": tokens,
                "body": js,
            }
        await asyncio.sleep(poll_interval)

async def run_phase(concurrency, runs, timeout, prompt, mode):
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    results = []
    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
        sem = asyncio.Semaphore(concurrency)
        async def one(i):
            async with sem:
                payload = build_payload(f"{prompt} [#{i}]")
                if mode == "sync":
                    return await call_runsync(client, payload, timeout)
                else:
                    return await call_run_async(client, payload, timeout)
        tasks = [asyncio.create_task(one(i)) for i in range(runs)]
        for t in asyncio.as_completed(tasks):
            results.append(await t)
    return results

def summarize(tag, results, metric="total"):
    if metric == "exec":
        lat = [r.get("exec_ms") for r in results if r["ok"] and r.get("exec_ms") is not None]
    elif metric == "delay":
        lat = [r.get("delay_ms") for r in results if r["ok"] and r.get("delay_ms") is not None]
    else:
        lat = [r["latency_ms"] for r in results if r["ok"] and r["latency_ms"] is not None]
    succ = sum(1 for r in results if r["ok"])
    fail = len(results) - succ
    return {
        "tag": tag,
        "runs": len(results),
        "success": succ,
        "fail": fail,
        "p50_ms": pct(lat,50) if lat else None,
        "p95_ms": pct(lat,95) if lat else None,
        "p99_ms": pct(lat,99) if lat else None,
        "avg_ms": sum(lat)/len(lat) if lat else None
    }

def print_table(rows):
    keys = ["tag","runs","success","fail","p50_ms","p95_ms","p99_ms","avg_ms"]
    print("| " + " | ".join(keys) + " |")
    print("|" + "|".join(["---"]*len(keys)) + "|")
    for r in rows:
        print("| " + " | ".join(str(r[k]) for k in keys) + " |")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concurrency", nargs="+", type=int, default=[1,4,8,16])
    ap.add_argument("--runs", type=int, default=40)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--prompt", type=str, default="Give concise career advice for switching from SDE to PM.")
    ap.add_argument("--mode", choices=["sync","async"], default="sync")
    ap.add_argument("--metric", choices=["total","exec","delay"], default="total",
                    help="Which latency to summarize: total client time, execution-only, or queue-only")
    ap.add_argument("--csv", type=str, default=f"rp_results_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.csv")
    args = ap.parse_args()

    if not API_KEY or not ENDPOINT_ID:
        print("Set RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID"); return

    all_rows = []
    all_results = []
    for c in args.concurrency:
        res = asyncio.run(run_phase(c, args.runs, args.timeout, args.prompt, args.mode))
        all_results.extend(res)
        summary = summarize(f"c={c}", res, metric=args.metric)
        all_rows.append(summary)
        print_table([summary])

    # thresholds
    # Thresholds for the selected metric
    # Defaults: total latency budget for serverless (adjust as needed)
    THR = {"p50_ms": 5000, "p95_ms": 8000, "fail_rate": 0.01}
    if args.metric == "exec":
        THR = {"p50_ms": 3000, "p95_ms": 5000, "fail_rate": 0.01}
    elif args.metric == "delay":
        THR = {"p50_ms": 2000, "p95_ms": 4000, "fail_rate": 0.05}
    print("\nTHRESHOLDS:", THR)
    for row in all_rows:
        fail_rate = 1 - (row["success"]/row["runs"])
        ok = (row["p50_ms"] and row["p50_ms"] <= THR["p50_ms"]) and \
             (row["p95_ms"] and row["p95_ms"] <= THR["p95_ms"]) and \
             (fail_rate <= THR["fail_rate"])
        print(f"{row['tag']}: {'PASS' if ok else 'FAIL'} "
              f"(p50={row['p50_ms']:.1f}ms, p95={row['p95_ms']:.1f}ms, fail_rate={fail_rate:.2%})")

    with open(args.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["ok","status","latency_ms","tokens_generated"])
        w.writeheader()
        for r in all_results:
            w.writerow({k:r.get(k) for k in ["ok","status","latency_ms","tokens_generated"]})
    print(f"\nSaved CSV -> {args.csv}")

if __name__ == "__main__":
    main()
