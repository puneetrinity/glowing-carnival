import json, os, sys, time, uuid
from datetime import datetime

APP = os.getenv("APP_NAME", "runpod_llm")
ENV = os.getenv("ENV", "prod")

def _emit(level, message, **fields):
    rec = {
        "ts": datetime.utcnow().isoformat() + "Z",
        "level": level,
        "app": APP,
        "env": ENV,
        "message": message
    }
    rec.update(fields)
    sys.stdout.write(json.dumps(rec, ensure_ascii=False) + "\n")
    sys.stdout.flush()

def log_start(intent: str, prompt_len: int, model: str):
    req_id = str(uuid.uuid4())
    t0 = time.time()
    _emit("info", "request_start", request_id=req_id, intent=intent, model=model, prompt_len=prompt_len)
    return req_id, t0

def log_progress(req_id: str, ttft_ms=None, tokens_so_far=None):
    _emit("info", "request_progress", request_id=req_id, ttft_ms=ttft_ms, tokens_so_far=tokens_so_far)

def log_end(req_id: str, ok: bool, total_ms: float, tokens_generated: int,
            p95_chunk_ms=None, error_type=None):
    _emit("info" if ok else "error", "request_end",
          request_id=req_id, ok=ok, total_ms=total_ms,
          tokens_generated=tokens_generated, p95_chunk_ms=p95_chunk_ms,
          error_type=error_type)
