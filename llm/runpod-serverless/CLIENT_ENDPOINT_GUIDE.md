# Client Guide: High‑Quality Calls to RunPod vLLM Endpoints

This guide shows how to call your RunPod vLLM endpoint (raw model) and your handler
endpoint (with validation/gates), using production ChatML prompts, per‑domain
exemplars, conservative decoding, micro‑retry, and quick quality checks.

Use this when testing and integrating your new model: Puneetrinity/qwen2.5-7b-careerv2.

---

## Quick Summary

- Always prepend a ChatML system prompt and seed where applicable. This is the
  single biggest quality lever for the raw vLLM endpoint.
- Use conservative decoding: temperature 0.3, repetition_penalty 1.0,
  stop ["<|im_end|>"], domain‑specific max_tokens.
- For JD/Match/Recruiting/ATS, include a 1‑shot exemplar in the system text for
  consistent structure. Add a micro‑retry with a clarifier on structural failures.
- Salary blocking and router safety only apply to the handler endpoint with
  validation enabled; raw vLLM will not block.

Targets (generations):
- Resume ≥ 85%, JD ≥ 80%, Match ≥ 80–90%, Recruiting ≥ 95%, ATS ≥ 95%

---

## Endpoints

- Raw vLLM (no validation): `https://api.runpod.ai/v2/<ENDPOINT_ID>/runsync`
- Handler (validation/gates): `https://api.runpod.ai/v2/<HANDLER_ENDPOINT_ID>/runsync`

Environment (suggested):
- `ENDPOINT_ID` (raw) or `HANDLER_ENDPOINT_ID` (handler)
- `API_KEY` (RunPod API key)

---

## Decoding Defaults (HR domains)

- `temperature`: 0.3
- `repetition_penalty`: 1.0
- `stop`: ["<|im_end|>"]
- `max_tokens`:
  - Resume: 200
  - Job Description: 230
  - Job‑Resume Match: 200
  - Recruiting: 180
  - ATS keywords: 150

---

## ChatML Builders (with 1‑shot exemplars)

Seeded domains: `resume_guidance`, `recruiting_strategy`, `ats_keywords` get "1. "

Resume exemplar:
```
1. Led migration of 50+ services to Kubernetes, reducing deployments by 60%
2. Built CI/CD with Jenkins + ArgoCD, enabling 20 daily deploys
3. Optimized AWS costs by 35% via auto-scaling and rightsizing
4. Designed REST APIs handling 10k+ req/sec with FastAPI
5. Implemented monitoring with Prometheus/Grafana and on-call rota
6. Mentored 3 junior engineers in code reviews and design
ATS Keywords: Kubernetes, Docker, Jenkins, ArgoCD, Terraform, AWS, FastAPI, Python, CI/CD, Prometheus
```

JD exemplar (sections):
```
**Summary:** Experienced backend engineer to build scalable APIs.
**Responsibilities:**
1. Design REST APIs
2. Optimize DB queries
3. Implement security
4. Collaborate with DevOps
5. Code reviews
6. Monitor SLIs/SLOs
**Requirements:**
1. 5+ yrs backend
2. Python/Go/Node
3. SQL/NoSQL
4. AWS/GCP
5. System design
6. Communication
```

System prompts per domain (short):
- Resume: "Provide 6–8 numbered bullets using action verbs. End with: ATS Keywords: …"
- JD: "Create JD with sections: Summary, Responsibilities (6–8), Requirements (6–8)."
- Match: "Provide: Score (0–100), Matches (5+), Gaps (3+), Next steps (3+)."
- Recruiting: "Provide 4–6 steps; include LinkedIn/referral/meetup/university/conference/GitHub/Stack Overflow, cadence weekly/monthly/quarterly, and metrics."
- ATS: "Provide 20–40 comma‑separated keywords (80–120 words)."

---

## cURL Cheat Sheet (Raw vLLM)

With ChatML + seed for resume:
```
curl -sS -X POST "https://api.runpod.ai/v2/<ENDPOINT_ID>/runsync" \
  -H "Authorization: Bearer <API_KEY>" -H "Content-Type: application/json" \
  -d '{
    "input": {
      "prompt": "<|im_start|>system\nYou are a career coach specializing in resumes. Provide 6-8 numbered bullets using action verbs. End with: ATS Keywords: ...\n<|im_end|>\n<|im_start|>user\nOptimize my resume for Senior Backend Engineer (Python, FastAPI, AWS).\n<|im_end|>\n<|im_start|>assistant\n1. ",
      "sampling_params": {"max_tokens": 200, "temperature": 0.3, "repetition_penalty": 1.0, "stop": ["<|im_end|>"]}
    }
  }'
```

Handler endpoint (validation on):
```
curl -sS -X POST "https://api.runpod.ai/v2/<HANDLER_ENDPOINT_ID>/runsync" \
  -H "Authorization: Bearer <API_KEY>" -H "Content-Type: application/json" \
  -d '{
    "input": {
      "prompt": "<|im_start|>system\n...\n<|im_end|>\n<|im_start|>user\n...\n<|im_end|>\n<|im_start|>assistant\n1. ",
      "sampling_params": {"max_tokens": 200, "temperature": 0.3, "repetition_penalty": 1.0, "stop": ["<|im_end|>"]},
      "enable_validation": true,
      "block_low_trust_intents": true
    }
  }'
```

---

## Python Client (Raw vLLM)

Save as `client_endpoint.py` (or reuse snippets in this doc). It:
- Builds ChatML per domain with exemplars and seeding
- Calls `/runsync` with conservative decoding
- Adds a single micro‑retry on structural failures for JD/Match/Recruiting/ATS

```
import os, re, json, requests

ENDPOINT_ID = os.getenv("ENDPOINT_ID", "<ENDPOINT_ID>")
API_KEY = os.getenv("API_KEY", "<API_KEY>")
BASE = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync"
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
GEN = {"max_tokens": 200, "temperature": 0.3, "top_p": 0.9, "repetition_penalty": 1.0, "stop": ["<|im_end|>"]}

SEEDED = {"resume_guidance","recruiting_strategy","ats_keywords"}
RESUME_EXAMPLE = ("1. Led migration...\n...\nATS Keywords: Kubernetes, Docker, Jenkins, ArgoCD, Terraform, AWS, FastAPI, Python, CI/CD, Prometheus")
JD_EXAMPLE = ("**Summary:** ...\n**Responsibilities:**...\n**Requirements:**...")

def build_chatml(domain, user):
    if domain == "resume_guidance":
        sys_txt = "You are a career coach specializing in resumes. Provide 6-8 numbered bullets using action verbs. End with: \"ATS Keywords: ...\".\n\nExample:\n" + RESUME_EXAMPLE
    elif domain == "job_description":
        sys_txt = "You are an HR specialist. Create JD with Summary, Responsibilities (6-8), Requirements (6-8).\n\nExample:\n" + JD_EXAMPLE
    elif domain == "job_resume_match":
        sys_txt = "You are a technical recruiter. Provide: Score (0-100), Matches (5+), Gaps (3+), Next steps (3+)."
    elif domain == "recruiting_strategy":
        sys_txt = ("You are a recruiting strategist. Provide 4–6 steps. Include LinkedIn, referral, meetup, university, conference, GitHub, Stack Overflow, and cadence weekly/monthly/quarterly. Add metrics.")
    else:
        sys_txt = "You are a resume optimization specialist. Provide 20–40 ATS keywords comma-separated (80–120 words)."
    seed = "1. " if domain in SEEDED else ""
    return f"<|im_start|>system\n{sys_txt}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n{seed}"

def call_endpoint(prompt, max_tokens=200):
    body = {"input": {"prompt": prompt, "sampling_params": {**GEN, "max_tokens": max_tokens}}}
    r = requests.post(BASE, headers=HEADERS, json=body, timeout=120)
    r.raise_for_status()
    return r.json()

# Minimal validators
_def_verbs = ["led","built","developed","managed","created","improved","optimized","designed","implemented","achieved"]

def v_resume(t):
    issues=[]
    if len(t.split())<50: issues.append("len")
    if not re.search(r'^\s*[\d\-\*\•]', t, re.M): issues.append("bullets")
    if not any(v in t.lower() for v in _def_verbs): issues.append("verbs")
    if ("ats keywords" not in t.lower()) and t.count(",")<5: issues.append("ats_line")
    return (len(issues)==0), issues

def v_jd(t):
    issues=[]
    if len(t.split())<100: issues.append("len")
    if sum(s in t.lower() for s in ["responsibilities","requirements"])<2: issues.append("sections")
    if len(re.findall(r'^\s*[\d\-\*\•]', t, re.M))<8: issues.append("bullets")
    return (len(issues)==0), issues

VALIDATORS = {
    "resume_guidance": v_resume,
    "job_description": v_jd,
}

# Micro-retry clarifier for structure
CLAR = {
    "resume_guidance": " Ensure 6–8 bullets with action verbs and end with: ATS Keywords: ...",
    "job_description": " Ensure sections: Summary + Responsibilities (6–8) + Requirements (6–8).",
}

def generate(domain, user_prompt, max_tokens=None, retry=True):
    chat = build_chatml(domain, user_prompt)
    mt = max_tokens or (230 if domain=="job_description" else 200)
    js = call_endpoint(chat, max_tokens=mt)
    text = (((js or {}).get("output") or {}).get("choices") or [{}])[0].get("text","")
    ok, issues = VALIDATORS.get(domain, lambda t:(True,[]))(text)
    if ok or not retry:
        return ok, issues, text
    # Retry with clarifier
    clar = CLAR.get(domain, "")
    chat2 = build_chatml(domain, user_prompt + clar)
    js2 = call_endpoint(chat2, max_tokens=mt)
    text2 = (((js2 or {}).get("output") or {}).get("choices") or [{}])[0].get("text","")
    ok2, issues2 = VALIDATORS.get(domain, lambda t:(True,[]))(text2)
    return ok2, issues2, text2
```

---

## Automated Quality Suite (CSV)

We include `test_quality_gates.py` in this repo. It:
- Generates HR prompts
- Builds ChatML + exemplars
- Calls `/runsync` and validates with training validators (or fallback)
- Writes a CSV with domain, ok, issues, preview, status, latency

Run:
```
export ENDPOINT_ID=<ENDPOINT_ID>
export API_KEY=<API_KEY>
python3 test_quality_gates.py \
  --endpoint-id "$ENDPOINT_ID" \
  --api-key "$API_KEY" \
  --count 10 \
  --out eval_results_endpoint.csv
```

Use the handler endpoint if you want salary blocking/router gates (`enable_validation`,
`block_low_trust_intents`); the raw vLLM endpoint will not block salary queries.

---

## Troubleshooting

- Empty outputs on raw endpoint: ensure ChatML format + seed, stop ["<|im_end|>"], temp=0.3.
- JD/Match underperform: add 1‑shot exemplars + micro‑retry clarifier for required sections.
- Salary/market not blocked: expected on raw endpoint; use handler with validation.
- Small‑talk oddities: add a small dedicated small_talk system prompt if you need it.
- Latency/timeouts: ensure `runsync` is used; set Min Workers=1, Idle Timeout ≥ 30s.

---

## Notes

- This client design mirrors the patterns that achieved ≥90% in your gen audits.
- For strongest guarantees in production: add a JSON‑first fallback client path for
  JD/Match/ATS (generate JSON → render → validate → micro‑retry), and keep the handler
  endpoint with validation enabled for first 1–2 weeks.

