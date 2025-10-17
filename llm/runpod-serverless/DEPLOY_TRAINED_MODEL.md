# Deploying Fine-Tuned Qwen2.5-7B Model to RunPod vLLM

**Complete guide for deploying your trained HR career guidance model from Colab to RunPod serverless.**

---

## Key Corrections Applied

✅ **GPU Sizing:** 24GB VRAM recommended (RTX A5000/L4), not 16GB
✅ **Validator Fixes:** Resume ATS (5+ commas), JD optional sections, HR seeding alignment
✅ **Test Temperatures:** 0.3-0.4 for HR domains (not 0.7) to avoid format drift
✅ **Env Variables:** Removed non-existent `USE_V3_VALIDATION`, `ENABLE_STREAMING` flags
✅ **Salary Blocking:** Verification test added, keep enabled in production
✅ **Pattern-Based Edits:** Using search/replace instead of line numbers

---

## Prerequisites

✅ Training completed in Colab (3 epochs, ~12,750 steps)
✅ Final validation pass rates:
- Resume guidance: 75-85%
- Job description: 75-85%
- Job resume match: 95-100%
- Recruiting strategy: 95-100%
- ATS keywords: 95-100%

---

## Step 1: Merge LoRA Adapter (In Colab)

After training completes, add this cell to your Colab notebook:

```python
# Merge LoRA adapter into base model for vLLM deployment
print("🔄 Merging LoRA adapter with base model...")

model.save_pretrained_merged(
    "/content/colab_training/qwen2.5-7b-career-hr-merged",
    tokenizer,
    save_method="merged_16bit"  # ~7GB, FP16 precision
)

print("✅ Merged model saved!")
print(f"📁 Location: /content/colab_training/qwen2.5-7b-career-hr-merged")
print(f"📊 Size: ~7GB (FP16)")
print("\n💡 This merged model is ready for vLLM deployment.")
```

**Why merge?**
- vLLM works best with merged models (no adapter overhead)
- Simpler deployment (single model directory)
- Better inference performance

---

## Step 2: Download Merged Model

### Option A: Via Google Drive (Recommended)

```python
# Backup merged model to Google Drive
import shutil
import os

drive_backup = '/content/drive/MyDrive/qwen2.5-7b-career-hr-merged'

print("💾 Copying merged model to Google Drive...")
if os.path.exists(drive_backup):
    shutil.rmtree(drive_backup)

shutil.copytree(
    "/content/colab_training/qwen2.5-7b-career-hr-merged",
    drive_backup
)

print("✅ Model backed up to Drive!")
print(f"📍 Download from: {drive_backup}")
```

Then download to WSL:
```bash
# On Windows, download from Drive to:
# Downloads/qwen2.5-7b-career-hr-merged/

# Move to WSL:
cd /mnt/c/Users/YourUsername/Downloads
mv qwen2.5-7b-career-hr-merged /home/ews/llm/
```

### Option B: Direct from Colab (Faster for testing)

```python
# Zip for faster download
!apt-get install -y zip
!cd /content/colab_training && zip -r qwen2.5-7b-career-hr-merged.zip qwen2.5-7b-career-hr-merged

# Download via Colab file browser:
# Files → qwen2.5-7b-career-hr-merged.zip
```

---

## Step 3: Fix Handler Validator Mismatches

**⚠️ CRITICAL:** Training validators differ from handler validators. Must fix before deployment.

### 3.1: Update Resume Validator

**File:** `/home/ews/llm/runpod-serverless/career_guidance_v3.py`

**Search for:** `has_ats = bool(re.search(r'ATS|keyword', response, re.IGNORECASE)) or (response.count(',') >= 8)`

**Replace with:**
```python
# Match training validator: 5+ commas OR "ATS Keywords:"
has_ats = ('ats keywords' in response.lower()) or (response.count(',') >= 5)
```

**Location:** In `validate_resume_guidance()` method, ATS keyword check section

### 3.2: Update Job Description Validator

**File:** `/home/ews/llm/runpod-serverless/career_guidance_v3.py`

**Search for:**
```python
# Check for nice-to-have or benefits (at least one)
optional_sections = ['nice-to-have', 'nice to have', 'benefits', 'perks']
has_optional = any(section in response.lower() for section in optional_sections)
if not has_optional:
    issues.append("Missing nice-to-have or benefits section")
```

**Replace with:**
```python
# Optional sections (not required - training validator only checks responsibilities + requirements)
# Training validator doesn't enforce nice-to-have, so commenting out this check
# optional_sections = ['nice-to-have', 'nice to have', 'benefits', 'perks']
# has_optional = any(section in response.lower() for section in optional_sections)
# if not has_optional:
#     issues.append("Missing nice-to-have or benefits section")
```

**Location:** In `validate_job_description()` method, optional sections check

### 3.3: Fix HR Domain Seeding

**File:** `/home/ews/llm/runpod-serverless/career_guidance_v3.py`

**Search for:**
```python
# Seed HR domains with "1. " to kickstart bullet/numbered lists
HR_DOMAINS = {"resume_guidance", "job_description", "job_resume_match",
              "recruiting_strategy", "ats_keywords"}
if domain in HR_DOMAINS:
```

**Replace with:**
```python
# Seed only domains that were seeded during training (match training setup)
HR_SEEDED_DOMAINS = {"resume_guidance", "recruiting_strategy", "ats_keywords"}
if domain in HR_SEEDED_DOMAINS:
```

**Location:** In `build_domain_prompt()` method, seeding logic section

---

## Step 4: Choose Deployment Option

### Option A: Upload to HuggingFace (Production)

**Best for:** Production deployment, team sharing, version control

**Steps:**

1. **Create/Update HuggingFace Repo:**
```bash
# Install HF CLI
pip install huggingface-hub

# Login
huggingface-cli login
# Enter your HF token
```

2. **Upload Model:**
```bash
cd /home/ews/llm/qwen2.5-7b-career-hr-merged

# Upload to your existing repo (or create new one)
huggingface-cli upload Puneetrinity/qwen2-7b-career-v2 . \
  --repo-type model \
  --commit-message "Fine-tuned v2: 75-85% HR pass rates, 3 epochs"

# For private repos, add --private flag:
# huggingface-cli upload Puneetrinity/qwen2-7b-career-v2 . --private
```

3. **Update Dockerfile (and add HF_TOKEN for private repos):**
```dockerfile
# Line 16-17: Update model name
ARG HF_MODEL=Puneetrinity/qwen2-7b-career-v2
ENV MODEL_PATH=/models/qwen2-7b-career-v2

# If using private HF repo, add before download:
# ARG HF_TOKEN
# ENV HF_TOKEN=${HF_TOKEN}
# Then build with: docker build --build-arg HF_TOKEN=your_token ...
```

**For private repos at runtime:**
Add to RunPod environment variables:
```
HF_TOKEN=your_hf_token_here
```

4. **Update handler.py:**
```python
# Line 94: Update default model path
model_path = os.getenv("MODEL_PATH", "/models/qwen2-7b-career-v2")
```

---

### Option B: Local Volume Mount (Testing)

**Best for:** Quick testing, iteration, avoiding HF upload

**Steps:**

1. **Verify model location:**
```bash
ls -lh /home/ews/llm/qwen2.5-7b-career-hr-merged
# Should show: config.json, model*.safetensors, tokenizer files
```

2. **Update Dockerfile for local mount:**
```dockerfile
# Replace lines 14-21 with:
# Model will be mounted at runtime from local path
ENV MODEL_PATH=/models/qwen2-7b-career
```

3. **Update deployment script to mount volume:**

**File:** `deploy.sh` or `deploy_runpodctl.sh`

Add volume mount flag:
```bash
# If using runpodctl:
runpodctl deploy \
  --name qwen-career-v2 \
  --imageName your-dockerhub-username/qwen-career:v2 \
  --gpuType "NVIDIA RTX A4000" \
  --gpuCount 1 \
  --volumePath /home/ews/llm/qwen2.5-7b-career-hr-merged:/models/qwen2-7b-career \
  --env MODEL_PATH=/models/qwen2-7b-career
```

**Note:** Volume mount only works if your RunPod pod has access to WSL filesystem. For serverless, use Option A (HuggingFace).

---

### Option C: Embed in Docker Image (Slower builds)

**Best for:** Final production image with everything embedded

1. **Copy model to runpod-serverless directory:**
```bash
cp -r /home/ews/llm/qwen2.5-7b-career-hr-merged /home/ews/llm/runpod-serverless/
```

2. **Update Dockerfile:**
```dockerfile
# Replace HF download (lines 14-21) with local copy:
COPY qwen2.5-7b-career-hr-merged /models/qwen2-7b-career

ENV MODEL_PATH=/models/qwen2-7b-career
```

**⚠️ Warning:** Docker image will be ~15GB. Build time: 30+ minutes.

---

## Step 5: Build and Push Docker Image

### 5.1: Apply Validator Fixes

```bash
cd /home/ews/llm/runpod-serverless

# Verify fixes applied
grep -n "has_ats" career_guidance_v3.py
# Should show line 420 with: response.count(',') >= 5

grep -n "HR_SEEDED_DOMAINS" career_guidance_v3.py
# Should show the corrected seeding
```

### 5.2: Build Image

```bash
# Build with your deployment option chosen above
docker build -t your-dockerhub-username/qwen-career:v2 .

# This takes 5-15 minutes depending on option:
# - Option A (HF): ~10 mins (downloads model during build)
# - Option B (mount): ~2 mins (no model in image)
# - Option C (embed): ~15 mins (copies large model)
```

### 5.3: Push to Docker Hub

```bash
# Login to Docker Hub
docker login

# Push image
docker push your-dockerhub-username/qwen-career:v2

# Note: Push time depends on image size
# - Option A/C: ~15-20 mins (large image)
# - Option B: ~3 mins (small image)
```

---

## Step 6: Deploy to RunPod

### 6.1: Via RunPod Web UI

1. Go to: https://www.runpod.io/console/serverless
2. Click "New Endpoint"
3. **Settings:**
   - **Name:** `qwen-career-v2`
   - **Container Image:** `your-dockerhub-username/qwen-career:v2`
   - **GPU Type:** **RTX A5000 or L4 (24GB VRAM recommended)**
     - FP16 7B model + KV cache rarely fits comfortably on 16GB
     - If using 16GB (A4000): Set `MAX_NUM_SEQS=4` and `GPU_MEMORY_UTILIZATION=0.85`
   - **Max Workers:** 3
   - **Idle Timeout:** 5 seconds
   - **Environment Variables:**
     ```
     MODEL_PATH=/models/qwen2-7b-career-v2  # (adjust for your option)
     MAX_MODEL_LEN=4096
     GPU_MEMORY_UTILIZATION=0.90  # (or 0.85 for 16GB GPUs)
     MAX_NUM_SEQS=8  # (or 4 for 16GB GPUs)
     NCCL_ASYNC_ERROR_HANDLING=1
     ```

     **Note:**
     - `ENABLE_STREAMING` and `USE_V3_VALIDATION` are **not** read by handler
     - Validation enabled by default (import-based), toggle per request via `enable_validation`
     - Streaming controlled by `sampling_params.stream` in request

4. Click "Deploy"
5. Wait for endpoint URL: `https://api.runpod.ai/v2/{endpoint-id}`

### 6.2: Via runpodctl CLI

```bash
runpodctl deploy \
  --name qwen-career-v2 \
  --imageName your-dockerhub-username/qwen-career:v2 \
  --gpuType "NVIDIA RTX A5000" \
  --gpuCount 1 \
  --env MODEL_PATH=/models/qwen2-7b-career-v2 \
  --env MAX_MODEL_LEN=4096 \
  --env GPU_MEMORY_UTILIZATION=0.90 \
  --env MAX_NUM_SEQS=8 \
  --env NCCL_ASYNC_ERROR_HANDLING=1

# For 24GB GPUs (A5000/L4) - recommended
# For 16GB GPUs (A4000) - use:
#   --env GPU_MEMORY_UTILIZATION=0.85 \
#   --env MAX_NUM_SEQS=4
```

---

## Step 7: Pre-Deployment Verification Checklist

**Run these tests BEFORE deploying to RunPod to ensure model quality.**

### 7.1: Test Set Audit

**What:** Validate against held-out test set to measure generalization

**How:**
```bash
cd /home/ews/llm/tinyllama_tools/staged_processing_fixed_v2/colab_training

# Run audit on test set (download from Colab first if needed)
python audit_dataset.py test_final_chatml.jsonl
```

**Expected Results:**
```
Resume guidance:     324/422 = 76.8% ✓
Job description:     367/451 = 81.4% ✓
Job resume match:    270/270 = 100.0% ✓
Recruiting strategy: 150/150 = 100.0% ✓
ATS keywords:        497/497 = 100.0% ✓
```

**Pass Criteria:** All domains ≥75%

---

### 7.2: Free-Form Probes (Unseen Prompts)

**What:** Test with 30-50 completely new prompts per HR domain to verify structure

**Create test file:** `free_form_test_prompts.txt`

```text
# Resume guidance (30-50 prompts)
Optimize resume for Senior Backend Engineer, 8 years Python/Django/AWS
Create resume bullets for ML Engineer transitioning from academia
Resume guidance for DevOps Lead with Kubernetes and Terraform
...

# Job description (30-50 prompts)
Write JD for Principal Software Architect at fintech startup
Create job posting for Staff Security Engineer, remote-first company
Draft JD for Engineering Manager, ML Infrastructure team
...

# Job resume match (30-50 prompts)
Score my 3 years React experience against Senior Frontend role
Evaluate fit: backend engineer vs full-stack position
Match analysis: data analyst background for ML engineer role
...

# Recruiting strategy (30-50 prompts)
Sourcing plan for hiring 5 senior engineers in 3 months
Recruiting strategy for niche role: embedded systems expert
Hiring approach for building data science team from scratch
...

# ATS keywords (30-50 prompts)
Extract ATS keywords from this cloud architect JD
List important keywords for data engineer resume
ATS optimization keywords for product manager role
...
```

**Test Script:** `test_free_form.py`

```python
import json
from transformers import AutoTokenizer
from unsloth import FastLanguageModel
from audit_dataset import VALIDATORS

MODEL_PATH = "/home/ews/llm/qwen2.5-7b-career-hr-merged"

# Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,
    max_seq_length=2048,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)

# Domain classification
def classify_domain(prompt):
    prompt_lower = prompt.lower()
    if "resume" in prompt_lower or "cv" in prompt_lower:
        return "resume_guidance"
    elif "job description" in prompt_lower or "jd" in prompt_lower:
        return "job_description"
    elif "match" in prompt_lower or "score" in prompt_lower:
        return "job_resume_match"
    elif "recruit" in prompt_lower or "sourcing" in prompt_lower:
        return "recruiting_strategy"
    elif "ats" in prompt_lower or "keywords" in prompt_lower:
        return "ats_keywords"
    return "unknown"

# Domains that need seeding
SEEDED_DOMAINS = {"resume_guidance", "recruiting_strategy", "ats_keywords"}

# Load prompts
with open("free_form_test_prompts.txt", "r") as f:
    prompts = [line.strip() for line in f if line.strip() and not line.startswith("#")]

# Test each prompt
results = {d: {"pass": 0, "fail": 0} for d in VALIDATORS.keys()}

for prompt in prompts:
    domain = classify_domain(prompt)
    if domain not in VALIDATORS:
        continue

    # Build ChatML prompt with seeding
    seed = "1. " if domain in SEEDED_DOMAINS else ""
    full_prompt = f"""<|im_start|>system
You are a helpful career guidance assistant.<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
{seed}"""

    # Generate
    inputs = tokenizer([full_prompt], return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=250,
        temperature=0.4,
        top_p=0.9,
        do_sample=True
    )

    # Extract response
    response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    assistant_start = response.find('<|im_start|>assistant\n') + len('<|im_start|>assistant\n')
    assistant_end = response.find('<|im_end|>', assistant_start)
    generated = response[assistant_start:assistant_end].strip()

    # Validate
    is_valid, issues = VALIDATORS[domain](generated)
    if is_valid:
        results[domain]["pass"] += 1
    else:
        results[domain]["fail"] += 1
        print(f"\n❌ Failed: {domain}")
        print(f"   Prompt: {prompt[:60]}...")
        print(f"   Issues: {issues}")

# Print results
print("\n" + "="*80)
print("FREE-FORM PROBE RESULTS")
print("="*80)
for domain in VALIDATORS.keys():
    total = results[domain]["pass"] + results[domain]["fail"]
    if total > 0:
        rate = results[domain]["pass"] / total * 100
        status = "✓" if rate >= 75 else "✗"
        print(f"{status} {domain:25s}: {results[domain]['pass']:2d}/{total:2d} = {rate:5.1f}%")
```

**Run:**
```bash
python test_free_form.py
```

**Expected:** All domains ≥75% pass rate

**Manual Checks:**
- Resume: 5+ bullets, action verbs, 5+ commas OR "ATS Keywords:"
- Job description: Responsibilities + Requirements sections, 8+ bullets
- Match: Score present, mentions matches/gaps/skills
- Recruiting: 3+ sourcing channels, timing words
- ATS: 15+ keywords (comma-separated or bulleted)

---

### 7.3: Robustness Sweep (Temperature Stability)

**What:** Test temperature range 0.2-0.6 to ensure format doesn't collapse

**Test Script:** `test_temperature_sweep.py`

```python
import json
from transformers import AutoTokenizer
from unsloth import FastLanguageModel
from audit_dataset import VALIDATORS

MODEL_PATH = "/home/ews/llm/qwen2.5-7b-career-hr-merged"

# Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,
    max_seq_length=2048,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)

# Test prompts (1 per domain)
test_cases = {
    "resume_guidance": "Optimize resume for Senior ML Engineer with Python, TensorFlow, AWS",
    "job_description": "Write job description for Staff DevOps Engineer at cloud company",
    "job_resume_match": "Score my 5 years React experience against Senior Frontend role",
    "recruiting_strategy": "Sourcing strategy for hiring senior backend engineers",
    "ats_keywords": "Extract ATS keywords from data engineer job description"
}

SEEDED_DOMAINS = {"resume_guidance", "recruiting_strategy", "ats_keywords"}

# Temperature sweep
temperatures = [0.2, 0.3, 0.4, 0.5, 0.6]

print("="*80)
print("TEMPERATURE ROBUSTNESS SWEEP")
print("="*80)

for temp in temperatures:
    print(f"\nTemperature: {temp}")
    print("-" * 60)

    for domain, prompt in test_cases.items():
        # Build ChatML prompt
        seed = "1. " if domain in SEEDED_DOMAINS else ""
        full_prompt = f"""<|im_start|>system
You are a helpful career guidance assistant.<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
{seed}"""

        # Generate
        inputs = tokenizer([full_prompt], return_tensors="pt").to("cuda")
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=temp,
            top_p=0.9,
            do_sample=True
        )

        # Extract response
        response = tokenizer.decode(outputs[0], skip_special_tokens=False)
        assistant_start = response.find('<|im_start|>assistant\n') + len('<|im_start|>assistant\n')
        assistant_end = response.find('<|im_end|>', assistant_start)
        generated = response[assistant_start:assistant_end].strip()

        # Validate
        is_valid, issues = VALIDATORS[domain](generated)
        status = "✓" if is_valid else "✗"

        print(f"  {status} {domain:25s}: {'PASS' if is_valid else f'FAIL ({issues})'}")

print("\n" + "="*80)
print("EXPECTED: All domains pass at all temperatures")
print("WARNING: If failures at temp >0.5, use temp=0.3-0.4 in production")
print("="*80)
```

**Run:**
```bash
python test_temperature_sweep.py
```

**Pass Criteria:**
- All domains pass at temperature 0.3-0.4 (production setting)
- Acceptable: Some failures at 0.6 (too high for structured output)
- Red flag: Failures at 0.2-0.3 (model not learning structure well)

---

### 7.4: Verification Checklist Summary

Before proceeding to RunPod deployment, confirm:

- [ ] **Test set audit:** All domains ≥75% on `test_final_chatml.jsonl`
- [ ] **Free-form probes:** 30-50 unseen prompts per domain, ≥75% pass rate
- [ ] **Temperature sweep:** All domains pass at temp 0.3-0.4
- [ ] **Structure checks:** Manual review confirms:
  - Resume: Bullets, action verbs, ATS line
  - JD: Sections (responsibilities, requirements), 8+ bullets
  - Match: Score, matches/gaps mentioned
  - Recruiting: Channels, timing
  - ATS: 15+ keywords

**If all pass:** ✅ Proceed to Step 8 (Deploy to RunPod)

**If any fail:** ⚠️ Review training logs, consider:
- Loading earlier checkpoint (if later ones degraded)
- Extending training (if still early, <8000 steps)
- Investigating failed prompts (dataset quality issues?)

---

## Step 8: Deploy to RunPod

**Note:** Only proceed after completing Step 7 Pre-Deployment Verification.

---

## Step 9: Test Deployed Endpoint

### 7.1: Test Resume Guidance (Seeded domain)

```bash
curl -X POST https://api.runpod.ai/v2/{endpoint-id}/runsync \
  -H "Authorization: Bearer YOUR_RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "prompt": "Optimize my resume for a Senior Machine Learning Engineer role with Python, TensorFlow, and AWS.",
      "sampling_params": {
        "max_tokens": 200,
        "temperature": 0.3,
        "stream": false
      },
      "enable_validation": true
    }
  }'
```

**Note:** Use `temperature: 0.3-0.4` for HR domains (not 0.7) to avoid format drift and ensure consistent structure.

**Expected:**
- Response starts with "1. " (seeded)
- 5+ bullet points
- Action verbs (Led, Built, Optimized)
- 5+ commas OR "ATS Keywords:" line
- Validation: `"is_valid": true`

### 7.2: Test Job Description (Non-seeded domain)

```bash
curl -X POST https://api.runpod.ai/v2/{endpoint-id}/runsync \
  -H "Authorization: Bearer YOUR_RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "prompt": "Write a job description for a Senior DevOps Engineer at a cloud infrastructure company.",
      "sampling_params": {
        "max_tokens": 230,
        "temperature": 0.4,
        "stream": false
      },
      "enable_validation": true
    }
  }'
```

**Expected:**
- Response does NOT start with "1. " (not seeded)
- Responsibilities section
- Requirements section
- 8+ bullet points
- Validation: `"is_valid": true`

### 7.3: Test Match Scoring (Non-seeded domain)

```bash
curl -X POST https://api.runpod.ai/v2/{endpoint-id}/runsync \
  -H "Authorization: Bearer YOUR_RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "prompt": "Score match between my 5 years of React development and this Senior Frontend Engineer role requiring React, TypeScript, and GraphQL.",
      "sampling_params": {
        "max_tokens": 200,
        "temperature": 0.3,
        "stream": false
      },
      "enable_validation": true
    }
  }'
```

**Expected:**
- Match score (0-100 or percentage)
- Matches/strengths section
- Gaps/missing skills section
- Validation: `"is_valid": true`

### 7.4: Test Salary/Market Blocking (Critical)

```bash
curl -X POST https://api.runpod.ai/v2/{endpoint-id}/runsync \
  -H "Authorization: Bearer YOUR_RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "prompt": "What is the salary range for Senior ML Engineers in San Francisco?",
      "sampling_params": {
        "max_tokens": 150,
        "temperature": 0.3,
        "stream": false
      },
      "enable_validation": true
    }
  }'
```

**Expected:**
- `"intent": "salary_intel"` or `"intent": "market_intel"`
- `"should_use_rag": true`
- `"recommendation": "REQUIRE RAG - Block model-only answers"`
- Confirms salary/market blocking still works

---

## Step 8: Monitor Pass Rates

### 8.1: Run Load Test

```bash
cd /home/ews/llm/runpod-serverless

# Update endpoint URL in rp_load_test.py
python rp_load_test.py \
  --endpoint-id YOUR_ENDPOINT_ID \
  --api-key YOUR_RUNPOD_API_KEY \
  --requests 100 \
  --workers 5
```

### 8.2: Check Validation Metrics

Look for in output:
```
Resume guidance: 48/60 validated (80%)
Job description: 32/40 validated (80%)
Job resume match: 40/40 validated (100%)
Recruiting strategy: 30/30 validated (100%)
ATS keywords: 30/30 validated (100%)
```

**Target pass rates:**
- Resume: ≥75%
- Job description: ≥75%
- Others: ≥90%

---

## Step 9: Production Cutover

### 9.1: Canary Deployment (Week 1)

**Route 5-10% of live traffic to new model and monitor metrics.**

#### Setup Canary Routing

**Option A: API Gateway/Load Balancer**
```python
import random

def route_request(user_request):
    # 10% to new model
    if random.random() < 0.10:
        endpoint = "qwen-career-v2"  # New model
    else:
        endpoint = "qwen-career-v1"  # Old model

    return call_endpoint(endpoint, user_request)
```

**Option B: Feature Flag (LaunchDarkly, Split.io)**
```python
from launchdarkly import LDClient

client = LDClient(sdk_key="your-key")

def route_request(user_id, user_request):
    use_v2 = client.variation("use-career-model-v2", {
        "key": user_id
    }, False)

    endpoint = "qwen-career-v2" if use_v2 else "qwen-career-v1"
    return call_endpoint(endpoint, user_request)
```

#### Monitor Canary Metrics

**Critical Metrics (Real-time dashboard):**

1. **Validator Pass Rate** (Target: ≥75%)
```sql
-- Query your logs/metrics DB
SELECT
    domain,
    COUNT(*) as total_requests,
    SUM(CASE WHEN is_valid = true THEN 1 ELSE 0 END) as valid_count,
    (SUM(CASE WHEN is_valid = true THEN 1 ELSE 0 END) * 100.0 / COUNT(*)) as pass_rate
FROM inference_logs
WHERE model_version = 'v2'
    AND timestamp > NOW() - INTERVAL '1 hour'
GROUP BY domain
ORDER BY domain;
```

**Alert if:** Any domain drops below 75% for >10 minutes

2. **Regeneration Rate** (Target: <10%)
```sql
SELECT
    COUNT(*) as total_requests,
    SUM(CASE WHEN regenerated = true THEN 1 ELSE 0 END) as regen_count,
    (SUM(CASE WHEN regenerated = true THEN 1 ELSE 0 END) * 100.0 / COUNT(*)) as regen_rate
FROM inference_logs
WHERE model_version = 'v2'
    AND timestamp > NOW() - INTERVAL '1 hour';
```

**Alert if:** Regen rate >15% for >30 minutes

3. **Response Latency** (Target: p95 <3s)
```sql
SELECT
    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY latency_ms) as p50,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY latency_ms) as p95,
    PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY latency_ms) as p99
FROM inference_logs
WHERE model_version = 'v2'
    AND timestamp > NOW() - INTERVAL '1 hour';
```

**Alert if:** p95 >4000ms for >15 minutes

4. **Safety Blocks** (Target: 100% blocked for salary/market)
```sql
SELECT
    intent,
    COUNT(*) as blocked_count
FROM inference_logs
WHERE model_version = 'v2'
    AND should_use_rag = true
    AND recommendation LIKE '%REQUIRE RAG%'
    AND timestamp > NOW() - INTERVAL '1 hour'
GROUP BY intent;
```

**Alert if:** Any salary/market query NOT blocked

#### Canary Success Criteria (Week 1)

Compare V2 vs V1 metrics:

| Metric | V1 (Baseline) | V2 (Target) | Status |
|--------|---------------|-------------|--------|
| Resume pass rate | 60% | ≥75% | Monitor |
| JD pass rate | 65% | ≥75% | Monitor |
| Match pass rate | 95% | ≥95% | Monitor |
| Recruiting pass rate | 95% | ≥95% | Monitor |
| ATS pass rate | 95% | ≥95% | Monitor |
| Regen rate | 30% | <10% | **Key** |
| Latency p95 | 2.1s | <3s | Monitor |
| Salary blocks | 100% | 100% | **Critical** |

**Decision Points:**

✅ **Proceed to 50% (Week 2) if:**
- All pass rates ≥75%
- Regen rate <10%
- Latency acceptable
- No salary/market leaks
- No major user complaints

⚠️ **Hold at 10% if:**
- Pass rates 70-75% (acceptable but not great)
- Regen rate 10-15% (better than v1 but not target)
- Minor issues being fixed

🛑 **Rollback to 0% if:**
- Any pass rate <70%
- Regen rate >20%
- Salary/market queries not blocked
- Major user complaints or business impact

#### Week 2: Increase to 50% if stable

Update canary percentage:
```python
if random.random() < 0.50:  # Increase to 50%
    endpoint = "qwen-career-v2"
```

Continue monitoring same metrics.

#### Week 3: Full cutover (100%)

If Week 2 metrics stable:
```python
endpoint = "qwen-career-v2"  # All traffic to v2
```

Keep v1 endpoint running for 1 more week as safety net.

### 9.2: Remove Validators (Optional)

If pass rates stay ≥95% for 2 weeks:

**File:** `handler.py` line 407
```python
# Disable validation for production (model is reliable)
enable_validation = job_input.get("enable_validation", False)  # Changed from True
```

**⚠️ IMPORTANT - Keep These Enabled:**
- **Seeding:** Keep enabled - helps with consistency and format
- **Salary/Market Blocking:** Keep enabled even after relaxing other validation
  - Model should never answer salary/market questions without RAG
  - This is a business requirement, not just quality check

**Only disable:** HR structure validation (resume bullets, JD sections, etc.) after 2 weeks of 95%+ pass rates.

---

## Troubleshooting

### Issue: "Model not found" error

**Cause:** Model path mismatch

**Fix:**
```bash
# Check model path in logs:
docker logs <container-id>

# Verify MODEL_PATH env var matches actual path:
# - HF option: /models/qwen2-7b-career-v2
# - Local mount: /models/qwen2-7b-career
# - Embedded: /models/qwen2-7b-career
```

### Issue: OOM (Out of Memory) on GPU

**Cause:** Model too large for GPU

**Fix:**
```python
# Reduce GPU memory utilization
ENV GPU_MEMORY_UTILIZATION=0.85  # (was 0.90)

# Or reduce max sequences
ENV MAX_NUM_SEQS=6  # (was 8)
```

### Issue: High validation failure rate (>30%)

**Cause:** Validator mismatch not fixed

**Fix:**
1. Re-check Step 3 validator fixes
2. Rebuild Docker image
3. Redeploy

Verify with:
```bash
grep -A2 "has_ats" career_guidance_v3.py
# Should show: response.count(',') >= 5 (not 8)
```

### Issue: Resume outputs missing "1. " seed

**Cause:** Seeding not applied correctly

**Fix:**
```bash
grep -A3 "HR_SEEDED_DOMAINS" career_guidance_v3.py
# Should show: {"resume_guidance", "recruiting_strategy", "ats_keywords"}
# Not all 5 domains
```

### Issue: Slow cold starts (>30 seconds)

**Solutions:**
- **Option A (HF):** Model downloads from HF on first start
  - Fix: Use Option C (embed in image)
- **Option B (mount):** Volume mount slower
  - Fix: Use Option A or C
- **Option C (embed):** Should be fastest (~5-10s cold start)

### Issue: Model outputs seem worse than training

**Cause:** Checkpoint gating may have blocked best checkpoints

**Fix:**
1. Check training logs for best step (highest pass rates)
2. In Colab, load that checkpoint manually:
```python
# Find best checkpoint
checkpoints = sorted(glob.glob(f"{output_dir}/checkpoint-*"))
best_checkpoint = checkpoints[-1]  # Or identify best from logs

# Load and merge that checkpoint
model = FastLanguageModel.from_pretrained(
    model_name=best_checkpoint,
    max_seq_length=2048,
    load_in_4bit=True,
)
model.save_pretrained_merged(...)
```

---

## Rollback Plan

If new model performs worse:

### Quick Rollback (5 minutes)

**Option 1: Revert endpoint to old image**
```bash
# Via RunPod UI:
# Endpoints → qwen-career-v2 → Settings → Container Image
# Change back to: your-dockerhub-username/qwen-career:v1
# Click "Update Endpoint"
```

**Option 2: Route traffic to old endpoint**
```python
# In API gateway:
endpoint = "qwen-career-v1"  # (was v2)
```

### Full Rollback (30 minutes)

1. Stop new endpoint
2. Scale up old endpoint workers
3. Update DNS/load balancer
4. Verify metrics recovered

---

## Success Criteria

✅ Cold start: <10 seconds
✅ Response latency: <2 seconds (non-streaming)
✅ Resume validation: ≥75%
✅ Job description validation: ≥75%
✅ Others validation: ≥90%
✅ Regeneration rate: <10% (was 30%)

**If all criteria met:** Training successful, model ready for production!

---

## Next Steps After Deployment

1. **Monitor for 1 week:**
   - Validation pass rates
   - User satisfaction
   - Error rates
   - Response quality

2. **Collect feedback:**
   - User ratings
   - Regeneration triggers
   - Edge cases

3. **Iterate if needed:**
   - Retrain with more data
   - Adjust validators
   - Fine-tune hyperparameters

4. **Scale if successful:**
   - Increase max workers
   - Add more GPU types
   - Enable autoscaling

---

## Files Modified Summary

**Required Changes:**

1. **`career_guidance_v3.py`** - 3 critical validator fixes:
   - Resume ATS check: `has_ats` logic (search for `response.count(',') >= 8`)
   - JD validator: Comment out nice-to-have requirement (search for `optional_sections`)
   - HR seeding: Change `HR_DOMAINS` to `HR_SEEDED_DOMAINS` with only 3 domains

2. **`Dockerfile`** - Update model source:
   - Line 16: `ARG HF_MODEL=Puneetrinity/qwen2-7b-career-v2`
   - Line 17: `ENV MODEL_PATH=/models/qwen2-7b-career-v2`
   - Add `HF_TOKEN` handling if using private repo

**Optional Changes:**

3. **`handler.py`** - Update default path (if changed):
   - Line 94: `model_path = os.getenv("MODEL_PATH", "/models/qwen2-7b-career-v2")`

4. **`deploy.sh`** - Update environment variables:
   - Remove `ENABLE_STREAMING=true` and `USE_V3_VALIDATION=true` (not used)
   - Add GPU sizing notes for 16GB vs 24GB

**Verification:**
```bash
# Check fixes applied:
cd /home/ews/llm/runpod-serverless

grep "response.count(',') >= 5" career_guidance_v3.py
# Should find the corrected ATS check

grep "HR_SEEDED_DOMAINS" career_guidance_v3.py
# Should find: {"resume_guidance", "recruiting_strategy", "ats_keywords"}
```

---

## Reference: Training Configuration

For documentation:
- **Base model:** unsloth/Qwen2.5-7B-Instruct-bnb-4bit
- **Training:** 3 epochs, 34,000 examples, 12,750 steps
- **LoRA:** r=16, alpha=32
- **Batch size:** 8 (effective)
- **Learning rate:** 1.5e-4 (cosine schedule)
- **Validation frequency:** Every 100 steps
- **Training time:** ~10-12 hours (T4 GPU)

---

**📞 Support:** If issues persist, check training logs and validation outputs from Step 100, 200, etc.

**🎉 Good luck with deployment!**
