"""
RunPod Serverless Handler for vLLM with Streaming Support
Optimized for Qwen 2.5 7B Career Model - V3 with Quality Validation

V3 Improvements:
- 87% validation for career guidance (13/15)
- Intent classification with false positive fixes
- Auto-sanitization with over-cleaning prevention
- Enhanced salary range detection (SGD, SEK, etc)
"""

import os
import runpod
from vllm import SamplingParams
from typing import Generator, Dict, Any
import asyncio
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
import sys
import json
import uuid
import time
import atexit
import signal
from datetime import datetime, timezone

# Import V3 validation logic (use symlinked path in container)
sys.path.append('/home/ews/llm')
try:
    from career_guidance_v3 import (
        QuestionIntent,
        IntentClassifier,
        ResponseValidator,
        AutoSanitizer,
        PromptBuilder,
    )
    USE_V3_VALIDATION = True
    print("✓ V3 validation imported")
except ImportError as e:
    print(f"⚠ V3 validation not available: {e}")
    USE_V3_VALIDATION = False

# Global vLLM engine instance
llm_engine = None

# Graceful shutdown handler to reduce NCCL warnings
# Note: Set NCCL_ASYNC_ERROR_HANDLING=1 in environment for better NCCL cleanup
def _shutdown_handler(signum=None, frame=None):
    """Gracefully shutdown vLLM engine to prevent NCCL teardown warnings

    Reduces "ProcessGroupNCCL has NOT been destroyed" warnings on worker teardown.
    Works with SIGTERM (container stop), SIGINT (Ctrl+C), and atexit (normal exit).
    """
    global llm_engine
    if llm_engine is not None:
        try:
            print("🔄 Gracefully shutting down vLLM engine...")
            # vLLM AsyncEngine doesn't have a public shutdown method in 0.6.4.post1
            # Best effort: let Python's GC handle cleanup properly
            llm_engine = None
            print("✓ Engine shutdown initiated")
        except Exception as e:
            print(f"⚠ Error during shutdown: {e}")

# Register shutdown handlers
atexit.register(_shutdown_handler)
signal.signal(signal.SIGTERM, _shutdown_handler)
signal.signal(signal.SIGINT, _shutdown_handler)

# Structured logging functions
def log_json(event: str, request_id: str, **kwargs):
    """Emit structured JSON log

    Note: Handler can only log exec_ms (processing time).
    Queue delay (delay_ms) happens before handler starts -
    only available from RunPod response metadata client-side.
    """
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "level": kwargs.pop("level", "info"),
        "event": event,
        "request_id": request_id,
        **kwargs
    }
    print(json.dumps(record), flush=True)

def initialize_engine():
    """Initialize vLLM engine with optimized settings"""
    global llm_engine

    if llm_engine is not None:
        return llm_engine

    model_path = os.getenv("MODEL_PATH", "/models/qwen2-7b-career")

    # vLLM AsyncEngine configuration
    max_model_len = int(os.getenv("MAX_MODEL_LEN", "4096"))
    env_mnbt = os.getenv("MAX_NUM_BATCHED_TOKENS")
    mnbt = None
    if env_mnbt is not None and env_mnbt != "":
        try:
            mnbt_val = int(env_mnbt)
            if mnbt_val < max_model_len:
                print(
                    f"⚠ MAX_NUM_BATCHED_TOKENS ({mnbt_val}) < MAX_MODEL_LEN ({max_model_len}); "
                    f"overriding to {max_model_len} to satisfy vLLM constraint."
                )
                mnbt = max_model_len
            else:
                mnbt = mnbt_val
        except ValueError:
            print(f"⚠ Invalid MAX_NUM_BATCHED_TOKENS='{env_mnbt}', ignoring.")

    engine_kwargs = dict(
        model=model_path,
        max_model_len=max_model_len,
        gpu_memory_utilization=float(os.getenv("GPU_MEMORY_UTILIZATION", "0.90")),
        max_num_seqs=int(os.getenv("MAX_NUM_SEQS", "8")),
        dtype="auto",
        trust_remote_code=True,
    )
    if mnbt is not None:
        engine_kwargs["max_num_batched_tokens"] = mnbt

    engine_args = AsyncEngineArgs(**engine_kwargs)

    llm_engine = AsyncLLMEngine.from_engine_args(engine_args)
    print(f"✓ vLLM engine initialized with model: {model_path}")
    return llm_engine


# --- Hybrid Router Configuration ---
USE_LLM_ROUTER = os.getenv("USE_LLM_ROUTER", "false").lower() == "true"


# --- Hybrid Router (Phase 1: Heuristics) ---
NON_CAREER_DOMAINS = {
    "cooking": [
        "recipe", "cook", "bake", "cake", "ingredients", "oven", "fry", "sauté", "saute", "boil"
    ],
    "travel": [
        "itinerary", "flight", "visa", "hotel", "travel", "trip", "tourist", "places to visit"
    ],
    "weather": [
        "weather", "forecast", "temperature", "rain", "sunny", "snow", "humidity"
    ],
    "small_talk": [
        "hello", "hi ", "hey ", "hiya", "yo ", "what's up", "whats up", "sup ",
        "thanks", "thank you", "how are you", "how's your day", "hows your day",
        "good morning", "good afternoon", "good evening", "good night"
    ],
    "general_qna": [
        # Generic patterns
        "how to ", "fix ", "make ", "build ", "install ", "create ", "explain ", "setup ", "configure ",
        # Tech/software specific
        "docker", "ubuntu", "linux", "windows", "macos", "mac ",
        "kubernetes", "k8s", "terraform", "nginx", "apache", "postgres", "mysql", "redis",
        "homebrew", "apt ", "yum", "pip ", "npm ", "yarn", "git ", "github", "gitlab",
        "aws ", "azure", "gcp", "cloud", "serverless", "lambda"
    ],
    # HR/Recruiting domains
    "resume_guidance": [
        "resume", "cv ", "curriculum vitae", "write resume", "create resume", "resume tips",
        "resume guidance", "resume help", "improve resume", "resume keywords", "resume bullet"
    ],
    "job_description": [
        "job description", "jd ", "job posting", "job ad", "write job description",
        "create job description", "draft jd", "job requirements"
    ],
    "job_resume_match": [
        "match resume", "match candidate", "resume match", "fit for role", "candidate fit",
        "evaluate resume", "assess candidate", "resume score"
    ],
    "recruiting_strategy": [
        "recruiting", "recruitment", "hiring strategy", "sourcing", "talent acquisition",
        "find candidates", "recruit ", "hiring plan", "candidate sourcing"
    ],
    "ats_keywords": [
        "ats keywords", "ats ", "applicant tracking", "resume keywords", "keyword optimization",
        "resume scanner", "ats system"
    ],
}


def heuristic_route(question: str):
    """Heuristic pre-check for obvious non-career intents.

    Returns (domain: Optional[str], confidence: float)

    Tie-break priority (when confidence equal):
    small_talk > ats_keywords > job_description > resume_guidance >
    job_resume_match > recruiting_strategy > cooking/travel/weather > general_qna
    """
    q = question.lower()
    matches = []

    # Domain priority for tie-breaking (higher = more specific)
    DOMAIN_PRIORITY = {
        "small_talk": 10,
        "ats_keywords": 9,
        "job_description": 8,
        "resume_guidance": 7,
        "job_resume_match": 6,
        "recruiting_strategy": 5,
        "cooking": 4,
        "travel": 4,
        "weather": 4,
        "general_qna": 1,
    }

    for domain, keywords in NON_CAREER_DOMAINS.items():
        for kw in keywords:
            if kw in q:
                # Confidence levels: small_talk > specific domains > general_qna
                if domain == "small_talk":
                    conf = 0.95
                elif domain == "general_qna":
                    conf = 0.8
                else:
                    conf = 0.9

                priority = DOMAIN_PRIORITY.get(domain, 0)
                matches.append((domain, conf, priority))
                break  # One match per domain is enough

    if not matches:
        return None, 0.0

    # Sort by confidence DESC, then priority DESC (tie-breaker)
    best_match = max(matches, key=lambda x: (x[1], x[2]))
    return best_match[0], best_match[1]


# --- Hybrid Router (Phase 2: LLM Router) ---
async def classify_with_llm(question: str) -> dict:
    """Use LLM to classify question intent when heuristics are uncertain.

    Returns: {"intent": str, "confidence": float}

    Routing thresholds:
    - conf >= 0.6: Trust classification
    - 0.4 <= conf < 0.6: Use heuristic hint if present, else general_qna
    - conf < 0.4: Fallback to general_qna
    """
    engine = initialize_engine()

    # Build strict JSON classification prompt
    classification_prompt = f"""<|im_start|>system
You are an intent classifier. Classify the user's question into ONE of these intents:
- career_guidance: Career paths, transitions, skill development
- interview_skills: Interview prep, resume, networking
- salary_intelligence: Salary, compensation, pay ranges
- market_intel: Job market trends, demand, hiring
- resume_guidance: Resume writing, CV tips, resume improvement
- job_description: Job posting creation, JD writing
- job_resume_match: Candidate-job matching, resume evaluation
- recruiting_strategy: Sourcing, hiring channels, recruitment planning
- ats_keywords: ATS optimization, resume keywords
- cooking: Recipes, food preparation
- travel: Travel planning, destinations, logistics
- weather: Weather forecasts, conditions
- small_talk: Greetings, thanks, casual chat
- general_qna: How-to, technical questions, explanations

Respond ONLY with valid JSON: {{"intent": "intent_name", "confidence": 0.0-1.0}}
<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
{{"""

    sampling_params = SamplingParams(
        max_tokens=60,
        temperature=0.0,
        stop=["<|im_end|>"],
    )

    request_id = f"router-{os.urandom(4).hex()}"

    try:
        # Generate classification
        result_generator = engine.generate(classification_prompt, sampling_params, request_id)

        final_output = None
        async for request_output in result_generator:
            if request_output.finished:
                final_output = request_output.outputs[0].text

        if not final_output:
            raise ValueError("LLM router returned no output")

        # Parse JSON response (add closing brace if missing)
        json_text = final_output.strip()
        if not json_text.startswith("{"):
            json_text = "{" + json_text
        if not json_text.endswith("}"):
            json_text = json_text + "}"

        result = json.loads(json_text)
        intent = result.get("intent", "general_qna")
        confidence = float(result.get("confidence", 0.5))

        return {"intent": intent, "confidence": confidence}

    except (json.JSONDecodeError, ValueError, KeyError) as e:
        # Parse-safe fallback
        print(f"⚠ LLM router parse error: {e}, falling back to general_qna")
        return {"intent": "general_qna", "confidence": 0.3}


async def generate_streaming(prompt: str, sampling_params: SamplingParams) -> Generator[Dict, None, None]:
    """Generate tokens with streaming - emits only new deltas"""
    engine = initialize_engine()
    request_id = f"req-{os.urandom(8).hex()}"

    # Start generation
    results_generator = engine.generate(prompt, sampling_params, request_id)

    prev_len = 0
    deltas = []
    async for request_output in results_generator:
        if request_output.finished:
            # Final output
            full_text = request_output.outputs[0].text
            final_delta = full_text[prev_len:]  # Any remaining text
            if final_delta:
                deltas.append(final_delta)

            # Normalize usage keys (consistent with non-streaming)
            usage_input = len(request_output.prompt_token_ids)
            usage_output = len(request_output.outputs[0].token_ids)
            yield {
                "delta": final_delta,
                "text": full_text,  # Full text for convenience
                "deltas": deltas,  # All deltas for debugging
                "finished": True,
                "usage": {
                    "input": usage_input,
                    "output": usage_output,
                    "total": usage_input + usage_output,
                }
            }
        else:
            # Streaming delta - only new text
            full_text = request_output.outputs[0].text
            delta = full_text[prev_len:]
            prev_len = len(full_text)
            deltas.append(delta)

            yield {
                "delta": delta,
                "finished": False,
                "offset": prev_len - len(delta)  # Starting position of this delta
            }


async def generate_non_streaming(prompt: str, sampling_params: SamplingParams) -> Dict:
    """Generate without streaming (batch mode)"""
    engine = initialize_engine()
    request_id = f"req-{os.urandom(8).hex()}"

    # Generate and wait for completion
    final_output = None
    async for request_output in engine.generate(prompt, sampling_params, request_id):
        if request_output.finished:
            final_output = request_output

    if final_output is None:
        raise RuntimeError("Generation failed to complete")

    text = final_output.outputs[0].text
    usage_input = len(final_output.prompt_token_ids)
    usage_output = len(final_output.outputs[0].token_ids)
    return {
        "choices": [{
            "text": text,
            "tokens": [text]
        }],
        "usage": {
            "input": usage_input,
            "output": usage_output,
            "total": usage_input + usage_output,
        }
    }


async def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """Main async handler for RunPod with V3 validation"""
    # Generate request ID for tracking
    request_id = str(uuid.uuid4())[:8]
    request_start = time.time()

    try:
        job_input = job.get("input", {})

        # Extract parameters
        user_question = job_input.get("prompt", "")
        if not user_question:
            return {"error": "No prompt provided"}

        # Guardrail: Prompt length limit (prevents KV cache bloat)
        if len(user_question) > 1024:
            return {"error": "Prompt too long (max 1024 chars)"}

        sampling_config = job_input.get("sampling_params", {})
        stream = sampling_config.get("stream", False)
        enable_validation = job_input.get("enable_validation", USE_V3_VALIDATION)
        block_low_trust = job_input.get("block_low_trust_intents", True)

        # Guardrail: Clamp max_tokens for chat (prevents abuse, maintains quality)
        # Per-domain defaults to match prompt length requirements
        domain_max_tokens = {
            "job_description": 230,
            "resume_guidance": 200,
            "job_resume_match": 200,
            "recruiting_strategy": 180,
            "ats_keywords": 150,
        }

        # Get user's max_tokens for logging (don't force default here - domain logic handles it)
        max_tokens_for_logging = sampling_config.get("max_tokens", 150)
        max_tokens_for_logging = max(64, min(256, max_tokens_for_logging))

        # Log request start
        log_json("start", request_id,
                 prompt_len=len(user_question),
                 max_tokens=max_tokens_for_logging,
                 stream=stream,
                 validation_enabled=enable_validation)

        # Hybrid routing: Heuristic pre-check for non-career domains
        routed_domain = None

        # Market intel routing veto: Always send to V3 validation for blocking
        MARKET_VETO_KEYWORDS = [
            "which industries", "hiring trends", "in-demand", "job market",
            "market trends", "demand for", "industries hiring", "tech trends",
            "job trends", "employment trends", "market analysis"
        ]
        has_market_veto = any(kw in user_question.lower() for kw in MARKET_VETO_KEYWORDS)

        if enable_validation and not has_market_veto:  # Only route when validation path is considered and not market intel
            h_domain, h_conf = heuristic_route(user_question)
            if h_domain and h_conf >= 0.80:
                routed_domain = h_domain
                prompt = PromptBuilder.build_domain_prompt(h_domain, user_question)
                # Disable career validation for non-career domains
                enable_validation = False
                log_json("route", request_id, router="heuristic", router_intent=h_domain, router_conf=h_conf)
            elif USE_LLM_ROUTER and h_conf < 0.80:
                # Phase 2: LLM router for uncertain cases
                llm_result = await classify_with_llm(user_question)
                llm_intent = llm_result["intent"]
                llm_conf = llm_result["confidence"]

                log_json("route", request_id, router="llm", router_intent=llm_intent, router_conf=llm_conf)

                # Routing thresholds
                if llm_conf >= 0.6:
                    # Trust LLM classification
                    non_career_intents = ["cooking", "travel", "weather", "small_talk", "general_qna",
                                         "resume_guidance", "job_description", "job_resume_match",
                                         "recruiting_strategy", "ats_keywords"]
                    if llm_intent in non_career_intents:
                        routed_domain = llm_intent
                        prompt = PromptBuilder.build_domain_prompt(llm_intent, user_question)
                        enable_validation = False
                    # else: let career intents fall through to V3 validation
                elif llm_conf >= 0.4:
                    # Use heuristic hint if present, else general_qna
                    if h_domain:
                        routed_domain = h_domain
                        prompt = PromptBuilder.build_domain_prompt(h_domain, user_question)
                        enable_validation = False
                    else:
                        routed_domain = "general_qna"
                        prompt = PromptBuilder.build_domain_prompt("general_qna", user_question)
                        enable_validation = False
                else:
                    # Low confidence: default to general_qna
                    routed_domain = "general_qna"
                    prompt = PromptBuilder.build_domain_prompt("general_qna", user_question)
                    enable_validation = False

            # If not routed by heuristics or LLM, proceed with V3 validation
            if routed_domain is None:
                # V3: Classify intent and block if needed
                intent = None
                if USE_V3_VALIDATION and enable_validation:
                    intent = IntentClassifier.classify(user_question)
                    print(f"Intent classified: {intent.value}")

                    # Block salary/market queries (low validation rates)
                    if block_low_trust and intent in [QuestionIntent.SALARY_INTEL, QuestionIntent.MARKET_INTEL]:
                        log_json("end", request_id,
                                 ok=False,
                                 exec_ms=int((time.time() - request_start) * 1000),
                                 intent=intent.value,
                                 blocked=True)
                        return {
                            "blocked": True,
                            "intent": intent.value,
                            "message": (
                                "This question requires real-time compensation/market data. "
                                "We're integrating with trusted data sources (Levels.fyi, BLS.gov) "
                                "to provide accurate, up-to-date information. "
                                "In the meantime, try asking about career transitions, skill development, "
                                "interview preparation, or learning paths. "
                                "Expected availability: 2-4 weeks."
                            )
                        }

                    # Build proper prompt with template
                    prompt = PromptBuilder.build_prompt(user_question, intent)
                    print(f"✓ Prompt formatted with {intent.value} template")
                else:
                    # No validation - use raw prompt
                    prompt = user_question
        else:
            # Validation disabled up-front
            prompt = user_question

        # Phase 4: Domain-specific decoding parameters
        # Map defaults per domain (can be overridden by sampling_config)
        # Use conservative, uniform stops; strip artifacts in sanitizer instead
        if routed_domain == "small_talk":
            domain_defaults = {
                "temperature": 0.4,
                "stop": ["<|im_end|>"],
                "repetition_penalty": 1.0,
                "frequency_penalty": 0.0,
            }
        elif routed_domain in ["resume_guidance", "job_description", "job_resume_match",
                               "recruiting_strategy", "ats_keywords"]:
            # HR domains: Relaxed penalties to avoid terse/early stopping
            domain_defaults = {
                "temperature": 0.3,
                "repetition_penalty": 1.0,
                "frequency_penalty": 0.0,
                "stop": ["<|im_end|>"],
            }
        elif routed_domain in ["cooking", "general_qna", "travel", "weather"]:
            domain_defaults = {
                "temperature": 0.3,
                "repetition_penalty": 1.15,
                "frequency_penalty": 0.2,
                "stop": ["<|im_end|>"],
            }
        else:
            # Career/interview defaults
            domain_defaults = {
                "temperature": 0.3,
                "repetition_penalty": 1.15,
                "frequency_penalty": 0.2,
                "stop": ["<|im_end|>"],
            }

        # Apply per-domain max_tokens if not explicitly provided by user
        effective_max_tokens = sampling_config.get("max_tokens")
        if effective_max_tokens is None and routed_domain in domain_max_tokens:
            effective_max_tokens = domain_max_tokens[routed_domain]
        elif effective_max_tokens is None:
            effective_max_tokens = 150  # Global default

        # Clamp to valid range [64, 256]
        effective_max_tokens = max(64, min(256, effective_max_tokens))

        # Build sampling parameters (user config overrides domain defaults)
        sampling_params = SamplingParams(
            max_tokens=effective_max_tokens,
            temperature=sampling_config.get("temperature", domain_defaults.get("temperature", 0.3)),
            top_p=sampling_config.get("top_p", 0.9),
            top_k=sampling_config.get("top_k", 50),
            presence_penalty=sampling_config.get("presence_penalty", domain_defaults.get("presence_penalty", 0.0)),
            frequency_penalty=sampling_config.get("frequency_penalty", domain_defaults.get("frequency_penalty", 0.2)),
            repetition_penalty=sampling_config.get("repetition_penalty", domain_defaults.get("repetition_penalty", 1.15)),
            stop=sampling_config.get("stop", domain_defaults.get("stop", ["<|im_end|>"])),
        )

        # Generate response
        deltas = []  # Initialize for both streaming and non-streaming
        usage = None  # Initialize for both paths

        if stream:
            # Streaming mode - aggregate all deltas
            full_text = ""

            async for chunk in generate_streaming(prompt, sampling_params):
                if chunk.get("finished"):
                    full_text = chunk["text"]
                    usage = chunk["usage"]
                    deltas = chunk.get("deltas", deltas)  # Use server-side deltas
                else:
                    deltas.append(chunk["delta"])

            result_text = full_text
        else:
            # Non-streaming mode
            result = await generate_non_streaming(prompt, sampling_params)
            result_text = result["choices"][0]["text"]
            usage = result["usage"]

        # V3: Auto-sanitize and validate
        if USE_V3_VALIDATION and enable_validation:
            # Sanitize
            sanitized, status = AutoSanitizer.sanitize(result_text)

            if status == "EMPTY":
                # Over-sanitized - use original text instead of regenerating
                print("⚠ Over-sanitization detected - using original response")
                sanitized = result_text
                status = "OK"

            # Validate according to intent
            if intent == QuestionIntent.SALARY_INTEL:
                is_valid, issues = ResponseValidator.validate_salary_response(user_question, sanitized)
            elif intent == QuestionIntent.MARKET_INTEL:
                # Validate market responses for industries and rationale
                is_valid, issues = ResponseValidator.validate_market_response(sanitized)
            else:
                is_valid, issues = ResponseValidator.validate_career_response(sanitized)

            # DISABLED: Regeneration causes 30x throughput drop (1.3 t/s vs 47.8 t/s)
            # Log validation failures but accept response to maintain performance
            if not is_valid:
                print(f"⚠ Validation failed: {issues} - accepting anyway (regeneration disabled for performance)")

            # Calculate metrics
            exec_ms = int((time.time() - request_start) * 1000)
            tokens_out = usage.get("output", 0) if usage else len(sanitized.split())

            # Log request end
            log_json("end", request_id,
                     ok=True,
                     exec_ms=exec_ms,
                     tokens_generated=tokens_out,
                     intent=intent.value if intent else "unknown",
                     valid=is_valid)

            return {
                "choices": [{
                    "text": sanitized,
                    "deltas": deltas if stream else None,  # Token deltas (only new text per chunk)
                    "tokens": deltas if stream else [sanitized]  # Backward compatibility alias
                }],
                "usage": usage or {"input": 0, "output": len(sanitized.split()), "total": len(sanitized.split())},
                "validation": {
                    "valid": is_valid,
                    "issues": issues if not is_valid else [],
                    "sanitized": sanitized != result_text,
                    "intent": intent.value if intent else "unknown"
                },
                "streaming": stream  # Indicate if response was streamed
            }
        else:
            # No V3 validation - routed domains (may have HR validation)
            exec_ms = int((time.time() - request_start) * 1000)

            # Sanitize for all routed domains
            final_text = result_text
            sanitized_flag = False
            try:
                if routed_domain is not None and 'AutoSanitizer' in globals():
                    s_text, _s_status = AutoSanitizer.sanitize(result_text)
                    if s_text:
                        final_text = s_text
                        sanitized_flag = (s_text != result_text)
                    else:
                        # Empty after sanitization, use original to avoid blank response
                        final_text = result_text
            except Exception:
                pass

            # HR domain validation
            HR_VALIDATORS = {
                "resume_guidance": ResponseValidator.validate_resume_guidance,
                "job_description": ResponseValidator.validate_job_description,
                "job_resume_match": ResponseValidator.validate_job_resume_match,
                "recruiting_strategy": ResponseValidator.validate_recruiting_strategy,
                "ats_keywords": ResponseValidator.validate_ats_keywords,
            }

            is_valid = True
            issues = []
            has_validation = False

            if routed_domain in HR_VALIDATORS:
                validator = HR_VALIDATORS[routed_domain]
                is_valid, issues = validator(final_text)
                has_validation = True

            tokens_out = usage.get("output", 0) if usage else len(final_text.split())

            # Log request end
            log_json("end", request_id,
                     ok=True,
                     exec_ms=exec_ms,
                     tokens_generated=tokens_out,
                     intent=routed_domain if routed_domain else "none",
                     valid=is_valid)

            response = {
                "choices": [{
                    "text": final_text,
                    "deltas": deltas if stream else None,
                    "tokens": deltas if stream else [final_text]
                }],
                "usage": usage or {"input": 0, "output": len(final_text.split()), "total": len(final_text.split())},
                "streaming": stream
            }

            # Add validation block for HR domains
            if has_validation:
                response["validation"] = {
                    "valid": is_valid,
                    "issues": issues if not is_valid else [],
                    "sanitized": sanitized_flag,
                    "intent": routed_domain
                }

            return response

    except Exception as e:
        exec_ms = int((time.time() - request_start) * 1000)
        log_json("end", request_id,
                 level="error",
                 ok=False,
                 exec_ms=exec_ms,
                 error=str(e))
        print(f"Error in handler: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


# RunPod serverless start
if __name__ == "__main__":
    print("Starting RunPod Serverless Handler...")
    print("Initializing vLLM engine...")
    initialize_engine()
    print("✓ Handler ready")
    runpod.serverless.start({"handler": handler})
