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
        max_tokens = sampling_config.get("max_tokens", 150)
        max_tokens = max(64, min(256, max_tokens))  # Clamp to [64, 256]
        sampling_config["max_tokens"] = max_tokens

        # Log request start
        log_json("start", request_id,
                 prompt_len=len(user_question),
                 max_tokens=max_tokens,
                 stream=stream,
                 validation_enabled=enable_validation)

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

        # Build sampling parameters
        sampling_params = SamplingParams(
            max_tokens=sampling_config.get("max_tokens", 150),
            temperature=sampling_config.get("temperature", 0.7),
            top_p=sampling_config.get("top_p", 0.9),
            top_k=sampling_config.get("top_k", 50),
            presence_penalty=sampling_config.get("presence_penalty", 0.0),
            frequency_penalty=sampling_config.get("frequency_penalty", 0.0),
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
                    "intent": intent.value
                },
                "streaming": stream  # Indicate if response was streamed
            }
        else:
            # No validation - return raw response
            exec_ms = int((time.time() - request_start) * 1000)
            tokens_out = usage.get("output", 0) if usage else len(result_text.split())

            # Log request end
            log_json("end", request_id,
                     ok=True,
                     exec_ms=exec_ms,
                     tokens_generated=tokens_out,
                     intent="none",
                     valid=True)

            return {
                "choices": [{
                    "text": result_text,
                    "deltas": deltas if stream else None,
                    "tokens": deltas if stream else [result_text]
                }],
                "usage": usage or {"input": 0, "output": len(result_text.split()), "total": len(result_text.split())},
                "streaming": stream
            }

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
