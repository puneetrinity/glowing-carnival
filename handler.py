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
from vllm import LLM, SamplingParams
from typing import Generator, Dict, Any
import asyncio
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
import json
import sys

# Import V3 validation logic
sys.path.append('/home/ews/llm')
try:
    from career_guidance_v3 import (
        QuestionIntent,
        classify_intent,
        validate_response,
        AutoSanitizer,
        improved_prompts
    )
    USE_V3_VALIDATION = True
    print("✓ V3 validation imported")
except ImportError as e:
    print(f"⚠ V3 validation not available: {e}")
    USE_V3_VALIDATION = False

# Global vLLM engine instance
llm_engine = None

def initialize_engine():
    """Initialize vLLM engine with optimized settings"""
    global llm_engine

    if llm_engine is not None:
        return llm_engine

    model_path = os.getenv("MODEL_PATH", "/models/qwen2-7b-career")

    # vLLM AsyncEngine configuration
    engine_args = AsyncEngineArgs(
        model=model_path,
        max_model_len=int(os.getenv("MAX_MODEL_LEN", "4096")),
        gpu_memory_utilization=float(os.getenv("GPU_MEMORY_UTILIZATION", "0.90")),
        max_num_seqs=int(os.getenv("MAX_NUM_SEQS", "8")),
        max_num_batched_tokens=2048,
        enable_chunked_prefill=True,
        dtype="auto",
        trust_remote_code=True,
        disable_log_requests=True,
    )

    llm_engine = AsyncLLMEngine.from_engine_args(engine_args)
    print(f"✓ vLLM engine initialized with model: {model_path}")
    return llm_engine


async def generate_streaming(prompt: str, sampling_params: SamplingParams) -> Generator[Dict, None, None]:
    """Generate tokens with streaming"""
    engine = initialize_engine()
    request_id = f"req-{os.urandom(8).hex()}"

    # Start generation
    results_generator = engine.generate(prompt, sampling_params, request_id)

    tokens = []
    async for request_output in results_generator:
        if request_output.finished:
            # Final output
            text = request_output.outputs[0].text
            yield {
                "text": text,
                "tokens": tokens,
                "finished": True,
                "usage": {
                    "prompt_tokens": len(request_output.prompt_token_ids),
                    "completion_tokens": len(request_output.outputs[0].token_ids),
                    "total_tokens": len(request_output.prompt_token_ids) + len(request_output.outputs[0].token_ids)
                }
            }
        else:
            # Streaming token
            new_text = request_output.outputs[0].text
            tokens.append(new_text)
            yield {
                "text": new_text,
                "finished": False
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
    return {
        "choices": [{
            "text": text,
            "tokens": [text]
        }],
        "usage": {
            "input": len(final_output.prompt_token_ids),
            "output": len(final_output.outputs[0].token_ids)
        }
    }


async def async_handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """Main async handler for RunPod with V3 validation"""
    try:
        job_input = job.get("input", {})

        # Extract parameters
        prompt = job_input.get("prompt", "")
        if not prompt:
            return {"error": "No prompt provided"}

        sampling_config = job_input.get("sampling_params", {})
        stream = sampling_config.get("stream", False)
        enable_validation = job_input.get("enable_validation", USE_V3_VALIDATION)

        # V3: Classify intent and block if needed
        if USE_V3_VALIDATION and enable_validation:
            intent = classify_intent(prompt)
            print(f"Intent classified: {intent.value}")

            # Block salary/market queries (low validation rates)
            if intent in [QuestionIntent.SALARY_INTEL, QuestionIntent.MARKET_INTEL]:
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
        if stream:
            # Streaming mode - aggregate all tokens
            tokens = []
            full_text = ""
            usage = None

            async for chunk in generate_streaming(prompt, sampling_params):
                if chunk.get("finished"):
                    full_text = chunk["text"]
                    usage = chunk["usage"]
                else:
                    tokens.append(chunk["text"])

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
                # Over-sanitized - regenerate with stricter params
                print("⚠ Over-sanitization detected - regenerating")
                sampling_params.temperature = 0.5
                sampling_params.max_tokens = 120
                result = await generate_non_streaming(prompt, sampling_params)
                result_text = result["choices"][0]["text"]
                sanitized, status = AutoSanitizer.sanitize(result_text)

            # Validate
            is_valid, issues = validate_response(sanitized, intent, prompt)

            if not is_valid and sampling_config.get("allow_regeneration", True):
                # Try once more with stricter parameters
                print(f"⚠ Validation failed: {issues} - regenerating")
                sampling_params.temperature = 0.5
                sampling_params.max_tokens = 120
                result = await generate_non_streaming(prompt, sampling_params)
                result_text = result["choices"][0]["text"]
                sanitized, status = AutoSanitizer.sanitize(result_text)
                is_valid, issues = validate_response(sanitized, intent, prompt)

            return {
                "choices": [{
                    "text": sanitized,
                    "tokens": tokens if stream else [sanitized]
                }],
                "usage": usage or {"input": 0, "output": len(sanitized.split())},
                "validation": {
                    "valid": is_valid,
                    "issues": issues if not is_valid else [],
                    "sanitized": sanitized != result_text,
                    "intent": intent.value
                }
            }
        else:
            # No validation - return raw response
            return {
                "choices": [{
                    "text": result_text,
                    "tokens": tokens if stream else [result_text]
                }],
                "usage": usage or {"input": 0, "output": len(result_text.split())}
            }

    except Exception as e:
        print(f"Error in handler: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """Synchronous wrapper for RunPod"""
    return asyncio.run(async_handler(job))


# RunPod serverless start
if __name__ == "__main__":
    print("Starting RunPod Serverless Handler...")
    print("Initializing vLLM engine...")
    initialize_engine()
    print("✓ Handler ready")
    runpod.serverless.start({"handler": handler})
