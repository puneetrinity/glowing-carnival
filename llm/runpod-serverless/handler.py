"""
RunPod Serverless Handler for vLLM with Enhanced Features
Combines:
- Career Model V3 optimizations (87% validation rate)
- Domain-aware task routing (ATS, Resume, Job Description, etc.)
- Structured output validation with auto-retry
- OpenAI-compatible messages format
- Profanity filtering and safety checks
- Streaming support with async vLLM engine

Version: 2.0.0
"""

import os
import runpod
from vllm import SamplingParams
from typing import Generator, Dict, Any, List, Optional, Tuple, Union
import asyncio
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
import sys
import json
import uuid
import time
import atexit
import signal
import re
import logging
from dataclasses import dataclass
from enum import Enum
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

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global vLLM engine instance
llm_engine = None

# Task types for domain-aware routing
class TaskType(str, Enum):
    # HR/Recruiting specific tasks
    ATS_KEYWORDS = "ats_keywords"
    RESUME_BULLETS = "resume_bullets"
    JOB_DESCRIPTION = "job_description"
    JOB_RESUME_MATCH = "job_resume_match"
    RECRUITING_STRATEGY = "recruiting_strategy"
    # Career guidance (from V3)
    CAREER_GUIDANCE = "career_guidance"
    # General chat types
    GENERIC_CHAT = "generic_chat"
    SMALL_TALK = "small_talk"
    LIFE_ADVICE = "life_advice"
    OTHER = "other"

# Task-specific configuration
@dataclass
class TaskConfig:
    system_prompt: str
    expected_format: str  # "json" or "text"
    temperature: float
    max_tokens: int
    top_p: float
    requires_disclaimer: bool = False
    validator_fn: Optional[callable] = None
    use_v3_validation: bool = False

# Profanity/safety filter patterns
PROFANITY_PATTERNS = [
    r'\b(fuck|shit|damn|bitch|ass|crap|piss|bastard|hell)\b',
    r'\b(nigga|nigger|faggot|retard|cunt)\b',  # slurs
]

DISCLAIMER_PATTERN = r'(not a licensed professional|not a therapist|not medical advice|seek professional help|consult a professional)'

# Configuration settings
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "1"))

# Graceful shutdown handler
def _shutdown_handler(signum=None, frame=None):
    """Gracefully shutdown vLLM engine to prevent NCCL teardown warnings"""
    global llm_engine
    if llm_engine is not None:
        try:
            print("🔄 Gracefully shutting down vLLM engine...")
            llm_engine = None
            print("✓ Engine shutdown initiated")
        except Exception as e:
            print(f"⚠ Error during shutdown: {e}")

# Register shutdown handlers
atexit.register(_shutdown_handler)
signal.signal(signal.SIGTERM, _shutdown_handler)
signal.signal(signal.SIGINT, _shutdown_handler)

# Validation functions for structured outputs
def validate_ats_keywords(data: Any) -> Tuple[bool, Optional[str]]:
    """Validate ATS keywords response"""
    if not isinstance(data, list):
        return False, "Expected array of strings"
    if len(data) != 25:
        return False, f"Expected exactly 25 keywords, got {len(data)}"
    if not all(isinstance(kw, str) and kw.islower() for kw in data):
        return False, "All keywords must be lowercase strings"
    return True, None

def validate_resume_bullets(data: Any) -> Tuple[bool, Optional[str]]:
    """Validate resume bullets response"""
    if not isinstance(data, list):
        return False, "Expected array of strings"
    if len(data) != 5:
        return False, f"Expected exactly 5 bullets, got {len(data)}"
    for bullet in data:
        if not isinstance(bullet, str):
            return False, "All bullets must be strings"
        words = bullet.split()
        if not (12 <= len(words) <= 20):
            return False, f"Each bullet must be 12-20 words, got {len(words)}"
    return True, None

def validate_job_description(data: Any) -> Tuple[bool, Optional[str]]:
    """Validate job description response"""
    if not isinstance(data, dict):
        return False, "Expected JSON object"
    required_keys = {"summary", "responsibilities", "requirements"}
    if not required_keys.issubset(data.keys()):
        return False, f"Missing required keys: {required_keys - set(data.keys())}"
    if not isinstance(data["responsibilities"], list) or len(data["responsibilities"]) < 3:
        return False, "responsibilities must be array with at least 3 items"
    if not isinstance(data["requirements"], list) or len(data["requirements"]) < 3:
        return False, "requirements must be array with at least 3 items"
    return True, None

def validate_job_resume_match(data: Any) -> Tuple[bool, Optional[str]]:
    """Validate job/resume match response"""
    if not isinstance(data, dict):
        return False, "Expected JSON object"
    required_keys = {"score", "matches", "gaps", "next_steps"}
    if not required_keys.issubset(data.keys()):
        return False, f"Missing required keys: {required_keys - set(data.keys())}"
    if not isinstance(data["score"], (int, float)) or not (0 <= data["score"] <= 100):
        return False, "score must be number between 0-100"
    for key in ["matches", "gaps", "next_steps"]:
        if not isinstance(data[key], list):
            return False, f"{key} must be an array"
    return True, None

def validate_recruiting_strategy(data: Any) -> Tuple[bool, Optional[str]]:
    """Validate recruiting strategy response"""
    if not isinstance(data, dict):
        return False, "Expected JSON object"
    required_keys = {"channels", "cadence", "metrics"}
    if not required_keys.issubset(data.keys()):
        return False, f"Missing required keys: {required_keys - set(data.keys())}"
    if not isinstance(data["channels"], list) or len(data["channels"]) < 4:
        return False, "channels must be array with at least 4 items"
    if not isinstance(data["metrics"], list) or len(data["metrics"]) < 3:
        return False, "metrics must be array with at least 3 items"
    if not isinstance(data["cadence"], str):
        return False, "cadence must be a string"
    return True, None

# Task configurations
TASK_CONFIGS: Dict[TaskType, TaskConfig] = {
    TaskType.ATS_KEYWORDS: TaskConfig(
        system_prompt="You are an ATS keyword extraction expert. Return a JSON array of exactly 25 lowercase keywords (strings only). No code fences, no markdown, no extra text. Just the raw JSON array.",
        expected_format="json",
        temperature=0.2,
        max_tokens=200,
        top_p=0.9,
        validator_fn=validate_ats_keywords
    ),
    TaskType.RESUME_BULLETS: TaskConfig(
        system_prompt="You are a professional resume writer. Return a JSON array of exactly 5 strings. Each string must be 12-20 words and start with a strong action verb. No code fences, no markdown. Just the raw JSON array.",
        expected_format="json",
        temperature=0.25,
        max_tokens=300,
        top_p=0.9,
        validator_fn=validate_resume_bullets
    ),
    TaskType.JOB_DESCRIPTION: TaskConfig(
        system_prompt="You are a job description expert. Return a JSON object with exactly these keys: summary (string), responsibilities (array of at least 3 strings), requirements (array of at least 3 strings). No extra keys, no code fences, no markdown. Just the raw JSON object.",
        expected_format="json",
        temperature=0.3,
        max_tokens=500,
        top_p=0.9,
        validator_fn=validate_job_description
    ),
    TaskType.JOB_RESUME_MATCH: TaskConfig(
        system_prompt="You are a hiring expert analyzing job-resume fit. Return a JSON object with exactly these keys: score (number 0-100), matches (array of strings), gaps (array of strings), next_steps (array of strings). No code fences, no markdown. Just the raw JSON object.",
        expected_format="json",
        temperature=0.25,
        max_tokens=400,
        top_p=0.9,
        validator_fn=validate_job_resume_match
    ),
    TaskType.RECRUITING_STRATEGY: TaskConfig(
        system_prompt="You are a recruiting strategist. Return a JSON object with exactly these keys: channels (array of at least 4 strings), cadence (string mentioning week/month/quarter/daily), metrics (array of at least 3 strings). No code fences, no markdown. Just the raw JSON object.",
        expected_format="json",
        temperature=0.3,
        max_tokens=400,
        top_p=0.9,
        validator_fn=validate_recruiting_strategy
    ),
    TaskType.CAREER_GUIDANCE: TaskConfig(
        system_prompt="You are a career guidance expert providing professional advice to help individuals advance their careers. Focus on practical, actionable guidance.",
        expected_format="text",
        temperature=0.5,
        max_tokens=700,
        top_p=0.9,
        use_v3_validation=True
    ),
    TaskType.LIFE_ADVICE: TaskConfig(
        system_prompt="You are an empathetic assistant providing life advice. Use a warm, supportive tone. IMPORTANT: You must include this disclaimer in your response: 'I am not a licensed professional. For serious concerns, please consult a qualified therapist or counselor.' Be helpful but safe.",
        expected_format="text",
        temperature=0.7,
        max_tokens=300,
        top_p=0.9,
        requires_disclaimer=True
    ),
    TaskType.SMALL_TALK: TaskConfig(
        system_prompt="You are a friendly, family-friendly assistant. Respond in 1-2 brief sentences. No profanity, no slurs, no inappropriate content. Be warm and helpful.",
        expected_format="text",
        temperature=0.7,
        max_tokens=80,
        top_p=0.9
    ),
    TaskType.GENERIC_CHAT: TaskConfig(
        system_prompt="You are a helpful, family-friendly assistant. Provide clear, accurate information. No profanity, no slurs, no inappropriate content. Be professional and helpful.",
        expected_format="text",
        temperature=0.6,
        max_tokens=500,
        top_p=0.9
    ),
    TaskType.OTHER: TaskConfig(
        system_prompt="You are a helpful, professional assistant. Provide accurate, family-friendly responses. No profanity or inappropriate content.",
        expected_format="text",
        temperature=0.6,
        max_tokens=500,
        top_p=0.9
    )
}

# Helper functions
def check_profanity(text: str) -> bool:
    """Check if text contains profanity or slurs"""
    text_lower = text.lower()
    for pattern in PROFANITY_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return True
    return False

def check_disclaimer(text: str) -> bool:
    """Check if text contains required disclaimer"""
    return bool(re.search(DISCLAIMER_PATTERN, text, re.IGNORECASE))

def strip_code_fences(text: str) -> str:
    """Remove markdown code fences from text"""
    # Pattern 1: Full fenced block
    fenced_pattern = r'^```(?:json|JSON)?\s*\n(.*?)\n```\s*$'
    match = re.search(fenced_pattern, text, flags=re.DOTALL | re.MULTILINE)
    if match:
        text = match.group(1)
    else:
        # Pattern 2: Leading fence
        text = re.sub(r'^```(?:json|JSON)?\s*\n', '', text, flags=re.MULTILINE)
        # Pattern 3: Trailing fence
        text = re.sub(r'\n```\s*$', '', text, flags=re.MULTILINE)
        # Pattern 4: Inline fences
        text = re.sub(r'```(?:json|JSON)?\s*', '', text)
    
    return text.strip()

def parse_and_validate_json(text: str, task_config: TaskConfig) -> Tuple[bool, Any, Optional[str]]:
    """Parse JSON and validate against task schema"""
    clean_text = strip_code_fences(text)
    
    try:
        data = json.loads(clean_text)
    except json.JSONDecodeError as e:
        return False, None, f"JSON parse error: {str(e)}"
    
    # Validate against task-specific validator
    if task_config.validator_fn:
        is_valid, error_msg = task_config.validator_fn(data)
        if not is_valid:
            return False, data, error_msg
    
    return True, data, None

def build_corrective_prompt(original_prompt: str, error_msg: str, task_config: TaskConfig) -> str:
    """Build a corrective system prompt for retry"""
    correction = f"{task_config.system_prompt}\n\nIMPORTANT: Previous attempt failed with error: {error_msg}\n"
    
    if "exactly" in task_config.system_prompt.lower():
        correction += "Ensure you return EXACTLY the requested number of items. "
    
    correction += "Return ONLY raw JSON with no code fences, no markdown, no extra text before or after."
    
    return correction

def detect_task_from_content(content: str) -> TaskType:
    """Auto-detect task type from user message content"""
    content_lower = content.lower()
    
    # HR/Recruiting task detection
    if any(kw in content_lower for kw in ["ats", "keyword", "applicant tracking"]):
        return TaskType.ATS_KEYWORDS
    elif any(kw in content_lower for kw in ["resume bullet", "achievement", "accomplishment"]):
        return TaskType.RESUME_BULLETS
    elif any(kw in content_lower for kw in ["job description", "job posting", "jd"]):
        return TaskType.JOB_DESCRIPTION
    elif any(kw in content_lower for kw in ["job fit", "resume match", "candidate fit"]):
        return TaskType.JOB_RESUME_MATCH
    elif any(kw in content_lower for kw in ["recruiting", "sourcing", "hiring strategy"]):
        return TaskType.RECRUITING_STRATEGY
    
    # Career guidance detection (V3 intents)
    elif any(kw in content_lower for kw in ["career", "salary", "skill", "interview", "promotion"]):
        return TaskType.CAREER_GUIDANCE
    
    # Life advice detection
    elif any(kw in content_lower for kw in ["life advice", "personal", "relationship", "mental health"]):
        return TaskType.LIFE_ADVICE
    
    # Small talk detection
    elif any(kw in content_lower for kw in ["joke", "weather", "hello", "hi there"]):
        return TaskType.SMALL_TALK
    
    return TaskType.GENERIC_CHAT

# Structured logging functions
def log_json(event: str, request_id: str, **kwargs):
    """Emit structured JSON log"""
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
    
    model_path = os.getenv("MODEL_PATH", "/models/Puneetrinity/qwen2.5-7b-careerv2")
    
    # vLLM AsyncEngine configuration
    max_model_len = int(os.getenv("MAX_MODEL_LEN", "4096"))
    env_mnbt = os.getenv("MAX_NUM_BATCHED_TOKENS")
    mnbt = None
    if env_mnbt is not None and env_mnbt != "":
        mnbt = int(env_mnbt)
    else:
        mnbt = min(max_model_len, 2048)
    
    engine_args = AsyncEngineArgs(
        model=model_path,
        tokenizer=model_path,
        trust_remote_code=True,
        max_model_len=max_model_len,
        max_num_batched_tokens=mnbt,
        max_num_seqs=int(os.getenv("MAX_NUM_SEQS", "8")),
        gpu_memory_utilization=float(os.getenv("GPU_MEMORY_UTILIZATION", "0.90")),
        dtype="auto",
        disable_log_requests=True,
        enable_chunked_prefill=True,
        max_num_on_the_fly=16,
    )
    
    print(f"🚀 Initializing vLLM AsyncEngine...")
    print(f"   Model: {model_path}")
    print(f"   Max model length: {max_model_len}")
    print(f"   Max batched tokens: {mnbt}")
    
    llm_engine = AsyncLLMEngine.from_engine_args(engine_args)
    print("✓ vLLM engine initialized")
    
    return llm_engine

async def generate_with_engine(prompt: str, sampling_params: SamplingParams, request_id: str) -> str:
    """Generate text using vLLM engine"""
    engine = initialize_engine()
    
    # Generate response
    request_output = None
    async for output in engine.generate(prompt, sampling_params, request_id):
        request_output = output
    
    if request_output is None:
        raise RuntimeError("No output generated")
    
    return request_output.outputs[0].text

async def stream_with_engine(prompt: str, sampling_params: SamplingParams, request_id: str) -> AsyncGenerator:
    """Stream text using vLLM engine"""
    engine = initialize_engine()
    
    async for output in engine.generate(prompt, sampling_params, request_id):
        if output.outputs:
            text = output.outputs[0].text
            yield text

def convert_messages_to_prompt(messages: List[Dict[str, str]]) -> str:
    """Convert OpenAI messages format to prompt string"""
    prompt_parts = []
    
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        
        if role == "system":
            prompt_parts.append(f"System: {content}")
        elif role == "user":
            prompt_parts.append(f"Human: {content}")
        elif role == "assistant":
            prompt_parts.append(f"Assistant: {content}")
    
    # Add final assistant prompt
    prompt_parts.append("Assistant:")
    
    return "\n\n".join(prompt_parts)

def handler(job):
    """
    Enhanced RunPod handler with task routing, structured outputs, and V3 validation
    
    Supports multiple input formats:
    
    1. Legacy format (backward compatible):
    {
        "input": {
            "prompt": "Your prompt here",
            "sampling_params": {...}
        }
    }
    
    2. OpenAI messages format:
    {
        "input": {
            "messages": [{"role": "user", "content": "..."}],
            "sampling_params": {...}
        }
    }
    
    3. Task-specific format:
    {
        "input": {
            "task": "ats_keywords",
            "messages": [...],
            "response_format": "json",
            "safety": {"family_friendly": true},
            "sampling_params": {...}
        }
    }
    """
    job_input = job.get("input", {})
    request_id = str(uuid.uuid4())
    
    # Track metrics
    start_time = time.time()
    retry_count = 0
    validation_passed = False
    task_type = None
    
    try:
        # Determine task type
        task_type_str = job_input.get("task")
        
        # Parse messages or prompt
        if "messages" in job_input:
            messages = job_input["messages"]
            # Auto-detect task if not specified
            if not task_type_str and messages:
                user_content = next((m["content"] for m in messages if m["role"] == "user"), "")
                task_type = detect_task_from_content(user_content)
            else:
                try:
                    task_type = TaskType(task_type_str) if task_type_str else TaskType.GENERIC_CHAT
                except ValueError:
                    task_type = TaskType.GENERIC_CHAT
        elif "prompt" in job_input:
            # Legacy format
            prompt = job_input["prompt"]
            task_type = detect_task_from_content(prompt) if not task_type_str else TaskType(task_type_str)
            # Convert to messages format
            messages = [{"role": "user", "content": prompt}]
        else:
            return {"error": "Either 'messages' or 'prompt' must be provided"}
        
        # Get task configuration
        task_config = TASK_CONFIGS[task_type]
        
        # Add system prompt if not present
        if not any(msg.get("role") == "system" for msg in messages):
            messages.insert(0, {
                "role": "system",
                "content": task_config.system_prompt
            })
        
        # Extract parameters
        sampling_params_dict = job_input.get("sampling_params", {})
        temperature = sampling_params_dict.get("temperature", task_config.temperature)
        max_tokens = sampling_params_dict.get("max_tokens", task_config.max_tokens)
        top_p = sampling_params_dict.get("top_p", task_config.top_p)
        stream = sampling_params_dict.get("stream", False)
        
        # Safety settings
        safety_config = job_input.get("safety", {})
        family_friendly = safety_config.get("family_friendly", True)
        
        response_format = job_input.get("response_format", task_config.expected_format)
        
        log_json("request_received", request_id, 
                task=task_type.value,
                format=response_format,
                stream=stream)
        
        # Convert messages to prompt
        prompt = convert_messages_to_prompt(messages)
        
        # V3 validation for career guidance
        if USE_V3_VALIDATION and task_config.use_v3_validation:
            # Use V3 prompt building and validation
            user_content = next((m["content"] for m in messages if m["role"] == "user"), "")
            
            classifier = IntentClassifier()
            intent = classifier.classify(user_content)
            
            prompt_builder = PromptBuilder()
            enhanced_prompt = prompt_builder.build(user_content, intent)
            prompt = enhanced_prompt
        
        # Create sampling params
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=["<|im_end|>", "Human:", "<|endoftext|>"]
        )
        
        # Main generation loop with retry
        for attempt in range(MAX_RETRIES + 1):
            try:
                # Generate response
                if stream:
                    # For streaming, return generator
                    async def generate_stream():
                        full_text = ""
                        async for text in stream_with_engine(prompt, sampling_params, request_id):
                            chunk = text[len(full_text):]
                            full_text = text
                            yield chunk
                        
                        # Validate final output if needed
                        if response_format == "json":
                            is_valid, parsed_data, error_msg = parse_and_validate_json(full_text, task_config)
                            if not is_valid and attempt < MAX_RETRIES:
                                # Can't retry in streaming mode
                                yield f"\n\n[Warning: JSON validation failed: {error_msg}]"
                    
                    return runpod.serverless.modules.rp_scale.job_stream_wrapper(generate_stream())
                else:
                    # Non-streaming generation
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    full_text = loop.run_until_complete(
                        generate_with_engine(prompt, sampling_params, request_id)
                    )
                    loop.close()
                
                # Safety checks
                if family_friendly and check_profanity(full_text):
                    if attempt < MAX_RETRIES:
                        retry_count += 1
                        messages[0]["content"] += "\n\nIMPORTANT: Do not use profanity or inappropriate language."
                        prompt = convert_messages_to_prompt(messages)
                        temperature = max(0.15, temperature * 0.5)
                        continue
                    else:
                        return {"error": "Response contains inappropriate content"}
                
                # Check disclaimer for life advice
                if task_config.requires_disclaimer and not check_disclaimer(full_text):
                    if attempt < MAX_RETRIES:
                        retry_count += 1
                        messages[0]["content"] += "\n\nREQUIRED: Include disclaimer about not being a licensed professional."
                        prompt = convert_messages_to_prompt(messages)
                        continue
                    else:
                        full_text += "\n\nI am not a licensed professional. For serious concerns, please consult a qualified therapist or counselor."
                
                # Validate JSON if expected
                if response_format == "json":
                    is_valid, parsed_data, error_msg = parse_and_validate_json(full_text, task_config)
                    
                    if not is_valid:
                        if attempt < MAX_RETRIES:
                            retry_count += 1
                            messages[0]["content"] = build_corrective_prompt(
                                messages[0]["content"],
                                error_msg,
                                task_config
                            )
                            prompt = convert_messages_to_prompt(messages)
                            temperature = 0.15
                            continue
                        else:
                            return {
                                "error": f"JSON validation failed: {error_msg}",
                                "raw_output": full_text
                            }
                    
                    validation_passed = True
                    full_text = json.dumps(parsed_data, ensure_ascii=False)
                
                # V3 validation for career guidance
                if USE_V3_VALIDATION and task_config.use_v3_validation:
                    validator = ResponseValidator()
                    sanitizer = AutoSanitizer()
                    
                    # Validate response
                    is_valid, issues = validator.validate(full_text, intent)
                    if not is_valid and attempt < MAX_RETRIES:
                        retry_count += 1
                        continue
                    
                    # Sanitize response
                    full_text = sanitizer.sanitize(full_text)
                
                # Calculate execution time
                exec_time = int((time.time() - start_time) * 1000)
                
                log_json("request_completed", request_id,
                        task=task_type.value,
                        exec_ms=exec_time,
                        retry_count=retry_count,
                        validation_passed=validation_passed)
                
                # Return response
                return {
                    "output": full_text,
                    "metadata": {
                        "task": task_type.value,
                        "retry_count": retry_count,
                        "validation_passed": validation_passed,
                        "exec_ms": exec_time
                    }
                }
                
            except Exception as e:
                if attempt < MAX_RETRIES:
                    retry_count += 1
                    logger.warning(f"Generation failed, retrying: {str(e)}")
                    continue
                else:
                    raise
        
        return {"error": "Max retries exceeded"}
        
    except Exception as e:
        logger.error(f"Handler error: {str(e)}", exc_info=True)
        log_json("request_failed", request_id,
                level="error",
                error=str(e),
                task=task_type.value if task_type else "unknown")
        return {"error": str(e)}

# RunPod serverless handler
runpod.serverless.start({"handler": handler})