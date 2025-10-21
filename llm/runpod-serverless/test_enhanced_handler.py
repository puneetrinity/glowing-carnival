#!/usr/bin/env python3
"""
Test script for the enhanced RunPod handler
Tests all task types and features
"""

import json
import os
import requests
import time
from typing import Dict, Any

# RunPod API configuration
ENDPOINT_ID = os.getenv("RUNPOD_ENDPOINT_ID", "31jyzzmzvcyja1")
API_KEY = os.getenv("RUNPOD_API_KEY", "your-api-key-here")
BASE_URL = f"https://api.runpod.ai/v2/{ENDPOINT_ID}"

def run_job(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Submit a job and wait for completion"""
    # Submit job
    response = requests.post(
        f"{BASE_URL}/run",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}"
        },
        json={"input": input_data}
    )
    
    if response.status_code != 200:
        return {"error": f"Failed to submit job: {response.text}"}
    
    job_data = response.json()
    job_id = job_data["id"]
    
    print(f"Job submitted: {job_id}")
    
    # Poll for completion
    max_attempts = 30
    for attempt in range(max_attempts):
        time.sleep(2)
        
        status_response = requests.get(
            f"{BASE_URL}/status/{job_id}",
            headers={"Authorization": f"Bearer {API_KEY}"}
        )
        
        if status_response.status_code != 200:
            return {"error": f"Failed to get status: {status_response.text}"}
        
        status_data = status_response.json()
        
        if status_data["status"] == "COMPLETED":
            return status_data.get("output", {})
        elif status_data["status"] == "FAILED":
            return {"error": "Job failed", "details": status_data}
    
    return {"error": "Job timed out"}

def test_legacy_format():
    """Test backward compatibility with legacy format"""
    print("\n" + "="*70)
    print("TEST: Legacy Format (backward compatible)")
    print("="*70)
    
    input_data = {
        "prompt": "What are the top 3 skills for a Python developer?",
        "sampling_params": {
            "temperature": 0.7,
            "max_tokens": 150,
            "stream": False
        }
    }
    
    result = run_job(input_data)
    print(f"Result: {json.dumps(result, indent=2)}")
    return result

def test_ats_keywords():
    """Test ATS keywords extraction with JSON validation"""
    print("\n" + "="*70)
    print("TEST: ATS Keywords (structured JSON output)")
    print("="*70)
    
    input_data = {
        "task": "ats_keywords",
        "messages": [
            {"role": "user", "content": "Generate ATS keywords for Senior DevOps Engineer position at AWS"}
        ],
        "response_format": "json",
        "sampling_params": {"stream": False}
    }
    
    result = run_job(input_data)
    print(f"Result: {json.dumps(result, indent=2)}")
    
    # Verify structured output
    if "output" in result:
        try:
            keywords = json.loads(result["output"])
            print(f"✓ Valid JSON with {len(keywords)} keywords")
        except:
            print("✗ Invalid JSON output")
    
    return result

def test_resume_bullets():
    """Test resume bullets generation"""
    print("\n" + "="*70)
    print("TEST: Resume Bullets (structured JSON output)")
    print("="*70)
    
    input_data = {
        "task": "resume_bullets",
        "messages": [
            {"role": "user", "content": "Create resume bullets for a Machine Learning Engineer at Google working on LLMs"}
        ],
        "response_format": "json",
        "sampling_params": {"stream": False}
    }
    
    result = run_job(input_data)
    print(f"Result: {json.dumps(result, indent=2)}")
    
    # Verify structured output
    if "output" in result:
        try:
            bullets = json.loads(result["output"])
            print(f"✓ Valid JSON with {len(bullets)} bullets")
        except:
            print("✗ Invalid JSON output")
    
    return result

def test_job_description():
    """Test job description generation"""
    print("\n" + "="*70)
    print("TEST: Job Description (structured JSON output)")
    print("="*70)
    
    input_data = {
        "task": "job_description",
        "messages": [
            {"role": "user", "content": "Create a job description for a Full Stack Developer"}
        ],
        "response_format": "json",
        "sampling_params": {"stream": False}
    }
    
    result = run_job(input_data)
    print(f"Result: {json.dumps(result, indent=2)}")
    
    # Verify structured output
    if "output" in result:
        try:
            jd = json.loads(result["output"])
            required_keys = {"summary", "responsibilities", "requirements"}
            if required_keys.issubset(jd.keys()):
                print(f"✓ Valid JSON with all required fields")
            else:
                print(f"✗ Missing required fields: {required_keys - set(jd.keys())}")
        except:
            print("✗ Invalid JSON output")
    
    return result

def test_openai_messages():
    """Test OpenAI-compatible messages format"""
    print("\n" + "="*70)
    print("TEST: OpenAI Messages Format")
    print("="*70)
    
    input_data = {
        "messages": [
            {"role": "system", "content": "You are a career counselor specializing in tech careers."},
            {"role": "user", "content": "What's the career path from junior to senior software engineer?"}
        ],
        "sampling_params": {
            "temperature": 0.6,
            "max_tokens": 200,
            "stream": False
        }
    }
    
    result = run_job(input_data)
    print(f"Result: {json.dumps(result, indent=2)}")
    return result

def test_auto_detection():
    """Test automatic task detection"""
    print("\n" + "="*70)
    print("TEST: Auto Task Detection")
    print("="*70)
    
    input_data = {
        "messages": [
            {"role": "user", "content": "I need ATS keywords for a data scientist role"}
        ],
        "sampling_params": {"stream": False}
    }
    
    result = run_job(input_data)
    print(f"Result: {json.dumps(result, indent=2)}")
    
    # Check if task was auto-detected
    if "metadata" in result and "task" in result["metadata"]:
        print(f"✓ Auto-detected task: {result['metadata']['task']}")
    
    return result

def test_profanity_filter():
    """Test profanity filtering"""
    print("\n" + "="*70)
    print("TEST: Profanity Filter")
    print("="*70)
    
    input_data = {
        "messages": [
            {"role": "user", "content": "Tell me a joke"}
        ],
        "task": "small_talk",
        "safety": {"family_friendly": True},
        "sampling_params": {"stream": False}
    }
    
    result = run_job(input_data)
    print(f"Result: {json.dumps(result, indent=2)}")
    return result

def test_career_guidance():
    """Test career guidance with V3 validation"""
    print("\n" + "="*70)
    print("TEST: Career Guidance (V3 validation)")
    print("="*70)
    
    input_data = {
        "task": "career_guidance",
        "messages": [
            {"role": "user", "content": "What's the average salary for a senior software engineer in San Francisco?"}
        ],
        "sampling_params": {"stream": False}
    }
    
    result = run_job(input_data)
    print(f"Result: {json.dumps(result, indent=2)}")
    return result

def main():
    """Run all tests"""
    print("\n" + "#"*70)
    print("# Enhanced RunPod Handler Test Suite")
    print("#"*70)
    
    tests = [
        ("Legacy Format", test_legacy_format),
        ("ATS Keywords", test_ats_keywords),
        ("Resume Bullets", test_resume_bullets),
        ("Job Description", test_job_description),
        ("OpenAI Messages", test_openai_messages),
        ("Auto Detection", test_auto_detection),
        ("Profanity Filter", test_profanity_filter),
        ("Career Guidance", test_career_guidance)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = "PASSED" if "output" in result else "FAILED"
        except Exception as e:
            print(f"Error in {test_name}: {str(e)}")
            results[test_name] = "ERROR"
    
    # Summary
    print("\n" + "#"*70)
    print("# Test Summary")
    print("#"*70)
    for test_name, status in results.items():
        symbol = "✓" if status == "PASSED" else "✗"
        print(f"{symbol} {test_name}: {status}")

if __name__ == "__main__":
    main()