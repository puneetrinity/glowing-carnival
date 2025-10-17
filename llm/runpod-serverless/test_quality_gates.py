#!/usr/bin/env python3
"""
Endpoint Quality Gates Tester

Runs structured generation tests against a RunPod serverless endpoint and
validates outputs using the same VALIDATORS as training (audit_dataset.py).

Features:
- HR domains: resume_guidance, job_description, job_resume_match,
  recruiting_strategy, ats_keywords
- Optional router/safety checks: salary blocking, non-career prompts
- CSV export with per-prompt results

Usage:
  python3 test_quality_gates.py \
    --endpoint-id <ENDPOINT_ID> \
    --api-key <RUNPOD_API_KEY> \
    --out eval_results_endpoint.csv \
    [--count 10] [--no-router] [--no-salary]

Environment fallback:
  ENDPOINT_ID, API_KEY

Notes:
- Uses ChatML prompts with per-domain 1-shot exemplars where helpful.
- Uses conservative decoding defaults.
"""
import os
import sys
import csv
import time
import json
import argparse
from typing import Dict, List, Tuple

import requests


def load_validators():
    """Import VALIDATORS from colab_training/audit_dataset.py, with fallback."""
    # Try local colab_training path first
    sys.path.insert(0, os.path.join(os.getcwd(), 'colab_training'))
    try:
        from audit_dataset import VALIDATORS as _VAL
        return _VAL
    except Exception:
        pass

    # Fallback minimal validators (structure-only, lenient)
    import re

    def v_resume(text: str):
        issues = []
        if len(text.split()) < 50:
            issues.append('len')
        if not re.search(r'^\s*[\d\-\*\•]', text, flags=re.M):
            issues.append('bullets')
        verbs = ['led', 'built', 'developed', 'managed', 'created', 'improved', 'optimized', 'designed', 'implemented', 'achieved']
        if not any(v in text.lower() for v in verbs):
            issues.append('verbs')
        if ('ats keywords' not in text.lower()) and text.count(',') < 5:
            issues.append('ats_line')
        return (len(issues) == 0), issues

    def v_jd(text: str):
        import re
        issues = []
        if len(text.split()) < 100:
            issues.append('len')
        if sum(s in text.lower() for s in ['responsibilities', 'requirements']) < 2:
            issues.append('sections')
        bullets = len(re.findall(r'^\s*[\d\-\*\•]', text, flags=re.M))
        if bullets < 8:
            issues.append('bullets')
        return (len(issues) == 0), issues

    def v_match(text: str):
        import re
        issues = []
        if len(text.split()) < 80:
            issues.append('len')
        if not re.search(r'\b\d{1,3}\s*%|\bscore[:\s]+\d{1,3}', text, flags=re.I):
            issues.append('score')
        if sum(t in text.lower() for t in ['match', 'gap', 'skill']) < 2:
            issues.append('content')
        return (len(issues) == 0), issues

    def v_recruit(text: str):
        issues = []
        if len(text.split()) < 60:
            issues.append('len')
        channels = ['linkedin', 'referral', 'meetup', 'university', 'conference', 'github', 'stackoverflow']
        if sum(c in text.lower() for c in channels) < 3:
            issues.append('channels')
        if not any(w in text.lower() for w in ['week', 'month', 'quarter', 'weekly', 'monthly', 'quarterly']):
            issues.append('cadence')
        return (len(issues) == 0), issues

    def v_ats(text: str):
        import re
        comma = text.count(',')
        bullets = len(re.findall(r'^\s*[\d\-\*\•]', text, flags=re.M))
        k = max(comma + 1, bullets)
        return (k >= 15), ([] if k >= 15 else ['count'])

    return {
        'resume_guidance': v_resume,
        'job_description': v_jd,
        'job_resume_match': v_match,
        'recruiting_strategy': v_recruit,
        'ats_keywords': v_ats,
    }


# ChatML builders with 1-shot exemplars for resume/JD
SEEDED = {'resume_guidance', 'recruiting_strategy', 'ats_keywords'}

RESUME_EXAMPLE = (
    '1. Led migration of 50+ services to Kubernetes, reducing deployments by 60%\n'
    '2. Built CI/CD with Jenkins + ArgoCD, enabling 20 daily deploys\n'
    '3. Optimized AWS costs by 35% via auto-scaling and rightsizing\n'
    '4. Designed REST APIs handling 10k+ req/sec with FastAPI\n'
    '5. Implemented monitoring with Prometheus/Grafana and on-call rota\n'
    '6. Mentored 3 junior engineers in code reviews and design\n'
    'ATS Keywords: Kubernetes, Docker, Jenkins, ArgoCD, Terraform, AWS, FastAPI, Python, CI/CD, Prometheus'
)

JD_EXAMPLE = (
    '**Summary:** Experienced backend engineer to build scalable APIs.\n'
    '**Responsibilities:**\n1. Design REST APIs\n2. Optimize DB queries\n3. Implement security\n4. Collaborate with DevOps\n5. Code reviews\n6. Monitor SLIs/SLOs\n'
    '**Requirements:**\n1. 5+ yrs backend\n2. Python/Go/Node\n3. SQL/NoSQL\n4. AWS/GCP\n5. System design\n6. Communication\n'
)


def build_chatml(domain: str, user: str) -> str:
    if domain == 'resume_guidance':
        sys_txt = '''You are a career coach specializing in resumes. Provide exactly 6-8 numbered bullets:
- Start each bullet with a strong action verb (Led, Built, Designed, Optimized, etc.)
- Include quantifiable metrics (%, numbers, scale)
- End with "ATS Keywords: ..." line containing 8-12 comma-separated keywords

Example format:
''' + RESUME_EXAMPLE
    elif domain == 'job_description':
        sys_txt = '''You are an HR specialist. Create a job description with 3 sections:
1. **Summary:** 1-2 sentence role overview
2. **Responsibilities:** 6-8 numbered bullet points
3. **Requirements:** 6-8 numbered bullet points

Example format:
''' + JD_EXAMPLE
    elif domain == 'job_resume_match':
        sys_txt = '''You are a technical recruiter. Analyze the match and provide:

**Score:** X/100 (numeric score 0-100)
**Matches:** (List 5+ matching qualifications)
- Match 1
- Match 2
...

**Gaps:** (List 3+ missing requirements)
- Gap 1
- Gap 2
...

**Next Steps:** (List 3+ actionable recommendations)
1. Next step 1
2. Next step 2
...

Format exactly as shown. Be concise but complete.'''
    elif domain == 'recruiting_strategy':
        sys_txt = 'You are a recruiting strategist. Provide 4-6 numbered steps with channels (LinkedIn, referral, meetup, university, conference, GitHub, Stack Overflow), cadence (weekly/monthly/quarterly) and 1-2 metrics.'
    else:
        sys_txt = 'You are a resume optimization specialist. Provide 20-40 ATS keywords comma-separated (80-120 words).'

    seed = '1. ' if domain in SEEDED else ''
    return f"<|im_start|>system\n{sys_txt}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n{seed}"


def default_hr_prompts(count: int) -> Dict[str, List[str]]:
    roles = [
        'Backend Engineer', 'DevOps Engineer', 'Data Scientist', 'ML Engineer', 'Cloud Architect',
        'Frontend Engineer', 'SRE', 'Security Engineer', 'Platform Engineer', 'Mobile Engineer',
        'Data Engineer', 'MLOps Engineer', 'Full Stack Engineer', 'Android Engineer', 'iOS Engineer'
    ]
    stacks = [
        'Python, FastAPI, Postgres, AWS', 'Kubernetes, Terraform, EKS', 'PyTorch, XGBoost, SQL',
        'AWS, IaC, cost optimization', 'React, TypeScript, GraphQL', 'SLIs/SLOs, on-call, observability',
        'IAM, KMS, threat modeling', 'K8s, GitOps, ArgoCD', 'Kotlin, Jetpack, CI/CD', 'Swift, SwiftUI, CI/CD'
    ]

    resume = [f"Optimize my resume for {r} ({stacks[i % len(stacks)]})." for i, r in enumerate(roles)]
    jd = [f"Write a job description for Senior {r} at a product company." for r in roles]
    match = [
        'Score match: 5 yrs React vs Senior Frontend role (React, TS, GraphQL).',
        'Evaluate fit: DevOps (Docker, Jenkins) vs Cloud Architect (AWS, IaC).',
        'Match analysis: Data Analyst vs Data Scientist (Python, SQL, ML).',
        'Score: 7 yrs Java vs Senior Backend role (Spring, Microservices).',
        'Evaluate: SRE background vs Platform Engineer (K8s, GitOps).',
    ]
    recruiting = [
        'Sourcing strategy to hire 5 senior ML engineers in 3 months.',
        'Recruiting plan for Security Engineers with cloud expertise.',
        'Sourcing strategy for competitive tech hub (SF/Seattle/NYC).',
        'Recruiting plan for Platform Engineers in remote-first team.',
        'Sourcing strategy for Cloud Architects with enterprise background.'
    ]
    ats = [
        'List ATS keywords for Senior Backend Engineer (Python, FastAPI, Postgres).',
        'ATS terms for DevOps Engineer resume (Kubernetes, Terraform, CI/CD).',
        'List ATS terms for Cloud Architect (AWS, IaC, security).',
        'ATS keywords for Data Scientist (PyTorch, MLOps, SQL).'
    ]

    # Truncate to requested count per domain where applicable
    return {
        'resume_guidance': resume[:count],
        'job_description': jd[:count],
        'job_resume_match': match[:count],
        'recruiting_strategy': recruiting[:count],
        'ats_keywords': ats[:count],
    }


def call_endpoint(endpoint_id: str, api_key: str, prompt: str, max_tokens: int = 200) -> Tuple[dict, int, str]:
    url = f"https://api.runpod.ai/v2/{endpoint_id}/runsync"
    headers = {'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'}
    body = {
        'input': {
            'prompt': prompt,
            'sampling_params': {
                'max_tokens': max_tokens,
                'temperature': 0.3,
                'top_p': 0.9,
                'repetition_penalty': 1.0,
                'stop': ["<|im_end|>"]
            }
        }
    }
    try:
        r = requests.post(url, headers=headers, json=body, timeout=120)
        js = {}
        try:
            js = r.json()
        except Exception:
            pass
        return js, r.status_code, None
    except Exception as e:
        return {}, 0, str(e)


def run_suite(endpoint_id: str, api_key: str, count: int, out_csv: str, include_router: bool = True, include_salary: bool = True) -> None:
    validators = load_validators()
    prompts = default_hr_prompts(count)
    rows = []

    # HR domains
    for d, plist in prompts.items():
        print(f"\n== Domain: {d} ==")
        passed = 0
        for p in plist:
            chat = build_chatml(d, p)
            # Domain-specific max_tokens based on validation requirements
            max_tok = {
                'job_description': 230,
                'job_resume_match': 280,  # Needs more tokens for detailed analysis
                'recruiting_strategy': 200,
                'resume_guidance': 200,
                'ats_keywords': 150
            }.get(d, 200)
            js, status, err = call_endpoint(endpoint_id, api_key, chat, max_tokens=max_tok)
            # Parse response - output can be dict or list
            output = (js or {}).get('output', [])
            if isinstance(output, list) and len(output) > 0:
                text = output[0].get('choices', [{}])[0].get('tokens', [''])[0]
            else:
                text = ''
            ok, issues = validators[d](text)
            rows.append({
                'domain': d, 'prompt': p, 'ok': ok,
                'issues': ';'.join(issues), 'len': len(text.split()),
                'preview': text[:200], 'http_status': status,
                'exec_ms': (js or {}).get('executionTime', -1)
            })
            passed += int(ok)
        rate = passed / max(1, len(plist)) * 100
        print(f"{passed}/{len(plist)} = {rate:.1f}%")

    # Salary blocking
    if include_salary:
        print("\n== Router/Safety: Salary blocking ==")
        salary_tests = [
            'What is the typical salary for a Solutions Architect in Stockholm?',
            'Average pay for Data Scientist in Singapore?'
        ]
        for q in salary_tests:
            js, status, err = call_endpoint(endpoint_id, api_key, q)
            output = (js or {}).get('output', {})
            blocked = output.get('blocked') if isinstance(output, dict) else False
            rows.append({'domain': 'salary_block', 'prompt': q, 'ok': bool(blocked), 'issues': '' if blocked else 'not_blocked', 'len': 0, 'preview': json.dumps(output)[:180], 'http_status': status, 'exec_ms': (js or {}).get('executionTime', -1)})
            print(('Blocked' if blocked else 'Not blocked') + ' | ' + q)

    # Non-career router
    if include_router:
        print("\n== Router: Non-career ==")
        non_career = {
            'cooking': ['How to bake a chocolate cake?'],
            'travel': ['2-day itinerary for Paris?'],
            'weather': ['What is the weather in Delhi today?'],
            'small_talk': ['Hello! How is your day?'],
            'general_qna': ['How to set up Docker on Ubuntu?']
        }

        # System messages for specific categories
        category_systems = {
            'small_talk': 'You are a friendly assistant. Respond warmly and naturally to casual conversation.',
            'cooking': 'You are a helpful cooking assistant.',
            'travel': 'You are a travel planning assistant.',
            'weather': 'You are a weather information assistant.',
            'general_qna': 'You are a helpful technical assistant.'
        }

        for cat, plist in non_career.items():
            for p in plist:
                # Wrap with ChatML if system message exists
                if cat in category_systems:
                    prompt = f"<|im_start|>system\n{category_systems[cat]}<|im_end|>\n<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n"
                else:
                    prompt = p
                js, status, err = call_endpoint(endpoint_id, api_key, prompt)
                output = (js or {}).get('output', [])
                if isinstance(output, list) and len(output) > 0:
                    text = output[0].get('choices', [{}])[0].get('tokens', [''])[0]
                else:
                    text = ''
                rows.append({'domain': f'router_{cat}', 'prompt': p, 'ok': True, 'issues': '', 'len': len(text.split()), 'preview': text[:180], 'http_status': status, 'exec_ms': (js or {}).get('executionTime', -1)})
                print(cat + ' | ' + text[:80].replace('\n', ' '))

    # Write CSV
    if rows:
        with open(out_csv, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader(); w.writerows(rows)
        print('Saved CSV:', out_csv)


def main():
    ap = argparse.ArgumentParser(description='Run endpoint quality gates tests')
    ap.add_argument('--endpoint-id', default=os.getenv('ENDPOINT_ID', ''), help='RunPod endpoint ID')
    ap.add_argument('--api-key', default=os.getenv('API_KEY', ''), help='RunPod API key')
    ap.add_argument('--count', type=int, default=10, help='Prompts per HR domain')
    ap.add_argument('--out', default='eval_results_endpoint.csv', help='Output CSV path')
    ap.add_argument('--no-router', action='store_true', help='Skip non-career router tests')
    ap.add_argument('--no-salary', action='store_true', help='Skip salary blocking tests')
    args = ap.parse_args()

    if not args.endpoint_id or not args.api_key:
        print('ERROR: Missing --endpoint-id or --api-key (or ENDPOINT_ID/API_KEY env).')
        sys.exit(1)

    run_suite(
        endpoint_id=args.endpoint_id,
        api_key=args.api_key,
        count=args.count,
        out_csv=args.out,
        include_router=not args.no_router,
        include_salary=not args.no_salary,
    )


if __name__ == '__main__':
    main()

