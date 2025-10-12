#!/usr/bin/env python3
"""
Career Guidance Model - V3 with Critical Fixes
Addresses issues found in deep V2 analysis:
1. Intent classification false positives
2. Salary range regex gaps (SGD, SEK formats)
3. Over-sanitization leaving empty responses
4. Improved prompts for weak categories
"""

import re
from typing import Dict, List, Tuple, Optional
from enum import Enum

class QuestionIntent(Enum):
    """Question intent categories"""
    SALARY_INTEL = "salary_intelligence"
    CAREER_GUIDANCE = "career_guidance"
    INTERVIEW_SKILLS = "interview_skills"
    MARKET_INTEL = "market_intelligence"

# City to currency mapping
CITY_CURRENCY_MAP = {
    "san francisco": "USD", "sf": "USD", "bay area": "USD",
    "new york": "USD", "nyc": "USD", "new york city": "USD",
    "seattle": "USD", "austin": "USD", "boston": "USD",
    "toronto": "CAD", "vancouver": "CAD",
    "london": "GBP",
    "berlin": "EUR", "munich": "EUR", "paris": "EUR", "amsterdam": "EUR",
    "barcelona": "EUR", "dublin": "EUR", "copenhagen": "DKK",
    "zurich": "CHF", "stockholm": "SEK",
    "bangalore": "INR", "bengaluru": "INR", "mumbai": "INR", "hyderabad": "INR",
    "delhi": "INR", "pune": "INR", "chennai": "INR", "noida": "INR",
    "singapore": "SGD",
    "sydney": "AUD", "melbourne": "AUD", "brisbane": "AUD",
    "tel aviv": "ILS",
}

# Currency patterns
CURRENCY_PATTERNS = {
    "USD": r'\$|USD|usd|dollars?',
    "EUR": r'€|EUR|eur|euros?',
    "GBP": r'£|GBP|gbp|pounds?',
    "INR": r'₹|INR|inr|rupees?',
    "CAD": r'C\$|CAD|cad',
    "SGD": r'S\$|SGD|sgd',
    "AUD": r'A\$|AU\$|AUD|aud',
    "CHF": r'CHF|chf|francs?',
    "SEK": r'SEK|sek|kr|kronor',
    "DKK": r'DKK|dkk|kr|kroner',
    "ILS": r'₪|ILS|ils|shekel',
}

class AutoSanitizer:
    """Auto-sanitize with over-cleaning prevention"""

    @staticmethod
    def sanitize(text: str) -> Tuple[str, str]:
        """
        Sanitize and return (cleaned_text, status)
        Status: "ALLOW", "REGENERATE", "EMPTY"
        """
        original_length = len(text)
        cleaned = text

        # Light prefix trimmer: Remove leading role labels and question fragments
        # Remove leading "Assistant:" or "User:" labels
        cleaned = re.sub(r'^\s*(Assistant|User)\s*:\s*', '', cleaned, flags=re.IGNORECASE)

        # Remove leading short question fragments (< 60 chars ending in ?)
        # Examples: "more meaningfully and purposefully?", "without eggs?"
        lines = cleaned.split('\n', 1)
        if lines and len(lines[0]) < 60 and lines[0].strip().endswith('?'):
            cleaned = lines[1] if len(lines) > 1 else ''

        # Remove common leading fragments
        cleaned = re.sub(r'^\s*(from scratch|without \w+)\?\s*', '', cleaned, flags=re.IGNORECASE)

        # Remove hashtags
        cleaned = re.sub(r'#\w+', '', cleaned)

        # Remove HTML
        cleaned = re.sub(r'<[^>]+>', '', cleaned)

        # Remove meta patterns
        cleaned = re.sub(r'Context:.*?(?:\n|$)', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'Answer according to:.*?(?:\n|$)', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'\|\s*Glassdoor.*?(?:\n|$)', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'What should I know about.*?(?:\n|$)', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"What'?s it like to work at.*?(?:\n|$)", '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'Answered.*?years? ago', '', cleaned, flags=re.IGNORECASE)

        # Remove standalone pipes (but preserve in ranges)
        cleaned = re.sub(r'^\s*\|', '', cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r'\|\s*$', '', cleaned, flags=re.MULTILINE)

        # Remove chat tokens
        cleaned = re.sub(r'<\|[^>]+\|>', '', cleaned)

        # Remove trailing artifacts: [END], [Truncated], [Read more], etc.
        cleaned = re.sub(r'\s*\[(END|Truncated|Read more|Continue reading)\].*$', '', cleaned, flags=re.IGNORECASE | re.DOTALL)

        # Remove trailing ellipsis spam (3+ consecutive dots or periods)
        cleaned = re.sub(r'(\.\s*){3,}$', '', cleaned)

        # Clean whitespace
        cleaned = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned)
        cleaned = re.sub(r'[ \t]+', ' ', cleaned)
        cleaned = re.sub(r'\.{4,}', '.', cleaned)

        cleaned = cleaned.strip()

        # Check for over-sanitization
        if len(cleaned) < 50:
            if len(cleaned) < 10:
                return cleaned, "EMPTY"
            # Check if we removed >80% of content
            if original_length > 0 and len(cleaned) / original_length < 0.2:
                return cleaned, "REGENERATE"

        # Check if has useful keywords (not just filler)
        useful_keywords = [
            'skill', 'learn', 'develop', 'practice', 'salary', 'range',
            'experience', 'company', 'career', 'role', 'position', 'tech',
            'cloud', 'software', 'engineer', 'developer', 'manager'
        ]
        if not any(kw in cleaned.lower() for kw in useful_keywords):
            return cleaned, "REGENERATE"

        return cleaned, "ALLOW"

class IntentClassifier:
    """Fixed intent classification - career keywords have priority"""

    # CRITICAL: Career guidance keywords checked FIRST to avoid false positives
    CAREER_KEYWORDS = [
        'transition', 'move from', 'skills should i develop', 'skills do i need',
        'learning path', 'prepare for transition', 'best path', 'career path',
        'certifications would help', 'how can i move', 'what skills',
        'how to move', 'how do i move', 'moving from'
    ]

    SALARY_KEYWORDS = [
        'salary', 'compensation', 'expect', 'pay', 'range', 'make in',
        'typical salary', 'average compensation', 'how much', 'earn'
    ]

    INTERVIEW_KEYWORDS = [
        'resume', 'interview', 'prepare for interview', 'showcase', 'demonstrate',
        'highlight', 'structure my resume', 'must-have skills for', 'projects should'
    ]

    MARKET_KEYWORDS = [
        'hiring trends', 'companies are hiring', 'in-demand', 'opportunities',
        'emerging technologies', 'industries are hiring', 'hiring practices'
    ]

    @staticmethod
    def classify(question: str) -> QuestionIntent:
        """
        Fixed classification - check career keywords FIRST
        Prevents misclassifying "What skills..." as salary
        """
        q_lower = question.lower()

        # Priority 1: Career guidance (most specific patterns)
        if any(kw in q_lower for kw in IntentClassifier.CAREER_KEYWORDS):
            return QuestionIntent.CAREER_GUIDANCE

        # Priority 2: Interview/skills (specific format)
        if any(kw in q_lower for kw in IntentClassifier.INTERVIEW_KEYWORDS):
            return QuestionIntent.INTERVIEW_SKILLS

        # Priority 3: Market intelligence
        if any(kw in q_lower for kw in IntentClassifier.MARKET_KEYWORDS):
            return QuestionIntent.MARKET_INTEL

        # Priority 4: Salary (only if not matched above)
        if any(kw in q_lower for kw in IntentClassifier.SALARY_KEYWORDS):
            return QuestionIntent.SALARY_INTEL

        # Default to career guidance (safest category)
        return QuestionIntent.CAREER_GUIDANCE

class ResponseValidator:
    """Fixed validator with improved regex patterns"""

    @staticmethod
    def extract_city(question: str) -> Optional[str]:
        """Extract city from question"""
        q_lower = question.lower()
        for city in CITY_CURRENCY_MAP:
            if city in q_lower:
                return city
        return None

    @staticmethod
    def detect_currencies(response: str) -> List[str]:
        """Detect currencies mentioned in response"""
        currencies = []
        for currency, pattern in CURRENCY_PATTERNS.items():
            if re.search(pattern, response, re.IGNORECASE):
                currencies.append(currency)
        return list(set(currencies))

    @staticmethod
    def has_numeric_range(response: str) -> bool:
        """
        FIXED: Comprehensive salary range detection
        Now matches: SGD 5,000 - SGD 13,000, SEK 145,000 and SEK 830,000, etc.
        """
        range_patterns = [
            # Standard k notation: 50k-70k, $50K-$70K
            r'\d+[kK]\s*[-–—to]+\s*\$?\d+[kK]',

            # With currency code: USD 100k-150k, EUR 60k-80k
            r'(USD|EUR|GBP|INR|CAD|AUD|SGD|SEK|CHF|DKK|ILS)\s*\d+[kK]?\s*[-–—to]+\s*\1?\s*\d+[kK]?',

            # With commas: SGD 5,000 - SGD 13,000, €60,000-€75,000
            r'(USD|EUR|GBP|INR|CAD|AUD|SGD|SEK|CHF|DKK|ILS|\$|€|£|₹)\s*\d{1,3}(?:,\d{3})+\s*[-–—to]+\s*\1?\s*\d{1,3}(?:,\d{3})+',

            # "between X and Y": between SEK 145,000 and SEK 830,000
            r'between\s+(USD|EUR|GBP|INR|CAD|AUD|SGD|SEK|CHF|DKK|ILS|\$|€|£|₹)\s*\d{1,3}(?:,?\d{3})*\s+and\s+\1?\s*\d{1,3}(?:,?\d{3})*',

            # Symbol format: $100k-$150k, €60k-€80k
            r'(\$|€|£|₹|CHF|SGD|SEK|DKK)\s*\d+[kK]?\s*[-–—to]+\s*\1?\s*\d+[kK]?',

            # Plain numbers (4+ digits): 50000-70000
            r'\d{4,}\s*[-–—to]+\s*\d{4,}',

            # Median format: median ~$97k, median ~€70k
            r'median\s*~?\s*(\$|€|£|₹|USD|EUR|GBP|INR|CAD|AUD|SGD|SEK|CHF)?\s*\d+[kK]?',

            # Monthly ranges: 8,000–12,000 per month
            r'\d{1,3}(?:,\d{3})*\s*[-–—to]+\s*\d{1,3}(?:,\d{3})*\s*(?:per month|monthly|/month)',

            # Fallback: "typically falls between X and Y"
            r'(?:falls|ranges?)\s+between\s+\d+[kK]?\s+and\s+\d+[kK]?',
        ]

        for pattern in range_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                return True

        return False

    @staticmethod
    def check_length(response: str) -> Tuple[bool, Optional[str]]:
        """Check response length"""
        word_count = len(response.split())

        if word_count < 20:
            return False, "Response too short (< 20 words)"
        elif word_count > 250:
            return False, "Response too long (> 250 words) - may be contaminated"

        return True, None

    @staticmethod
    def check_repetition(response: str) -> Tuple[bool, Optional[str]]:
        """Check for repetition"""
        words = response.split()
        if len(words) > 20:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.35:  # More lenient
                return False, f"Repetitive content (unique ratio: {unique_ratio:.2f})"

        return True, None

    @staticmethod
    def validate_salary_response(question: str, response: str) -> Tuple[bool, List[str]]:
        """Validate salary response"""
        issues = []

        # 1. Currency check
        city = ResponseValidator.extract_city(question)
        if city:
            expected_currency = CITY_CURRENCY_MAP[city]
            detected_currencies = ResponseValidator.detect_currencies(response)

            if not detected_currencies:
                issues.append("No currency detected")
            elif expected_currency not in detected_currencies:
                issues.append(
                    f"Currency mismatch: expected {expected_currency} for {city}, "
                    f"found {', '.join(detected_currencies)}"
                )

        # 2. Numeric range check (FIXED)
        if not ResponseValidator.has_numeric_range(response):
            issues.append("No numeric salary range found")

        # 3. Length check
        is_length_ok, length_issue = ResponseValidator.check_length(response)
        if not is_length_ok:
            issues.append(length_issue)

        # 4. Repetition check
        is_rep_ok, rep_issue = ResponseValidator.check_repetition(response)
        if not is_rep_ok:
            issues.append(rep_issue)

        return len(issues) == 0, issues

    @staticmethod
    def validate_career_response(response: str) -> Tuple[bool, List[str]]:
        """Validate career guidance response"""
        issues = []

        # Skill keywords (more lenient)
        skill_keywords = [
            'skill', 'learn', 'practice', 'develop', 'master',
            'study', 'certification', 'experience', 'knowledge'
        ]
        if not any(kw in response.lower() for kw in skill_keywords):
            issues.append("Missing skill guidance keywords")

        # Length
        is_length_ok, length_issue = ResponseValidator.check_length(response)
        if not is_length_ok:
            issues.append(length_issue)

        # Repetition
        is_rep_ok, rep_issue = ResponseValidator.check_repetition(response)
        if not is_rep_ok:
            issues.append(rep_issue)

        return len(issues) == 0, issues

class PromptBuilder:
    """Improved prompts for weak categories"""

    INTERVIEW_TEMPLATE = """You are a helpful career coach. Answer with numbered, actionable steps. No metadata. No "Context:" lines. Keep answers 80-150 words.

Name 4-5 specific tools/frameworks (e.g., "Terraform", "Kubernetes", not generic terms). Focus on concrete preparation steps.

Question: {question}

Answer:"""

    CAREER_TEMPLATE = """You are a helpful career coach. Answer with numbered, actionable steps. No metadata. No "Context:" lines. Keep answers 80-150 words.

Name 4-5 specific skills/tools (e.g., "AWS", "Kubernetes"). Include realistic timeline (weeks/months).

Question: {question}

Answer:"""

    SALARY_TEMPLATE = """You are a helpful career coach. Answer with numbered, actionable steps. No metadata. No "Context:" lines. Keep answers 80-150 words.

Provide salary guidance with correct local currency (USD/EUR/GBP/INR/SGD/SEK/CHF). Include numeric range and 2-3 factors affecting compensation.

Question: {question}

Answer:"""

    @staticmethod
    def build_prompt(question: str, intent: QuestionIntent) -> str:
        """Build improved prompt based on intent"""
        templates = {
            QuestionIntent.SALARY_INTEL: PromptBuilder.SALARY_TEMPLATE,
            QuestionIntent.CAREER_GUIDANCE: PromptBuilder.CAREER_TEMPLATE,
            QuestionIntent.INTERVIEW_SKILLS: PromptBuilder.INTERVIEW_TEMPLATE,
            QuestionIntent.MARKET_INTEL: PromptBuilder.CAREER_TEMPLATE,  # Reuse career template
        }

        template = templates.get(intent, PromptBuilder.CAREER_TEMPLATE)
        return template.format(question=question)

def process_question_v3(question: str, raw_response: str) -> Dict:
    """
    V3: Fixed classification + improved sanitization + better validation
    """
    # Step 1: Classify (FIXED - career keywords have priority)
    intent = IntentClassifier.classify(question)

    # Step 2: Sanitize with over-cleaning detection
    cleaned_response, sanitize_status = AutoSanitizer.sanitize(raw_response)

    # Step 3: Validate
    if intent == QuestionIntent.SALARY_INTEL:
        is_valid, issues = ResponseValidator.validate_salary_response(question, cleaned_response)
    else:
        is_valid, issues = ResponseValidator.validate_career_response(cleaned_response)

    # Step 4: Routing decision
    should_use_rag = intent in [QuestionIntent.SALARY_INTEL, QuestionIntent.MARKET_INTEL]

    if sanitize_status == "EMPTY":
        recommendation = "REGENERATE - Over-sanitization (empty response)"
    elif sanitize_status == "REGENERATE":
        recommendation = "REGENERATE - Lost >80% content in sanitization"
    elif should_use_rag:
        recommendation = "REQUIRE RAG - Block model-only answers"
    elif is_valid:
        recommendation = "ALLOW (validated)"
    else:
        recommendation = "REGENERATE with stricter prompt"

    return {
        'question': question,
        'intent': intent.value,
        'raw_response': raw_response,
        'cleaned_response': cleaned_response,
        'sanitize_status': sanitize_status,
        'is_valid': is_valid,
        'issues': issues,
        'should_use_rag': should_use_rag,
        'recommendation': recommendation,
        'sanitized': raw_response != cleaned_response
    }

if __name__ == '__main__':
    # Test cases from deep analysis
    test_cases = [
        # Misclassified as salary (should be career)
        ("I want to move from Manual Testing to Automation Testing. What's the best learning path?",
         "Learn Selenium, Pytest, Jenkins. Practice writing test automation scripts."),

        # Missing range detection (SEK format)
        ("Stockholm Solutions Architect salary?",
         "typically falls between SEK 145,000 and SEK 830,000"),

        # Missing range detection (SGD format)
        ("Singapore Backend Engineer salary?",
         "Salary range: SGD 5,000 - SGD 13,000"),

        # Over-sanitization case
        ("DevOps resume structure?",
         "#devops #resume Context: resume structure"),
    ]

    print("Testing V3 Fixes\n" + "="*80)
    for question, response in test_cases:
        result = process_question_v3(question, response)
        print(f"\nQuestion: {question[:60]}...")
        print(f"Intent: {result['intent']}")
        print(f"Sanitize status: {result['sanitize_status']}")
        print(f"Valid: {result['is_valid']}")
        print(f"Issues: {', '.join(result['issues']) if result['issues'] else 'None'}")
        print(f"Recommendation: {result['recommendation']}")
