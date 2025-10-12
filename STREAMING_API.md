# Streaming API

This project supports delta-style streaming internally for faster, smaller responses.
Because RunPod Serverless returns a single response payload, the handler aggregates
all deltas and includes them in the final JSON.

Summary:
- Streaming emits only new text (delta), not full snapshots.
- The final response includes `choices[0].deltas` (list of deltas) and
  `choices[0].text` (full concatenated text). For backward compatibility,
  `choices[0].tokens` mirrors `deltas` when streaming is enabled.

## Request

POST to your RunPod endpoint (`/runsync` or `/run`) with `stream=true`:

```json
{
  "input": {
    "prompt": "What skills should I learn for backend development?",
    "sampling_params": {
      "max_tokens": 150,
      "temperature": 0.7,
      "stream": true
    },
    "enable_validation": true,
    "block_low_trust_intents": true
  }
}
```

**Request parameters:**
- `prompt` (required): User question
- `sampling_params.stream` (optional): Enable delta streaming (default: `false`)
- `enable_validation` (optional): Enable V3 validation (default: `true`)
- `block_low_trust_intents` (optional): Block salary/market intents (default: `true`)

**Notes:**
- `stream: true` enables delta generation internally. The network response is a
  single JSON with all deltas aggregated.
- Validation (V3) runs on the final text. If sanitization modifies text, deltas
  reflect the model output, while `choices[0].text` reflects the sanitized final.
- When `block_low_trust_intents: true`, salary/market queries return a `blocked` response.

## Response (stream=true)

```json
{
  "choices": [
    {
      "text": "final sanitized text ...",
      "deltas": ["First ", "chunk ", "only-new-text"],
      "tokens": ["First ", "chunk ", "only-new-text"]
    }
  ],
  "usage": {
    "input": 42,
    "output": 128
  },
  "validation": {
    "valid": true,
    "issues": [],
    "sanitized": false,
    "intent": "career_guidance"
  },
  "streaming": true
}
```

Field semantics:
- `choices[0].text`: Final full text (after any sanitization)
- `choices[0].deltas`: List of only-new text chunks (model-side deltas)
- `choices[0].tokens`: Alias of `deltas` for backward compatibility
- `usage`: Token usage from vLLM (prompt/output)
- `validation`: Present when V3 validation is enabled
- `streaming`: Echoes whether streaming mode was enabled

## Response (stream=false)

```json
{
  "choices": [
    {
      "text": "final sanitized text ...",
      "tokens": ["final sanitized text ..."]
    }
  ],
  "usage": { "input": 42, "output": 128 },
  "validation": { ... },
  "streaming": false
}
```

## Blocked Response (salary/market intents)

When `block_low_trust_intents: true` and the intent is salary or market intel:

```json
{
  "blocked": true,
  "intent": "salary_intel",
  "message": "This question requires real-time compensation/market data. We're integrating with trusted data sources (Levels.fyi, BLS.gov) to provide accurate, up-to-date information. In the meantime, try asking about career transitions, skill development, interview preparation, or learning paths. Expected availability: 2-4 weeks."
}
```

To allow these queries, set `block_low_trust_intents: false` or disable validation entirely with `enable_validation: false`.

## Rationale

- Delta streaming reduces payload size and simplifies clients (no diffing).
- Mirrors common provider semantics (e.g., OpenAI chat delta events).
- `tokens` remains for compatibility; prefer `deltas` going forward.

