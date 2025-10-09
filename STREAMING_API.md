# Streaming API Documentation

## Delta Streaming (Efficient Token Delivery)

The handler now emits **only new text deltas** instead of cumulative chunks, reducing bandwidth and simplifying client implementation.

### Response Format

#### Streaming Response (`stream: true`)
```json
{
  "choices": [{
    "text": "I recommend learning Python for backend development...",
    "deltas": [
      "I",
      " recommend",
      " learning",
      " Python",
      " for",
      " backend",
      " development",
      "..."
    ],
    "tokens": ["I", " recommend", ...]  // Alias for backward compatibility
  }],
  "usage": {
    "prompt_tokens": 15,
    "completion_tokens": 42,
    "total_tokens": 57
  },
  "validation": {
    "valid": true,
    "issues": [],
    "sanitized": false,
    "intent": "CAREER_GUIDANCE"
  },
  "streaming": true
}
```

#### Non-Streaming Response (`stream: false`)
```json
{
  "choices": [{
    "text": "I recommend learning Python for backend development...",
    "deltas": null,
    "tokens": ["I recommend learning Python for backend development..."]
  }],
  "usage": {
    "input": 15,
    "output": 42
  },
  "streaming": false
}
```

### During Streaming (Individual Chunks)

Each chunk emitted during streaming:

**Intermediate Chunk:**
```json
{
  "delta": " Python",      // Only new text
  "finished": false,
  "offset": 18             // Starting position in full text
}
```

**Final Chunk:**
```json
{
  "delta": ".",            // Any remaining text
  "text": "I recommend learning Python for backend development.",  // Full text
  "deltas": ["I", " recommend", " learning", " Python", ...],      // All deltas
  "finished": true,
  "usage": {
    "prompt_tokens": 15,
    "completion_tokens": 42,
    "total_tokens": 57
  }
}
```

## Client Implementation

### JavaScript/TypeScript Example

```javascript
// Simple delta rendering
const responseDiv = document.getElementById('response');
let fullText = '';

async function streamResponse(prompt) {
  const response = await fetch(ENDPOINT_URL, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${API_KEY}`,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      input: {
        prompt: prompt,
        sampling_params: { stream: true, max_tokens: 150 },
        enable_validation: true
      }
    })
  });

  const reader = response.body.getReader();
  const decoder = new TextDecoder();

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    const chunk = JSON.parse(decoder.decode(value));

    if (chunk.delta) {
      // Append only new text - no diffing needed!
      fullText += chunk.delta;
      responseDiv.textContent = fullText;
    }

    if (chunk.finished) {
      console.log('Usage:', chunk.usage);
      console.log('Validation:', chunk.validation);
    }
  }
}
```

### Python Example

```python
import requests
import json

def stream_response(prompt: str):
    response = requests.post(
        ENDPOINT_URL,
        headers={
            'Authorization': f'Bearer {API_KEY}',
            'Content-Type': 'application/json'
        },
        json={
            'input': {
                'prompt': prompt,
                'sampling_params': {'stream': True, 'max_tokens': 150},
                'enable_validation': True
            }
        },
        stream=True
    )

    full_text = ''
    for line in response.iter_lines():
        if line:
            chunk = json.loads(line)

            if 'delta' in chunk:
                # Append only new text
                delta = chunk['delta']
                full_text += delta
                print(delta, end='', flush=True)

            if chunk.get('finished'):
                print(f"\n\nUsage: {chunk['usage']}")
                print(f"Validation: {chunk.get('validation', {})}")
```

## Benefits of Delta Streaming

### 1. Lower Bandwidth
```
Old (cumulative): "I" + "I am" + "I am happy" = 13 chars transmitted
New (deltas):     "I" + " am" + " happy"      = 10 chars transmitted
```
With 150-token responses, this saves ~40% bandwidth.

### 2. Simpler Clients
- No diffing logic needed
- No edge cases with Unicode, whitespace, or special chars
- Just append each delta: `fullText += chunk.delta`

### 3. Clear Semantics
- Each chunk represents exactly one "token" or text segment
- Matches OpenAI and other LLM streaming APIs
- `offset` field provides position for advanced UI (cursors, highlights)

### 4. Better UX
- Smoother rendering (no re-rendering entire text)
- Can show typing animation with precise control
- Easy to implement "stop generation" at any delta

## Backward Compatibility

The `tokens` field is aliased to `deltas` for backward compatibility:
- Old clients using `tokens` will still work
- New clients should use `deltas` for clarity
- Both contain the same delta array when streaming

## Error Handling

If a delta is empty (rare), it's still included in the array:
```json
{
  "delta": "",
  "finished": false,
  "offset": 42
}
```

This preserves streaming semantics and allows clients to track offsets accurately.
