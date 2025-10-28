# 🚀 Quick Reference - Function Calling API

## Basic Usage

### TypeScript
```typescript
import { LLMService } from './llm-service-example';

const llm = new LLMService('http://localhost:8000');

const functions = [{
    name: "sendPersonaImage",
    description: "Retrieves persona images from database",
    parameters: {
        type: "object",
        properties: {
            personaId: { type: "string", description: "Persona ID" }
        },
        required: ["personaId"]
    }
}];

const response = await llm.generateResponse(
    "Send me images for xyz-123",
    "models/gemini-2.0-flash",
    [],
    "vertexai",
    functions,
    "xyz-123",
    "user-456",
    "http://localhost:3000/webhooks/ai-gateway/database-query"
);
```

### Python
```python
import httpx

async with httpx.AsyncClient() as client:
    r = await client.post("http://localhost:8000/api/llm/generate", json={
        "prompt": "Calculate sqrt(144)",
        "llm_provider": "vertexai",
        "model": "models/gemini-2.0-flash",
        "functions": [{
            "name": "calculator",
            "description": "Math calculations",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"]
            }
        }]
    })
    print(r.json())
```

### cURL
```bash
curl -X POST http://localhost:8000/api/llm/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Your prompt here",
    "llm_provider": "vertexai",
    "model": "models/gemini-2.0-flash",
    "functions": [...]
  }'
```

## Function Format

```typescript
{
    name: string;              // Function identifier
    description: string;       // What it does
    parameters: {
        type: "object";
        properties: {
            paramName: {
                type: string;       // "string", "number", "boolean", "array", "object"
                description?: string;
            }
        };
        required?: string[];    // Required parameter names
    };
}
```

## Available Tools

| Tool | Description | Parameters | Example |
|------|-------------|------------|---------|
| `sendPersonaImage` | Database query via webhook | `personaId` (string) | Query persona images |
| `database_query` | Generic DB query | `personaId` (string) | Same as above |
| `calculator` | Safe math eval | `expression` (string) | `"sqrt(144) + 10"` |

## Request Parameters

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `prompt` | string | ✅ | User prompt |
| `llm_provider` | string | ❌ | Provider name (default: config) |
| `model` | string | ❌ | Model name (default: config) |
| `history` | array | ❌ | Chat history |
| `temperature` | float | ❌ | Temperature (0-1) |
| `max_tokens` | int | ❌ | Max response tokens |
| `functions` | array | ❌ | Function definitions |
| `persona_id` | string | ❌ | Persona context |
| `user_id` | string | ❌ | User context |
| `webhook_url` | string | ❌ | Webhook endpoint |

## Response Format

```typescript
{
    response: string;           // Tool result or LLM response
    provider: string;           // Provider used
    model?: string;             // Model used
    usage?: {                   // Token usage
        prompt_tokens?: number;
        completion_tokens?: number;
        total_tokens?: number;
    };
    function_call?: {           // If function was called
        name: string;
        arguments: object;
        result: string;
    };
}
```

## Supported Providers

| Provider | Model Example | Notes |
|----------|---------------|-------|
| `vertexai` | `models/gemini-2.0-flash` | ✅ Recommended |
| `vertexai` | `models/gemini-2.5-pro` | ✅ Recommended |
| `openai` | `gpt-3.5-turbo` | ✅ Supported |
| `openai` | `gpt-4` | ✅ Supported |
| `gemini` | `models/gemini-2.5-flash` | ✅ Supported |
| `anthropic` | `claude-3-haiku-20240307` | ✅ Supported |
| `ollama` | `llama3.1:latest` | ✅ Supported |

## Environment Variables

```bash
# VertexAI (Required for VertexAI provider)
VERTEXAI_PROJECT_ID=your-project-id
VERTEXAI_SERVICE_ACCOUNT_JSON=/path/to/service-account.json

# OpenAI (Optional)
OPENAI_API_KEY=sk-...

# Gemini (Optional)
GOOGLE_API_KEY=AI...

# Anthropic (Optional)
ANTHROPIC_API_KEY=sk-ant-...
```

## Testing

```bash
# Run tests
cd /Users/sadikhanecioglu/Documents/Works/AIBackend
python test_function_calling.py

# Or specific test
python -c "
import asyncio
import httpx

async def test():
    async with httpx.AsyncClient() as client:
        r = await client.post('http://localhost:8000/api/llm/generate', json={
            'prompt': 'Calculate 10 * 5',
            'llm_provider': 'vertexai',
            'model': 'models/gemini-2.0-flash',
            'functions': [{
                'name': 'calculator',
                'description': 'Math',
                'parameters': {
                    'type': 'object',
                    'properties': {'expression': {'type': 'string'}},
                    'required': ['expression']
                }
            }]
        })
        print(r.json())

asyncio.run(test())
"
```

## Webhook Implementation

```typescript
// Node.js webhook endpoint
app.post('/webhooks/ai-gateway/database-query', async (req, res) => {
    const { query_type, persona_id, user_id, limit } = req.body;
    
    if (query_type === 'image') {
        const images = await db.image.findMany({
            where: { personaId: persona_id, userId: user_id },
            take: limit || 5
        });
        
        return res.json({
            success: true,
            data: images.map(img => ({
                id: img.id,
                url: img.url,
                createdAt: img.createdAt
            }))
        });
    }
    
    res.status(400).json({ success: false, error: 'Invalid query_type' });
});
```

## Common Patterns

### Pattern 1: Calculator
```typescript
const functions = [{
    name: "calculator",
    description: "Performs calculations",
    parameters: {
        type: "object",
        properties: {
            expression: { type: "string", description: "Math expression" }
        },
        required: ["expression"]
    }
}];

await llm.generateResponse(
    "What is sqrt(144)?",
    "models/gemini-2.0-flash",
    [],
    "vertexai",
    functions
);
```

### Pattern 2: Database Query
```typescript
const functions = [{
    name: "sendPersonaImage",
    description: "Gets persona images",
    parameters: {
        type: "object",
        properties: {
            personaId: { type: "string" }
        },
        required: ["personaId"]
    }
}];

await llm.generateResponse(
    "Show images for xyz-123",
    "models/gemini-2.0-flash",
    [],
    "vertexai",
    functions,
    "xyz-123",
    "user-456",
    "http://localhost:3000/webhooks/ai-gateway/database-query"
);
```

### Pattern 3: Multiple Functions
```typescript
const functions = [
    {
        name: "calculator",
        description: "Math calculations",
        parameters: { /* ... */ }
    },
    {
        name: "sendPersonaImage",
        description: "Gets images",
        parameters: { /* ... */ }
    }
];

await llm.generateResponse(
    "Calculate 5*5 and show images for xyz-123",
    "models/gemini-2.0-flash",
    [],
    "vertexai",
    functions,
    "xyz-123",
    "user-456",
    "http://localhost:3000/webhooks/ai-gateway/database-query"
);
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Function not called | Add explicit instruction in prompt |
| Webhook timeout | Increase timeout in execute_tool() |
| Invalid function format | Validate against OpenAI schema |
| Tool not found | Check tool name in execute_tool() |
| VertexAI auth error | Verify VERTEXAI env vars |

## Documentation

- 📖 Full Usage Guide: `FUNCTION_CALLING_USAGE.md`
- 🏗️ Architecture: `ARCHITECTURE.md`
- 📝 Summary: `IMPLEMENTATION_SUMMARY.md`
- 🧪 Tests: `test_function_calling.py`
- 💻 Examples: `llm-service-example.ts`

## Status

✅ **Production Ready**
- OpenAI-compatible function format
- VertexAI/Gemini support
- Safe calculator (AST-based)
- Webhook integration
- Multi-provider support
- Full TypeScript types
- Comprehensive tests
- Complete documentation
