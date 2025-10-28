# Function Calling with VertexAI Support

## Overview

The `/api/llm/generate` endpoint now supports **OpenAI-style function calling** with **VertexAI/Gemini** support!

## API Updates

### Request Model
```python
class LLMRequest(BaseModel):
    prompt: str
    history: Optional[List[Dict[str, str]]] = None
    llm_provider: Optional[str] = None
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    
    # 🔥 NEW: Function calling support
    functions: Optional[List[Dict[str, Any]]] = None
    persona_id: Optional[str] = None
    user_id: Optional[str] = None
    webhook_url: Optional[str] = None
```

### Response Model
```python
class LLMResponse(BaseModel):
    response: str
    provider: str
    usage: Optional[Dict[str, Any]] = None
    model: Optional[str] = None
    
    # 🔥 NEW: Function call result
    function_call: Optional[Dict[str, Any]] = None
```

## Function Format

Use **OpenAI-style** function definitions:

```typescript
const functions = [
    {
        name: "sendPersonaImage",
        description: "Sends persona image to the user",
        parameters: {
            type: "object",
            properties: {
                personaId: {
                    type: "string",
                    description: "The ID of the persona"
                },
                limit: {
                    type: "number",
                    description: "Maximum number of images to return"
                }
            },
            required: ["personaId"]
        }
    }
];
```

## Usage Examples

### TypeScript Client

```typescript
import { LLMService } from './services/llm-service';

const llmService = new LLMService('http://localhost:8000');

// Define functions
const functions = [
    {
        name: "sendPersonaImage",
        description: "Sends persona images from database to the user",
        parameters: {
            type: "object",
            properties: {
                personaId: {
                    type: "string",
                    description: "The ID of the persona whose images to retrieve"
                },
                limit: {
                    type: "number",
                    description: "Maximum number of images to return (default: 5)"
                }
            },
            required: ["personaId"]
        }
    },
    {
        name: "calculator",
        description: "Performs mathematical calculations",
        parameters: {
            type: "object",
            properties: {
                expression: {
                    type: "string",
                    description: "Mathematical expression to evaluate (e.g., '2 + 2', 'sqrt(16)')"
                }
            },
            required: ["expression"]
        }
    }
];

// Generate with function calling
const response = await llmService.generateResponse(
    "Please send me images of persona xyz-123",
    "models/gemini-2.0-flash",  // VertexAI model
    [],  // history
    "vertexai",  // provider
    functions,  // 🔥 Functions parameter
    "xyz-123",  // persona_id
    "user-456",  // user_id
    "http://localhost:3000/webhooks/ai-gateway/database-query"  // webhook_url
);

console.log(response);
// {
//   response: "Found 3 images:\n1. https://...\n2. https://...\n3. https://...",
//   provider: "vertexai",
//   function_call: {
//     name: "sendPersonaImage",
//     arguments: { personaId: "xyz-123", limit: 5 },
//     result: "Found 3 images:..."
//   }
// }
```

### Python Example

```python
import httpx

# Define functions
functions = [
    {
        "name": "sendPersonaImage",
        "description": "Sends persona images from database",
        "parameters": {
            "type": "object",
            "properties": {
                "personaId": {
                    "type": "string",
                    "description": "The ID of the persona"
                }
            },
            "required": ["personaId"]
        }
    }
]

# Make request
async with httpx.AsyncClient() as client:
    response = await client.post(
        "http://localhost:8000/api/llm/generate",
        json={
            "prompt": "Send me images of persona abc-123",
            "llm_provider": "vertexai",
            "model": "models/gemini-2.0-flash",
            "functions": functions,
            "persona_id": "abc-123",
            "user_id": "user-789",
            "webhook_url": "http://localhost:3000/webhooks/ai-gateway/database-query"
        }
    )
    
    data = response.json()
    print(data)
```

### cURL Example

```bash
curl -X POST http://localhost:8000/api/llm/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Send me images for persona xyz-123",
    "llm_provider": "vertexai",
    "model": "models/gemini-2.0-flash",
    "functions": [
      {
        "name": "sendPersonaImage",
        "description": "Retrieves persona images from database",
        "parameters": {
          "type": "object",
          "properties": {
            "personaId": {
              "type": "string",
              "description": "The persona ID"
            }
          },
          "required": ["personaId"]
        }
      }
    ],
    "persona_id": "xyz-123",
    "user_id": "user-456",
    "webhook_url": "http://localhost:3000/webhooks/ai-gateway/database-query"
  }'
```

## Supported Providers

- ✅ **VertexAI** (Gemini 2.0 Flash/Pro)
- ✅ **OpenAI** (GPT-3.5/4)
- ✅ **Anthropic** (Claude)
- ✅ **Google** (Gemini API)
- ✅ **Ollama** (Local models)

## Available Functions

### 1. `sendPersonaImage` / `database_query`

Queries database for persona-related data via webhook.

**Parameters:**
- `personaId` (string, required): Persona identifier
- `limit` (number, optional): Max results (default: 5)

**Requirements:**
- `webhook_url`: Endpoint to send query
- `persona_id`: Can be in arguments or request
- `user_id`: Optional user context

**Webhook Payload:**
```json
{
    "query_type": "image",
    "persona_id": "xyz-123",
    "user_id": "user-456",
    "limit": 5
}
```

**Expected Response:**
```json
{
    "success": true,
    "data": [
        {"url": "https://...", "id": "img-1"},
        {"url": "https://...", "id": "img-2"}
    ]
}
```

### 2. `calculator`

Safe mathematical expression evaluator.

**Parameters:**
- `expression` (string, required): Math expression

**Allowed functions:**
- Basic: `abs`, `round`, `min`, `max`, `sum`, `len`
- Math: `sqrt`, `sin`, `cos`, `tan`, `log`, `exp`, `pi`, `e`

**Examples:**
- `"2 + 2"` → `"4"`
- `"sqrt(16)"` → `"4.0"`
- `"pi * 2"` → `"6.283185307179586"`

## How It Works

1. **Client sends request** with `functions` array
2. **FastAPI converts** OpenAI format → internal tool format
3. **Prompt is enhanced** with tool documentation
4. **LLM decides** whether to use a tool
5. **LLM responds** with JSON tool call (if needed):
   ```json
   {
     "tool": "sendPersonaImage",
     "arguments": {"personaId": "xyz-123"}
   }
   ```
6. **FastAPI extracts** JSON from response (supports markdown code blocks)
7. **FastAPI executes** the tool:
   - `sendPersonaImage` → Webhook to Node.js → Database query
   - `calculator` → Safe AST evaluation
8. **FastAPI returns** tool result to client

## Node.js Webhook Handler

Your Node.js server should implement:

```typescript
// webhook-handler.ts
import express from 'express';
import { ImageService } from './services/image-service';

const router = express.Router();
const imageService = new ImageService();

router.post('/webhooks/ai-gateway/database-query', async (req, res) => {
    try {
        const { query_type, persona_id, user_id, limit } = req.body;
        
        if (query_type === 'image') {
            const images = await imageService.getPersonaImages(
                persona_id,
                user_id,
                limit || 5
            );
            
            return res.json({
                success: true,
                data: images.map(img => ({
                    id: img.id,
                    url: img.url,
                    createdAt: img.createdAt
                }))
            });
        }
        
        res.status(400).json({
            success: false,
            error: 'Invalid query_type'
        });
        
    } catch (error) {
        console.error('Webhook error:', error);
        res.status(500).json({
            success: false,
            error: error.message
        });
    }
});

export default router;
```

```typescript
// services/image-service.ts
import { PrismaClient } from '@prisma/client';

export class ImageService {
    private prisma = new PrismaClient();
    
    async getPersonaImages(personaId: string, userId: string, limit: number) {
        return this.prisma.image.findMany({
            where: {
                personaId: personaId,
                userId: userId
            },
            orderBy: {
                createdAt: 'desc'
            },
            take: limit
        });
    }
}
```

## LLM Service Client

```typescript
// services/llm-service.ts
export class LLMService {
    private baseUrl: string;
    
    constructor(baseUrl: string) {
        this.baseUrl = baseUrl;
    }
    
    async generateResponse(
        prompt: string,
        model: string,
        history: Array<{ role: string; content: string }>,
        provider: string,
        functions?: Array<{
            name: string;
            description: string;
            parameters: any;
        }>,
        personaId?: string,
        userId?: string,
        webhookUrl?: string
    ) {
        const response = await fetch(`${this.baseUrl}/api/llm/generate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                prompt,
                model,
                history,
                llm_provider: provider,
                functions,
                persona_id: personaId,
                user_id: userId,
                webhook_url: webhookUrl
            })
        });
        
        return response.json();
    }
}
```

## Testing

### Test with VertexAI

```bash
# Make sure VertexAI is configured
export VERTEXAI_PROJECT_ID="your-project-id"
export VERTEXAI_SERVICE_ACCOUNT_JSON="path/to/service-account.json"

# Test function calling
curl -X POST http://localhost:8000/api/llm/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Calculate sqrt(144) + 10",
    "llm_provider": "vertexai",
    "model": "models/gemini-2.0-flash",
    "functions": [
      {
        "name": "calculator",
        "description": "Performs math calculations",
        "parameters": {
          "type": "object",
          "properties": {
            "expression": {"type": "string"}
          },
          "required": ["expression"]
        }
      }
    ]
  }'
```

### Test with Image Query

```bash
curl -X POST http://localhost:8000/api/llm/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Show me images for persona test-123",
    "llm_provider": "vertexai",
    "model": "models/gemini-2.0-flash",
    "functions": [
      {
        "name": "sendPersonaImage",
        "description": "Retrieves persona images",
        "parameters": {
          "type": "object",
          "properties": {
            "personaId": {"type": "string"}
          },
          "required": ["personaId"]
        }
      }
    ],
    "persona_id": "test-123",
    "user_id": "user-456",
    "webhook_url": "http://localhost:3000/webhooks/ai-gateway/database-query"
  }'
```

## Benefits

✅ **OpenAI-compatible**: Use same function format across all providers  
✅ **VertexAI support**: Works with Gemini 2.0 Flash/Pro  
✅ **Webhook integration**: Query Node.js databases seamlessly  
✅ **Safe execution**: AST-based calculator, no `eval()` injection  
✅ **Flexible**: Easy to add new tools  
✅ **Type-safe**: Full TypeScript support  

## Next Steps

1. ✅ Implement function calling in `/generate` endpoint
2. ✅ Add VertexAI support
3. ✅ Integrate webhook system
4. 🔄 Test with your Node.js client
5. 📝 Add more functions as needed
