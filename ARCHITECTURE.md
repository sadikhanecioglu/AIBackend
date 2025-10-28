# Function Calling Architecture

## System Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                          CLIENT APPLICATION                          │
│                      (TypeScript/Node.js/Python)                    │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 │ POST /api/llm/generate
                                 │ {
                                 │   prompt: "Send images for xyz-123",
                                 │   llm_provider: "vertexai",
                                 │   model: "models/gemini-2.0-flash",
                                 │   functions: [{...}],
                                 │   persona_id: "xyz-123",
                                 │   webhook_url: "http://..."
                                 │ }
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      FASTAPI AI GATEWAY                             │
│                   /api/llm/generate endpoint                        │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ Convert OpenAI Functions│
                    │ to Internal Tool Format │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │  Enhance Prompt with    │
                    │  Tool Documentation     │
                    └────────────┬────────────┘
                                 │
                                 │ Enhanced Prompt
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      LLM PROVIDER                                   │
│               (VertexAI/Gemini 2.0 Flash)                          │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 │ Response:
                                 │ ```json
                                 │ {
                                 │   "tool": "sendPersonaImage",
                                 │   "arguments": {
                                 │     "personaId": "xyz-123"
                                 │   }
                                 │ }
                                 │ ```
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    TOOL EXECUTION ENGINE                            │
│                    execute_tool() function                          │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │                         │
                    ▼                         ▼
        ┌──────────────────────┐  ┌──────────────────────┐
        │   Calculator Tool    │  │  Database Query Tool │
        │   (AST Evaluation)   │  │   (Webhook to Node)  │
        └──────────────────────┘  └──────────┬───────────┘
                    │                         │
                    │                         │ POST webhook
                    │                         │ {
                    │                         │   query_type: "image",
                    │                         │   persona_id: "xyz-123",
                    │                         │   user_id: "user-456"
                    │                         │ }
                    │                         ▼
                    │             ┌──────────────────────────┐
                    │             │   NODE.JS BACKEND        │
                    │             │   Webhook Handler        │
                    │             └────────────┬─────────────┘
                    │                         │
                    │                         │ Query Database
                    │                         ▼
                    │             ┌──────────────────────────┐
                    │             │      DATABASE            │
                    │             │   (Prisma/Images)        │
                    │             └────────────┬─────────────┘
                    │                         │
                    │                         │ Results
                    │                         ▼
                    │             ┌──────────────────────────┐
                    │             │  {                       │
                    │             │    success: true,        │
                    │             │    data: [               │
                    │             │      {url: "...", ...}   │
                    │             │    ]                     │
                    │             │  }                       │
                    │             └────────────┬─────────────┘
                    │                         │
                    ▼                         ▼
        ┌────────────────────────────────────────┐
        │         Format Results                 │
        │    "Found 3 images:..."                │
        └────────────────┬───────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    FASTAPI RESPONSE                                 │
│  {                                                                  │
│    response: "Found 3 images:\n1. ...\n2. ...\n3. ...",           │
│    provider: "vertexai",                                           │
│    function_call: {                                                │
│      name: "sendPersonaImage",                                     │
│      arguments: { personaId: "xyz-123" },                          │
│      result: "Found 3 images:..."                                  │
│    }                                                               │
│  }                                                                  │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          CLIENT APPLICATION                          │
│                    Receives formatted results                       │
└─────────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Client Application
- **Languages:** TypeScript, Python, JavaScript
- **Sends:** OpenAI-style function definitions
- **Receives:** Formatted results with optional function_call metadata

### 2. FastAPI AI Gateway
- **Endpoint:** `/api/llm/generate`
- **Functions:**
  - `convert_openai_functions_to_tools()` - Format conversion
  - `execute_tool()` - Tool execution dispatcher
  - `generate_text()` - Main endpoint with function calling

### 3. LLM Provider
- **Supported:** VertexAI, OpenAI, Anthropic, Gemini, Ollama
- **Input:** Enhanced prompt with tool documentation
- **Output:** Tool call in JSON format or standard text

### 4. Tool Execution
- **Calculator:** AST-based safe math evaluation
- **Database Query:** Webhook to Node.js backend
- **Extensible:** Easy to add new tools

### 5. Node.js Backend
- **Webhook Endpoint:** `/webhooks/ai-gateway/database-query`
- **Database:** Prisma ORM
- **Returns:** Formatted query results

## Data Flow Example

### Request
```json
{
  "prompt": "Send me images for persona xyz-123",
  "llm_provider": "vertexai",
  "model": "models/gemini-2.0-flash",
  "functions": [
    {
      "name": "sendPersonaImage",
      "description": "Retrieves persona images",
      "parameters": {
        "type": "object",
        "properties": {
          "personaId": { "type": "string" }
        },
        "required": ["personaId"]
      }
    }
  ],
  "persona_id": "xyz-123",
  "user_id": "user-456",
  "webhook_url": "http://localhost:3000/webhooks/ai-gateway/database-query"
}
```

### Enhanced Prompt
```
Send me images for persona xyz-123

🔧 AVAILABLE TOOLS:

**sendPersonaImage**
Description: Retrieves persona images
Parameters:
  - personaId (string) (required): The persona ID

📋 TO USE A TOOL, respond with JSON:
```json
{
  "tool": "tool_name",
  "arguments": {...}
}
```
```

### LLM Response
```json
{
  "tool": "sendPersonaImage",
  "arguments": {
    "personaId": "xyz-123"
  }
}
```

### Webhook Payload
```json
{
  "query_type": "image",
  "persona_id": "xyz-123",
  "user_id": "user-456",
  "limit": 5
}
```

### Database Response
```json
{
  "success": true,
  "data": [
    {
      "id": "img-1",
      "url": "https://example.com/image1.jpg",
      "createdAt": "2024-01-15T10:30:00Z"
    },
    {
      "id": "img-2",
      "url": "https://example.com/image2.jpg",
      "createdAt": "2024-01-14T15:20:00Z"
    }
  ]
}
```

### Final Response
```json
{
  "response": "Found 2 images:\n1. https://example.com/image1.jpg\n2. https://example.com/image2.jpg",
  "provider": "vertexai",
  "model": "models/gemini-2.0-flash",
  "usage": {
    "prompt_tokens": 150,
    "completion_tokens": 45,
    "total_tokens": 195
  },
  "function_call": {
    "name": "sendPersonaImage",
    "arguments": {
      "personaId": "xyz-123"
    },
    "result": "Found 2 images:\n1. https://example.com/image1.jpg\n2. https://example.com/image2.jpg"
  }
}
```

## Security Features

✅ **Safe Calculator** - AST-based, no `eval()` injection  
✅ **Webhook Validation** - Requires explicit webhook_url  
✅ **Parameter Validation** - Type checking via Pydantic  
✅ **Error Handling** - Comprehensive try/catch blocks  
✅ **Timeout Protection** - 10s webhook timeout  

## Extensibility

### Adding New Tools

1. **Define tool in client:**
```typescript
{
  name: "weather",
  description: "Gets current weather",
  parameters: {
    type: "object",
    properties: {
      city: { type: "string" }
    },
    required: ["city"]
  }
}
```

2. **Implement in `execute_tool()`:**
```python
elif tool_name == "weather":
    city = arguments.get("city")
    # Call weather API
    weather_data = get_weather(city)
    return f"Weather in {city}: {weather_data}"
```

3. **Use it:**
```typescript
const response = await llmService.generateResponse(
    "What's the weather in Paris?",
    "models/gemini-2.0-flash",
    [],
    "vertexai",
    [weatherFunction]
);
```

## Performance

- **Latency:** ~2-5s for function execution (including LLM + tool execution)
- **Webhook Timeout:** 10s (configurable)
- **Concurrent Requests:** Async/await for parallel processing
- **Caching:** Can be added at LLM or database layer

## Monitoring

Logs include:
- 🔧 Function calling enabled
- 🤖 LLM Response preview
- 🔍 Extracted tool call
- 🛠️ Tool execution details
- 🔔 Webhook requests
- ✅ Success confirmations
- ❌ Error details

Example log:
```
INFO: 🔧 Function calling enabled with 1 functions
INFO: 🤖 LLM Response: ```json\n{"tool": "sendPersonaImage", ...
INFO: 🔍 Extracted tool call from markdown: {'tool': 'sendPersonaImage', ...}
INFO: 🛠️ Executing tool: sendPersonaImage with args: {'personaId': 'xyz-123'}
INFO: 🔔 Sending webhook to http://localhost:3000/webhooks/...
INFO: ✅ Webhook response: {'success': True, 'data': [...]}
```
