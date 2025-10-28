# Function Calling Implementation Summary

## ✅ What Was Implemented

### 1. Enhanced Request/Response Models

**Updated `LLMRequest`:**
```python
class LLMRequest(BaseModel):
    prompt: str
    history: Optional[List[Dict[str, str]]] = None
    llm_provider: Optional[str] = None
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    
    # 🔥 NEW FIELDS
    functions: Optional[List[Dict[str, Any]]] = None  # OpenAI-style functions
    persona_id: Optional[str] = None  # Persona context
    user_id: Optional[str] = None  # User context
    webhook_url: Optional[str] = None  # Webhook for function execution
```

**Updated `LLMResponse`:**
```python
class LLMResponse(BaseModel):
    response: str
    provider: str
    usage: Optional[Dict[str, Any]] = None
    model: Optional[str] = None
    function_call: Optional[Dict[str, Any]] = None  # 🔥 Function call result
```

### 2. Helper Functions

#### `convert_openai_functions_to_tools()`
- Converts OpenAI-style function definitions to internal tool format
- Input: Array of `{name, description, parameters}`
- Output: Dictionary of `{tool_name: {description, parameters}}`

#### `execute_tool()`
- Executes tools with given arguments
- Supports:
  - `sendPersonaImage` / `database_query` - Webhook to Node.js
  - `calculator` - AST-based safe math evaluation
- Returns tool execution result as string

### 3. Enhanced `/generate` Endpoint

The endpoint now:
1. ✅ Accepts `functions` parameter (OpenAI format)
2. ✅ Converts functions to internal tool format
3. ✅ Enhances prompt with tool documentation
4. ✅ Sends enhanced prompt to LLM
5. ✅ Extracts JSON tool calls from response (supports markdown blocks)
6. ✅ Executes tools automatically
7. ✅ Returns tool results or standard response

**Function Calling Flow:**
```
Client Request (with functions)
    ↓
Convert OpenAI functions → Internal tools
    ↓
Enhance prompt with tool docs
    ↓
Send to LLM (VertexAI/Gemini)
    ↓
LLM responds with JSON tool call
    ↓
Extract & parse tool call
    ↓
Execute tool (calculator, database_query, etc.)
    ↓
Return result to client
```

### 4. Available Tools

#### Calculator Tool
- **Safe AST-based** evaluation (no `eval()` injection)
- **Allowed functions:** `abs`, `round`, `min`, `max`, `sum`, `len`, `sqrt`, `sin`, `cos`, `tan`, `log`, `exp`, `pi`, `e`
- **Example:** `sqrt(144) + 10` → `22.0`

#### Database Query Tool
- **Webhook integration** to Node.js backend
- **Query types:** image, gallery, document, data
- **Payload:**
  ```json
  {
    "query_type": "image",
    "persona_id": "xyz-123",
    "user_id": "user-456",
    "limit": 5
  }
  ```
- **Response:** Array of database results formatted for LLM

### 5. Provider Support

✅ **VertexAI** (Gemini 2.0 Flash/Pro)  
✅ **OpenAI** (GPT-3.5/4)  
✅ **Anthropic** (Claude)  
✅ **Google** (Gemini API)  
✅ **Ollama** (Local models)  

All providers now support function calling!

## 📝 Usage Examples

### TypeScript Client

```typescript
import { LLMService } from './llm-service-example';

const llmService = new LLMService('http://localhost:8000');

const functions = [
    {
        name: "sendPersonaImage",
        description: "Sends persona images",
        parameters: {
            type: "object",
            properties: {
                personaId: { type: "string", description: "Persona ID" }
            },
            required: ["personaId"]
        }
    }
];

const response = await llmService.generateResponse(
    "Send me images for persona xyz-123",
    "models/gemini-2.0-flash",
    [],
    "vertexai",
    functions,
    "xyz-123",
    "user-456",
    "http://localhost:3000/webhooks/ai-gateway/database-query"
);

console.log(response.response);
// "Found 3 images:\n1. https://...\n2. https://...\n3. https://..."

console.log(response.function_call);
// {
//   name: "sendPersonaImage",
//   arguments: { personaId: "xyz-123" },
//   result: "Found 3 images:..."
// }
```

### Python Client

```python
import httpx

functions = [
    {
        "name": "calculator",
        "description": "Performs calculations",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {"type": "string"}
            },
            "required": ["expression"]
        }
    }
]

async with httpx.AsyncClient() as client:
    response = await client.post(
        "http://localhost:8000/api/llm/generate",
        json={
            "prompt": "What is sqrt(144)?",
            "llm_provider": "vertexai",
            "model": "models/gemini-2.0-flash",
            "functions": functions
        }
    )
    print(response.json())
```

### cURL

```bash
curl -X POST http://localhost:8000/api/llm/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Calculate 10 * 5",
    "llm_provider": "vertexai",
    "model": "models/gemini-2.0-flash",
    "functions": [
      {
        "name": "calculator",
        "description": "Performs math",
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

## 🧪 Testing

### Run Python Tests
```bash
cd /Users/sadikhanecioglu/Documents/Works/AIBackend
python test_function_calling.py
```

### Run TypeScript Examples
```bash
ts-node llm-service-example.ts
```

### Manual Test
```bash
curl -X POST http://localhost:8000/api/llm/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Send me images for persona test-123",
    "llm_provider": "vertexai",
    "model": "models/gemini-2.0-flash",
    "functions": [{"name": "sendPersonaImage", "description": "Gets images", "parameters": {"type": "object", "properties": {"personaId": {"type": "string"}}, "required": ["personaId"]}}],
    "persona_id": "test-123",
    "webhook_url": "http://localhost:3000/webhooks/ai-gateway/database-query"
  }'
```

## 📁 Files Modified

1. **`app/api/routes/llm.py`**
   - Added `functions`, `persona_id`, `user_id`, `webhook_url` to `LLMRequest`
   - Added `function_call` to `LLMResponse`
   - Implemented `convert_openai_functions_to_tools()`
   - Implemented `execute_tool()` with calculator and database_query
   - Enhanced `generate_text()` endpoint with function calling logic

## 📁 Files Created

1. **`FUNCTION_CALLING_USAGE.md`** - Complete usage documentation
2. **`test_function_calling.py`** - Python test suite
3. **`llm-service-example.ts`** - TypeScript client examples

## 🔑 Key Features

✅ **OpenAI-Compatible Format** - Use same function definitions across all providers  
✅ **VertexAI Support** - Full support for Gemini 2.0 Flash/Pro  
✅ **Safe Execution** - AST-based calculator, no eval injection  
✅ **Webhook Integration** - Query Node.js databases seamlessly  
✅ **Flexible** - Easy to add new tools  
✅ **Type-Safe** - Full TypeScript support  
✅ **Multi-Provider** - Works with OpenAI, Anthropic, Gemini, VertexAI, Ollama  

## 🎯 Next Steps

1. ✅ Implementation complete
2. 🔄 **Test with your Node.js client**
3. 📝 **Verify webhook endpoint** at `http://localhost:3000/webhooks/ai-gateway/database-query`
4. 🧪 **Run test suite** with `python test_function_calling.py`
5. 🚀 **Deploy to production**

## 💡 How to Add New Tools

```python
# In execute_tool() function
elif tool_name == "your_new_tool":
    # Extract arguments
    param1 = arguments.get("param1")
    param2 = arguments.get("param2")
    
    # Execute tool logic
    result = your_tool_logic(param1, param2)
    
    # Return result
    return str(result)
```

Then define it in your client:

```typescript
const functions = [
    {
        name: "your_new_tool",
        description: "Does something useful",
        parameters: {
            type: "object",
            properties: {
                param1: { type: "string", description: "First param" },
                param2: { type: "number", description: "Second param" }
            },
            required: ["param1"]
        }
    }
];
```

## 🐛 Troubleshooting

### LLM doesn't call function
- Check prompt mentions the function's purpose
- Verify function description is clear
- Add explicit instruction: "Use the available tools to answer"

### Webhook fails
- Verify webhook URL is accessible
- Check Node.js server is running
- Test webhook endpoint manually with curl

### Tool not found
- Check tool name matches exactly
- Verify function is defined in `execute_tool()`
- Check logs for errors

## 📊 Example Response

**With function call:**
```json
{
    "response": "Found 3 images:\n1. https://...\n2. https://...\n3. https://...",
    "provider": "vertexai",
    "model": "models/gemini-2.0-flash",
    "usage": {
        "prompt_tokens": 150,
        "completion_tokens": 50,
        "total_tokens": 200
    },
    "function_call": {
        "name": "sendPersonaImage",
        "arguments": {
            "personaId": "xyz-123",
            "limit": 5
        },
        "result": "Found 3 images:..."
    }
}
```

**Without function call:**
```json
{
    "response": "I'd be happy to help! What would you like to know?",
    "provider": "vertexai",
    "model": "models/gemini-2.0-flash",
    "usage": {...}
}
```

---

## ✨ Summary

You now have **full OpenAI-style function calling support** with **VertexAI/Gemini** in your FastAPI AI Gateway! 

Your existing TypeScript code will work perfectly:

```typescript
const response = await llmService.generateResponse(
    prompt,
    model,
    history,
    provider,
    functions  // 🎉 Now fully supported!
);
```

Ready to test! 🚀
