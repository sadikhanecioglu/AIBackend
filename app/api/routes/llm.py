"""
LLM endpoints for text generation
"""

import logging
from typing import Dict, Any, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from llm_provider import LLMProviderFactory, GenerationRequest, ToolFunction
from llm_provider.providers import VertexAI
from llm_provider.utils import run_with_tools_async
from llm_provider.utils.config import (
    VertexAIConfig,
    OpenAIConfig,
    AnthropicConfig,
    ProviderConfig,
    GeminiConfig,
    OllamaConfig,
)
from app.config.models import AIGatewayConfig
from app.api.dependencies import get_config


# Standard factory using llm-provider-factory for all providers


logger = logging.getLogger(__name__)
router = APIRouter()


class LLMRequest(BaseModel):
    """Request model for LLM endpoint"""

    prompt: str
    history: Optional[List[Dict[str, str]]] = None
    llm_provider: Optional[str] = None
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    functions: Optional[List[Dict[str, Any]]] = None  # 🔥 Function calling support


class LLMResponse(BaseModel):
    """Response model for LLM endpoint"""

    response: str
    provider: str
    usage: Optional[Dict[str, Any]] = None
    model: Optional[str] = None
    function_call: Optional[Dict[str, Any]] = None  # 🔥 Function call result


class EmbeddingRequest(BaseModel):
    """Request model for embedding endpoint"""

    text: Optional[str] = None
    texts: Optional[List[str]] = None
    embedding_provider: Optional[str] = None
    model: Optional[str] = None


class EmbeddingResponse(BaseModel):
    """Response model for embedding endpoint"""

    embeddings: List[List[float]]
    provider: str
    model: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None
    dimensions: Optional[int] = None


def convert_openai_functions_to_tools(
    functions: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """
    Convert OpenAI-style functions to internal tool format.

    OpenAI format:
    {
        "name": "sendPersonaImage",
        "description": "...",
        "parameters": {
            "type": "object",
            "properties": {...},
            "required": [...]
        }
    }

    Internal format:
    {
        "sendPersonaImage": {
            "description": "...",
            "parameters": {...}
        }
    }
    """
    tools = {}
    for func in functions:
        name = func.get("name")
        if not name:
            continue

        tools[name] = {
            "description": func.get("description", ""),
            "parameters": func.get("parameters", {}),
        }

    return tools


@router.post("/generate", response_model=LLMResponse)
async def generate_text(
    request: LLMRequest, config: AIGatewayConfig = Depends(get_config)
) -> LLMResponse:
    """Generate text response using specified or configured LLM provider with optional function calling"""

    if not request.prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")

    # Use specified provider or default from config
    provider_name = request.llm_provider or config.llm_provider
    temperature = request.temperature or config.llm_temperature
    max_tokens = request.max_tokens or config.llm_max_tokens

    logger.info(
        f"LLM request - provider: {provider_name}, prompt length: {len(request.prompt)}"
    )

    try:
        # Get API key and model for the provider
        api_key = None
        model = None

        if provider_name == "openai":
            api_key = config.openai_api_key
            model = request.model or "gpt-3.5-turbo"
        elif provider_name == "anthropic":
            api_key = config.anthropic_api_key
            model = request.model or "claude-3-haiku-20240307"
        elif provider_name == "google" or provider_name == "gemini":
            api_key = config.google_api_key
            # Support both gemini models, default to flash
            if hasattr(request, "model") and request.model:
                if "2.5-pro" in request.model.lower():
                    model = "models/gemini-2.5-pro"
                elif "2.5-flash" in request.model.lower():
                    model = "models/gemini-2.5-flash"
                else:
                    model = (
                        request.model
                        if request.model.startswith("models/")
                        else f"models/{request.model}"
                    )
            else:
                model = "models/gemini-2.5-flash"  # default
        elif provider_name == "azure":
            api_key = config.azure_openai_api_key
            model = request.model or "gpt-35-turbo"
        elif provider_name == "vertexai":
            projectid = config.vertexai_project_id
            service_account_json = config.vertexai_service_account_json
            model = request.model or config.vertexai_model or "mistral-large-2411"

            if not projectid:
                raise HTTPException(
                    status_code=500,
                    detail="VertexAI Project ID not configured (VERTEXAI_PROJECT_ID)",
                )
            if not service_account_json:
                raise HTTPException(
                    status_code=500,
                    detail="VertexAI service account JSON path not configured (VERTEXAI_SERVICE_ACCOUNT_JSON)",
                )
        elif provider_name == "ollama":
            base_url = config.ollama_base_url or "http://localhost:11434"
            model = request.model or config.ollama_model or "llama3.1:latest"

        if not api_key and provider_name not in ["vertexai", "ollama"]:
            raise HTTPException(
                status_code=500,
                detail=f"API key not configured for provider: {provider_name}",
            )

        # Create LLM provider using enhanced factory
        if provider_name == "vertexai":
            logger.info(
                f"🔍 Creating VertexAI provider with config: project_id={projectid}, model={model}"
            )

            vertexai_config = VertexAIConfig(
                project_id=projectid,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                service_account_json=service_account_json,
            )

            # Use standard factory
            factory = LLMProviderFactory()
            llm_provider = factory.create_provider(provider_name, vertexai_config)
            logger.info(f"✅ VertexAI provider created: {type(llm_provider)}")
        elif provider_name == "ollama":
            logger.info(
                f"🔍 Creating Ollama provider with config: base_url={base_url}, model={model}"
            )

            ollama_config = OllamaConfig(
                base_url=base_url,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            # Use standard factory
            factory = LLMProviderFactory()
            llm_provider = factory.create_provider(provider_name, ollama_config)
            logger.info(f"✅ Ollama provider created: {type(llm_provider)}")
        else:
            # Standard providers - use config classes like test file
            llm_provider = None  # Initialize to ensure it's set

            if provider_name == "openai":
                provider_config = OpenAIConfig(
                    api_key=api_key,
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                # Use factory exactly like test file
                factory = LLMProviderFactory()
                llm_provider = factory.create_provider(provider_name, provider_config)

            elif provider_name == "anthropic":
                provider_config = AnthropicConfig(
                    api_key=api_key,
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                # Use factory exactly like test file
                factory = LLMProviderFactory()
                llm_provider = factory.create_provider(provider_name, provider_config)

            elif provider_name == "google" or provider_name == "gemini":
                # For Gemini, use GeminiConfig
                provider_config = GeminiConfig(
                    api_key=api_key,
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                # Use 'gemini' as provider name for factory
                factory = LLMProviderFactory()
                llm_provider = factory.create_provider("gemini", provider_config)
            else:
                # Fallback to ProviderConfig for other providers
                provider_config = ProviderConfig(
                    api_key=api_key,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    organization=None,
                    base_url=None,
                    timeout=30,
                )

                # Use factory exactly like test file
                factory = LLMProviderFactory()
                llm_provider = factory.create_provider(provider_name, provider_config)

        if llm_provider is None:
            raise HTTPException(
                status_code=500, detail=f"Failed to create provider: {provider_name}"
            )

        # Initialize provider
        await llm_provider.initialize()

        # 🔥 Function Calling Support
        if request.functions and len(request.functions) > 0:
            logger.info(
                f"🔧 Function calling enabled with {len(request.functions)} functions"
            )

            # Convert OpenAI-style functions to tools
            tools = convert_openai_functions_to_tools(request.functions)

            # 🔥 Convert dict tools to ToolFunction list for provider
            tool_functions = []
            for tool_name, tool_info in tools.items():
                tool_func = ToolFunction(
                    name=tool_name,
                    description=tool_info.get("description", ""),
                    parameters=tool_info.get("parameters", {}),
                )
                tool_functions.append(tool_func)

            # Build enhanced prompt with tool documentation
            tool_docs = "\n\n🔧 AVAILABLE TOOLS:\n"
            for tool_name, tool_info in tools.items():
                tool_docs += f"\n**{tool_name}**\n"
                tool_docs += f"Description: {tool_info['description']}\n"

                params = tool_info.get("parameters", {})
                if params:
                    tool_docs += f"Parameters:\n"
                    properties = params.get("properties", {})
                    required = params.get("required", [])

                    for param_name, param_info in properties.items():
                        param_type = param_info.get("type", "any")
                        param_desc = param_info.get("description", "")
                        is_required = " (required)" if param_name in required else ""
                        tool_docs += f"  - {param_name} ({param_type}){is_required}: {param_desc}\n"

            tool_docs += "\n📋 TO USE A TOOL, respond with JSON:\n"
            tool_docs += (
                '```json\n{\n  "tool": "tool_name",\n  "arguments": {...}\n}\n```\n'
            )

            # Add tool documentation to prompt
            enhanced_prompt = request.prompt + tool_docs

            # Create generation request with enhanced prompt AND tools
            generation_request = GenerationRequest(
                prompt=enhanced_prompt,
                history=request.history,
                tools=tool_functions,  # 🔥 Pass tools to provider!
            )

            # Generate response
            response = await llm_provider.generate(generation_request)
            response_text = response.content

            logger.info(f"🤖 LLM Response: {response_text[:200]}...")

            # Check if response contains tool call
            tool_call_data = None

            # Try to extract JSON from markdown code blocks
            import re

            json_match = re.search(
                r"```json\s*(\{.*?\})\s*```", response_text, re.DOTALL
            )
            if json_match:
                try:
                    import json

                    tool_call_data = json.loads(json_match.group(1))
                    logger.info(
                        f"🔍 Extracted tool call from markdown: {tool_call_data}"
                    )
                except:
                    pass

            # Try to parse as direct JSON
            if not tool_call_data:
                try:
                    import json

                    tool_call_data = json.loads(response_text.strip())
                except:
                    pass

            # Return function call info if found (client will execute)
            if tool_call_data and "tool" in tool_call_data:
                tool_name = tool_call_data["tool"]
                arguments = tool_call_data.get("arguments", {})

                logger.info(
                    f"� Function call detected: {tool_name} with args: {arguments}"
                )

                # Get usage information
                usage_info = None
                if hasattr(response, "usage"):
                    usage_info = response.usage

                model_name = None
                if hasattr(response, "model"):
                    model_name = response.model

                # Return function call info - client will execute
                return LLMResponse(
                    response=response_text,  # Original response with JSON
                    provider=provider_name,
                    usage=usage_info,
                    model=model_name,
                    function_call={
                        "name": tool_name,
                        "arguments": arguments,
                    },
                )

            # No tool call found, return standard response
            usage_info = None
            if hasattr(response, "usage"):
                usage_info = response.usage

            model_name = None
            if hasattr(response, "model"):
                model_name = response.model

            return LLMResponse(
                response=response_text,
                provider=provider_name,
                usage=usage_info,
                model=model_name,
            )

        # Standard generation without functions
        # Create generation request - use standard format for all providers
        generation_request = GenerationRequest(
            prompt=request.prompt, history=request.history
        )

        # Generate response
        response = await llm_provider.generate(generation_request)

        # Get usage information if available
        usage_info = None
        if hasattr(response, "usage"):
            usage_info = response.usage

        # Get model name if available
        model_name = None
        if hasattr(response, "model"):
            model_name = response.model

        logger.info(f"LLM response generated successfully with {provider_name}")

        return LLMResponse(
            response=response.content,  # Use response.content instead of response
            provider=provider_name,
            usage=usage_info,
            model=model_name,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"LLM generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate/vertexai/tools")
async def generate_vertexai_with_tools(
    request_data: Dict[str, Any] = Body(...),
    config: AIGatewayConfig = Depends(get_config),
) -> JSONResponse:
    """Generate text using VertexAI (Gemini) with tool support via run_with_tools_async."""

    try:
        # Build VertexAI config from gateway config + request override
        project_id = config.vertexai_project_id
        service_account_json = config.vertexai_service_account_json
        model = (
            request_data.get("model")
            or config.vertexai_model
            or "models/gemini-2.0-flash"
        )

        if not project_id or not service_account_json:
            raise HTTPException(
                status_code=500,
                detail="VertexAI configuration missing: project_id or service_account_json",
            )

        vertex_cfg = VertexAIConfig(
            project_id=project_id,
            location=request_data.get("location", "us-central1"),
            model=model,
            service_account_json=service_account_json,
            max_tokens=request_data.get("max_tokens", config.llm_max_tokens),
            temperature=request_data.get("temperature", config.llm_temperature),
        )

        # Create provider. VertexAI tools often work via the Gemini API, so
        # fall back to creating a 'gemini' provider if the VertexAI provider
        # implementation is incompatible with tools.
        factory = LLMProviderFactory()
        google_api_key = config.google_api_key
        if google_api_key:
            from llm_provider.utils.config import GeminiConfig

            gemini_cfg = GeminiConfig(
                api_key=google_api_key,
                model=model,
                max_tokens=vertex_cfg.max_tokens,
                temperature=vertex_cfg.temperature,
            )
            provider = factory.create_provider("gemini", gemini_cfg)
        else:
            # Try VertexAI provider as a last resort
            provider = factory.create_provider("vertexai", vertex_cfg)

        if provider is None:
            raise HTTPException(
                status_code=500,
                detail="Failed to create LLM provider for VertexAI/tools",
            )

        await provider.initialize()

        # Define available tools
        available_tools_map = {
            "calculator": ToolFunction(
                name="calculator",
                description="Hesaplama yapar",
                parameters={
                    "type": "object",
                    "properties": {"expression": {"type": "string"}},
                    "required": ["expression"],
                },
            ),
            "weather": ToolFunction(
                name="get_weather",
                description="Hava durumu getirir",
                parameters={
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            ),
        }

        # Select requested tools
        requested_tools = request_data.get("available_tools", []) or []
        tools: List[ToolFunction] = [
            available_tools_map[t] for t in requested_tools if t in available_tools_map
        ]

        def execute_tool(name: str, args: Dict[str, Any]) -> str:
            """Synchronous tool executor (run_with_tools_async calls it synchronously)."""
            try:
                if name == "calculator":
                    expr = args.get("expression", "")
                    # safe-ish eval: no builtins, allow math functions
                    import math as _math

                    safe_locals = {
                        k: getattr(_math, k)
                        for k in dir(_math)
                        if not k.startswith("__")
                    }
                    # Evaluate expression (operators are allowed)
                    return str(eval(expr, {"__builtins__": None}, safe_locals))
                elif name == "get_weather":
                    city = args.get("city", "unknown")
                    return f"{city}: 22°C, sunny"
                else:
                    return "Unknown tool"
            except Exception as e:
                logger.exception("Tool execution error")
                return f"Tool error: {e}"

        # If no tools requested, fallback to direct generate
        if not request_data.get("use_tools") or not tools:
            req = GenerationRequest(prompt=request_data.get("prompt", ""))
            resp = await provider.generate(req)
            return JSONResponse(
                {
                    "content": resp.content,
                    "model": getattr(resp, "model", None),
                    "usage": getattr(resp, "usage", None),
                }
            )

        # Run with tools
        resp = await run_with_tools_async(
            provider,
            prompt=request_data.get("prompt", ""),
            tools=tools,
            execute_tool=execute_tool,
            max_iterations=request_data.get("max_iterations", 5),
            temperature=request_data.get("temperature", config.llm_temperature),
        )

        return JSONResponse(
            {
                "content": resp.content,
                "model": getattr(resp, "model", None),
                "usage": getattr(resp, "usage", None),
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("VertexAI tools generation error")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate/vertexai/tools_protocol")
async def generate_vertexai_with_tools_protocol(
    request_data: Dict[str, Any] = Body(...),
    config: AIGatewayConfig = Depends(get_config),
) -> JSONResponse:
    """A simpler, provider-agnostic tool-calling protocol for VertexAI/Gemini.

    The model is instructed to return a JSON object when it wants to call a tool:
    {"tool_call": {"name": "calculator", "arguments": {"expression": "2+2"}}}

    We parse that JSON, run the tool locally, then provide the tool result back to the model
    and ask it to continue. This avoids depending on provider-specific tool_call implementations.
    """

    try:
        # Prepare provider (use Gemini fallback if available)
        project_id = config.vertexai_project_id
        service_account_json = config.vertexai_service_account_json
        model = (
            request_data.get("model")
            or config.vertexai_model
            or "models/gemini-2.0-flash"
        )

        vertex_cfg = VertexAIConfig(
            project_id=project_id,
            location=request_data.get("location", "us-central1"),
            model=model,
            service_account_json=service_account_json,
            max_tokens=request_data.get("max_tokens", config.llm_max_tokens),
            temperature=request_data.get("temperature", config.llm_temperature),
        )

        factory = LLMProviderFactory()
        google_api_key = config.google_api_key
        if google_api_key:
            from llm_provider.utils.config import GeminiConfig

            provider_cfg = GeminiConfig(
                api_key=google_api_key,
                model=model,
                max_tokens=vertex_cfg.max_tokens,
                temperature=vertex_cfg.temperature,
            )
            provider = factory.create_provider("gemini", provider_cfg)
        else:
            provider = factory.create_provider("vertexai", vertex_cfg)

        if provider is None:
            raise HTTPException(status_code=500, detail="Failed to create provider")

        await provider.initialize()

        # Define all available tools with their descriptions and schemas
        def safe_calculator(args: Dict[str, Any]) -> str:
            """Güvenli matematiksel hesaplama (eval yerine ast kullanarak)"""
            import ast
            import operator
            import math as _math

            expr = args.get("expression", "")

            # Allowed operations
            operators_map = {
                ast.Add: operator.add,
                ast.Sub: operator.sub,
                ast.Mult: operator.mul,
                ast.Div: operator.truediv,
                ast.Pow: operator.pow,
                ast.USub: operator.neg,
                ast.UAdd: operator.pos,
                ast.Mod: operator.mod,
                ast.FloorDiv: operator.floordiv,
            }

            # Allowed functions (math module)
            allowed_functions = {
                "sqrt": _math.sqrt,
                "sin": _math.sin,
                "cos": _math.cos,
                "tan": _math.tan,
                "log": _math.log,
                "log10": _math.log10,
                "exp": _math.exp,
                "abs": abs,
                "round": round,
                "floor": _math.floor,
                "ceil": _math.ceil,
                "pi": _math.pi,
                "e": _math.e,
            }

            def eval_node(node):
                if isinstance(node, ast.Num):
                    return node.n
                elif isinstance(node, ast.Constant):
                    return node.value
                elif isinstance(node, ast.BinOp):
                    left = eval_node(node.left)
                    right = eval_node(node.right)
                    op = operators_map.get(type(node.op))
                    if op is None:
                        raise ValueError(
                            f"Desteklenmeyen operatör: {type(node.op).__name__}"
                        )
                    return op(left, right)
                elif isinstance(node, ast.UnaryOp):
                    operand = eval_node(node.operand)
                    op = operators_map.get(type(node.op))
                    if op is None:
                        raise ValueError(
                            f"Desteklenmeyen unary operatör: {type(node.op).__name__}"
                        )
                    return op(operand)
                elif isinstance(node, ast.Call):
                    func_name = (
                        node.func.id if isinstance(node.func, ast.Name) else None
                    )
                    if func_name not in allowed_functions:
                        raise ValueError(f"Desteklenmeyen fonksiyon: {func_name}")
                    func = allowed_functions[func_name]
                    args = [eval_node(arg) for arg in node.args]
                    return func(*args)
                elif isinstance(node, ast.Name):
                    if node.id in allowed_functions:
                        return allowed_functions[node.id]
                    raise ValueError(f"Desteklenmeyen değişken: {node.id}")
                else:
                    raise ValueError(
                        f"Desteklenmeyen ifade tipi: {type(node).__name__}"
                    )

            try:
                tree = ast.parse(expr, mode="eval")
                result = eval_node(tree.body)
                return str(result)
            except Exception as e:
                return f"Hesaplama hatası: {str(e)}"

        all_tools = {
            "calculator": {
                "description": "Matematiksel hesaplamalar yapar. İfadelerde +, -, *, /, ** (üs), % (mod), // (tam bölme), () parantez ve math fonksiyonları kullanılabilir: sqrt, sin, cos, tan, log, log10, exp, abs, round, floor, ceil, pi, e",
                "parameters": {
                    "expression": {
                        "type": "string",
                        "description": "Hesaplanacak matematiksel ifade (örn: '50*20+100', 'sqrt(144)', 'pi*2', 'sin(pi/2)')",
                    }
                },
                "executor": safe_calculator,
            },
            "get_weather": {
                "description": "Belirtilen şehir için hava durumu bilgisi getirir.",
                "parameters": {
                    "city": {
                        "type": "string",
                        "description": "Hava durumu sorgulanacak şehir adı (örn: 'Istanbul', 'Ankara')",
                    }
                },
                "executor": lambda a: f"{a.get('city','unknown')}: 22°C, güneşli",
            },
        }

        # Filter requested tools
        requested_tool_names = request_data.get("available_tools", []) or []
        active_tools = {
            name: all_tools[name] for name in requested_tool_names if name in all_tools
        }

        if not active_tools:
            # No tools, just do normal generation
            req = GenerationRequest(prompt=request_data.get("prompt", ""))
            resp = await provider.generate(req)
            return JSONResponse(
                {
                    "content": resp.content,
                    "model": getattr(resp, "model", None),
                    "usage": getattr(resp, "usage", None),
                }
            )

        # Build tools map for execution
        tools_map = {name: tool["executor"] for name, tool in active_tools.items()}

        # Build system instruction with tool descriptions
        tool_descriptions = []
        for name, tool in active_tools.items():
            params_desc = ", ".join(
                [
                    f"{pname}: {pinfo['description']}"
                    for pname, pinfo in tool["parameters"].items()
                ]
            )
            tool_descriptions.append(
                f"- **{name}**: {tool['description']}\n  Parametreler: {params_desc}"
            )

        tools_info = "\n".join(tool_descriptions)

        system_msg = f"""Kullanılabilir Araçlar:
{tools_info}

Bir araç çağırmak istediğinde, SADECE şu formatta bir JSON nesnesi döndür (başka metin ekleme):
{{"tool_call": {{"name": "ARAÇ_ADI", "arguments": {{"parametre": "değer"}}}}}}

Örnek calculator çağrısı:
{{"tool_call": {{"name": "calculator", "arguments": {{"expression": "50*20+100"}}}}}}

Örnek weather çağrısı:
{{"tool_call": {{"name": "get_weather", "arguments": {{"city": "Istanbul"}}}}}}

Eğer araç gerekmiyorsa veya araç sonucunu aldıktan sonra, doğrudan cevabı yaz."""

        # We'll use a prompt-based loop (concatenate system/user/assistant/tool texts)
        current_prompt = system_msg + "\n\nUser: " + request_data.get("prompt", "")

        max_iterations = int(request_data.get("max_iterations", 5))

        for _ in range(max_iterations):
            req = GenerationRequest(
                prompt=current_prompt,
                max_tokens=request_data.get("max_tokens", config.llm_max_tokens),
                temperature=request_data.get("temperature", config.llm_temperature),
            )
            resp = await provider.generate(req)

            content = (resp.content or "").strip()

            # Try parse JSON for tool_call (handle markdown code blocks)
            import json as _json
            import re

            # Remove markdown code blocks if present
            content_clean = re.sub(r"```json\s*|\s*```", "", content).strip()

            parsed = None
            try:
                parsed = _json.loads(content_clean)
            except Exception:
                # Try original content
                try:
                    parsed = _json.loads(content)
                except Exception:
                    parsed = None

            if isinstance(parsed, dict) and "tool_call" in parsed:
                tc = parsed["tool_call"]
                name = tc.get("name")
                args = tc.get("arguments", {})
                # execute tool synchronously
                tool_result = "Unknown tool"
                try:
                    if name in tools_map:
                        tool_result = tools_map[name](args)
                    else:
                        tool_result = f"Tool {name} not found"
                except Exception as e:
                    tool_result = f"Tool error: {e}"

                # Append the assistant's JSON tool_call and the tool result into the prompt, then ask model to continue
                current_prompt += "\n\nAssistant (tool_call): " + content
                current_prompt += (
                    "\n\nTool result (" + str(name) + "): " + str(tool_result)
                )
                current_prompt += (
                    "\n\nUser: Please continue and produce the final answer."
                )
                continue
            else:
                # Final answer
                return JSONResponse(
                    {
                        "content": content,
                        "model": getattr(resp, "model", None),
                        "usage": getattr(resp, "usage", None),
                    }
                )

        # If we reach here, iterations exhausted
        return JSONResponse(
            {
                "content": "Max iterations reached",
                "model": getattr(resp, "model", None),
                "usage": getattr(resp, "usage", None),
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("VertexAI protocol tools error")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/embed", response_model=EmbeddingResponse)
async def generate_embeddings(
    request: EmbeddingRequest, config: AIGatewayConfig = Depends(get_config)
) -> EmbeddingResponse:
    """Generate embeddings for text(s) using specified or configured provider"""

    # Validate input
    if not request.text and not request.texts:
        raise HTTPException(
            status_code=400, detail="Either 'text' or 'texts' is required"
        )

    # Prepare texts for embedding
    texts_to_embed = []
    if request.text:
        texts_to_embed = [request.text]
    elif request.texts:
        texts_to_embed = request.texts

    # Use specified provider or default from config (default: openai)
    provider_name = request.embedding_provider or "openai"
    model = request.model or "text-embedding-3-small"

    logger.info(
        f"Embedding request - provider: {provider_name}, texts: {len(texts_to_embed)}, model: {model}"
    )

    try:
        # Get API key for the provider
        if provider_name == "openai":
            api_key = config.openai_api_key
        elif provider_name == "google" or provider_name == "gemini":
            api_key = config.google_api_key
        elif provider_name == "anthropic":
            api_key = config.anthropic_api_key
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported embedding provider: {provider_name}. Supported: openai, google, anthropic",
            )

        if not api_key:
            raise HTTPException(
                status_code=500,
                detail=f"API key not configured for provider: {provider_name}",
            )

        # Generate embeddings based on provider
        if provider_name == "openai":
            embeddings, usage_info = await _generate_openai_embeddings(
                api_key, texts_to_embed, model
            )
            response_model = model or "text-embedding-3-small"

        elif provider_name == "google" or provider_name == "gemini":
            embeddings, usage_info = await _generate_google_embeddings(
                api_key, texts_to_embed, model
            )
            response_model = model or "text-embedding-004"

        elif provider_name == "anthropic":
            embeddings, usage_info = await _generate_anthropic_embeddings(
                api_key, texts_to_embed, model
            )
            response_model = model or "default-anthropic-model"

        else:
            raise HTTPException(
                status_code=500, detail=f"Provider not supported: {provider_name}"
            )

        logger.info(
            f"Embeddings generated successfully - {len(embeddings)} vectors, dimensions: {len(embeddings[0])}"
        )

        return EmbeddingResponse(
            embeddings=embeddings,
            provider=provider_name,
            model=response_model,
            usage=usage_info,
            dimensions=len(embeddings[0]) if embeddings else 0,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Embedding generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _generate_openai_embeddings(
    api_key: str, texts: List[str], model: str
) -> tuple:
    """Generate embeddings using OpenAI API"""
    try:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=api_key)

        # Generate embeddings
        response = await client.embeddings.create(input=texts, model=model)

        # Extract embeddings
        embeddings = [item.embedding for item in response.data]

        # Build usage info
        usage_info = None
        if hasattr(response, "usage"):
            usage_info = {
                "prompt_tokens": response.usage.prompt_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        return embeddings, usage_info

    except Exception as e:
        logger.error(f"OpenAI embedding error: {e}")
        raise


async def _generate_google_embeddings(
    api_key: str, texts: List[str], model: str
) -> tuple:
    """Generate embeddings using Google API"""
    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)

        # Use default model if not specified
        if not model or model == "text-embedding-004":
            model = "models/text-embedding-004"
        elif not model.startswith("models/"):
            model = f"models/{model}"

        embeddings = []
        for text in texts:
            result = genai.embed_content(
                model=model,
                content=text,
                task_type="RETRIEVAL_DOCUMENT",
            )
            embeddings.append(result["embedding"])

        return embeddings, None

    except Exception as e:
        logger.error(f"Google embedding error: {e}")
        raise


async def _generate_anthropic_embeddings(
    api_key: str, texts: List[str], model: str
) -> tuple:
    """Generate embeddings using Anthropic API"""
    try:
        from anthropic import Anthropic

        client = Anthropic(api_key=api_key)

        # Anthropic uses Claude model for embeddings
        embeddings = []

        # Simple approach: use model to generate fixed-size semantic vectors
        for text in texts:
            # This is a placeholder - Anthropic doesn't have direct embeddings API
            # You would typically use another service or implement custom logic
            logger.warning(
                "Anthropic embedding not fully implemented - consider using OpenAI or Google"
            )
            # Return dummy embedding for now
            embeddings.append([0.0] * 1536)

        return embeddings, None

    except Exception as e:
        logger.error(f"Anthropic embedding error: {e}")
        raise


@router.get("/providers")
async def get_llm_providers() -> Dict[str, Any]:
    """Get available LLM providers"""

    try:
        # Get available providers from llm-provider-factory
        available_providers = LLMProviderFactory().get_available_providers()

        return {
            "available_providers": available_providers,
            "supported_features": {
                "conversation_history": True,
                "streaming": False,  # Could be implemented
                "temperature_control": True,
                "max_tokens_control": True,
            },
        }

    except Exception as e:
        logger.error(f"Error getting LLM providers: {e}")
        return {
            "available_providers": ["openai", "anthropic", "google", "azure"],
            "error": str(e),
        }


@router.post("/test")
async def test_llm_provider(
    provider_name: str = Body(...),
    api_key: str = Body(...),
    test_prompt: str = Body(default="Hello, this is a test message."),
) -> Dict[str, Any]:
    """Test an LLM provider with given credentials"""

    try:
        # Create test provider
        test_model = "gpt-3.5-turbo" if provider_name == "openai" else "default-model"

        test_config = ProviderConfig(
            api_key=api_key,
            model=test_model,
            temperature=0.7,
            max_tokens=100,
            organization=None,  # Required for OpenAI
            base_url=None,  # Required for OpenAI
            timeout=30,
        )

        test_provider = LLMProviderFactory().create_provider(
            provider_name=provider_name, config=test_config
        )

        if test_provider is None:
            return {
                "success": False,
                "error": f"Failed to create provider: {provider_name}",
            }

        # Initialize provider
        await test_provider.initialize()

        # Create generation request
        generation_request = GenerationRequest(prompt=test_prompt, history=None)

        # Test with generation request
        response = await test_provider.generate(generation_request)

        return {
            "success": True,
            "provider": provider_name,
            "test_response": (
                response.content[:200] + "..."
                if len(response.content) > 200
                else response.content
            ),
            "message": "Provider test successful",
        }

    except Exception as e:
        logger.error(f"Provider test error: {e}")
        return {
            "success": False,
            "provider": provider_name,
            "error": str(e),
            "message": "Provider test failed",
        }
