"""
LLM endpoints for text generation
"""

import logging
from typing import Dict, Any, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from llm_provider import LLMProviderFactory, GenerationRequest
from llm_provider.providers import VertexAI
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


class LLMResponse(BaseModel):
    """Response model for LLM endpoint"""

    response: str
    provider: str
    usage: Optional[Dict[str, Any]] = None
    model: Optional[str] = None


@router.post("/generate", response_model=LLMResponse)
async def generate_text(
    request: LLMRequest, config: AIGatewayConfig = Depends(get_config)
) -> LLMResponse:
    """Generate text response using specified or configured LLM provider"""

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


@router.post("/chat")
async def chat_conversation(
    request: LLMRequest, config: AIGatewayConfig = Depends(get_config)
) -> Dict[str, Any]:
    """Chat-style conversation with conversation history management"""

    try:
        # Generate response using the main endpoint logic
        response = await generate_text(request, config)

        # Build updated conversation history
        updated_history = request.history.copy() if request.history else []
        updated_history.append({"role": "user", "content": request.prompt})
        updated_history.append({"role": "assistant", "content": response.response})

        # Keep history manageable (last 20 messages)
        if len(updated_history) > 20:
            updated_history = updated_history[-20:]

        return {
            "response": response.content,  # Use response.content
            "provider": response.provider,
            "conversation_history": updated_history,
            "usage": response.usage,
            "model": response.model,
        }

    except Exception as e:
        logger.error(f"Chat conversation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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
