"""
LLM endpoints for text generation
"""

import logging
from typing import Dict, Any, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from llm_provider import LLMProviderFactory, GenerationRequest
from llm_provider.utils.config import ProviderConfig, VertexAIConfig
from app.config.models import AIGatewayConfig
from app.api.dependencies import get_config

logger = logging.getLogger(__name__)
router = APIRouter()


class LLMRequest(BaseModel):
    """Request model for LLM endpoint"""

    prompt: str
    history: Optional[List[Dict[str, str]]] = None
    llm_provider: Optional[str] = None
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
            model = "gpt-3.5-turbo"
        elif provider_name == "anthropic":
            api_key = config.anthropic_api_key
            model = "claude-3-sonnet-20240229"
        elif provider_name == "google":
            api_key = config.google_api_key
            model = "gemini-pro"
        elif provider_name == "azure":
            api_key = config.azure_openai_api_key
            model = "gpt-35-turbo"
        elif provider_name == "vertexai":
            projectid = config.vertexai_project_id
            service_account_json = config.vertexai_service_account_json
            model = "gemini-pro"

        if not api_key:
            raise HTTPException(
                status_code=500,
                detail=f"API key not configured for provider: {provider_name}",
            )

        config = VertexAIConfig(
            project_id=projectid,
            location="us-central1",
            model=model,
            credentials_path=service_account_json,
        )

        # Create LLM provider using llm-provider-factory
        provider_config = ProviderConfig(
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            organization=None,  # Required for OpenAI
            base_url=None,  # Required for OpenAI
            timeout=30,  # Default timeout
        )

        llm_provider = LLMProviderFactory().create_provider(
            provider_name=provider_name, config=provider_config
        )

        if llm_provider is None:
            raise HTTPException(
                status_code=500, detail=f"Failed to create provider: {provider_name}"
            )

        # Initialize provider
        await llm_provider.initialize()

        # Create generation request
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
