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
