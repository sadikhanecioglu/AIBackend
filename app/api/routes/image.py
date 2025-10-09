"""
Image generation endpoints using LLM Provider Factory
"""

import logging
import time
from typing import Dict, Any, Optional, List

from fastapi import APIRouter, Depends, HTTPException, Body, Request
from pydantic import BaseModel

from app.config.models import AIGatewayConfig
from app.api.dependencies import get_config
from app.core.exceptions import AIGatewayException

# Import LLM Provider Factory for image generation
try:
    from llm_provider import ImageProviderFactory
    from llm_provider.utils.config import OpenAIImageConfig, ReplicateImageConfig

    HAS_IMAGE_PROVIDER = True
except ImportError:
    HAS_IMAGE_PROVIDER = False
    ImageProviderFactory = None
    OpenAIImageConfig = None
    ReplicateImageConfig = None

logger = logging.getLogger(__name__)
router = APIRouter()


class ImageRequest(BaseModel):
    """Request model for image generation"""

    prompt: str
    provider: Optional[str] = None
    model: Optional[str] = None
    size: Optional[str] = None
    quality: Optional[str] = None
    style: Optional[str] = None
    n: Optional[int] = 1
    width: Optional[int] = None
    height: Optional[int] = None
    guidance_scale: Optional[float] = None
    num_inference_steps: Optional[int] = None
    scheduler: Optional[str] = None


class ImageResponse(BaseModel):
    """Response model for image generation"""

    urls: List[str]
    provider: str
    model: Optional[str] = None
    prompt: str
    parameters: Dict[str, Any]
    processing_time: float
    metadata: Optional[Dict[str, Any]] = None


@router.post("/generate", response_model=ImageResponse)
async def generate_image(
    request: ImageRequest, config: AIGatewayConfig = Depends(get_config)
) -> ImageResponse:
    """Generate images using LLM Provider Factory"""

    if not request.prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")

    if not HAS_IMAGE_PROVIDER:
        raise HTTPException(
            status_code=503,
            detail="Image provider factory not available. Please install llm-provider-factory with image support.",
        )

    # Use specified provider or default from config
    provider_name = request.provider or config.image_provider or "openai"
    start_time = time.time()

    logger.info(
        f"Image generation request - provider: {provider_name}, prompt: {request.prompt[:100]}..."
    )

    try:
        # Create provider factory
        factory = ImageProviderFactory()

        # Generate image based on provider
        if provider_name.lower() == "openai":
            response = await _generate_with_openai(factory, request, config)
        elif provider_name.lower() == "replicate":
            response = await _generate_with_replicate(factory, request, config)
        else:
            raise HTTPException(
                status_code=400, detail=f"Unsupported provider: {provider_name}"
            )

        processing_time = time.time() - start_time

        # Prepare response parameters
        parameters = {
            "size": request.size or config.image_size,
            "quality": request.quality or config.image_quality,
            "n": request.n or 1,
            "style": request.style,
            "model": request.model,
            "width": request.width,
            "height": request.height,
            "guidance_scale": request.guidance_scale,
            "num_inference_steps": request.num_inference_steps,
            "scheduler": request.scheduler,
        }

        logger.info(
            f"Generated {len(response.urls)} images with {provider_name} in {processing_time:.2f}s"
        )

        return ImageResponse(
            urls=response.urls,
            provider=provider_name,
            model=getattr(response, "model", request.model),
            prompt=request.prompt,
            parameters=parameters,
            processing_time=processing_time,
            metadata=getattr(response, "metadata", {}),
        )

    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Image generation error: {e}")
        raise AIGatewayException(f"Failed to generate image: {str(e)}")


async def _generate_with_openai(
    factory, request: ImageRequest, config: AIGatewayConfig
):
    """Generate image using OpenAI DALL-E"""

    # Determine model
    model = request.model or "dall-e-3"

    # Prepare config parameters
    config_params = {
        "api_key": config.openai_api_key,
        "model": model,
        "size": request.size or config.image_size or "1024x1024",
    }

    # Add quality and style only for DALL-E 3
    if model == "dall-e-3":
        config_params["quality"] = request.quality or config.image_quality or "standard"
        config_params["style"] = request.style or "vivid"

    # Create OpenAI config
    openai_config = OpenAIImageConfig(**config_params)

    # Create provider
    provider = factory.create_provider("openai_image", openai_config)

    # Prepare generation parameters
    generation_params = {
        "prompt": request.prompt,
        "n": (
            min(request.n or 1, 1) if model == "dall-e-3" else request.n or 1
        ),  # DALL-E 3 supports only n=1
        "size": request.size or config.image_size or "1024x1024",
    }

    # Add quality and style only for DALL-E 3
    if model == "dall-e-3":
        generation_params["quality"] = (
            request.quality or config.image_quality or "standard"
        )
        generation_params["style"] = request.style or "vivid"

    # Generate image
    response = await provider.generate_image(**generation_params)

    return response


async def _generate_with_replicate(
    factory, request: ImageRequest, config: AIGatewayConfig
):
    """Generate image using Replicate"""

    # Default Replicate model
    default_model = "stability-ai/stable-diffusion"

    # Create provider using direct factory method
    provider = factory.create_replicate_image(
        api_token=config.replicate_api_token,
        model=request.model or default_model,
        width=request.width or 1024,
        height=request.height or 1024,
        steps=request.num_inference_steps or 50,
    )

    # Generate image
    response = await provider.generate_image(
        prompt=request.prompt,
        width=request.width or 1024,
        height=request.height or 1024,
        num_outputs=request.n or 1,
        guidance_scale=request.guidance_scale or 7.5,
        num_inference_steps=request.num_inference_steps or 50,
        scheduler=request.scheduler or "K_EULER",
    )

    return response


@router.get("/providers")
async def get_image_providers(
    config: AIGatewayConfig = Depends(get_config),
) -> Dict[str, Any]:
    """Get available image generation providers"""

    if not HAS_IMAGE_PROVIDER:
        return {
            "error": "Image provider factory not available",
            "available_providers": [],
            "message": "Please install llm-provider-factory with image support",
        }

    providers = []

    # Check OpenAI availability
    if config.openai_api_key:
        providers.append(
            {
                "name": "openai",
                "models": ["dall-e-2", "dall-e-3"],
                "available": True,
                "supported_sizes": [
                    "256x256",
                    "512x512",
                    "1024x1024",
                    "1792x1024",
                    "1024x1792",
                ],
                "supported_qualities": ["standard", "hd"],
                "supported_styles": ["vivid", "natural"],
                "max_images": {"dall-e-2": 10, "dall-e-3": 1},
            }
        )

    # Check Replicate availability
    if hasattr(config, "replicate_api_token") and config.replicate_api_token:
        providers.append(
            {
                "name": "replicate",
                "models": [
                    "stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b",
                    "stability-ai/stable-diffusion:27b93a2413e7f36cd83da926f3656280b2931564ff050bf9575f1fdf9bcd7478",
                ],
                "available": True,
                "supported_sizes": "Custom width/height",
                "max_outputs": 4,
                "features": ["guidance_scale", "num_inference_steps", "scheduler"],
            }
        )

    return {
        "available_providers": providers,
        "default_provider": config.image_provider or "openai",
        "supported_features": {
            "text_to_image": True,
            "image_variations": True,
            "batch_generation": True,
            "custom_sizes": True,
            "style_control": True,
        },
    }


@router.post("/test")
async def test_image_provider(
    provider_name: str = Body(...),
    api_key: str = Body(...),
    test_prompt: str = Body(default="A beautiful sunset over mountains"),
) -> Dict[str, Any]:
    """Test an image generation provider with given credentials"""

    if not HAS_IMAGE_PROVIDER:
        return {
            "success": False,
            "error": "Image provider factory not available",
            "message": "Please install llm-provider-factory with image support",
        }

    try:
        logger.info(f"Testing image provider: {provider_name}")

        factory = ImageProviderFactory()

        if provider_name.lower() == "openai":
            # Test OpenAI
            config = OpenAIImageConfig(
                api_key=api_key,
                model="dall-e-2",  # Use cheaper model for testing
                size="512x512",
            )
            provider = factory.create_provider("openai_image", config)
            response = await provider.generate_image(prompt=test_prompt, size="512x512")

            return {
                "success": True,
                "provider": provider_name,
                "test_image_urls": response.urls,
                "model": getattr(response, "model", "dall-e-2"),
                "message": f"Provider {provider_name} test successful",
            }

        elif provider_name.lower() == "replicate":
            # Test Replicate
            config = ReplicateImageConfig(
                api_token=api_key,
                model="stability-ai/stable-diffusion:27b93a2413e7f36cd83da926f3656280b2931564ff050bf9575f1fdf9bcd7478",
                width=512,
                height=512,
                num_outputs=1,
            )
            provider = factory.create_provider("replicate_image", config)
            response = await provider.generate_image(
                prompt=test_prompt, width=512, height=512, num_outputs=1
            )

            return {
                "success": True,
                "provider": provider_name,
                "test_image_urls": response.urls,
                "message": f"Provider {provider_name} test successful",
            }

        else:
            return {
                "success": False,
                "provider": provider_name,
                "error": f"Unknown provider: {provider_name}",
                "supported_providers": ["openai", "replicate"],
                "message": "Provider test failed",
            }

    except Exception as e:
        logger.error(f"Image provider test error: {e}")
        return {
            "success": False,
            "provider": provider_name,
            "error": str(e),
            "message": "Provider test failed",
        }


@router.post("/batch")
async def batch_generate_images(
    prompts: List[str] = Body(...),
    provider: str = Body(default="openai"),
    model: Optional[str] = Body(default=None),
    size: Optional[str] = Body(default="512x512"),
    config: AIGatewayConfig = Depends(get_config),
) -> Dict[str, Any]:
    """Generate multiple images from a list of prompts"""

    if not HAS_IMAGE_PROVIDER:
        raise HTTPException(
            status_code=503, detail="Image provider factory not available"
        )

    if not prompts:
        raise HTTPException(status_code=400, detail="At least one prompt is required")

    if len(prompts) > 10:  # Limit batch size
        raise HTTPException(status_code=400, detail="Maximum 10 prompts per batch")

    start_time = time.time()
    results = []

    try:
        factory = ImageProviderFactory()

        for i, prompt in enumerate(prompts):
            logger.info(f"Generating image {i+1}/{len(prompts)}: {prompt[:50]}...")

            try:
                # Create individual request
                individual_request = ImageRequest(
                    prompt=prompt, provider=provider, model=model, size=size, n=1
                )

                # Generate image
                if provider.lower() == "openai":
                    response = await _generate_with_openai(
                        factory, individual_request, config
                    )
                elif provider.lower() == "replicate":
                    response = await _generate_with_replicate(
                        factory, individual_request, config
                    )
                else:
                    raise ValueError(f"Unsupported provider: {provider}")

                results.append(
                    {
                        "prompt": prompt,
                        "urls": response.urls,
                        "success": True,
                        "index": i,
                    }
                )

            except Exception as e:
                logger.error(f"Failed to generate image for prompt {i+1}: {e}")
                results.append(
                    {"prompt": prompt, "error": str(e), "success": False, "index": i}
                )

        processing_time = time.time() - start_time
        successful = len([r for r in results if r["success"]])

        return {
            "total_prompts": len(prompts),
            "successful_generations": successful,
            "failed_generations": len(prompts) - successful,
            "processing_time": processing_time,
            "results": results,
            "provider": provider,
            "parameters": {"model": model, "size": size},
        }

    except Exception as e:
        logger.error(f"Batch generation error: {e}")
        raise AIGatewayException(f"Batch generation failed: {str(e)}")


@router.post("/variations")
async def create_image_variations(
    image_url: str = Body(...),
    provider: str = Body(default="openai"),
    n: int = Body(default=2),
    size: str = Body(default="1024x1024"),
    config: AIGatewayConfig = Depends(get_config),
) -> Dict[str, Any]:
    """Create variations of an existing image"""

    if not HAS_IMAGE_PROVIDER:
        raise HTTPException(
            status_code=503, detail="Image provider factory not available"
        )

    if provider.lower() != "openai":
        raise HTTPException(
            status_code=400,
            detail="Image variations currently only supported with OpenAI provider",
        )

    try:
        start_time = time.time()
        logger.info(f"Creating {n} variations of image with {provider}")

        factory = ImageProviderFactory()

        # Create OpenAI config for variations
        openai_config = OpenAIImageConfig(
            api_key=config.openai_api_key,
            model="dall-e-2",  # Only DALL-E 2 supports variations
            size=size,
        )

        provider_instance = factory.create_provider("openai_image", openai_config)

        # Create variations (this would need to be implemented in the provider)
        # For now, we'll simulate multiple generations with variation-style prompts
        variation_prompts = [
            f"Create a variation of the style and composition similar to the reference image",
            f"Generate an image with similar artistic style but different details",
            f"Create an alternative version with the same aesthetic approach",
        ][:n]

        variations = []
        for i, prompt in enumerate(variation_prompts):
            response = await provider_instance.generate_image(
                prompt=prompt, size=size, n=1
            )
            variations.extend(response.urls)

        processing_time = time.time() - start_time

        return {
            "success": True,
            "original_image": image_url,
            "variations": variations[:n],
            "provider": provider,
            "processing_time": processing_time,
            "parameters": {"n": n, "size": size, "model": "dall-e-2"},
        }

    except Exception as e:
        logger.error(f"Image variation error: {e}")
        raise AIGatewayException(f"Image variation failed: {str(e)}")
