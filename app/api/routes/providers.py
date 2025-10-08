"""
Provider management endpoints
"""

import logging
from typing import Dict, Any

from fastapi import APIRouter, Request, Depends, HTTPException
from fastapi.responses import JSONResponse

from app.config.models import AIGatewayConfig
from app.config.manager import ConfigManager
from app.api.dependencies import get_config, get_config_manager

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("")
async def get_providers(
    config: AIGatewayConfig = Depends(get_config)
) -> Dict[str, Any]:
    """Get current and available providers"""
    
    current = {
        "stt": config.stt_provider,
        "llm": config.llm_provider,
        "tts": config.tts_provider,
        "image": config.image_provider,
        "audio": config.audio_processor
    }
    
    # List of available providers (could be dynamic based on installed packages)
    available = {
        "stt": ["openai", "azure", "google"],
        "llm": ["openai", "anthropic", "google", "azure"],
        "tts": ["openai", "azure", "google"],
        "image": ["openai", "azure", "stability"],
        "audio": ["whisper", "azure", "google"]
    }
    
    return {
        "current": current,
        "available": available,
        "status": "active"
    }


@router.post("/config/update")
async def update_config(
    new_config: dict,
    config_manager: ConfigManager = Depends(get_config_manager),
    request: Request = None
) -> Dict[str, Any]:
    """Update configuration with new provider settings"""
    
    try:
        # Update configuration
        updated_config = config_manager.update_config(new_config)
        
        # Update app state
        request.app.state.app_config = updated_config
        
        # Save to file
        config_manager.save_config()
        
        logger.info(f"Configuration updated: {new_config}")
        
        return {
            "status": "success",
            "message": "Configuration updated successfully",
            "config": {
                "stt_provider": updated_config.stt_provider,
                "llm_provider": updated_config.llm_provider,
                "tts_provider": updated_config.tts_provider,
                "image_provider": updated_config.image_provider,
                "audio_processor": updated_config.audio_processor
            }
        }
        
    except Exception as e:
        logger.error(f"Configuration update error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/config")
async def get_config_info(
    config: AIGatewayConfig = Depends(get_config)
) -> Dict[str, Any]:
    """Get current configuration (without sensitive data)"""
    
    config_dict = config.to_dict()
    
    # Remove sensitive information
    sensitive_keys = [
        "openai_api_key", "anthropic_api_key", "google_api_key",
        "azure_openai_api_key", "azure_speech_key"
    ]
    
    for key in sensitive_keys:
        if key in config_dict and config_dict[key]:
            config_dict[key] = "***configured***"
        elif key in config_dict:
            config_dict[key] = "***not_set***"
    
    return {
        "config": config_dict,
        "status": "active"
    }


@router.post("/validate")
async def validate_provider_config(
    provider_config: dict
) -> Dict[str, Any]:
    """Validate provider configuration without applying it"""
    
    try:
        # Create temporary config to validate
        test_config = AIGatewayConfig.from_dict(provider_config)
        
        # Basic validation
        valid_providers = {
            "stt": ["openai", "azure", "google"],
            "llm": ["openai", "anthropic", "google", "azure"], 
            "tts": ["openai", "azure", "google"],
            "image": ["openai", "azure", "stability"],
            "audio": ["whisper", "azure", "google"]
        }
        
        validation_results = {}
        
        if test_config.stt_provider in valid_providers["stt"]:
            validation_results["stt"] = "valid"
        else:
            validation_results["stt"] = f"invalid - must be one of {valid_providers['stt']}"
            
        if test_config.llm_provider in valid_providers["llm"]:
            validation_results["llm"] = "valid"
        else:
            validation_results["llm"] = f"invalid - must be one of {valid_providers['llm']}"
            
        if test_config.tts_provider in valid_providers["tts"]:
            validation_results["tts"] = "valid"
        else:
            validation_results["tts"] = f"invalid - must be one of {valid_providers['tts']}"
            
        if test_config.image_provider in valid_providers["image"]:
            validation_results["image"] = "valid"
        else:
            validation_results["image"] = f"invalid - must be one of {valid_providers['image']}"
        
        # Check if all validations passed
        all_valid = all(result == "valid" for result in validation_results.values())
        
        return {
            "valid": all_valid,
            "validation_results": validation_results,
            "message": "Configuration is valid" if all_valid else "Configuration has validation errors"
        }
        
    except Exception as e:
        logger.error(f"Configuration validation error: {e}")
        return {
            "valid": False,
            "error": str(e),
            "message": "Configuration validation failed"
        }