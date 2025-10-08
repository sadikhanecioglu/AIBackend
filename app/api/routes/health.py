"""
Health check endpoints
"""

import logging
from datetime import datetime
from typing import Dict, Any

from fastapi import APIRouter, Request, Depends
from fastapi.responses import JSONResponse

from app.config.models import AIGatewayConfig
from app.api.dependencies import get_config, get_active_sessions

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("")
async def health_check(
    request: Request,
    config: AIGatewayConfig = Depends(get_config),
    active_sessions: Dict = Depends(get_active_sessions)
) -> Dict[str, Any]:
    """Health check endpoint"""
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0",
        "active_sessions": len(active_sessions),
        "config": {
            "stt_provider": config.stt_provider,
            "llm_provider": config.llm_provider,
            "tts_provider": config.tts_provider,
            "image_provider": config.image_provider,
            "audio_processor": config.audio_processor
        },
        "uptime": "Available",  # Could implement actual uptime tracking
        "services": {
            "stt": "available",
            "llm": "available", 
            "tts": "available",
            "image": "available"
        }
    }


@router.get("/detailed")
async def detailed_health_check(
    request: Request,
    config: AIGatewayConfig = Depends(get_config),
    active_sessions: Dict = Depends(get_active_sessions)
) -> Dict[str, Any]:
    """Detailed health check with more information"""
    
    session_info = {}
    for session_id, session_data in active_sessions.items():
        if "session" in session_data:
            session_info[session_id] = session_data["session"].get_session_info()
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0",
        "system": {
            "active_sessions": len(active_sessions),
            "max_sessions": config.max_sessions,
            "session_timeout": config.session_timeout
        },
        "providers": {
            "stt": {
                "current": config.stt_provider,
                "status": "available"
            },
            "llm": {
                "current": config.llm_provider,
                "status": "available",
                "temperature": config.llm_temperature,
                "max_tokens": config.llm_max_tokens
            },
            "tts": {
                "current": config.tts_provider,
                "status": "available",
                "voice": config.tts_voice,
                "speed": config.tts_speed
            },
            "image": {
                "current": config.image_provider,
                "status": "available",
                "size": config.image_size,
                "quality": config.image_quality
            }
        },
        "audio": {
            "processor": config.audio_processor,
            "sample_rate": config.audio_sample_rate,
            "chunk_size": config.audio_chunk_size
        },
        "sessions": session_info
    }