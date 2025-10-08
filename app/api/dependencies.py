"""
FastAPI dependencies for dependency injection
"""

from typing import Dict
from fastapi import Request

from app.config.models import AIGatewayConfig
from app.config.manager import ConfigManager


def get_config(request: Request) -> AIGatewayConfig:
    """Get the application configuration"""
    return request.app.state.app_config


def get_config_manager(request: Request) -> ConfigManager:
    """Get the configuration manager"""
    return request.app.state.config_manager


def get_active_sessions(request: Request) -> Dict:
    """Get the active sessions dictionary"""
    return request.app.state.active_sessions