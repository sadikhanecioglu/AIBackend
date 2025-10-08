"""
Test configuration module
"""

import pytest
import os
from app.config.manager import ConfigManager
from app.config.models import AIGatewayConfig


def test_config_creation():
    """Test basic config creation"""
    config = AIGatewayConfig()
    assert config.llm_provider == "openai"
    assert config.stt_provider == "openai"
    assert config.tts_provider == "openai"


def test_config_to_dict():
    """Test config serialization"""
    config = AIGatewayConfig(llm_provider="anthropic")
    config_dict = config.to_dict()
    assert config_dict["llm_provider"] == "anthropic"


def test_config_from_dict():
    """Test config deserialization"""
    data = {"llm_provider": "google", "llm_temperature": 0.5}
    config = AIGatewayConfig.from_dict(data)
    assert config.llm_provider == "google"
    assert config.llm_temperature == 0.5


def test_config_manager():
    """Test config manager basic functionality"""
    manager = ConfigManager("test_config.json")
    config = manager.load_config()
    assert isinstance(config, AIGatewayConfig)


def test_config_manager_with_override():
    """Test config manager with override"""
    manager = ConfigManager("test_config.json")
    override = {"llm_provider": "anthropic"}
    config = manager.load_config(override)
    assert config.llm_provider == "anthropic"