"""
Configuration manager for the AI Gateway
"""

import json
import os
import logging
from pathlib import Path
from typing import Dict, Optional, Any

from .models import AIGatewayConfig

logger = logging.getLogger(__name__)


class ConfigManager:
    """Manages configuration loading, saving, and updates"""
    
    def __init__(self, config_file: str = "config.json"):
        self.config_file = Path(config_file)
        self._config: Optional[AIGatewayConfig] = None
    
    def load_config(self, override_config: Optional[Dict[str, Any]] = None) -> AIGatewayConfig:
        """Load configuration from file and environment variables"""
        
        # Start with default config
        config_data = {}
        
        # Load from file if exists
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    file_config = json.load(f)
                    config_data.update(file_config)
                logger.info(f"Loaded configuration from {self.config_file}")
            except Exception as e:
                logger.warning(f"Failed to load config file: {e}")
        
        # Load from environment variables
        env_config = self._load_from_env()
        config_data.update(env_config)
        
        # Apply override config if provided
        if override_config:
            config_data.update(override_config)
        
        # Create config object
        self._config = AIGatewayConfig(**{k: v for k, v in config_data.items() 
                                        if k in AIGatewayConfig.__dataclass_fields__})
        
        return self._config
    
    def _load_from_env(self) -> Dict[str, Any]:
        """Load configuration from environment variables"""
        env_config = {}
        
        # Provider settings
        if os.getenv("DEFAULT_STT_PROVIDER"):
            env_config["stt_provider"] = os.getenv("DEFAULT_STT_PROVIDER")
        if os.getenv("DEFAULT_LLM_PROVIDER"):
            env_config["llm_provider"] = os.getenv("DEFAULT_LLM_PROVIDER")
        if os.getenv("DEFAULT_TTS_PROVIDER"):
            env_config["tts_provider"] = os.getenv("DEFAULT_TTS_PROVIDER")
        if os.getenv("DEFAULT_IMAGE_PROVIDER"):
            env_config["image_provider"] = os.getenv("DEFAULT_IMAGE_PROVIDER")
        if os.getenv("DEFAULT_AUDIO_PROCESSOR"):
            env_config["audio_processor"] = os.getenv("DEFAULT_AUDIO_PROCESSOR")
        
        # API Keys
        if os.getenv("OPENAI_API_KEY"):
            env_config["openai_api_key"] = os.getenv("OPENAI_API_KEY")
        if os.getenv("ANTHROPIC_API_KEY"):
            env_config["anthropic_api_key"] = os.getenv("ANTHROPIC_API_KEY")
        if os.getenv("GOOGLE_API_KEY"):
            env_config["google_api_key"] = os.getenv("GOOGLE_API_KEY")
        if os.getenv("AZURE_OPENAI_API_KEY"):
            env_config["azure_openai_api_key"] = os.getenv("AZURE_OPENAI_API_KEY")
        if os.getenv("AZURE_OPENAI_ENDPOINT"):
            env_config["azure_openai_endpoint"] = os.getenv("AZURE_OPENAI_ENDPOINT")
        if os.getenv("AZURE_SPEECH_KEY"):
            env_config["azure_speech_key"] = os.getenv("AZURE_SPEECH_KEY")
        if os.getenv("AZURE_SPEECH_REGION"):
            env_config["azure_speech_region"] = os.getenv("AZURE_SPEECH_REGION")
        
        # Numeric settings
        if os.getenv("MAX_SESSIONS"):
            env_config["max_sessions"] = int(os.getenv("MAX_SESSIONS"))
        if os.getenv("SESSION_TIMEOUT"):
            env_config["session_timeout"] = int(os.getenv("SESSION_TIMEOUT"))
        if os.getenv("LLM_TEMPERATURE"):
            env_config["llm_temperature"] = float(os.getenv("LLM_TEMPERATURE"))
        if os.getenv("LLM_MAX_TOKENS"):
            env_config["llm_max_tokens"] = int(os.getenv("LLM_MAX_TOKENS"))
        
        return env_config
    
    def save_config(self) -> None:
        """Save current configuration to file"""
        if self._config is None:
            logger.warning("No configuration to save")
            return
        
        try:
            config_dict = self._config.to_dict()
            # Remove sensitive data from saved config
            sensitive_keys = [
                "openai_api_key", "anthropic_api_key", "google_api_key",
                "azure_openai_api_key", "azure_speech_key"
            ]
            for key in sensitive_keys:
                config_dict.pop(key, None)
            
            with open(self.config_file, 'w') as f:
                json.dump(config_dict, f, indent=2)
            logger.info(f"Configuration saved to {self.config_file}")
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
    
    def update_config(self, updates: Dict[str, Any]) -> AIGatewayConfig:
        """Update configuration with new values"""
        if self._config is None:
            raise ValueError("Configuration not loaded")
        
        # Create new config with updates
        current_dict = self._config.to_dict()
        current_dict.update(updates)
        
        # Validate and create new config
        self._config = AIGatewayConfig(**{k: v for k, v in current_dict.items() 
                                        if k in AIGatewayConfig.__dataclass_fields__})
        
        return self._config
    
    def print_config(self) -> None:
        """Print current configuration (without sensitive data)"""
        if self._config is None:
            logger.warning("No configuration loaded")
            return
        
        config_dict = self._config.to_dict()
        
        # Mask sensitive data
        sensitive_keys = [
            "openai_api_key", "anthropic_api_key", "google_api_key",
            "azure_openai_api_key", "azure_speech_key"
        ]
        
        for key in sensitive_keys:
            if key in config_dict and config_dict[key]:
                config_dict[key] = f"***{config_dict[key][-4:]}"
        
        logger.info("Current Configuration:")
        logger.info("=" * 50)
        for key, value in config_dict.items():
            if not key.startswith("_"):
                logger.info(f"  {key}: {value}")
        logger.info("=" * 50)
    
    @property
    def config(self) -> Optional[AIGatewayConfig]:
        """Get current configuration"""
        return self._config