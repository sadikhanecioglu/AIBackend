"""
Configuration models for the AI Gateway
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Any


@dataclass
class AIGatewayConfig:
    """Main configuration class for the AI Gateway"""

    # Provider configurations
    stt_provider: str = "openai"
    llm_provider: str = "openai"
    tts_provider: str = "openai"
    image_provider: str = "openai"
    audio_processor: str = "whisper"

    # Provider-specific settings
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    google_api_key: Optional[str] = None
    azure_openai_api_key: Optional[str] = None
    azure_openai_endpoint: Optional[str] = None
    azure_speech_key: Optional[str] = None
    azure_speech_region: Optional[str] = None
    vertexai_service_account_json: Optional[str] = None
    vertexai_project_id: Optional[str] = None
    vertexai_model: Optional[str] = (
        "mistral-large-2411"  # Default to mistral-large-2411
    )

    # Ollama settings
    ollama_base_url: Optional[str] = "http://localhost:11434"  # Default Ollama server
    ollama_model: Optional[str] = "llama3.1:latest"  # Default Ollama model

    # Application settings
    max_sessions: int = 100
    session_timeout: int = 3600  # seconds
    audio_sample_rate: int = 16000
    audio_chunk_size: int = 1024

    # LLM settings
    llm_temperature: float = 0.7
    llm_max_tokens: int = 1000

    # Image generation settings
    image_size: str = "1024x1024"
    image_quality: str = "standard"

    # TTS settings
    tts_voice: str = "alloy"
    tts_speed: float = 1.0

    # Additional provider configurations
    provider_configs: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            "stt_provider": self.stt_provider,
            "llm_provider": self.llm_provider,
            "tts_provider": self.tts_provider,
            "image_provider": self.image_provider,
            "audio_processor": self.audio_processor,
            "openai_api_key": self.openai_api_key,
            "anthropic_api_key": self.anthropic_api_key,
            "google_api_key": self.google_api_key,
            "azure_openai_api_key": self.azure_openai_api_key,
            "azure_openai_endpoint": self.azure_openai_endpoint,
            "azure_speech_key": self.azure_speech_key,
            "azure_speech_region": self.azure_speech_region,
            "vertexai_service_account_json": self.vertexai_service_account_json,
            "vertexai_project_id": self.vertexai_project_id,
            "vertexai_model": self.vertexai_model,
            "ollama_base_url": self.ollama_base_url,
            "ollama_model": self.ollama_model,
            "max_sessions": self.max_sessions,
            "session_timeout": self.session_timeout,
            "audio_sample_rate": self.audio_sample_rate,
            "audio_chunk_size": self.audio_chunk_size,
            "llm_temperature": self.llm_temperature,
            "llm_max_tokens": self.llm_max_tokens,
            "image_size": self.image_size,
            "image_quality": self.image_quality,
            "tts_voice": self.tts_voice,
            "tts_speed": self.tts_speed,
            "provider_configs": self.provider_configs,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AIGatewayConfig":
        """Create config from dictionary"""
        return cls(**data)
