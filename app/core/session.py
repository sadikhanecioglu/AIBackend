"""
Voice session management for the AI Gateway
"""

import asyncio
import base64
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Optional, Any, List
from io import BytesIO

from llm_provider import LLMProviderFactory
from llm_provider.utils.config import ProviderConfig

from app.config.models import AIGatewayConfig
from app.core.exceptions import SessionException, ProviderException

logger = logging.getLogger(__name__)


class VoiceSession:
    """Manages a voice conversation session with modular provider support"""

    def __init__(self, session_id: str, config: AIGatewayConfig):
        self.session_id = session_id
        self.config = config
        self.created_at = datetime.now()
        self.last_activity = datetime.now()
        self.message_count = 0
        self.conversation_history: List[Dict[str, str]] = []

        # Audio processing state
        self.audio_buffer = BytesIO()
        self.is_recording = False
        self.speech_detected = False

        # Initialize providers
        self._llm_provider = None
        self._stt_provider = None
        self._tts_provider = None

        logger.info(f"Created voice session: {session_id}")

    def _get_llm_provider(self):
        """Get or create LLM provider using llm-provider-factory"""
        if self._llm_provider is None:
            try:
                # Use llm-provider-factory for LLM
                api_key = getattr(
                    self.config, f"{self.config.llm_provider}_api_key", None
                )

                # Get default model for provider
                model_mapping = {
                    "openai": "gpt-3.5-turbo",
                    "anthropic": "claude-3-sonnet-20240229",
                    "google": "gemini-pro",
                    "azure": "gpt-35-turbo",
                }
                model = model_mapping.get(self.config.llm_provider, "gpt-3.5-turbo")

                provider_config = ProviderConfig(
                    api_key=api_key,
                    model=model,
                    temperature=self.config.llm_temperature,
                    max_tokens=self.config.llm_max_tokens,
                )

                self._llm_provider = LLMProviderFactory().create_provider(
                    provider_name=self.config.llm_provider, config=provider_config
                )
                logger.info(f"Initialized LLM provider: {self.config.llm_provider}")
            except Exception as e:
                logger.error(f"Failed to initialize LLM provider: {e}")
                raise ProviderException(f"LLM provider initialization failed: {e}")

        return self._llm_provider

    async def process_audio(self, audio_data: bytes) -> Optional[Dict[str, Any]]:
        """Process incoming audio data and return response"""
        try:
            self.last_activity = datetime.now()

            # Add audio to buffer
            self.audio_buffer.write(audio_data)

            # Simple voice activity detection (placeholder)
            if len(audio_data) > 0:
                self.speech_detected = True
                self.is_recording = True

                # Return speech in progress status
                return {
                    "type": "speech_in_progress",
                    "message": "Speech detected, listening...",
                }

            # If speech ended (simplified logic)
            if self.speech_detected and len(audio_data) == 0:
                return await self._process_complete_audio()

        except Exception as e:
            logger.error(f"Audio processing error: {e}")
            return {"type": "error", "message": f"Audio processing failed: {str(e)}"}

        return None

    async def _process_complete_audio(self) -> Dict[str, Any]:
        """Process complete audio recording through STT -> LLM -> TTS pipeline"""
        try:
            # Get audio buffer
            audio_bytes = self.audio_buffer.getvalue()
            if len(audio_bytes) == 0:
                return {"type": "error", "message": "No audio data to process"}

            # Reset buffer and state
            self.audio_buffer = BytesIO()
            self.speech_detected = False
            self.is_recording = False

            # Step 1: Speech-to-Text (placeholder)
            # In real implementation, you would use the configured STT provider
            transcription = await self._mock_stt(audio_bytes)

            if not transcription:
                return {"type": "error", "message": "Speech transcription failed"}

            logger.info(f"Transcribed: {transcription}")

            # Step 2: Generate LLM response
            response_text = await self.generate_response(transcription)

            # Step 3: Text-to-Speech (placeholder)
            # In real implementation, you would use the configured TTS provider
            audio_response = await self._mock_tts(response_text)

            # Step 4: Return complete response
            return {
                "type": "audio_response",
                "transcription": transcription,
                "text": response_text,
                "audio_base64": base64.b64encode(audio_response).decode("utf-8"),
                "providers": {
                    "stt": self.config.stt_provider,
                    "llm": self.config.llm_provider,
                    "tts": self.config.tts_provider,
                },
            }

        except Exception as e:
            logger.error(f"Complete audio processing error: {e}")
            return {"type": "error", "message": f"Audio pipeline failed: {str(e)}"}

    async def generate_response(
        self, prompt: str, history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """Generate text response using LLM provider"""
        try:
            self.last_activity = datetime.now()
            self.message_count += 1

            # Get LLM provider
            llm_provider = self._get_llm_provider()

            # Use conversation history if available
            if history is None:
                history = self.conversation_history

            # Generate response
            if hasattr(llm_provider, "generate"):
                if history:
                    response = await llm_provider.generate(prompt, history=history)
                else:
                    response = await llm_provider.generate(prompt)
            else:
                # Fallback for providers without history support
                response = await llm_provider.generate(prompt)

            # Update conversation history
            self.conversation_history.append({"role": "user", "content": prompt})
            self.conversation_history.append({"role": "assistant", "content": response})

            # Keep history manageable
            if len(self.conversation_history) > 20:
                self.conversation_history = self.conversation_history[-20:]

            logger.info(f"Generated response for session {self.session_id}")
            return response

        except Exception as e:
            logger.error(f"Response generation error: {e}")
            raise ProviderException(f"LLM response generation failed: {e}")

    async def _mock_stt(self, audio_data: bytes) -> str:
        """Mock STT implementation - replace with real provider"""
        # Simulate processing delay
        await asyncio.sleep(0.1)

        # Return mock transcription
        return f"Mock transcription of {len(audio_data)} bytes of audio"

    async def _mock_tts(self, text: str) -> bytes:
        """Mock TTS implementation - replace with real provider"""
        # Simulate processing delay
        await asyncio.sleep(0.2)

        # Return mock audio data
        return f"Mock audio for: {text}".encode("utf-8")

    def get_session_info(self) -> Dict[str, Any]:
        """Get session information"""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "message_count": self.message_count,
            "is_recording": self.is_recording,
            "providers": {
                "stt": self.config.stt_provider,
                "llm": self.config.llm_provider,
                "tts": self.config.tts_provider,
            },
            "conversation_length": len(self.conversation_history),
        }

    def is_expired(self, timeout_seconds: int = 3600) -> bool:
        """Check if session has expired"""
        expiry_time = self.last_activity + timedelta(seconds=timeout_seconds)
        return datetime.now() > expiry_time

    def cleanup(self) -> None:
        """Clean up session resources"""
        self.audio_buffer.close()
        self.conversation_history.clear()
        self._llm_provider = None
        self._stt_provider = None
        self._tts_provider = None
        logger.info(f"Cleaned up session: {self.session_id}")
