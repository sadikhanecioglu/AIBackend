"""
Speech-to-Text API routes for the AI Gateway.
Supports multiple STT providers including OpenAI Whisper, AssemblyAI, and Deepgram.
"""

import asyncio
import tempfile
import logging
import os
import time
from typing import List, Dict, Any, Optional
from pathlib import Path

from fastapi import (
    APIRouter,
    HTTPException,
    Depends,
    UploadFile,
    File,
    Form,
    Query,
    Body,
)
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from llm_provider import SpeechFactory, BaseSpeechProvider, SpeechRequest
from app.api.dependencies import get_config
from app.config.models import AIGatewayConfig

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()


# Request/Response models
class TranscriptionResponse(BaseModel):
    """STT transcription response"""

    text: str
    language: Optional[str] = None
    confidence: Optional[float] = None
    duration: Optional[float] = None
    words: Optional[List[Dict[str, Any]]] = None
    provider: str
    processing_time: float


class STTProvidersResponse(BaseModel):
    """Available STT providers response"""

    providers: List[str]
    default: str


class STTTestResponse(BaseModel):
    """STT provider test response"""

    provider: str
    available: bool
    error: Optional[str] = None
    supported_formats: List[str]


class Base64AudioRequest(BaseModel):
    """Base64 encoded audio request"""

    audio_base64: str
    format: str = "wav"  # Audio format: wav, mp3, m4a, etc.
    language: Optional[str] = "tr"
    timestamps: bool = False
    word_timestamps: bool = False


class URLAudioRequest(BaseModel):
    """URL audio request"""

    audio_url: str
    language: Optional[str] = "tr"
    timestamps: bool = False
    word_timestamps: bool = False


async def get_speech_provider(
    provider: str, config: AIGatewayConfig = Depends(get_config)
) -> BaseSpeechProvider:
    """Get configured speech provider"""

    factory = SpeechFactory()

    try:
        if provider == "openai":
            if not config.openai_api_key:
                raise HTTPException(
                    status_code=400, detail="OpenAI API key not configured"
                )
            return factory.create_openai_speech(api_key=config.openai_api_key)

        elif provider == "assemblyai":
            if not config.assemblyai_api_key:
                raise HTTPException(
                    status_code=400, detail="AssemblyAI API key not configured"
                )
            return factory.create_assemblyai_speech(api_key=config.assemblyai_api_key)

        elif provider == "deepgram":
            if not config.deepgram_api_key:
                raise HTTPException(
                    status_code=400, detail="Deepgram API key not configured"
                )
            return factory.create_deepgram_speech(api_key=config.deepgram_api_key)

        else:
            raise HTTPException(
                status_code=400, detail=f"Unsupported STT provider: {provider}"
            )

    except Exception as e:
        logger.error(f"Failed to create STT provider {provider}: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to initialize STT provider: {str(e)}"
        )


def validate_audio_file(file: UploadFile, config: AIGatewayConfig) -> None:
    """Validate uploaded audio file"""

    # Check file size
    if file.size and file.size > config.stt_max_file_size:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {config.stt_max_file_size / (1024*1024):.1f}MB",
        )

    # Check file format
    if file.filename:
        file_ext = Path(file.filename).suffix.lower().lstrip(".")
        if file_ext not in config.stt_supported_formats:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format. Supported: {', '.join(config.stt_supported_formats)}",
            )


@router.get("/providers", response_model=STTProvidersResponse)
async def get_available_providers(
    config: AIGatewayConfig = Depends(get_config),
) -> STTProvidersResponse:
    """Get list of available STT providers"""

    available_providers = []

    # Check which providers are configured
    if config.openai_api_key:
        available_providers.append("openai")
    if config.assemblyai_api_key:
        available_providers.append("assemblyai")
    if config.deepgram_api_key:
        available_providers.append("deepgram")

    if not available_providers:
        raise HTTPException(status_code=500, detail="No STT providers configured")

    return STTProvidersResponse(
        providers=available_providers, default=available_providers[0]
    )


@router.get("/test/{provider}", response_model=STTTestResponse)
async def test_provider(
    provider: str, config: AIGatewayConfig = Depends(get_config)
) -> STTTestResponse:
    """Test STT provider availability"""

    try:
        speech_provider = await get_speech_provider(provider, config)

        # Get supported formats from config
        supported_formats = config.stt_supported_formats

        return STTTestResponse(
            provider=provider, available=True, supported_formats=supported_formats
        )

    except Exception as e:
        logger.error(f"STT provider {provider} test failed: {str(e)}")
        return STTTestResponse(
            provider=provider, available=False, error=str(e), supported_formats=[]
        )


@router.post("/transcribe/{provider}", response_model=TranscriptionResponse)
async def transcribe_audio(
    provider: str,
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),
    timestamps: bool = Form(False),
    config: AIGatewayConfig = Depends(get_config),
) -> TranscriptionResponse:
    """
    Transcribe audio file to text using specified STT provider

    Args:
        provider: STT provider name (openai, assemblyai, deepgram)
        file: Audio file to transcribe
        language: Language code (e.g., 'tr' for Turkish, 'en' for English)
        timestamps: Whether to include word-level timestamps
    """

    start_time = time.time()

    # Validate file
    validate_audio_file(file, config)

    # Get provider
    speech_provider = await get_speech_provider(provider, config)

    temp_file_path = None
    try:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=f".{Path(file.filename).suffix}"
        ) as temp_file:
            temp_file_path = temp_file.name
            content = await file.read()
            temp_file.write(content)

        # Create speech request
        request = SpeechRequest(
            audio_data=temp_file_path,
            language=language or "auto",
            timestamps=timestamps,
        )

        # Transcribe audio
        logger.info(f"Starting transcription with {provider} provider")
        response = await speech_provider.transcribe(request)

        processing_time = time.time() - start_time

        # Extract word-level timestamps if available
        words_data = None
        if timestamps and hasattr(response, "words") and response.words:
            words_data = []
            for word in response.words:
                # Handle both dict and object formats
                if isinstance(word, dict):
                    word_data = {
                        "word": word.get("word", ""),
                        "start": word.get("start", 0.0),
                        "end": word.get("end", 0.0),
                    }
                    if "confidence" in word:
                        word_data["confidence"] = word["confidence"]
                else:
                    # Object format
                    word_data = {
                        "word": getattr(word, "word", ""),
                        "start": getattr(word, "start", 0.0),
                        "end": getattr(word, "end", 0.0),
                    }
                    if hasattr(word, "confidence"):
                        word_data["confidence"] = word.confidence
                words_data.append(word_data)

        logger.info(f"Transcription completed in {processing_time:.2f}s")

        return TranscriptionResponse(
            text=response.text,
            language=getattr(response, "language", language),
            confidence=getattr(response, "confidence", None),
            duration=getattr(response, "duration", None),
            words=words_data,
            provider=provider,
            processing_time=processing_time,
        )

    except Exception as e:
        logger.error(f"Transcription failed with {provider}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

    finally:
        # Clean up temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                logger.warning(f"Failed to delete temporary file: {e}")


@router.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio_auto(
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),
    timestamps: bool = Form(False),
    preferred_provider: Optional[str] = Form(None),
    config: AIGatewayConfig = Depends(get_config),
) -> TranscriptionResponse:
    """
    Transcribe audio file using the best available provider

    Args:
        file: Audio file to transcribe
        language: Language code (e.g., 'tr' for Turkish, 'en' for English)
        timestamps: Whether to include word-level timestamps
        preferred_provider: Preferred provider if available
    """

    # Get available providers
    providers_response = await get_available_providers(config)
    available_providers = providers_response.providers

    if not available_providers:
        raise HTTPException(status_code=500, detail="No STT providers available")

    # Choose provider
    provider = (
        preferred_provider
        if preferred_provider in available_providers
        else available_providers[0]
    )

    # Delegate to specific provider endpoint
    return await transcribe_audio(
        provider=provider,
        file=file,
        language=language,
        timestamps=timestamps,
        config=config,
    )


@router.post("/transcribe-base64/{provider}", response_model=TranscriptionResponse)
async def transcribe_base64_audio(
    provider: str,
    request: Base64AudioRequest,
    config: AIGatewayConfig = Depends(get_config),
) -> TranscriptionResponse:
    """
    Transcribe Base64 encoded audio to text using specified STT provider

    Args:
        provider: STT provider name (openai, assemblyai, deepgram)
        request: Base64AudioRequest with audio_base64, format, language, timestamps
    """

    start_time = time.time()

    try:
        import base64
        import tempfile

        # Validate and decode base64 audio
        try:
            audio_data = base64.b64decode(request.audio_base64)
        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"Invalid base64 audio data: {str(e)}"
            )

        # Validate audio format
        if request.format not in config.stt_supported_formats:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported audio format: {request.format}. Supported: {', '.join(config.stt_supported_formats)}",
            )

        # Check audio data size (same as file upload limit)
        if len(audio_data) > config.stt_max_file_size:
            raise HTTPException(
                status_code=413,
                detail=f"Audio data too large. Maximum size: {config.stt_max_file_size / (1024*1024):.1f}MB",
            )

        # Get provider
        speech_provider = await get_speech_provider(provider, config)

        temp_file_path = None
        try:
            # Save base64 audio to temporary file
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=f".{request.format}"
            ) as temp_file:
                temp_file_path = temp_file.name
                temp_file.write(audio_data)

            # Create speech request
            stt_request = SpeechRequest(
                audio_data=temp_file_path,
                language=request.language or "auto",
                timestamps=request.timestamps or request.word_timestamps,
            )

            # Transcribe audio
            logger.info(f"Starting Base64 transcription with {provider} provider")
            response = await speech_provider.transcribe(stt_request)

            processing_time = time.time() - start_time

            # Extract word-level timestamps if available
            words_data = None
            if (
                (request.timestamps or request.word_timestamps)
                and hasattr(response, "words")
                and response.words
            ):
                words_data = []
                for word in response.words:
                    # Handle both dict and object formats
                    if isinstance(word, dict):
                        word_data = {
                            "word": word.get("word", ""),
                            "start": word.get("start", 0.0),
                            "end": word.get("end", 0.0),
                        }
                        if "confidence" in word:
                            word_data["confidence"] = word["confidence"]
                    else:
                        # Object format
                        word_data = {
                            "word": getattr(word, "word", ""),
                            "start": getattr(word, "start", 0.0),
                            "end": getattr(word, "end", 0.0),
                        }
                        if hasattr(word, "confidence"):
                            word_data["confidence"] = word.confidence
                    words_data.append(word_data)

            logger.info(f"Base64 transcription completed in {processing_time:.2f}s")

            return TranscriptionResponse(
                text=response.text,
                language=getattr(response, "language", request.language),
                confidence=getattr(response, "confidence", None),
                duration=getattr(response, "duration", None),
                words=words_data,
                provider=provider,
                processing_time=processing_time,
            )

        finally:
            # Clean up temporary file
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                except Exception as e:
                    logger.warning(f"Failed to delete temporary file: {e}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Base64 transcription failed with {provider}: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Base64 transcription failed: {str(e)}"
        )


@router.post("/transcribe-url/{provider}", response_model=TranscriptionResponse)
async def transcribe_url_audio(
    provider: str,
    request: URLAudioRequest,
    config: AIGatewayConfig = Depends(get_config),
) -> TranscriptionResponse:
    """
    Download audio from URL and transcribe to text using specified STT provider

    Args:
        provider: STT provider name (openai, assemblyai, deepgram)
        request: URLAudioRequest with audio_url, language, timestamps
    """

    start_time = time.time()

    try:
        import httpx
        import tempfile
        from urllib.parse import urlparse

        # Validate URL
        parsed_url = urlparse(request.audio_url)
        if not all([parsed_url.scheme, parsed_url.netloc]):
            raise HTTPException(status_code=400, detail="Invalid audio URL provided")

        # Get provider
        speech_provider = await get_speech_provider(provider, config)

        temp_file_path = None
        try:
            # Download audio from URL
            logger.info(f"Downloading audio from URL: {request.audio_url}")
            async with httpx.AsyncClient(timeout=30.0) as client:
                audio_response = await client.get(request.audio_url)
                audio_response.raise_for_status()

                audio_data = audio_response.content

                # Check file size
                if len(audio_data) > config.stt_max_file_size:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Downloaded audio too large. Maximum size: {config.stt_max_file_size / (1024*1024):.1f}MB",
                    )

                # Determine file format from URL or Content-Type
                content_type = audio_response.headers.get("content-type", "")
                file_ext = "wav"  # default

                if "mp3" in content_type or request.audio_url.endswith(".mp3"):
                    file_ext = "mp3"
                elif "wav" in content_type or request.audio_url.endswith(".wav"):
                    file_ext = "wav"
                elif "m4a" in content_type or request.audio_url.endswith(".m4a"):
                    file_ext = "m4a"
                elif "ogg" in content_type or request.audio_url.endswith(".ogg"):
                    file_ext = "ogg"
                elif "flac" in content_type or request.audio_url.endswith(".flac"):
                    file_ext = "flac"
                elif "webm" in content_type or request.audio_url.endswith(".webm"):
                    file_ext = "webm"

                # Save to temporary file
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=f".{file_ext}"
                ) as temp_file:
                    temp_file_path = temp_file.name
                    temp_file.write(audio_data)

            # Create speech request
            stt_request = SpeechRequest(
                audio_data=temp_file_path,
                language=request.language or "auto",
                timestamps=request.timestamps or request.word_timestamps,
            )

            # Transcribe audio
            logger.info(f"Starting URL transcription with {provider} provider")
            response = await speech_provider.transcribe(stt_request)

            processing_time = time.time() - start_time

            # Extract word-level timestamps if available
            words_data = None
            if (
                (request.timestamps or request.word_timestamps)
                and hasattr(response, "words")
                and response.words
            ):
                words_data = []
                for word in response.words:
                    # Handle both dict and object formats
                    if isinstance(word, dict):
                        word_data = {
                            "word": word.get("word", ""),
                            "start": word.get("start", 0.0),
                            "end": word.get("end", 0.0),
                        }
                        if "confidence" in word:
                            word_data["confidence"] = word["confidence"]
                    else:
                        # Object format
                        word_data = {
                            "word": getattr(word, "word", ""),
                            "start": getattr(word, "start", 0.0),
                            "end": getattr(word, "end", 0.0),
                        }
                        if hasattr(word, "confidence"):
                            word_data["confidence"] = word.confidence
                    words_data.append(word_data)

            logger.info(f"URL transcription completed in {processing_time:.2f}s")

            return TranscriptionResponse(
                text=response.text,
                language=getattr(response, "language", request.language),
                confidence=getattr(response, "confidence", None),
                duration=getattr(response, "duration", None),
                words=words_data,
                provider=provider,
                processing_time=processing_time,
            )

        finally:
            # Clean up temporary file
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                except Exception as e:
                    logger.warning(f"Failed to delete temporary file: {e}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"URL transcription failed with {provider}: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"URL transcription failed: {str(e)}"
        )


@router.get("/")
async def stt_root():
    """STT API root endpoint"""
    return {
        "message": "Speech-to-Text API",
        "version": "2.0.0",
        "endpoints": {
            "providers": "/providers",
            "test": "/test/{provider}",
            "transcribe": "/transcribe/{provider}",
            "transcribe_auto": "/transcribe",
            "transcribe_base64": "/transcribe-base64/{provider}",
            "transcribe_url": "/transcribe-url/{provider}",
        },
        "supported_providers": ["openai", "assemblyai", "deepgram"],
        "supported_formats": ["mp3", "wav", "m4a", "ogg", "flac", "webm"],
        "input_methods": ["file_upload", "base64", "url"],
    }
