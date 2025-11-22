#!/usr/bin/env python3
"""
Multi-modal Modular AI Gateway
FastAPI application with modular provider support using llm-provider-factory
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Dict, Optional

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from app.config.manager import ConfigManager
from app.config.models import AIGatewayConfig
from app.api.routes import health, llm, image, providers, websocket, stt, callagent
from app.core.exceptions import setup_exception_handlers

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global state
config_manager: Optional[ConfigManager] = None
app_config: Optional[AIGatewayConfig] = None
active_sessions: Dict[str, Dict] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global config_manager, app_config

    # Startup
    logger.info("🚀 Starting AI Gateway...")

    # Initialize configuration
    config_manager = ConfigManager()
    app_config = config_manager.load_config()

    # Store in app state for dependency injection
    app.state.config_manager = config_manager
    app.state.app_config = app_config
    app.state.active_sessions = active_sessions

    # Print configuration
    config_manager.print_config()

    logger.info("✅ AI Gateway started successfully")

    yield

    # Shutdown
    logger.info("🛑 Shutting down AI Gateway...")
    # Clean up active sessions
    active_sessions.clear()
    logger.info("✅ AI Gateway shutdown complete")


def create_app() -> FastAPI:
    """Create and configure FastAPI application"""

    app = FastAPI(
        title="Multi-modal AI Gateway",
        description="A modular AI Gateway supporting STT → LLM → TTS workflows",
        version="2.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure based on your needs
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Exception handlers
    setup_exception_handlers(app)

    # Include routers
    app.include_router(health.router, prefix="/health", tags=["Health"])
    app.include_router(providers.router, prefix="/api/providers", tags=["Providers"])
    app.include_router(llm.router, prefix="/api/llm", tags=["LLM"])
    app.include_router(image.router, prefix="/api/image", tags=["Image"])
    app.include_router(stt.router, prefix="/api/stt", tags=["Speech-to-Text"])
    app.include_router(websocket.router, tags=["WebSocket"])
    app.include_router(callagent.router, prefix="/api/callagent", tags=["Call Agent"])

    return app


# Create the app instance
app = create_app()


def main():
    """Main entry point"""
    print(
        """
🚀 Multi-modal AI Gateway v2.0
===============================
🎯 Features:
  ✅ Modular provider architecture
  ✅ STT → LLM → TTS pipeline
  ✅ WebSocket voice communication
  ✅ REST API endpoints
  ✅ Dynamic configuration
  ✅ Session management

🌐 Endpoints:
  🔗 WebSocket: ws://localhost:8000/voice
  📊 Health: http://localhost:8000/health
  🔧 Providers: http://localhost:8000/providers
  🤖 LLM: http://localhost:8000/llm
  🖼️ Image: http://localhost:8000/image
  📚 Docs: http://localhost:8000/docs

🔧 Usage:
  # Default configuration
  python -m app.main
  
  # Custom port
  python -m app.main --port 3000
===============================
    """
    )

    uvicorn.run(
        "app.main:app", host="0.0.0.0", port=3000, reload=True, log_level="info"
    )
