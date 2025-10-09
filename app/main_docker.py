#!/usr/bin/env python3
"""
Multi-modal Modular AI Gateway - Docker Version
FastAPI application with direct provider support for Docker deployment
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Dict, Optional
import os

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from app.config.manager import ConfigManager
from app.config.models import AIGatewayConfig
from app.api.routes import health, image, providers, websocket
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
    logger.info("🚀 Starting AI Gateway (Docker)...")

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

    # Use Docker-specific LLM router
    if os.getenv("DOCKER_ENV", "false").lower() == "true":
        from app.api.routes import llm_docker

        app.include_router(llm_docker.router, prefix="/api/llm", tags=["LLM"])
    else:
        from app.api.routes import llm

        app.include_router(llm.router, prefix="/api/llm", tags=["LLM"])

    app.include_router(image.router, prefix="/api/image", tags=["Image"])
    app.include_router(websocket.router, tags=["WebSocket"])

    return app


# Create the app instance
app = create_app()


def main():
    """Main entry point"""
    print(
        """
🐳 Multi-modal AI Gateway v2.0 (Docker)
=======================================
🎯 Features:
  ✅ Direct provider integration
  ✅ OpenAI, Anthropic, Gemini support
  ✅ REST API endpoints
  ✅ Dynamic configuration
  ✅ Session management

🌐 Endpoints:
  📊 Health: http://localhost:8000/health
  🔧 Providers: http://localhost:8000/api/providers
  🤖 LLM: http://localhost:8000/api/llm
  🖼️ Image: http://localhost:8000/api/image
  📚 Docs: http://localhost:8000/docs

🔧 Usage:
  # Docker deployment
  docker-compose up --build
=======================================
    """
    )

    uvicorn.run(
        "app.main_docker:app", host="0.0.0.0", port=8000, reload=False, log_level="info"
    )


if __name__ == "__main__":
    main()
