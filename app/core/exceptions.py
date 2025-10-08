"""
Custom exceptions and error handlers for the AI Gateway
"""

import logging
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger(__name__)


class AIGatewayException(Exception):
    """Base exception for AI Gateway"""
    pass


class ProviderException(AIGatewayException):
    """Exception for provider-related errors"""
    pass


class ConfigurationException(AIGatewayException):
    """Exception for configuration-related errors"""
    pass


class SessionException(AIGatewayException):
    """Exception for session-related errors"""
    pass


def setup_exception_handlers(app: FastAPI) -> None:
    """Set up global exception handlers for the FastAPI app"""
    
    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        logger.error(f"HTTP error: {exc.status_code} - {exc.detail}")
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.detail,
                "status_code": exc.status_code,
                "path": str(request.url)
            }
        )
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        logger.error(f"Validation error: {exc.errors()}")
        return JSONResponse(
            status_code=422,
            content={
                "error": "Validation Error",
                "details": exc.errors(),
                "path": str(request.url)
            }
        )
    
    @app.exception_handler(AIGatewayException)
    async def ai_gateway_exception_handler(request: Request, exc: AIGatewayException):
        logger.error(f"AI Gateway error: {str(exc)}")
        return JSONResponse(
            status_code=500,
            content={
                "error": str(exc),
                "type": exc.__class__.__name__,
                "path": str(request.url)
            }
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unexpected error: {str(exc)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal Server Error",
                "message": "An unexpected error occurred",
                "path": str(request.url)
            }
        )