"""
WebSocket endpoints for real-time voice communication
"""

import asyncio
import base64
import json
import logging
import time
from datetime import datetime
from typing import Dict

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, Request

from app.core.session import VoiceSession
from app.config.models import AIGatewayConfig
from app.api.dependencies import get_config

logger = logging.getLogger(__name__)
router = APIRouter()


@router.websocket("/voice")
async def voice_websocket(
    websocket: WebSocket,
    request: Request = None
) -> None:
    """WebSocket endpoint for voice communication"""
    
    await websocket.accept()
    session_id = f"session_{int(time.time() * 1000)}"
    
    # Get config and active sessions from app state
    config: AIGatewayConfig = request.app.state.app_config
    active_sessions: Dict = request.app.state.active_sessions
    
    # Create voice session
    session = VoiceSession(session_id, config)
    
    # Store session
    active_sessions[session_id] = {
        "websocket": websocket,
        "session": session
    }
    
    logger.info(f"New WebSocket connection: {session_id}")
    logger.info(f"Providers: STT={config.stt_provider}, LLM={config.llm_provider}, TTS={config.tts_provider}")
    
    try:
        # Send connection confirmation
        await websocket.send_text(json.dumps({
            "type": "connection_established",
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "providers": {
                "stt": config.stt_provider,
                "llm": config.llm_provider,
                "tts": config.tts_provider
            }
        }))
        
        while True:
            # Receive message
            message = await websocket.receive_text()
            
            try:
                data = json.loads(message)
                response_data = None
                
                # Handle audio data
                if "user_audio_chunk" in data:
                    # Decode base64 audio
                    try:
                        audio_base64 = data["user_audio_chunk"]
                        audio_data = base64.b64decode(audio_base64)
                        
                        # Process audio through the session
                        response_data = await session.process_audio(audio_data)
                        
                    except Exception as audio_error:
                        logger.error(f"Audio processing error: {audio_error}")
                        response_data = {
                            "type": "error",
                            "message": f"Audio processing failed: {str(audio_error)}"
                        }
                
                # Handle text messages
                elif "text" in data:
                    text = data["text"]
                    
                    try:
                        # Generate text response
                        response_text = await session.generate_response(text)
                        
                        response_data = {
                            "type": "text_response",
                            "text": response_text,
                            "provider": config.llm_provider
                        }
                        
                    except Exception as llm_error:
                        logger.error(f"LLM processing error: {llm_error}")
                        response_data = {
                            "type": "error",
                            "message": f"Text processing failed: {str(llm_error)}"
                        }
                
                # Handle session info requests
                elif "action" in data and data["action"] == "get_session_info":
                    response_data = {
                        "type": "session_info",
                        "session_info": session.get_session_info()
                    }
                
                # Handle session reset
                elif "action" in data and data["action"] == "reset_session":
                    session.conversation_history.clear()
                    session.message_count = 0
                    response_data = {
                        "type": "session_reset",
                        "message": "Session reset successfully"
                    }
                
                # Unknown message format
                else:
                    response_data = {
                        "type": "error",
                        "message": "Unknown message format"
                    }
                
                # Send response if we have one
                if response_data:
                    response_data.update({
                        "session_id": session_id,
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    await websocket.send_text(json.dumps(response_data))
                    
                    # Log successful responses
                    if response_data["type"] not in ["error", "speech_in_progress"]:
                        logger.info(f"Session {session_id}: Response sent ({response_data['type']})")
                
            except json.JSONDecodeError:
                logger.error("JSON decode error")
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "Invalid JSON format",
                    "session_id": session_id
                }))
            
            except Exception as e:
                logger.error(f"Message processing error: {e}")
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": f"Message processing failed: {str(e)}",
                    "session_id": session_id
                }))
    
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {session_id}")
    
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    
    finally:
        # Clean up session
        if session_id in active_sessions:
            session.cleanup()
            del active_sessions[session_id]
        logger.info(f"Session cleaned up: {session_id}")


@router.get("/sessions")
async def get_active_sessions(request: Request) -> Dict:
    """Get information about active WebSocket sessions"""
    
    active_sessions: Dict = request.app.state.active_sessions
    
    sessions_info = {}
    for session_id, session_data in active_sessions.items():
        if "session" in session_data:
            sessions_info[session_id] = session_data["session"].get_session_info()
    
    return {
        "active_sessions": len(active_sessions),
        "sessions": sessions_info,
        "timestamp": datetime.now().isoformat()
    }


@router.delete("/sessions/{session_id}")
async def terminate_session(session_id: str, request: Request) -> Dict:
    """Terminate a specific WebSocket session"""
    
    active_sessions: Dict = request.app.state.active_sessions
    
    if session_id not in active_sessions:
        return {
            "success": False,
            "message": f"Session {session_id} not found"
        }
    
    try:
        # Close WebSocket connection
        session_data = active_sessions[session_id]
        if "websocket" in session_data:
            await session_data["websocket"].close()
        
        # Clean up session
        if "session" in session_data:
            session_data["session"].cleanup()
        
        # Remove from active sessions
        del active_sessions[session_id]
        
        logger.info(f"Session {session_id} terminated manually")
        
        return {
            "success": True,
            "message": f"Session {session_id} terminated successfully"
        }
        
    except Exception as e:
        logger.error(f"Error terminating session {session_id}: {e}")
        return {
            "success": False,
            "message": f"Error terminating session: {str(e)}"
        }


@router.post("/sessions/cleanup")
async def cleanup_expired_sessions(request: Request) -> Dict:
    """Clean up expired sessions"""
    
    active_sessions: Dict = request.app.state.active_sessions
    config: AIGatewayConfig = request.app.state.app_config
    
    expired_sessions = []
    
    for session_id, session_data in list(active_sessions.items()):
        if "session" in session_data:
            session = session_data["session"]
            if session.is_expired(config.session_timeout):
                expired_sessions.append(session_id)
                
                # Clean up expired session
                session.cleanup()
                if "websocket" in session_data:
                    try:
                        await session_data["websocket"].close()
                    except:
                        pass  # Connection might already be closed
                
                del active_sessions[session_id]
    
    logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
    
    return {
        "cleaned_sessions": len(expired_sessions),
        "expired_session_ids": expired_sessions,
        "remaining_sessions": len(active_sessions),
        "timestamp": datetime.now().isoformat()
    }