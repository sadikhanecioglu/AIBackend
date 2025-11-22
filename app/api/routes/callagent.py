"""Call agent WebSocket bridge for ElevenLabs ConvAI"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import aiohttp
import websockets
from fastapi import (
    APIRouter,
    HTTPException,
    Request,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from starlette.websockets import WebSocketState
from websockets.client import WebSocketClientProtocol
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK

from pydantic import BaseModel, Field

from app.config.models import AIGatewayConfig


class AriCallRequest(BaseModel):
    """Payload for initiating an outbound ARI call."""

    to_number: str = Field(..., description="Destination number to dial")
    from_number: Optional[str] = Field(
        None, description="Optional caller ID number presented on the call"
    )
    caller_id: Optional[str] = Field(
        None,
        description="Explicit caller ID string; overrides from_number if provided",
    )
    endpoint: Optional[str] = Field(
        None,
        description=(
            "Full ARI endpoint (e.g. PJSIP/trunk/1000). If omitted, the configured "
            "trunk will be combined with to_number."
        ),
    )
    trunk: Optional[str] = Field(
        None,
        description="Override for configured trunk when constructing the endpoint",
    )
    ari_url: Optional[str] = Field(
        None,
        description="Override for configured ari_url when connecting to ARI",
    )
    ari_username: Optional[str] = Field(
        None,
        description="Override for configured ari_username when connecting to ARI",
    )
    ari_password: Optional[str] = Field(
        None,
        description="Override for configured ari_password when connecting to ARI",
    )
    context: Optional[str] = Field(
        None, description="Dialplan context. Defaults to configured ari_context"
    )
    extension: Optional[str] = Field(
        None,
        description="Dialplan extension. Defaults to configured ari_extension",
    )
    priority: Optional[int] = Field(
        None, description="Dialplan priority. Defaults to configured ari_priority"
    )
    timeout: Optional[int] = Field(
        None,
        description="Dial timeout in seconds. Defaults to configured ari_timeout",
    )
    variables: Dict[str, str] = Field(
        default_factory=dict,
        description="Optional channel variables to set when originating the call",
    )

    class Config:
        anystr_strip_whitespace = True


def _get_config_from_request(request: Request) -> AIGatewayConfig:
    config: Optional[AIGatewayConfig] = getattr(request.app.state, "app_config", None)
    if config is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Application configuration not available",
        )
    return config


logger = logging.getLogger(__name__)


router = APIRouter()


@dataclass
class CallAgentContext:
    """Represents metadata passed along with the call."""

    to_number: str
    from_number: str
    first_message: Optional[str] = None
    prompt: Optional[str] = None
    client_tools: Optional[str] = None
    server_tools: Optional[str] = None
    agent_id: Optional[str] = None
    provider: Optional[str] = None
    llm_provider: Optional[str] = None
    call_status_hook: Optional[str] = None
    call_end_hook: Optional[str] = None

    @classmethod
    def from_websocket(
        cls, websocket: WebSocket, config: AIGatewayConfig
    ) -> "CallAgentContext":
        params = websocket.query_params

        to_number = params.get("exten", "").strip()
        from_number = params.get("caller", "").strip()

        return cls(
            to_number=to_number,
            from_number=from_number,
            first_message=params.get("FirstMessage"),
            prompt=params.get("Prompt"),
            client_tools=params.get("ClientTools"),
            server_tools=params.get("ServerTools"),
            agent_id=params.get("AgentId") or config.elevenlabs_agent_id,
            provider=params.get("Provider"),
            llm_provider=params.get("LlmProvider"),
            call_status_hook=params.get("CallStatusHook"),
            call_end_hook=params.get("CallEndHook"),
        )

    def is_valid(self) -> bool:
        return bool(self.to_number and self.from_number)

    def to_payload(self) -> Dict[str, Optional[str]]:
        return {
            "ToNumber": self.to_number,
            "FromNumber": self.from_number,
            "FirstMessage": self.first_message,
            "Prompt": self.prompt,
            "ClientTools": self.client_tools,
            "ServerTools": self.server_tools,
            "AgentId": self.agent_id,
            "Provider": self.provider,
            "LlmProvider": self.llm_provider,
            "CallStatusHook": self.call_status_hook,
            "CallEndHook": self.call_end_hook,
        }


async def notify_hook(url: Optional[str], payload: Dict[str, Any]) -> None:
    if not url:
        return

    try:
        async with aiohttp.ClientSession() as session:
            await session.post(url, json=payload, timeout=10)
    except Exception as exc:  # pragma: no cover - best effort notification
        logger.warning("Call agent hook notification failed: %s", exc)


class ElevenLabsClient:
    """Thin wrapper around ElevenLabs ConvAI WebSocket client."""

    def __init__(self, config: AIGatewayConfig, context: CallAgentContext):
        self._config = config
        self._context = context
        self.ws: Optional[WebSocketClientProtocol] = None
        self.chunk_size = 160
        self.media_paused = False

    async def _get_signed_url(self) -> str:
        if not self._config.elevenlabs_api_key:
            raise RuntimeError("ElevenLabs API key is not configured")
        if not self._config.elevenlabs_agent_id:
            raise RuntimeError("ElevenLabs agent ID is not configured")

        endpoint = self._config.elevenlabs_signed_url_endpoint
        url = f"{endpoint}?agent_id={self._config.elevenlabs_agent_id}"
        headers = {
            "xi-api-key": self._config.elevenlabs_api_key,
            "accept": "application/json",
        }

        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, timeout=15) as response:
                if response.status >= 400:
                    text = await response.text()
                    raise RuntimeError(
                        f"ElevenLabs signed URL request failed ({response.status}): {text}"
                    )
                payload = await response.json()
                signed_url = payload.get("signed_url")
                if not signed_url:
                    raise RuntimeError("ElevenLabs response missing signed_url")
                return signed_url

    async def connect(self) -> None:
        signed_url = await self._get_signed_url()
        logger.info("Connecting to ElevenLabs ConvAI via signed URL")

        self.ws = await websockets.connect(
            signed_url,
            ping_interval=20,
            ping_timeout=10,
            max_size=10 * 1024 * 1024,
        )

        initiation_payload: Dict[str, Any] = {
            "type": "conversation_initiation_client_data",
            "conversation_config_override": {"tts": {"output_format": "ulaw_8000"}},
        }

        if self._context.prompt:
            initiation_payload["conversation_config_override"]["conversation"] = {
                "voice_prompt": self._context.prompt
            }

        await self.ws.send(json.dumps(initiation_payload))
        logger.info("ElevenLabs ConvAI WebSocket established")

    async def close(self) -> None:
        if self.ws is not None:
            try:
                await self.ws.close()
            except Exception:  # pragma: no cover - cleanup best effort
                pass
            finally:
                self.ws = None

    def set_chunk_size(self, size: int) -> None:
        if size > 0:
            self.chunk_size = size
            logger.debug("Using chunk size %s bytes for ElevenLabs audio", size)

    async def send_audio(self, audio_bytes: bytes) -> None:
        if not self.ws:
            return

        message = {
            "type": "user_audio_chunk",
            "user_audio_chunk": base64.b64encode(audio_bytes).decode(),
        }

        await self.ws.send(json.dumps(message))

    async def listen(self, asterisk_ws: WebSocket) -> None:
        if not self.ws:
            return

        try:
            while True:
                payload = await self.ws.recv()
                data = json.loads(payload)
                event_type = data.get("type")

                if event_type == "audio":
                    audio_b64 = data["audio_event"]["audio_base_64"]
                    audio = base64.b64decode(audio_b64)

                    if self.media_paused:
                        if asterisk_ws.application_state == WebSocketState.CONNECTED:
                            await asterisk_ws.send_text("CONTINUE_MEDIA")
                        self.media_paused = False

                    chunk_size = max(1, self.chunk_size)
                    for index in range(0, len(audio), chunk_size):
                        chunk = audio[index : index + chunk_size]
                        if (
                            chunk
                            and asterisk_ws.application_state
                            == WebSocketState.CONNECTED
                        ):
                            await asterisk_ws.send_bytes(chunk)

                elif event_type == "ping":
                    pong_msg = {
                        "type": "pong",
                        "event_id": data["ping_event"]["event_id"],
                    }
                    await self.ws.send(json.dumps(pong_msg))

                elif event_type == "interruption":
                    self.media_paused = True
                    if asterisk_ws.application_state == WebSocketState.CONNECTED:
                        await asterisk_ws.send_text("PAUSE_MEDIA")
                        await asterisk_ws.send_text("FLUSH_MEDIA")

                elif event_type == "agent_response":
                    logger.info(
                        "ElevenLabs agent response: %s",
                        data.get("agent_response_event", {}).get("agent_response"),
                    )

                elif event_type == "user_transcript":
                    logger.info(
                        "User transcript: %s",
                        data.get("user_transcription_event", {}).get("user_transcript"),
                    )

                else:
                    logger.debug("Unhandled ElevenLabs event: %s", event_type)

        except (ConnectionClosedError, ConnectionClosedOK):
            logger.info("ElevenLabs WebSocket closed")
        except asyncio.CancelledError:  # pragma: no cover - background task cancel
            logger.debug("ElevenLabs listener task cancelled")
        except Exception as exc:
            logger.error("Error in ElevenLabs listener: %s", exc)


@router.post("/ari/call", status_code=status.HTTP_202_ACCEPTED)
async def originate_ari_call(
    request: Request, payload: AriCallRequest
) -> Dict[str, Any]:
    """Originate an outbound call via Asterisk ARI."""

    config = _get_config_from_request(request)
    config.ari_url = payload.ari_url or config.ari_url
    config.ari_username = payload.ari_username or config.ari_username
    config.ari_password = payload.ari_password or config.ari_password
    missing = [
        name
        for name, value in [
            ("ari_url", config.ari_url),
            ("ari_username", config.ari_username),
            ("ari_password", config.ari_password),
        ]
        if not value
    ]
    if missing:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"ARI configuration missing: {', '.join(missing)}",
        )

    endpoint = payload.endpoint
    if not endpoint:
        trunk = payload.trunk or config.ari_trunk
        if not trunk:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="ARI trunk is not configured",
            )
        endpoint = f"PJSIP/{payload.to_number}@{trunk}"

    context = payload.context or config.ari_context
    extension = payload.extension or config.ari_extension or payload.to_number
    priority = payload.priority or config.ari_priority or 1
    timeout = payload.timeout or config.ari_timeout or 30

    if not context:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="ARI context must be provided",
        )
    if not extension:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="ARI extension must be provided",
        )

    caller_id = payload.caller_id or payload.from_number

    variables_param = None
    if payload.variables:
        variables_param = ",".join(
            f"{key}={str(value)}" for key, value in payload.variables.items()
        )

    params = {
        "endpoint": endpoint,
        "extension": extension,
        "context": context,
        "priority": str(priority),
        "timeout": str(timeout),
    }
    if caller_id:
        params["callerId"] = caller_id
    if variables_param:
        params["variables"] = variables_param

    ari_url = config.ari_url.rstrip("/") + "/ari/channels"

    logger.info(
        "Initiating ARI outbound call to %s via %s (context=%s, extension=%s)",
        payload.to_number,
        endpoint,
        context,
        extension,
    )

    content: Any = None

    try:
        async with aiohttp.ClientSession(
            auth=aiohttp.BasicAuth(config.ari_username, config.ari_password)
        ) as session:
            async with session.post(ari_url, params=params, timeout=15) as response:
                if response.status >= 400:
                    error_text = await response.text()
                    raise HTTPException(
                        status_code=response.status,
                        detail=(
                            f"ARI originate failed ({response.status}): {error_text}"
                        ),
                    )

                content: Any
                try:
                    content = await response.json(content_type=None)
                except aiohttp.ContentTypeError:
                    content = {"raw": await response.text()}

    except aiohttp.ClientError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to contact ARI endpoint: {exc}",
        ) from exc

    channel_id = content.get("id") if isinstance(content, dict) else None

    return {
        "status": "initiated",
        "channel_id": channel_id,
        "endpoint": endpoint,
        "context": context,
        "extension": extension,
        "response": content,
    }


@router.websocket("/bridge")
async def call_agent_bridge(
    websocket: WebSocket,
) -> None:
    """WebSocket endpoint that bridges Asterisk audio with ElevenLabs ConvAI."""

    subprotocol_header = websocket.headers.get("sec-websocket-protocol", "")
    accepted_subprotocol = None
    if subprotocol_header:
        for proto in (p.strip() for p in subprotocol_header.split(",")):
            if proto == "media":
                accepted_subprotocol = "media"
                break

    await websocket.accept(subprotocol=accepted_subprotocol)

    app_state = getattr(websocket, "app", None)
    config: Optional[AIGatewayConfig] = None
    if app_state and hasattr(app_state, "state"):
        config = getattr(app_state.state, "app_config", None)

    if config is None:
        await websocket.send_json(
            {
                "type": "error",
                "message": "Application configuration not available",
            }
        )
        await websocket.close(code=1011)
        return

    context = CallAgentContext.from_websocket(websocket, config)
    if not context.is_valid():
        await websocket.send_json(
            {
                "type": "error",
                "message": "ToNumber and FromNumber query parameters are required",
            }
        )
        await websocket.close(code=1008)
        return

    await websocket.send_json(
        {
            "type": "call_context",
            "ToNumber": context.to_number,
            "FromNumber": context.from_number,
        }
    )

    asyncio.create_task(
        notify_hook(
            context.call_status_hook,
            {"event": "call_started", "context": context.to_payload()},
        )
    )

    eleven_client = ElevenLabsClient(config, context)

    try:
        await eleven_client.connect()
    except Exception as exc:
        logger.error("Unable to connect to ElevenLabs: %s", exc)
        await websocket.send_json({"type": "error", "message": str(exc)})
        await websocket.close(code=1011)
        return

    listener_task = asyncio.create_task(eleven_client.listen(websocket))
    send_allowed = asyncio.Event()
    send_allowed.set()

    try:
        while True:
            message = await websocket.receive()
            message_type = message.get("type")

            if message_type == "websocket.disconnect":
                break

            text_data = message.get("text")
            binary_data = message.get("bytes")

            if text_data is not None:
                logger.debug("MEDIA SIGNAL: %s", text_data)

                if text_data.startswith("MEDIA_START"):
                    for part in text_data.split():
                        if part.startswith("optimal_frame_size:"):
                            try:
                                frame_size = int(part.split(":", maxsplit=1)[1])
                                eleven_client.set_chunk_size(frame_size)
                            except ValueError:
                                logger.warning(
                                    "Failed to parse optimal_frame_size from %s", part
                                )
                            break
                    continue

                if text_data.startswith("MEDIA_XOFF"):
                    send_allowed.clear()
                    continue

                if text_data.startswith("MEDIA_XON"):
                    send_allowed.set()
                    continue

                if text_data.startswith("MEDIA_BUFFERING_COMPLETED"):
                    logger.debug("MEDIA buffering completed")
                    continue

                # Unhandled signalling message - ignore
                continue

            if binary_data:
                if send_allowed.is_set():
                    await eleven_client.send_audio(binary_data)

    except WebSocketDisconnect:
        logger.info("Call agent websocket disconnected")
    except Exception as exc:
        logger.error("Call agent bridge error: %s", exc)
    finally:
        listener_task.cancel()
        await eleven_client.close()

        asyncio.create_task(
            notify_hook(
                context.call_end_hook,
                {"event": "call_ended", "context": context.to_payload()},
            )
        )

        send_allowed.set()
