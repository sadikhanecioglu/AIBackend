"""Call agent WebSocket bridge for ElevenLabs ConvAI"""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import logging
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse
from xml.etree import ElementTree as ET

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

try:  # Twilio SDK is optional until outbound calls are used
    from twilio.base.exceptions import TwilioRestException  # type: ignore[import]
    from twilio.rest import Client as TwilioClient  # type: ignore[import]
except ImportError:  # pragma: no cover - optional dependency
    TwilioClient = None  # type: ignore
    TwilioRestException = None  # type: ignore


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


class TwilioCallRequest(BaseModel):
    """Payload for initiating an outbound Twilio call using media streams."""

    to_number: str = Field(..., description="Destination number to dial")
    from_number: Optional[str] = Field(
        None, description="Optional caller ID. Defaults to configured Twilio number"
    )
    agent_id: Optional[str] = Field(
        None,
        description="ElevenLabs agent identifier to use for this call",
    )
    call_id: Optional[str] = Field(
        None,
        description="Unique identifier for the call. Generated automatically when omitted",
    )
    company_id: Optional[str] = Field(
        None, description="Optional company identifier forwarded to the stream"
    )
    first_message: Optional[str] = Field(
        None,
        description="Initial message the agent should speak when the call starts",
    )
    prompt: Optional[str] = Field(
        None,
        description="Prompt text providing additional instructions for the agent",
    )
    call_status_hook: Optional[str] = Field(
        None, description="Webhook invoked when the call starts"
    )
    call_end_hook: Optional[str] = Field(
        None, description="Webhook invoked when the call ends"
    )
    conversation_data: Optional[Dict[str, Any]] = Field(
        None,
        description="Conversation initiation payload forwarded to ElevenLabs",
    )
    custom_parameters: Dict[str, str] = Field(
        default_factory=dict,
        description="Additional Twilio stream parameters to include",
    )
    twilio_account_sid: Optional[str] = Field(
        None, description="Override configured Twilio Account SID"
    )
    twilio_auth_token: Optional[str] = Field(
        None, description="Override configured Twilio Auth Token"
    )
    twilio_from_number: Optional[str] = Field(
        None, description="Override configured Twilio caller ID"
    )
    stream_url: Optional[str] = Field(
        None,
        description="Explicit WebSocket URL for the media stream. Overrides base host",
    )
    stream_host: Optional[str] = Field(
        None,
        description=(
            "Host domain used to build the media stream URL when stream_url is not provided"
        ),
    )
    status_callback_url: Optional[str] = Field(
        None, description="Twilio status callback URL"
    )
    status_callback_method: Optional[str] = Field(
        None, description="HTTP method for Twilio status callback"
    )
    status_callback_events: List[str] = Field(
        default_factory=list,
        description="Specific Twilio status events to subscribe to",
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


@router.post("/twilio/call", status_code=status.HTTP_202_ACCEPTED)
async def originate_twilio_call(
    request: Request, payload: TwilioCallRequest
) -> Dict[str, Any]:
    """Originate an outbound call via Twilio Voice and connect to our media stream."""

    if TwilioClient is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Twilio client library is not installed",
        )

    config = _get_config_from_request(request)

    account_sid = payload.twilio_account_sid or config.twilio_account_sid
    auth_token = payload.twilio_auth_token or config.twilio_auth_token
    api_key = config.twilio_api_key
    api_secret = config.twilio_api_secret

    if not account_sid:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Twilio Account SID is not configured",
        )

    caller_id = (
        payload.twilio_from_number or payload.from_number or config.twilio_from_number
    )
    if not caller_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Caller ID (from_number) must be provided",
        )

    to_number = _normalize_phone_number(payload.to_number)
    from_number = _normalize_phone_number(caller_id)

    call_id = payload.call_id or str(uuid.uuid4())

    stream_url = _resolve_twilio_stream_url(
        request,
        call_id,
        payload.stream_url,
        payload.stream_host,
        config.twilio_stream_url,
        config.twilio_stream_host,
    )

    stream_parameters: Dict[str, str] = {
        "agent": payload.agent_id
        or config.twilio_default_agent_id
        or config.elevenlabs_agent_id,
        "toNumber": to_number,
        "fromNumber": from_number,
        "ditacallid": call_id,
        "companyId": payload.company_id or config.twilio_default_company_id,
        "FirstMessage": payload.first_message
        or config.twilio_default_first_message
        or "",
        "Prompt": payload.prompt or config.twilio_default_prompt or "",
        "callStatusHook": payload.call_status_hook,
        "callEndHook": payload.call_end_hook,
    }

    if payload.conversation_data:
        try:
            stream_parameters["conversation_initiation_client_data"] = json.dumps(
                payload.conversation_data
            )
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"conversation_data is not JSON serializable: {exc}",
            )

    for key, value in payload.custom_parameters.items():
        stream_parameters[key] = str(value)

    stream_parameters = {
        key: value
        for key, value in stream_parameters.items()
        if value not in (None, "")
    }

    twiml = _build_twilio_twiml(stream_url, stream_parameters)

    if api_key and api_secret:
        client = TwilioClient(api_key, api_secret, account_sid=account_sid)
    elif auth_token:
        client = TwilioClient(account_sid, auth_token)
    else:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Twilio credentials are not configured",
        )

    if config.twilio_region:
        client.region = config.twilio_region
    if config.twilio_edge:
        client.edge = config.twilio_edge

    call_kwargs: Dict[str, Any] = {
        "from_": from_number,
        "to": to_number,
        "twiml": twiml,
    }

    status_callback_url = (
        payload.status_callback_url or config.twilio_status_callback_url
    )
    if status_callback_url:
        call_kwargs["status_callback"] = status_callback_url

    status_callback_method = (
        payload.status_callback_method or config.twilio_status_callback_method
    )
    if status_callback_method:
        call_kwargs["status_callback_method"] = status_callback_method

    status_events = (
        payload.status_callback_events or config.twilio_status_callback_events
    )
    if status_events:
        call_kwargs["status_callback_event"] = status_events

    try:
        twilio_call = await _twilio_call_create(client, call_kwargs)
    except TwilioRestException as exc:  # type: ignore[arg-type]
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Twilio originate failed: {exc.msg if hasattr(exc, 'msg') else exc}",
        ) from exc

    response_payload = {
        "status": "initiated",
        "call_id": call_id,
        "call_sid": getattr(twilio_call, "sid", None),
        "to": to_number,
        "from": from_number,
        "stream_url": stream_url,
        "parameters": stream_parameters,
    }

    if payload.call_status_hook:
        asyncio.create_task(
            notify_hook(
                payload.call_status_hook,
                {
                    "event": "call_requested",
                    "call_id": call_id,
                    "call_sid": getattr(twilio_call, "sid", None),
                },
            )
        )

    return response_payload


@dataclass
class TwilioCallSession:
    """Represents an active Twilio streaming session."""

    call_id: str
    stream_sid: Optional[str]
    to_number: Optional[str]
    from_number: Optional[str]
    company_id: Optional[str]
    call_sid: Optional[str]
    status_hook: Optional[str] = None
    end_hook: Optional[str] = None
    agent: Optional[str] = None
    conversation_id: Optional[str] = None


# Track active Twilio sockets by call identifier
twilio_active_connections: Dict[str, WebSocket] = {}
twilio_sessions: Dict[str, TwilioCallSession] = {}
_twilio_active_lock: Optional[asyncio.Lock] = None


def _get_twilio_lock() -> asyncio.Lock:
    global _twilio_active_lock
    if _twilio_active_lock is None:
        _twilio_active_lock = asyncio.Lock()
    return _twilio_active_lock


def _build_twilio_twiml(stream_url: str, parameters: Dict[str, str]) -> str:
    response_el = ET.Element("Response")
    connect_el = ET.SubElement(response_el, "Connect")
    stream_el = ET.SubElement(connect_el, "Stream", attrib={"url": stream_url})

    for name, value in parameters.items():
        if value is None or value == "":
            continue
        ET.SubElement(
            stream_el,
            "Parameter",
            attrib={"name": name, "value": str(value)},
        )

    return ET.tostring(response_el, encoding="unicode")


def _normalize_phone_number(number: str) -> str:
    if not number:
        return number
    trimmed = number.strip()
    if trimmed.startswith(("sip:", "client:")):
        return trimmed
    if trimmed.startswith("+"):
        return trimmed
    return f"+{trimmed}"


def _resolve_twilio_stream_url(
    request: Request,
    call_id: str,
    explicit_url: Optional[str],
    host_override: Optional[str],
    config_base_url: Optional[str],
    config_host: Optional[str],
) -> str:
    base_url = explicit_url or config_base_url

    if base_url:
        parsed = urlparse(base_url)
        query = dict(parse_qsl(parsed.query))
        query["callid"] = call_id
        new_query = urlencode(query, doseq=True)
        parsed = parsed._replace(query=new_query)
        if not parsed.path:
            parsed = parsed._replace(path="/api/callagent/twilio")
        return urlunparse(parsed)

    resolved_host = host_override or config_host
    if resolved_host:
        parsed_host = urlparse(resolved_host)
        if parsed_host.scheme:
            base = resolved_host.rstrip("/")
        else:
            scheme = "wss" if request.url.scheme == "https" else "ws"
            base = f"{scheme}://{resolved_host.rstrip('/')}"
        stream_url = f"{base}/api/callagent/twilio"
        return f"{stream_url}?callid={call_id}"

    request_host = request.headers.get("host") or request.url.hostname
    if not request_host:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unable to determine stream host for Twilio call",
        )

    scheme = "wss" if request.url.scheme == "https" else "ws"
    stream_url = f"{scheme}://{request_host}/api/callagent/twilio"
    return f"{stream_url}?callid={call_id}"


async def _twilio_call_create(
    client: Any,
    call_kwargs: Dict[str, Any],
) -> Any:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: client.calls.create(**call_kwargs))


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
        self.current_agent_id: Optional[str] = (
            context.agent_id or config.elevenlabs_agent_id
        )

    async def _get_signed_url(self, agent_id: Optional[str]) -> str:
        if not self._config.elevenlabs_api_key:
            raise RuntimeError("ElevenLabs API key is not configured")
        if not agent_id:
            raise RuntimeError("ElevenLabs agent ID is not configured")

        endpoint = self._config.elevenlabs_signed_url_endpoint
        url = f"{endpoint}?agent_id={agent_id}"
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

    async def connect(
        self,
        agent_id_override: Optional[str] = None,
        conversation_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        agent_id = (
            agent_id_override
            or self._context.agent_id
            or self._config.elevenlabs_agent_id
        )
        self.current_agent_id = agent_id

        signed_url = await self._get_signed_url(agent_id)
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

        if conversation_data:
            initiation_payload["conversation_initiation_client_data"] = (
                conversation_data
            )

        if agent_id and self._context.agent_id != agent_id:
            self._context.agent_id = agent_id

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

    async def listen_twilio(
        self,
        websocket: WebSocket,
        stream_sid: Optional[str],
        call_id: Optional[str] = None,
    ) -> None:
        if not self.ws:
            return

        try:
            while True:
                payload = await self.ws.recv()
                data = json.loads(payload)
                event_type = data.get("type")

                if event_type == "audio":
                    audio_b64 = data["audio_event"]["audio_base_64"]
                    if websocket.application_state == WebSocketState.CONNECTED:
                        message = {
                            "event": "media",
                            "streamSid": stream_sid,
                            "media": {"payload": audio_b64},
                        }
                        await websocket.send_text(json.dumps(message))

                elif event_type == "ping":
                    pong_msg = {
                        "type": "pong",
                        "event_id": data["ping_event"]["event_id"],
                    }
                    await self.ws.send(json.dumps(pong_msg))

                elif event_type == "interruption":
                    if websocket.application_state == WebSocketState.CONNECTED:
                        clear_msg = {"event": "clear"}
                        if stream_sid:
                            clear_msg["streamSid"] = stream_sid
                        await websocket.send_text(json.dumps(clear_msg))

                elif event_type == "conversation_initiation":
                    conversation_event = data.get("conversation_initiation_event", {})
                    conversation_id = conversation_event.get("conversation_id")
                    if not conversation_id:
                        conversation_id = conversation_event.get(
                            "conversation", {}
                        ).get("id")

                    if call_id and conversation_id:
                        session = twilio_sessions.get(call_id)
                        if session:
                            session.conversation_id = conversation_id
                            if session.status_hook:
                                asyncio.create_task(
                                    notify_hook(
                                        session.status_hook,
                                        {
                                            "event": "conversation_initiated",
                                            "call_id": call_id,
                                            "conversation_id": conversation_id,
                                        },
                                    )
                                )

                    logger.info(
                        "ElevenLabs conversation initiated for call %s", call_id
                    )

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
            logger.info("ElevenLabs WebSocket closed for call %s", call_id)
        except asyncio.CancelledError:  # pragma: no cover - background task cancel
            logger.debug("ElevenLabs Twilio listener task cancelled")
        except Exception as exc:
            logger.error("Error in ElevenLabs Twilio listener: %s", exc)


@router.websocket("/twilio")
async def twilio_call_bridge(websocket: WebSocket) -> None:
    """WebSocket endpoint that bridges Twilio Media Streams with ElevenLabs ConvAI."""

    await websocket.accept()

    app_state = getattr(websocket, "app", None)
    config: Optional[AIGatewayConfig] = None
    if app_state and hasattr(app_state, "state"):
        config = getattr(app_state.state, "app_config", None)

    if config is None:
        await websocket.send_text(
            json.dumps({"error": "Application configuration not available"})
        )
        await websocket.close(code=1011)
        return

    call_id: Optional[str] = None
    listener_task: Optional[asyncio.Task] = None
    eleven_client: Optional[ElevenLabsClient] = None
    session: Optional[TwilioCallSession] = None
    stream_sid: Optional[str] = None

    try:
        while True:
            message = await websocket.receive()
            message_type = message.get("type")

            if message_type == "websocket.disconnect":
                break

            text_data = message.get("text")
            if text_data is None:
                continue

            try:
                payload = json.loads(text_data)
            except json.JSONDecodeError:
                logger.warning("Invalid JSON payload received from Twilio")
                continue

            event = payload.get("event")

            if event == "start":
                start_info = payload.get("start") or {}
                stream_sid = start_info.get("streamSid")
                custom_params = start_info.get("customParameters") or {}

                proposed_call_id = (
                    custom_params.get("ditacallid")
                    or custom_params.get("callId")
                    or stream_sid
                    or str(uuid.uuid4())
                )

                lock = _get_twilio_lock()
                async with lock:
                    existing_ws = twilio_active_connections.get(proposed_call_id)
                    if (
                        existing_ws
                        and existing_ws is not websocket
                        and existing_ws.application_state == WebSocketState.CONNECTED
                    ):
                        await websocket.send_text(
                            json.dumps({"error": "This call is already connected."})
                        )
                        await websocket.close(
                            code=4000, reason="Duplicate socket for same callid"
                        )
                        return

                    twilio_active_connections[proposed_call_id] = websocket

                call_id = proposed_call_id
                status_hook = custom_params.get("callStatusHook")
                end_hook = custom_params.get("callEndHook")

                conversation_data_raw = custom_params.get(
                    "conversation_initiation_client_data"
                ) or custom_params.get("conversationInitiationClientData")
                conversation_data: Optional[Dict[str, Any]] = None
                if isinstance(conversation_data_raw, str) and conversation_data_raw:
                    try:
                        conversation_data = json.loads(conversation_data_raw)
                    except json.JSONDecodeError:
                        logger.warning(
                            "Failed to parse conversation initiation data for call %s",
                            proposed_call_id,
                        )
                elif isinstance(conversation_data_raw, dict):
                    conversation_data = conversation_data_raw

                session = TwilioCallSession(
                    call_id=call_id,
                    stream_sid=stream_sid,
                    to_number=custom_params.get("toNumber"),
                    from_number=custom_params.get("fromNumber"),
                    company_id=custom_params.get("companyId"),
                    call_sid=start_info.get("callSid"),
                    status_hook=status_hook,
                    end_hook=end_hook,
                    agent=custom_params.get("agent"),
                )

                twilio_sessions[call_id] = session

                if session.status_hook:
                    asyncio.create_task(
                        notify_hook(
                            session.status_hook,
                            {
                                "event": "call_started",
                                "call_id": call_id,
                                "stream_sid": stream_sid,
                            },
                        )
                    )

                first_message = custom_params.get("FirstMessage") or custom_params.get(
                    "firstMessage"
                )
                prompt = custom_params.get("Prompt") or custom_params.get("prompt")
                client_tools = custom_params.get("ClientTools") or custom_params.get(
                    "clientTools"
                )
                server_tools = custom_params.get("ServerTools") or custom_params.get(
                    "serverTools"
                )
                provider = custom_params.get("Provider") or custom_params.get(
                    "provider"
                )
                llm_provider = custom_params.get("LlmProvider") or custom_params.get(
                    "llmProvider"
                )

                call_context = CallAgentContext(
                    to_number=session.to_number or "",
                    from_number=session.from_number or "",
                    first_message=first_message,
                    prompt=prompt,
                    client_tools=client_tools,
                    server_tools=server_tools,
                    agent_id=session.agent or config.elevenlabs_agent_id,
                    provider=provider,
                    llm_provider=llm_provider,
                    call_status_hook=session.status_hook,
                    call_end_hook=session.end_hook,
                )

                eleven_client = ElevenLabsClient(config, call_context)

                try:
                    await eleven_client.connect(
                        agent_id_override=session.agent,
                        conversation_data=conversation_data,
                    )
                except Exception as exc:
                    logger.error(
                        "Unable to connect ElevenLabs for Twilio call %s: %s",
                        call_id,
                        exc,
                    )
                    await websocket.send_text(
                        json.dumps({"error": f"Unable to connect ElevenLabs: {exc}"})
                    )
                    await websocket.close(code=1011)
                    return

                listener_task = asyncio.create_task(
                    eleven_client.listen_twilio(websocket, stream_sid, call_id)
                )

                logger.info(
                    "Twilio call %s started (streamSid=%s)", call_id, stream_sid
                )

            elif event == "media":
                if not eleven_client:
                    logger.warning("Received media before ElevenLabs connection")
                    continue

                media_info = payload.get("media") or {}
                audio_payload = media_info.get("payload")
                if not audio_payload:
                    continue

                try:
                    audio_bytes = base64.b64decode(audio_payload)
                except (binascii.Error, ValueError):
                    logger.warning(
                        "Failed to decode media payload from Twilio call %s", call_id
                    )
                    continue

                await eleven_client.send_audio(audio_bytes)

            elif event in {"stop", "validate"}:
                logger.info("Twilio call %s received %s event", call_id, event)
                try:
                    await websocket.close()
                except RuntimeError:
                    pass
                break

            elif event == "mark":
                continue

            else:
                logger.debug("Unhandled Twilio event: %s", event)

    except WebSocketDisconnect:
        logger.info("Twilio websocket disconnected for call %s", call_id)
    except Exception as exc:
        logger.error("Error in Twilio call bridge: %s", exc)
    finally:
        if listener_task:
            listener_task.cancel()

        if eleven_client:
            await eleven_client.close()

        if call_id:
            lock = _get_twilio_lock()
            async with lock:
                existing_ws = twilio_active_connections.get(call_id)
                if existing_ws is websocket:
                    twilio_active_connections.pop(call_id, None)

            session = twilio_sessions.pop(call_id, None) or session

            if session and session.end_hook:
                asyncio.create_task(
                    notify_hook(
                        session.end_hook,
                        {
                            "event": "call_ended",
                            "call_id": call_id,
                            "stream_sid": session.stream_sid,
                            "conversation_id": session.conversation_id,
                        },
                    )
                )

        if session is None and call_id:
            twilio_sessions.pop(call_id, None)


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
