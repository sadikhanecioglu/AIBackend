# Multi-modal Modular AI Gateway 🎯

A FastAPI-based Multi-modal AI Gateway that orchestrates Speech-to-Text (STT) → Large Language Model (LLM) → Text-to-Speech (TTS) workflows through both REST and WebSocket APIs. Built with Strategy Pattern and Dependency Injection for maximum modularity.

## ✨ Features

- 🎙️ **Speech-to-Text Integration**: Support for OpenAI Whisper, Azure Speech, Google Speech
- 🤖 **LLM Integration**: OpenAI GPT, Anthropic Claude, Google Gemini, Azure OpenAI, VertexAI, Ollama
- 🔊 **Text-to-Speech**: OpenAI TTS, Azure Speech, Google TTS
- 🖼️ **Image Generation**: OpenAI DALL-E, Stability AI, Azure OpenAI
- 🔄 **Real-time Communication**: WebSocket support for voice conversations
- ⚙️ **Dynamic Configuration**: Runtime provider switching
- 📊 **Session Management**: Multi-session support with tracking
- 🏗️ **Modular Architecture**: Strategy Pattern with Factory injection

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   STT Provider  │───▶│   LLM Provider  │───▶│   TTS Provider  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                    AI Gateway FastAPI                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │ WebSocket   │  │    REST     │  │   Config    │            │
│  │ /voice      │  │ /llm /image │  │ Management  │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- API keys for your chosen providers (OpenAI, Anthropic, Google, etc.)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/your-username/ai-backend.git
cd ai-backend
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env with your API keys
# Never commit .env to version control!
```

> **⚠️ Security Warning**: The `.env` file contains sensitive API keys and should **NEVER** be committed to version control. It is automatically excluded by `.gitignore`. Use `.env.example` as a reference for required configuration variables.

4. **Run the application**
```bash
python main.py
```

The API will be available at:
- **HTTP**: http://localhost:8000
- **WebSocket**: ws://localhost:8000/voice
- **Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## 📋 Environment Configuration

Create a `.env` file with your API keys:

```env
# OpenAI
OPENAI_API_KEY=your_openai_api_key

# Anthropic
ANTHROPIC_API_KEY=your_anthropic_api_key

# Google
GOOGLE_API_KEY=your_google_api_key
GOOGLE_APPLICATION_CREDENTIALS=path/to/service-account.json

# Azure
AZURE_OPENAI_API_KEY=your_azure_openai_key
AZURE_OPENAI_ENDPOINT=your_azure_endpoint
AZURE_SPEECH_KEY=your_azure_speech_key
AZURE_SPEECH_REGION=your_azure_region

# Default Providers
DEFAULT_STT_PROVIDER=openai
DEFAULT_LLM_PROVIDER=openai
DEFAULT_TTS_PROVIDER=openai
DEFAULT_AUDIO_PROCESSOR=whisper

# VertexAI
VERTEXAI_PROJECT_ID=your_project_id
VERTEXAI_SERVICE_ACCOUNT_JSON=path/to/service-account.json
VERTEXAI_MODEL=mistral-large-2411

# Ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:latest
```

## 🎯 Supported Models

### OpenAI Models
The system supports dynamic model selection for OpenAI. You can specify any OpenAI model in your requests:

**Popular Models:**
- `gpt-3.5-turbo` (default)
- `gpt-4` 
- `gpt-4-turbo`
- `gpt-4o`
- `gpt-4o-mini`
- `gpt-3.5-turbo-16k`
- `gpt-4-32k`

**Usage:**
```json
{
  "prompt": "Your question",
  "llm_provider": "openai",
  "model": "gpt-4o"
}
```

### Anthropic Models
**Supported Models:**
- `claude-3-haiku-20240307` (default)
- `claude-3-sonnet-20240229`
- `claude-3-opus-20240229`
- `claude-3-5-sonnet-20241022`
- `claude-2.1`
- `claude-2.0`

**Usage:**
```json
{
  "prompt": "Your question",
  "llm_provider": "anthropic",
  "model": "claude-3-opus-20240229"
}
```

### Google Gemini Models
**Supported Models:**
- `models/gemini-2.5-flash` (default)
- `models/gemini-2.5-pro`
- `models/gemini-1.5-pro`
- `models/gemini-1.5-flash`

**Usage:**
```json
{
  "prompt": "Your question",
  "llm_provider": "gemini",
  "model": "models/gemini-2.5-pro"
}
```

### VertexAI Models
**Supported Models:**
- `mistral-large-2411` (default)
- `claude-3-5-sonnet@20241022`
- `claude-3-opus@20240229`
- `claude-3-sonnet@20240229`
- `claude-3-haiku@20240307`
- `gemini-1.5-pro`
- `gemini-1.5-flash`

**Usage:**
```json
{
  "prompt": "Your question",
  "llm_provider": "vertexai",
  "model": "claude-3-5-sonnet@20241022"
}
```

### Ollama Models
**Popular Models:** (requires local Ollama installation)
- `llama3.1:latest` (default)
- `llama3.1:8b`
- `llama3.1:70b`
- `llama2:latest`
- `codellama:latest`
- `mistral:latest`
- `phi3:latest`
- `qwen2:latest`

**Usage:**
```json
{
  "prompt": "Your question",
  "llm_provider": "ollama",
  "model": "llama3.1:8b"
}
```

### Azure OpenAI Models
**Supported Models:**
- `gpt-35-turbo` (default)
- `gpt-4`
- `gpt-4-turbo`
- `gpt-4o`

**Usage:**
```json
{
  "prompt": "Your question",
  "llm_provider": "azure",
  "model": "gpt-4"
}
```

## 🔧 API Endpoints

### Health Check
```http
GET /health
```

### Provider Management
```http
GET /providers
POST /config/update
```

### Session Management
```http
GET /sessions
```

### LLM Chat
```http
POST /api/v1/llm/generate
Content-Type: application/json

{
  "prompt": "Hello, how are you?",
  "history": [{"role": "user", "content": "Previous message"}],
  "llm_provider": "openai",
  "model": "gpt-4o",
  "temperature": 0.7,
  "max_tokens": 1000
}
```

### Image Generation
```http
POST /image
Content-Type: application/json

{
  "prompt": "A beautiful sunset over mountains",
  "image_provider": "openai",
  "size": "1024x1024"
}
```

### WebSocket Voice Chat
```javascript
const ws = new WebSocket('ws://localhost:8000/voice');

// Send audio data
ws.send(JSON.stringify({
  "user_audio_chunk": "base64_encoded_audio_data"
}));

// Send text message
ws.send(JSON.stringify({
  "text": "Hello, how are you?"
}));
```

## 📁 Project Structure

```
ai-backend/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application entry point
│   ├── config/
│   │   ├── __init__.py
│   │   ├── manager.py          # Configuration management
│   │   └── models.py           # Configuration data models
│   ├── core/
│   │   ├── __init__.py
│   │   ├── session.py          # Voice session management
│   │   └── exceptions.py       # Custom exceptions
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── health.py       # Health check endpoints
│   │   │   ├── llm.py          # LLM endpoints
│   │   │   ├── image.py        # Image generation endpoints
│   │   │   ├── providers.py    # Provider management
│   │   │   └── websocket.py    # WebSocket voice endpoint
│   │   └── dependencies.py     # FastAPI dependencies
│   └── providers/
│       ├── __init__.py
│       ├── factory.py          # Provider factory
│       ├── base/
│       │   ├── __init__.py
│       │   ├── stt.py          # STT base interface
│       │   ├── llm.py          # LLM base interface
│       │   ├── tts.py          # TTS base interface
│       │   └── image.py        # Image generation base
│       ├── openai/
│       │   ├── __init__.py
│       │   ├── stt.py
│       │   ├── llm.py
│       │   ├── tts.py
│       │   └── image.py
│       └── azure/
│           ├── __init__.py
│           ├── stt.py
│           ├── llm.py
│           └── tts.py
├── tests/
│   ├── __init__.py
│   ├── test_config.py
│   ├── test_providers.py
│   └── test_api.py
├── docs/
│   ├── api.md
│   ├── configuration.md
│   └── deployment.md
├── scripts/
│   ├── setup.sh
│   └── test_endpoints.py
├── .env.example
├── .gitignore
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## 🎮 Usage Examples

### 1. Chat with Different LLM Providers

```python
import requests

# Chat with OpenAI GPT-4o
response = requests.post("http://localhost:8000/api/v1/llm/generate", json={
    "prompt": "Explain quantum computing",
    "llm_provider": "openai",
    "model": "gpt-4o"
})

# Chat with Anthropic Claude Opus
response = requests.post("http://localhost:8000/api/v1/llm/generate", json={
    "prompt": "Explain quantum computing",
    "llm_provider": "anthropic",
    "model": "claude-3-opus-20240229"
})

# Chat with Google Gemini Pro
response = requests.post("http://localhost:8000/api/v1/llm/generate", json={
    "prompt": "Explain quantum computing",
    "llm_provider": "gemini",
    "model": "models/gemini-2.5-pro"
})

# Chat with VertexAI Mistral
response = requests.post("http://localhost:8000/api/v1/llm/generate", json={
    "prompt": "Explain quantum computing",
    "llm_provider": "vertexai",
    "model": "mistral-large-2411"
})

# Chat with Ollama Llama
response = requests.post("http://localhost:8000/api/v1/llm/generate", json={
    "prompt": "Explain quantum computing",
    "llm_provider": "ollama",
    "model": "llama3.1:8b"
})
```

### 2. Generate Images

```python
import requests

response = requests.post("http://localhost:8000/image", json={
    "prompt": "A cyberpunk cityscape at night",
    "image_provider": "openai",
    "size": "1024x1024",
    "quality": "hd"
})
```

### 3. WebSocket Voice Conversation

```javascript
const ws = new WebSocket('ws://localhost:8000/voice');

ws.onopen = function() {
    console.log('Connected to voice assistant');
};

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    
    if (data.type === 'audio_response') {
        // Play the audio response
        const audio = new Audio('data:audio/mp3;base64,' + data.audio_base64);
        audio.play();
    } else if (data.type === 'text_response') {
        console.log('AI Response:', data.text);
    }
};

// Send audio data
function sendAudio(audioBlob) {
    const reader = new FileReader();
    reader.onload = function() {
        const base64Audio = btoa(reader.result);
        ws.send(JSON.stringify({
            user_audio_chunk: base64Audio
        }));
    };
    reader.readAsBinaryString(audioBlob);
}
```

### 4. Dynamic Provider Switching

```python
import requests

# Switch to different providers
config_update = {
    "llm_provider": "anthropic",
    "stt_provider": "azure",
    "tts_provider": "google"
}

response = requests.post("http://localhost:8000/config/update", json=config_update)
```

## 🐳 Docker Deployment

### Build and Run

```bash
# Build the image
docker build -t ai-gateway .

# Run with docker-compose
docker-compose up -d
```

### Environment Variables for Docker

```yaml
# docker-compose.yml
version: '3.8'
services:
  ai-gateway:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - GOOGLE_API_KEY=${GOOGLE_API_KEY}
    volumes:
      - ./logs:/app/logs
```

## 🧪 Testing

### Run Unit Tests
```bash
# Run all tests
python -m pytest tests/

# Run specific test
python -m pytest tests/test_providers.py

# Test endpoints manually
python scripts/test_endpoints.py
```

### Test Different Models
```bash
# Test OpenAI GPT-4o
curl -X POST "http://localhost:8000/api/v1/llm/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is AI?",
    "llm_provider": "openai",
    "model": "gpt-4o"
  }'

# Test Anthropic Claude Opus
curl -X POST "http://localhost:8000/api/v1/llm/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is AI?",
    "llm_provider": "anthropic", 
    "model": "claude-3-opus-20240229"
  }'

# Test Google Gemini Pro
curl -X POST "http://localhost:8000/api/v1/llm/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is AI?",
    "llm_provider": "gemini",
    "model": "models/gemini-2.5-pro"
  }'

# Test VertexAI
curl -X POST "http://localhost:8000/api/v1/llm/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is AI?",
    "llm_provider": "vertexai",
    "model": "mistral-large-2411"
  }'

# Test Ollama (requires local Ollama installation)
curl -X POST "http://localhost:8000/api/v1/llm/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is AI?",
    "llm_provider": "ollama",
    "model": "llama3.1:8b"
  }'
```

## 📊 Monitoring and Logging

The application provides comprehensive logging and session tracking:

- **Health monitoring** at `/health`
- **Active sessions** at `/sessions`
- **Provider status** at `/providers`
- **Structured logging** with timestamps and session IDs

## 🔒 Security Considerations

- API keys are managed through environment variables
- Input validation on all endpoints
- Session isolation and cleanup
- Rate limiting (configurable)
- CORS support for web clients

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- 📧 Email: support@ai-gateway.com
- 💬 Discord: [AI Gateway Community](https://discord.gg/ai-gateway)
- 📖 Documentation: [docs.ai-gateway.com](https://docs.ai-gateway.com)
- 🐛 Issues: [GitHub Issues](https://github.com/your-username/ai-backend/issues)

---

Made with ❤️ by the AI Gateway team