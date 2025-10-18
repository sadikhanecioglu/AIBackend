# 🎤 AI Gateway STT API - Kullanım Kılavuzu

## 📋 Genel Bakış

AI Gateway STT (Speech-to-Text) API'si, ses dosyalarını metne dönüştürmek için çoklu provider desteği sunar. OpenAI Whisper, AssemblyAI, ve Deepgram gibi popüler STT servislerini tek bir API üzerinden kullanabilirsiniz.

## 🌟 Özellikler

- **🔀 Multi-Provider Support**: OpenAI, AssemblyAI, Deepgram
- **📁 Çoklu Input Method**: File Upload, Base64, URL
- **⏱️ Word-level Timestamps**: Kelime bazında zaman damgaları
- **🌍 Multi-Language**: Türkçe, İngilizce, ve 50+ dil desteği
- **🚀 Fast Processing**: Optimize edilmiş işlem süreleri
- **📊 Detailed Response**: Confidence score, duration, processing time

## 🛠️ API Endpoints

```
BASE_URL: http://localhost:8000/api/stt

🔗 Endpoints:
├── GET  /                              # API bilgisi
├── GET  /providers                     # Mevcut provider'lar
├── GET  /test/{provider}              # Provider testi
├── POST /transcribe/{provider}        # File upload transcription
├── POST /transcribe                   # Auto-provider transcription
├── POST /transcribe-base64/{provider} # Base64 transcription
└── POST /transcribe-url/{provider}    # URL transcription
```

## 📝 Kullanım Örnekleri

### 1. **File Upload (Multipart/Form-Data)**

#### **cURL:**
```bash
curl -X POST "http://localhost:8000/api/stt/transcribe/openai" \\
  -H "accept: application/json" \\
  -F "file=@audio.wav" \\
  -F "language=tr" \\
  -F "timestamps=true"
```

#### **Node.js:**
```javascript
const FormData = require('form-data');
const fs = require('fs');
const axios = require('axios');

async function transcribeFile() {
    const form = new FormData();
    form.append('file', fs.createReadStream('./audio.wav'));
    form.append('language', 'tr');
    form.append('timestamps', 'true');
    
    const response = await axios.post(
        'http://localhost:8000/api/stt/transcribe/openai',
        form,
        {
            headers: form.getHeaders(),
            timeout: 30000
        }
    );
    
    console.log('Transcription:', response.data.text);
    return response.data;
}
```

#### **Python:**
```python
import httpx

async def transcribe_file():
    with open("audio.wav", "rb") as f:
        files = {"file": f}
        data = {"language": "tr", "timestamps": "true"}
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://localhost:8000/api/stt/transcribe/openai",
                files=files,
                data=data
            )
            
            result = response.json()
            print("Transcription:", result["text"])
            return result
```

### 2. **Base64 Encoded Audio**

#### **Node.js:**
```javascript
const fs = require('fs');
const axios = require('axios');

async function transcribeBase64() {
    // Ses dosyasını base64'e çevir
    const audioBuffer = fs.readFileSync('./audio.wav');
    const audioBase64 = audioBuffer.toString('base64');
    
    const response = await axios.post(
        'http://localhost:8000/api/stt/transcribe-base64/openai',
        {
            audio_base64: audioBase64,
            format: 'wav',
            language: 'tr',
            timestamps: true,
            word_timestamps: true
        },
        {
            headers: { 'Content-Type': 'application/json' }
        }
    );
    
    return response.data;
}
```

#### **Python:**
```python
import base64
import httpx

async def transcribe_base64():
    with open("audio.wav", "rb") as f:
        audio_data = f.read()
        audio_base64 = base64.b64encode(audio_data).decode('utf-8')
    
    request_data = {
        "audio_base64": audio_base64,
        "format": "wav",
        "language": "tr",
        "timestamps": True,
        "word_timestamps": True
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/api/stt/transcribe-base64/openai",
            json=request_data
        )
        
        return response.json()
```

### 3. **URL'den Audio İndirme**

#### **cURL:**
```bash
curl -X POST "http://localhost:8000/api/stt/transcribe-url/openai" \\
  -H "Content-Type: application/json" \\
  -d '{
    "audio_url": "https://example.com/audio.wav",
    "language": "tr",
    "timestamps": true
  }'
```

#### **JavaScript/Node.js:**
```javascript
async function transcribeFromUrl() {
    const response = await axios.post(
        'http://localhost:8000/api/stt/transcribe-url/openai',
        {
            audio_url: 'https://example.com/audio.wav',
            language: 'tr',
            timestamps: true,
            word_timestamps: true
        }
    );
    
    return response.data;
}
```

## 📊 Response Format

```json
{
    "text": "Merhaba, bu bir test mesajıdır.",
    "language": "turkish",
    "confidence": 0.95,
    "duration": 5.32,
    "words": [
        {
            "word": "Merhaba",
            "start": 0.0,
            "end": 0.54,
            "confidence": 0.98
        },
        {
            "word": "bu",
            "start": 0.92,
            "end": 1.0,
            "confidence": 0.95
        }
    ],
    "provider": "openai",
    "processing_time": 2.45
}
```

## 🔧 Provider Konfigürasyonu

### **Environment Variables (.env):**
```bash
# OpenAI Whisper
OPENAI_API_KEY=sk-proj-xxx

# AssemblyAI
ASSEMBLYAI_API_KEY=xxx

# Deepgram
DEEPGRAM_API_KEY=xxx
```

### **Mevcut Provider'ları Kontrol:**
```bash
curl http://localhost:8000/api/stt/providers
```

### **Provider Test:**
```bash
curl http://localhost:8000/api/stt/test/openai
```

## 🎯 Desteklenen Formatlar

- **Audio Formats**: MP3, WAV, M4A, OGG, FLAC, WebM
- **Max File Size**: 25MB
- **Languages**: 50+ dil (tr, en, es, fr, de, it, pt, ru, ja, ko, zh, ar...)

## 💡 Express.js Middleware Örneği

```javascript
const multer = require('multer');
const axios = require('axios');

// Multer config
const upload = multer({ 
    storage: multer.memoryStorage(),
    limits: { fileSize: 25 * 1024 * 1024 }
});

// STT Middleware
const sttMiddleware = (provider = 'openai') => {
    return async (req, res, next) => {
        try {
            const FormData = require('form-data');
            const form = new FormData();
            
            form.append('file', req.file.buffer, {
                filename: req.file.originalname,
                contentType: req.file.mimetype
            });
            form.append('language', req.body.language || 'tr');
            
            const response = await axios.post(
                \`http://localhost:8000/api/stt/transcribe/\${provider}\`,
                form,
                { headers: form.getHeaders() }
            );
            
            req.transcription = response.data;
            next();
        } catch (error) {
            res.status(500).json({ error: 'Transcription failed' });
        }
    };
};

// Route kullanımı
app.post('/upload-audio', 
    upload.single('audio'), 
    sttMiddleware('openai'), 
    (req, res) => {
        res.json({
            success: true,
            text: req.transcription.text,
            words: req.transcription.words
        });
    }
);
```

## 🚀 Flask/FastAPI Integration

### **Flask Örneği:**
```python
from flask import Flask, request, jsonify
import httpx
import asyncio

app = Flask(__name__)

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    audio_file = request.files['audio']
    
    async def async_transcribe():
        async with httpx.AsyncClient() as client:
            files = {'file': audio_file.read()}
            data = {'language': 'tr'}
            
            response = await client.post(
                'http://localhost:8000/api/stt/transcribe/openai',
                files=files,
                data=data
            )
            return response.json()
    
    result = asyncio.run(async_transcribe())
    return jsonify(result)
```

## 🛡️ Error Handling

```json
// Hata Response Formatı
{
    "error": "Transcription failed: Invalid audio format",
    "status_code": 400,
    "path": "http://localhost:8000/api/stt/transcribe/openai"
}
```

## 📈 Performance Tips

1. **Format Seçimi**: WAV ve FLAC en iyi kalite, MP3 küçük boyut
2. **File Size**: Büyük dosyalar için URL method kullanın
3. **Language**: Doğru dil kodu belirtmek accuracy'yi artırır
4. **Timestamps**: Sadece gerektiğinde kullanın (işlem süresini artırır)

## 🧪 Test Scripts

### **Node.js Test:**
```bash
npm install
node nodejs_stt_test.js
```

### **Python Test:**
```bash
pip install httpx
python python_stt_test.py
```

## 📚 API Documentation

Detaylı API dokümantasyonu için:
```
http://localhost:8000/docs
```

## ❓ Troubleshooting

### **Yaygın Hatalar:**

1. **"OpenAI API key not configured"**
   - `.env` dosyasında `OPENAI_API_KEY` ayarlayın

2. **"Unsupported file format"**
   - Desteklenen formatları kontrol edin: MP3, WAV, M4A, OGG, FLAC, WebM

3. **"File too large"**
   - Max 25MB limit. Büyük dosyalar için URL method kullanın

4. **"Provider not available"**
   - Provider API key'lerini kontrol edin
   - `/api/stt/providers` endpoint'ini kullanın

### **Debug:**
```bash
# Provider durumu
curl http://localhost:8000/api/stt/test/openai

# API bilgisi
curl http://localhost:8000/api/stt/

# Health check
curl http://localhost:8000/health
```

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 License

MIT License - see LICENSE file for details.