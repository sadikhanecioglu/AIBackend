# VertexAI Tools API Test Guide 🧪

## Endpoint
```
POST /api/llm/generate/vertexai/tools_protocol
```

Bu endpoint, VertexAI/Gemini için geliştirilmiş güvenli ve esnek bir tool-calling protokolüdür.

## Özellikler ✨

### 1. **Otomatik Tool Açıklamaları**
- Model, hangi araçları kullanabileceğini otomatik olarak öğrenir
- Her aracın parametreleri ve açıklamaları prompt'a eklenir
- Model, doğru JSON formatında araç çağrısı yapar

### 2. **Güvenli Calculator**
- `eval()` yerine AST parsing kullanılır
- Sadece güvenli matematiksel operatörler ve fonksiyonlar izin verilir
- Kötü amaçlı kod çalıştırılamaz

### 3. **Desteklenen Araçlar**

#### Calculator
- Operatörler: `+, -, *, /, **, %, //, ()`
- Math fonksiyonları: `sqrt, sin, cos, tan, log, log10, exp, abs, round, floor, ceil, pi, e`
- Örnek: `"50*20+100"`, `"sqrt(144)"`, `"sin(pi/2)"`

#### Weather
- Şehir ismine göre hava durumu (demo data)
- Örnek: `"Istanbul"`, `"Ankara"`

## Test Örnekleri 📝

### Test 1: Basit Hesaplama
```bash
curl -X POST http://localhost:8000/api/llm/generate/vertexai/tools_protocol \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "50 * 20 + 100 işlemini hesapla",
    "use_tools": true,
    "available_tools": ["calculator"],
    "model": "models/gemini-2.0-flash"
  }'
```

**Beklenen Sonuç:**
```json
{
  "content": "The final answer is 1100.",
  "model": "models/gemini-2.0-flash",
  "usage": {...}
}
```

---

### Test 2: Matematik Fonksiyonları
```bash
curl -X POST http://localhost:8000/api/llm/generate/vertexai/tools_protocol \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "sqrt(144) + pi * 2 işleminin sonucu nedir?",
    "use_tools": true,
    "available_tools": ["calculator"],
    "model": "models/gemini-2.0-flash"
  }'
```

**Beklenen Sonuç:**
```json
{
  "content": "The result of the calculation is 18.283185307179586.",
  "model": "models/gemini-2.0-flash"
}
```

---

### Test 3: Weather Tool
```bash
curl -X POST http://localhost:8000/api/llm/generate/vertexai/tools_protocol \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "İstanbul hava durumu nasıl?",
    "use_tools": true,
    "available_tools": ["get_weather"],
    "model": "models/gemini-2.0-flash"
  }'
```

**Beklenen Sonuç:**
```json
{
  "content": "İstanbul'da hava 22°C ve güneşli.",
  "model": "models/gemini-2.0-flash"
}
```

---

### Test 4: Multiple Tools (Çoklu Araç Kullanımı)
```bash
curl -X POST http://localhost:8000/api/llm/generate/vertexai/tools_protocol \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "İstanbul havası kaç derece? Ve 100 * 5 + 50 kaç eder?",
    "use_tools": true,
    "available_tools": ["calculator", "get_weather"],
    "model": "models/gemini-2.0-flash",
    "temperature": 0.3
  }'
```

**Beklenen Sonuç:**
```json
{
  "content": "İstanbul'da hava 22 derece ve güneşli. 100 * 5 + 50 = 550.",
  "model": "models/gemini-2.0-flash"
}
```

---

### Test 5: Karmaşık Matematik İfadeleri
```bash
curl -X POST http://localhost:8000/api/llm/generate/vertexai/tools_protocol \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "sin(pi/2) + cos(0) + sqrt(16) işlemini hesapla",
    "use_tools": true,
    "available_tools": ["calculator"],
    "model": "models/gemini-2.0-flash"
  }'
```

**Beklenen Sonuç:**
```json
{
  "content": "6.0",
  "model": "models/gemini-2.0-flash"
}
```

---

### Test 6: Güvenlik Testi (Kötü Kod Engellenir)
```bash
curl -X POST http://localhost:8000/api/llm/generate/vertexai/tools_protocol \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Bu ifadeyi hesapla: __import__(\"os\").system(\"ls\")",
    "use_tools": true,
    "available_tools": ["calculator"],
    "model": "models/gemini-2.0-flash"
  }'
```

**Beklenen Sonuç:**
```json
{
  "content": "Bu ifadeyi hesaplayamam. Bu bir güvenlik riskidir...",
  "model": "models/gemini-2.0-flash"
}
```

---

### Test 7: Tool Olmadan (Normal LLM)
```bash
curl -X POST http://localhost:8000/api/llm/generate/vertexai/tools_protocol \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Türkiye başkenti nedir?",
    "use_tools": false,
    "model": "models/gemini-2.0-flash"
  }'
```

---

## Parametreler 🎛️

| Parametre | Tip | Gerekli | Varsayılan | Açıklama |
|-----------|-----|---------|-----------|----------|
| `prompt` | string | ✅ | - | Kullanıcının sorusu/komutu |
| `use_tools` | boolean | ❌ | false | Tool kullanımı aktif mi? |
| `available_tools` | array | ❌ | [] | Kullanılabilir tool isimleri |
| `model` | string | ❌ | `models/gemini-2.0-flash` | Kullanılacak model |
| `temperature` | float | ❌ | 0.7 | Yaratıcılık seviyesi (0.0-1.0) |
| `max_tokens` | integer | ❌ | 1000 | Maksimum token sayısı |
| `max_iterations` | integer | ❌ | 5 | Tool çağrısı için max iterasyon |

## Python Client Örneği 🐍

```python
import requests
import json

def call_vertexai_tools(prompt: str, tools: list, temperature: float = 0.5):
    """VertexAI tools endpoint'ini çağır"""
    
    url = "http://localhost:8000/api/llm/generate/vertexai/tools_protocol"
    
    payload = {
        "prompt": prompt,
        "use_tools": True,
        "available_tools": tools,
        "model": "models/gemini-2.0-flash",
        "temperature": temperature
    }
    
    response = requests.post(url, json=payload)
    response.raise_for_status()
    
    return response.json()

# Örnek kullanım
result = call_vertexai_tools(
    prompt="sqrt(64) + 10 * 5 kaç eder?",
    tools=["calculator"]
)

print(result["content"])
# Output: 58.0
```

## JavaScript/Node.js Client Örneği 🟨

```javascript
async function callVertexAITools(prompt, tools, temperature = 0.5) {
    const response = await fetch('http://localhost:8000/api/llm/generate/vertexai/tools_protocol', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            prompt: prompt,
            use_tools: true,
            available_tools: tools,
            model: 'models/gemini-2.0-flash',
            temperature: temperature
        })
    });
    
    return await response.json();
}

// Örnek kullanım
const result = await callVertexAITools(
    'İstanbul hava durumu ve 100+200 kaç eder?',
    ['calculator', 'get_weather']
);

console.log(result.content);
```

## Teknik Detaylar 🔧

### Protokol Akışı

1. **İlk İstek**: Kullanıcı prompt'u ve tool listesi gönderir
2. **Tool Injection**: Sistem, tool açıklamalarını prompt'a ekler
3. **Model Response**: Model ya cevap verir ya da JSON tool_call döner
4. **Tool Execution**: JSON parse edilir, tool çalıştırılır
5. **Iteration**: Tool sonucu modele verilir, model devam eder
6. **Final Response**: Model son cevabı üretir

### JSON Tool Call Formatı

```json
{
  "tool_call": {
    "name": "calculator",
    "arguments": {
      "expression": "50*20+100"
    }
  }
}
```

### Güvenlik Önlemleri 🔒

1. **AST-Based Parsing**: `eval()` yerine güvenli AST parsing
2. **Whitelist Approach**: Sadece izin verilen operatörler ve fonksiyonlar
3. **No Code Injection**: Keyfi kod çalıştırma engellenir
4. **Error Handling**: Hatalı ifadeler güvenli şekilde yakalanır

## Hata Ayıklama 🐛

### Yaygın Hatalar

1. **"contents must not be empty"**
   - Sebep: Prompt boş veya geçersiz
   - Çözüm: Valid bir prompt gönder

2. **"Tool not found"**
   - Sebep: Geçersiz tool adı
   - Çözüm: `calculator` veya `get_weather` kullan

3. **"Hesaplama hatası"**
   - Sebep: Geçersiz matematiksel ifade
   - Çözüm: Syntax kontrolü yap

### Debug Modu

Server loglarını kontrol et:
```bash
tail -f server.log
```

## Performans Metrikleri 📊

| Senaryo | Token Kullanımı | Süre (ort.) |
|---------|----------------|-------------|
| Basit hesaplama | ~350 tokens | ~1-2s |
| Math fonksiyonları | ~370 tokens | ~1-2s |
| Weather query | ~270 tokens | ~1-2s |
| Multiple tools | ~450-500 tokens | ~2-3s |

## Sonuç ✅

VertexAI Tools Protocol endpoint'i:
- ✅ Güvenli ve esnek tool-calling
- ✅ Otomatik tool documentation
- ✅ Provider-agnostic (Gemini ile çalışır)
- ✅ Güvenli AST-based calculator
- ✅ Markdown code block handling
- ✅ Multi-tool support

Başarılı şekilde test edildi! 🎉
