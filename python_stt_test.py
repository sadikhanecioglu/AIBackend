# python_stt_test.py - Python'dan AI Gateway STT API Test Script
import asyncio
import httpx
import base64
import os
import json
from pathlib import Path

# API Base URL
API_BASE_URL = "http://localhost:8000/api/stt"

async def test_file_upload():
    """File upload ile STT test"""
    print("🎤 Testing File Upload STT...")
    
    audio_file = "test_audio.wav"
    if not os.path.exists(audio_file):
        print(f"❌ Audio file not found: {audio_file}")
        return None
    
    try:
        async with httpx.AsyncClient() as client:
            with open(audio_file, "rb") as f:
                files = {"file": f}
                data = {
                    "language": "tr",
                    "timestamps": "true"
                }
                
                response = await client.post(
                    f"{API_BASE_URL}/transcribe/openai",
                    files=files,
                    data=data,
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    result = response.json()
                    print("✅ File Upload Success:")
                    print(f"   Text: {result['text']}")
                    print(f"   Provider: {result['provider']}")
                    print(f"   Processing Time: {result['processing_time']:.2f}s")
                    
                    if result.get('words'):
                        print("   Word Timestamps:")
                        for word in result['words']:
                            print(f"     {word['word']}: {word['start']:.2f}s-{word['end']:.2f}s")
                    
                    return result
                else:
                    print(f"❌ File Upload Error: {response.status_code} - {response.text}")
                    return None
                    
    except Exception as e:
        print(f"❌ File Upload Exception: {e}")
        return None

async def test_base64_upload():
    """Base64 ile STT test"""
    print("\\n🔐 Testing Base64 STT...")
    
    audio_file = "test_audio.wav"
    if not os.path.exists(audio_file):
        print(f"❌ Audio file not found: {audio_file}")
        return None
    
    try:
        # Ses dosyasını base64'e çevir
        with open(audio_file, "rb") as f:
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
                f"{API_BASE_URL}/transcribe-base64/openai",
                json=request_data,
                timeout=30.0
            )
            
            if response.status_code == 200:
                result = response.json()
                print("✅ Base64 Upload Success:")
                print(f"   Text: {result['text']}")
                print(f"   Language: {result['language']}")
                print(f"   Duration: {result['duration']:.2f}s" if result['duration'] else "   Duration: N/A")
                print(f"   Processing Time: {result['processing_time']:.2f}s")
                
                if result.get('words'):
                    print("   Word Timestamps:")
                    for word in result['words']:
                        print(f"     {word['word']}: {word['start']:.2f}s-{word['end']:.2f}s")
                
                return result
            else:
                print(f"❌ Base64 Upload Error: {response.status_code} - {response.text}")
                return None
                
    except Exception as e:
        print(f"❌ Base64 Upload Exception: {e}")
        return None

async def test_url_upload():
    """URL ile STT test"""
    print("\\n🌐 Testing URL STT...")
    
    try:
        # Test için bir örnek ses dosyası URL'i
        test_audio_url = "https://www.soundjay.com/misc/sounds/bell-ringing-05.wav"
        
        request_data = {
            "audio_url": test_audio_url,
            "language": "en",  # English for this test
            "timestamps": True,
            "word_timestamps": True
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{API_BASE_URL}/transcribe-url/openai",
                json=request_data,
                timeout=45.0  # URL download için uzun timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                print("✅ URL Upload Success:")
                print(f"   Text: {result['text']}")
                print(f"   Language: {result['language']}")
                print(f"   Duration: {result['duration']:.2f}s" if result['duration'] else "   Duration: N/A")
                print(f"   Processing Time: {result['processing_time']:.2f}s")
                
                return result
            else:
                print(f"❌ URL Upload Error: {response.status_code} - {response.text}")
                return None
                
    except Exception as e:
        print(f"❌ URL Upload Exception: {e}")
        # URL test başarısız olursa devam et
        return None

async def test_providers():
    """Provider'ları ve API bilgilerini test et"""
    print("\\n🔧 Testing Providers...")
    
    try:
        async with httpx.AsyncClient() as client:
            # API bilgisi
            api_info = await client.get(f"{API_BASE_URL}/")
            if api_info.status_code == 200:
                info = api_info.json()
                print("✅ API Info:")
                print(f"   Version: {info['version']}")
                print(f"   Supported Providers: {', '.join(info['supported_providers'])}")
                print(f"   Supported Formats: {', '.join(info['supported_formats'])}")
                print(f"   Input Methods: {', '.join(info['input_methods'])}")
            
            # Mevcut provider'lar
            providers = await client.get(f"{API_BASE_URL}/providers")
            if providers.status_code == 200:
                prov = providers.json()
                print("\\n✅ Available Providers:")
                print(f"   Active Providers: {', '.join(prov['providers'])}")
                print(f"   Default Provider: {prov['default']}")
            
            # Provider test
            if 'openai' in prov.get('providers', []):
                test = await client.get(f"{API_BASE_URL}/test/openai")
                if test.status_code == 200:
                    test_result = test.json()
                    print("\\n✅ OpenAI Provider Test:")
                    print(f"   Available: {test_result['available']}")
                    print(f"   Supported Formats: {', '.join(test_result['supported_formats'])}")
    
    except Exception as e:
        print(f"❌ Provider Test Exception: {e}")

async def main():
    """Ana test fonksiyonu"""
    print("🚀 AI Gateway STT API Test Suite (Python)")
    print("==========================================\\n")
    
    try:
        # Ses dosyasının varlığını kontrol et
        if not os.path.exists("test_audio.wav"):
            print("❌ Test audio file not found: ./test_audio.wav")
            print("Please create a test audio file first.")
            return
        
        # 1. Provider testleri
        await test_providers()
        
        # 2. File upload test
        await test_file_upload()
        
        # 3. Base64 upload test
        await test_base64_upload()
        
        # 4. URL upload test (optional)
        await test_url_upload()
        
        print("\\n✅ All Python tests completed successfully!")
        
    except Exception as error:
        print(f"\\n❌ Python test suite failed: {error}")

def create_flask_example():
    """Flask example endpoint"""
    return '''
# flask_stt_example.py
from flask import Flask, request, jsonify
import httpx
import base64
import asyncio

app = Flask(__name__)

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    """Flask endpoint for audio transcription"""
    
    try:
        # File upload'dan ses dosyasını al
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({'error': 'No audio file selected'}), 400
        
        # Parametreleri al
        language = request.form.get('language', 'tr')
        provider = request.form.get('provider', 'openai')
        timestamps = request.form.get('timestamps', 'false').lower() == 'true'
        
        # Sync function içinde async call yapma
        def sync_transcribe():
            return asyncio.run(async_transcribe())
        
        async def async_transcribe():
            # AI Gateway'e gönder
            async with httpx.AsyncClient() as client:
                files = {'file': (audio_file.filename, audio_file.read(), audio_file.content_type)}
                data = {
                    'language': language,
                    'timestamps': str(timestamps).lower()
                }
                
                response = await client.post(
                    f'http://localhost:8000/api/stt/transcribe/{provider}',
                    files=files,
                    data=data,
                    timeout=30.0
                )
                
                return response.json() if response.status_code == 200 else None
        
        result = sync_transcribe()
        
        if result:
            return jsonify({
                'success': True,
                'transcription': result['text'],
                'language': result.get('language'),
                'provider': result['provider'],
                'processing_time': result['processing_time'],
                'words': result.get('words', [])
            })
        else:
            return jsonify({'error': 'Transcription failed'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/transcribe-base64', methods=['POST'])
def transcribe_base64():
    """Flask endpoint for base64 audio transcription"""
    
    try:
        data = request.get_json()
        
        if not data or 'audio_base64' not in data:
            return jsonify({'error': 'No base64 audio data provided'}), 400
        
        # Sync function içinde async call yapma
        def sync_transcribe():
            return asyncio.run(async_transcribe())
        
        async def async_transcribe():
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    'http://localhost:8000/api/stt/transcribe-base64/openai',
                    json=data,
                    timeout=30.0
                )
                
                return response.json() if response.status_code == 200 else None
        
        result = sync_transcribe()
        
        if result:
            return jsonify({
                'success': True,
                'transcription': result['text'],
                'language': result.get('language'),
                'duration': result.get('duration'),
                'words': result.get('words', [])
            })
        else:
            return jsonify({'error': 'Base64 transcription failed'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
'''

# Script çalıştır
if __name__ == "__main__":
    # Flask örneğini göster
    print("\\n📋 Flask Example:")
    print(create_flask_example())
    print("\\n" + "="*50)
    
    # Testleri çalıştır
    asyncio.run(main())