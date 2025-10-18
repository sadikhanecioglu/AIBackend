# speech_test.py - STT API Test Script
import asyncio
import os
from llm_provider import SpeechFactory, SpeechRequest

async def direct_speech_example():
    """Doğrudan llm-provider kullanarak STT test"""
    print("=== Doğrudan LLM Provider STT Test ===")
    
    # Speech factory oluştur
    factory = SpeechFactory()
    
    # OpenAI Whisper provider (API key'inizi .env'den alır)
    provider = factory.create_openai_speech(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Ses dosyasını transcribe et
    audio_file = "test_audio.mp3"  # Ses dosyanızın yolu
    if not os.path.exists(audio_file):
        print(f"❌ Ses dosyası bulunamadı: {audio_file}")
        return
    
    request = SpeechRequest(
        audio_data=audio_file,
        language="tr",  # Türkçe
        timestamps=True
    )
    
    try:
        response = await provider.transcribe(request)
        print(f"✅ Transcription: {response.text}")
        
        # Word-level timestamps
        if hasattr(response, 'words') and response.words:
            print("\n📝 Word-level timestamps:")
            for word in response.words:
                print(f"  {word.word}: {word.start}s-{word.end}s")
    except Exception as e:
        print(f"❌ Hata: {e}")

async def api_speech_example():
    """API üzerinden STT test (httpx kullanarak)"""
    print("\n=== AI Gateway STT API Test ===")
    import httpx
    
    audio_file = "test_audio.mp3"
    if not os.path.exists(audio_file):
        print(f"❌ Ses dosyası bulunamadı: {audio_file}")
        return
    
    # API endpoint'ine ses dosyası gönder
    api_url = "http://localhost:8000/api/stt/transcribe/openai"
    
    try:
        async with httpx.AsyncClient() as client:
            with open(audio_file, "rb") as f:
                files = {"file": f}
                data = {
                    "language": "tr",
                    "timestamps": True
                }
                
                response = await client.post(api_url, files=files, data=data)
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"✅ API Transcription: {result['text']}")
                    print(f"📊 Provider: {result['provider']}")
                    print(f"⏱️ Processing time: {result['processing_time']:.2f}s")
                    
                    if result.get('words'):
                        print("\n📝 Word-level timestamps:")
                        for word in result['words']:
                            print(f"  {word['word']}: {word['start']:.2f}s-{word['end']:.2f}s")
                else:
                    print(f"❌ API Hatası: {response.status_code} - {response.text}")
                    
    except Exception as e:
        print(f"❌ API connection hatası: {e}")

def create_test_audio():
    """Test için basit ses dosyası oluştur"""
    print("=== Test Ses Dosyası Oluşturma ===")
    
    try:
        # macOS'ta say komutu ile ses dosyası oluştur
        os.system('say "Merhaba, bu bir test mesajıdır. AI Gateway STT özelliğini test ediyoruz." --output-file=test_audio.aiff')
        
        # AIFF'i MP3'e çevir (ffmpeg gerekli)
        if os.path.exists("test_audio.aiff"):
            print("✅ Test ses dosyası oluşturuldu: test_audio.aiff")
            # İsterseniz ffmpeg ile MP3'e çevirebilirsiniz
            # os.system('ffmpeg -i test_audio.aiff test_audio.mp3')
            return "test_audio.aiff"
        
    except Exception as e:
        print(f"❌ Test ses dosyası oluşturulamadı: {e}")
    
    return None

async def main():
    """Ana test fonksiyonu"""
    print("🎤 STT API Test Başlıyor...")
    
    # Test ses dosyası kontrol et veya oluştur
    audio_file = "test_audio.aiff"
    if not os.path.exists(audio_file):
        print("📁 Test ses dosyası bulunamadı, oluşturuluyor...")
        audio_file = create_test_audio()
    
    if audio_file and os.path.exists(audio_file):
        # Doğrudan provider test
        await direct_speech_example()
        
        # API üzerinden test
        await api_speech_example()
    else:
        print("❌ Test için ses dosyası gerekli. Lütfen 'test_audio.mp3' veya 'test_audio.wav' dosyası ekleyin.")

if __name__ == "__main__":
    asyncio.run(main())