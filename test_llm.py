#!/usr/bin/env python3
"""
LLM API Test Script
"""

import requests
import json
import time


def test_llm_endpoint():
    """Test the LLM endpoint with OpenAI"""

    url = "http://localhost:8000/llm"

    # Test data
    test_requests = [
        {"prompt": "Merhaba! Nasılsın?", "llm_provider": "openai"},
        {
            "prompt": "Python hakkında kısa bir bilgi ver",
            "llm_provider": "openai",
            "temperature": 0.5,
            "max_tokens": 150,
        },
        {
            "prompt": "İstanbul'un tarihi hakkında 2 cümle yaz",
            "llm_provider": "openai",
            "temperature": 0.7,
        },
    ]

    print("🚀 LLM API Test Başlıyor...")
    print("=" * 50)

    for i, test_data in enumerate(test_requests, 1):
        print(f"\n📤 Test {i}: {test_data['prompt'][:30]}...")
        print(f"   Provider: {test_data['llm_provider']}")

        try:
            # Send request
            response = requests.post(
                url,
                json=test_data,
                headers={"Content-Type": "application/json"},
                timeout=30,
            )

            print(f"   Status: {response.status_code}")

            if response.status_code == 200:
                data = response.json()
                print(f"✅ Başarılı!")
                print(f"   📨 Cevap: {data['response'][:100]}...")
                print(f"   🔧 Provider: {data['provider']}")
                if "model" in data and data["model"]:
                    print(f"   🤖 Model: {data['model']}")
                if "usage" in data and data["usage"]:
                    print(f"   📊 Usage: {data['usage']}")
            else:
                print(f"❌ Hata: {response.status_code}")
                try:
                    error_data = response.json()
                    print(f"   Detay: {error_data.get('detail', 'Bilinmeyen hata')}")
                except:
                    print(f"   Raw response: {response.text}")

        except requests.exceptions.ConnectionError:
            print("❌ Bağlantı hatası - Server çalışıyor mu?")
            break
        except requests.exceptions.Timeout:
            print("❌ Timeout hatası - Request çok uzun sürdü")
        except Exception as e:
            print(f"❌ Beklenmeyen hata: {e}")

        # Wait between requests
        if i < len(test_requests):
            print("   ⏳ 2 saniye bekleniyor...")
            time.sleep(2)

    print("\n" + "=" * 50)
    print("🏁 Test tamamlandı!")


def test_health():
    """Test health endpoint first"""
    print("🔍 Health check yapılıyor...")
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Server sağlıklı - Status: {data['status']}")
            print(f"   Aktif sessionlar: {data['active_sessions']}")
            return True
        else:
            print(f"❌ Health check başarısız: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check hatası: {e}")
        return False


def main():
    print("🎯 AI Gateway LLM Test Suite")
    print("=" * 50)

    # Wait for server to be ready
    print("⏳ Server'ın hazır olması bekleniyor...")
    time.sleep(3)

    # Health check first
    if not test_health():
        print("❌ Server'a bağlanılamıyor. main.py çalışıyor mu?")
        return

    print()
    # Run LLM tests
    test_llm_endpoint()


if __name__ == "__main__":
    main()
