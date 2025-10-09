#!/usr/bin/env python3
"""
Test AI Gateway Image API
"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"


def test_health():
    """Test health endpoint"""
    print("🏥 Testing Health Endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False


def test_image_providers():
    """Test image providers endpoint"""
    print("\n🖼️ Testing Image Providers Endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/api/image/providers", timeout=5)
        print(f"Status: {response.status_code}")
        data = response.json()
        print(f"Response: {json.dumps(data, indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Image providers test failed: {e}")
        return False


def test_openai_image_generation():
    """Test OpenAI image generation"""
    print("\n🎨 Testing OpenAI Image Generation...")
    try:
        payload = {
            "prompt": "A cute cat sitting on a table",
            "provider": "openai",
            "model": "dall-e-2",
            "size": "512x512",
            "n": 1,
            # Note: quality and style not supported by DALL-E 2
        }

        response = requests.post(
            f"{BASE_URL}/api/image/generate", json=payload, timeout=30
        )

        print(f"Status: {response.status_code}")
        data = response.json()
        print(f"Response: {json.dumps(data, indent=2)}")

        if response.status_code == 200:
            print(f"✅ Generated {len(data.get('urls', []))} image(s)")
            for i, url in enumerate(data.get("urls", []), 1):
                print(f"   {i}. {url}")

        return response.status_code == 200
    except Exception as e:
        print(f"❌ OpenAI image generation failed: {e}")
        return False


def test_image_test_endpoint():
    """Test image provider test endpoint"""
    print("\n🧪 Testing Image Provider Test Endpoint...")
    try:
        # Get API key from environment or prompt
        import os

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("⚠️ No OPENAI_API_KEY found in environment")
            return False

        payload = {
            "provider_name": "openai",
            "api_key": api_key,
            "test_prompt": "A simple test image",
        }

        response = requests.post(f"{BASE_URL}/api/image/test", json=payload, timeout=30)

        print(f"Status: {response.status_code}")
        data = response.json()
        print(f"Response: {json.dumps(data, indent=2)}")

        return response.status_code == 200
    except Exception as e:
        print(f"❌ Image provider test failed: {e}")
        return False


def test_batch_generation():
    """Test batch image generation"""
    print("\n🔄 Testing Batch Image Generation...")
    try:
        prompts = ["A red cat", "A blue dog", "A green bird"]

        payload = {
            "prompts": prompts,
            "provider": "openai",
            "model": "dall-e-2",
            "size": "256x256",
        }

        response = requests.post(
            f"{BASE_URL}/api/image/batch", json=payload, timeout=60
        )

        print(f"Status: {response.status_code}")
        data = response.json()
        print(f"Response: {json.dumps(data, indent=2)}")

        if response.status_code == 200:
            print(
                f"✅ Batch completed: {data.get('successful_generations', 0)}/{data.get('total_prompts', 0)}"
            )

        return response.status_code == 200
    except Exception as e:
        print(f"❌ Batch generation failed: {e}")
        return False


def test_replicate_generation():
    """Test Replicate image generation"""
    print("\n🎨 Testing Replicate Image Generation...")
    try:
        payload = {
            "prompt": "A futuristic cityscape at sunset",
            "provider": "replicate",
            "width": 512,
            "height": 512,
            "num_outputs": 1,
        }

        response = requests.post(
            f"{BASE_URL}/api/image/generate", json=payload, timeout=60
        )

        print(f"Status: {response.status_code}")
        data = response.json()
        print(f"Response: {json.dumps(data, indent=2)}")

        if response.status_code == 200:
            print(f"✅ Generated {len(data.get('urls', []))} image(s)")
            for i, url in enumerate(data.get("urls", []), 1):
                print(f"   {i}. {url}")

        return response.status_code == 200
    except Exception as e:
        print(f"❌ Replicate generation failed: {e}")
        return False


def main():
    """Run all tests"""
    print("🚀 AI Gateway Image API Test Suite")
    print("=" * 50)

    # Wait for server to start
    print("⏳ Waiting for server to start...")
    time.sleep(2)

    tests = [
        ("Health Check", test_health),
        ("Image Providers", test_image_providers),
        ("Provider Test", test_image_test_endpoint),
        ("OpenAI Generation", test_openai_image_generation),
        ("Batch Generation", test_batch_generation),
        ("Replicate Generation", test_replicate_generation),
    ]

    results = {}
    for test_name, test_func in tests:
        results[test_name] = test_func()

    print("\n📊 Test Results:")
    print("=" * 50)
    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name}: {status}")

    total_tests = len(results)
    passed_tests = sum(results.values())
    print(f"\nTotal: {passed_tests}/{total_tests} tests passed")

    if passed_tests == total_tests:
        print("🎉 All tests passed!")
    else:
        print("⚠️ Some tests failed. Check server logs.")


if __name__ == "__main__":
    main()
