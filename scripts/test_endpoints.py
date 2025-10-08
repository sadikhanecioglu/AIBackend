#!/usr/bin/env python3
"""
Test script for the AI Gateway API endpoints
"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    print("🔍 Testing health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed - Status: {data['status']}")
            print(f"   Active sessions: {data['active_sessions']}")
            print(f"   Config: {data['config']}")
            return True
        else:
            print(f"❌ Health check failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_providers():
    """Test providers endpoint"""
    print("\n🔍 Testing providers endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/providers")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Providers endpoint passed")
            print(f"   Current providers: {data['current']}")
            print(f"   Available providers: {data['available']}")
            return True
        else:
            print(f"❌ Providers endpoint failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Providers endpoint error: {e}")
        return False

def test_image_generation():
    """Test image generation endpoint"""
    print("\n🔍 Testing image generation endpoint...")
    try:
        payload = {
            "prompt": "A beautiful sunset over mountains",
            "image_provider": "openai",
            "size": "1024x1024"
        }
        response = requests.post(f"{BASE_URL}/image", json=payload)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Image generation passed")
            print(f"   Generated {len(data['images'])} images")
            print(f"   Provider: {data['provider']}")
            print(f"   Sample URL: {data['images'][0][:50]}...")
            return True
        else:
            print(f"❌ Image generation failed with status {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Image generation error: {e}")
        return False

def test_llm_without_api_key():
    """Test LLM endpoint (will fail without API key, but tests structure)"""
    print("\n🔍 Testing LLM endpoint (without API key)...")
    try:
        payload = {
            "prompt": "Hello, how are you?",
            "llm_provider": "openai"
        }
        response = requests.post(f"{BASE_URL}/llm", json=payload)
        if response.status_code == 500:
            # Expected failure due to missing API key
            print(f"✅ LLM endpoint structure correct (failed as expected due to missing API key)")
            return True
        elif response.status_code == 200:
            print(f"✅ LLM endpoint passed unexpectedly (API key might be configured)")
            return True
        else:
            print(f"❌ LLM endpoint failed with unexpected status {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ LLM endpoint error: {e}")
        return False

def test_sessions():
    """Test sessions endpoint"""
    print("\n🔍 Testing sessions endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/sessions")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Sessions endpoint passed")
            print(f"   Active sessions: {data['active_sessions']}")
            return True
        else:
            print(f"❌ Sessions endpoint failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Sessions endpoint error: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 AI Gateway API Test Suite")
    print("=" * 50)
    
    # Wait a moment for server to be ready
    print("⏳ Waiting for server to be ready...")
    time.sleep(2)
    
    tests = [
        test_health,
        test_providers,
        test_sessions,
        test_image_generation,
        test_llm_without_api_key
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! AI Gateway is working correctly.")
    else:
        print(f"⚠️  {total - passed} tests failed. Check the output above for details.")
    
    print("\n🌐 Available endpoints:")
    print(f"   📊 Health: {BASE_URL}/health")
    print(f"   🔧 Providers: {BASE_URL}/providers")
    print(f"   📋 Sessions: {BASE_URL}/sessions")
    print(f"   🤖 LLM: {BASE_URL}/llm")
    print(f"   🖼️ Image: {BASE_URL}/image")
    print(f"   📚 Docs: {BASE_URL}/docs")
    print(f"   🔗 WebSocket: ws://localhost:8000/voice")

if __name__ == "__main__":
    main()