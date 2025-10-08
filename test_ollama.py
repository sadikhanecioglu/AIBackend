#!/usr/bin/env python3
"""
Test script for Ollama provider
"""

import asyncio
import aiohttp
import json


async def test_ollama_provider():
    """Test Ollama provider with different models"""

    url = "http://localhost:8000/api/v1/llm/generate"

    # Test different models
    test_cases = [
        {
            "name": "Default Ollama Model",
            "payload": {
                "prompt": "What is artificial intelligence? Give a brief explanation.",
                "llm_provider": "ollama",
                "temperature": 0.7,
                "max_tokens": 200,
            },
        },
        {
            "name": "Specific Ollama Model",
            "payload": {
                "prompt": "Explain machine learning in simple terms.",
                "llm_provider": "ollama",
                "model": "llama3.1:latest",
                "temperature": 0.5,
                "max_tokens": 150,
            },
        },
    ]

    headers = {"Content-Type": "application/json"}

    for test_case in test_cases:
        print(f"\n🔄 Testing: {test_case['name']}")
        print(
            f"📤 Request: {json.dumps(test_case['payload'], indent=2, ensure_ascii=False)}"
        )

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url, json=test_case["payload"], headers=headers
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        print(f"✅ Success! Status: {response.status}")
                        print(f"🏭 Provider: {result.get('provider', 'N/A')}")
                        print(f"🎯 Model: {result.get('model', 'N/A')}")
                        print(f"🤖 Response: {result.get('response', 'N/A')[:200]}...")
                        print(f"📈 Usage: {result.get('usage', 'N/A')}")
                    else:
                        error_text = await response.text()
                        print(f"❌ Error! Status: {response.status}")
                        print(f"📥 Error Response: {error_text}")

        except Exception as e:
            print(f"💥 Exception occurred: {e}")


if __name__ == "__main__":
    asyncio.run(test_ollama_provider())
