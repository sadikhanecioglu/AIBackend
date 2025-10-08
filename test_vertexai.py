#!/usr/bin/env python3
"""
Test script for VertexAI provider with mistral-large-2411
"""

import asyncio
import aiohttp
import json


async def test_vertexai_provider():
    """Test VertexAI provider with mistral-large-2411 model"""

    url = "http://localhost:8000/api/v1/llm/generate"

    payload = {
        "prompt": "Merhaba, Türkçe olarak kendini tanıt ve hangi model olduğunu söyle",
        "llm_provider": "vertexai",
        "temperature": 0.7,
        "max_tokens": 300,
    }

    headers = {"Content-Type": "application/json"}

    print("🔄 Testing VertexAI provider with mistral-large-2411...")
    print(f"📤 Request: {json.dumps(payload, indent=2, ensure_ascii=False)}")

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers) as response:
                if response.status == 200:
                    result = await response.json()
                    print(f"✅ Success! Status: {response.status}")
                    print(
                        f"📥 Response: {json.dumps(result, indent=2, ensure_ascii=False)}"
                    )
                else:
                    error_text = await response.text()
                    print(f"❌ Error! Status: {response.status}")
                    print(f"📥 Error Response: {error_text}")

    except Exception as e:
        print(f"💥 Exception occurred: {e}")


if __name__ == "__main__":
    asyncio.run(test_vertexai_provider())
