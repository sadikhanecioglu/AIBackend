"""
Test Function Calling with VertexAI Support
"""

import asyncio
import httpx
import json


async def test_calculator():
    """Test calculator function"""
    print("\n🧪 Testing Calculator Function...")

    functions = [
        {
            "name": "calculator",
            "description": "Performs mathematical calculations",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate",
                    }
                },
                "required": ["expression"],
            },
        }
    ]

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            "http://localhost:8000/api/llm/generate",
            json={
                "prompt": "Calculate sqrt(144) + 10",
                "llm_provider": "vertexai",
                "model": "models/gemini-2.0-flash",
                "functions": functions,
            },
        )

        data = response.json()
        print(f"✅ Response: {json.dumps(data, indent=2)}")

        if data.get("function_call"):
            print(f"\n🔧 Function Called: {data['function_call']['name']}")
            print(f"📝 Arguments: {data['function_call']['arguments']}")
            print(f"📊 Result: {data['function_call']['result']}")
        else:
            print(f"\n💬 LLM Response: {data.get('response', 'No response')}")


async def test_database_query():
    """Test database query function (requires webhook endpoint)"""
    print("\n🧪 Testing Database Query Function...")

    functions = [
        {
            "name": "sendPersonaImage",
            "description": "Retrieves persona images from database",
            "parameters": {
                "type": "object",
                "properties": {
                    "personaId": {
                        "type": "string",
                        "description": "The ID of the persona",
                    },
                    "limit": {
                        "type": "number",
                        "description": "Maximum number of images",
                    },
                },
                "required": ["personaId"],
            },
        }
    ]

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            "http://localhost:8000/api/llm/generate",
            json={
                "prompt": "Send me images for persona xyz-123",
                "llm_provider": "vertexai",
                "model": "models/gemini-2.0-flash",
                "functions": functions,
                "persona_id": "xyz-123",
                "user_id": "user-456",
                "webhook_url": "http://localhost:3000/webhooks/ai-gateway/database-query",
            },
        )

        data = response.json()
        print(f"✅ Response: {json.dumps(data, indent=2)}")

        if data.get("function_call"):
            print(f"\n🔧 Function Called: {data['function_call']['name']}")
            print(f"📝 Arguments: {data['function_call']['arguments']}")
            print(f"📊 Result: {data['function_call']['result'][:200]}...")
        else:
            print(f"\n💬 LLM Response: {data.get('response', 'No response')[:200]}...")


async def test_no_functions():
    """Test standard generation without functions"""
    print("\n🧪 Testing Standard Generation (No Functions)...")

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            "http://localhost:8000/api/llm/generate",
            json={
                "prompt": "What is 2 + 2?",
                "llm_provider": "vertexai",
                "model": "models/gemini-2.0-flash",
            },
        )

        data = response.json()
        print(f"✅ Response: {data.get('response', 'No response')[:200]}...")
        print(f"📊 Provider: {data.get('provider')}")
        print(f"🤖 Model: {data.get('model')}")


async def main():
    """Run all tests"""
    print("🚀 Function Calling Tests\n")
    print("=" * 60)

    try:
        # Test 1: Calculator
        await test_calculator()

        print("\n" + "=" * 60)

        # Test 2: Database query (may fail if webhook not running)
        try:
            await test_database_query()
        except Exception as e:
            print(f"⚠️  Database query test failed (webhook may not be running): {e}")

        print("\n" + "=" * 60)

        # Test 3: Standard generation
        await test_no_functions()

        print("\n" + "=" * 60)
        print("\n✅ All tests completed!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
