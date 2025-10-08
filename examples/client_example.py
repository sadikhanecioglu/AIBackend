#!/usr/bin/env python3
"""
Simple client example for the AI Gateway
"""

import asyncio
import json
import requests
import websockets
from typing import Optional

class AIGatewayClient:
    """Simple client for the AI Gateway"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.ws_url = base_url.replace("http", "ws") + "/voice"
    
    def health_check(self) -> dict:
        """Check API health"""
        response = requests.get(f"{self.base_url}/health")
        return response.json()
    
    def get_providers(self) -> dict:
        """Get available providers"""
        response = requests.get(f"{self.base_url}/providers")
        return response.json()
    
    def generate_text(self, prompt: str, provider: str = "openai", history: Optional[list] = None) -> dict:
        """Generate text using LLM"""
        payload = {
            "prompt": prompt,
            "llm_provider": provider
        }
        if history:
            payload["history"] = history
            
        response = requests.post(f"{self.base_url}/llm", json=payload)
        return response.json()
    
    def generate_image(self, prompt: str, provider: str = "openai", size: str = "1024x1024") -> dict:
        """Generate image"""
        payload = {
            "prompt": prompt,
            "image_provider": provider,
            "size": size
        }
        response = requests.post(f"{self.base_url}/image", json=payload)
        return response.json()
    
    async def voice_chat(self):
        """Example WebSocket voice chat session"""
        try:
            async with websockets.connect(self.ws_url) as websocket:
                print("🔗 Connected to voice chat")
                
                # Listen for initial connection message
                response = await websocket.recv()
                data = json.loads(response)
                print(f"📨 {data}")
                
                # Send text message
                text_message = {
                    "text": "Hello, this is a test message from the Python client!"
                }
                await websocket.send(json.dumps(text_message))
                print(f"📤 Sent: {text_message}")
                
                # Receive response
                response = await websocket.recv()
                data = json.loads(response)
                print(f"📨 Response: {data}")
                
                # Send session info request
                info_request = {
                    "action": "get_session_info"
                }
                await websocket.send(json.dumps(info_request))
                
                # Receive session info
                response = await websocket.recv()
                data = json.loads(response)
                print(f"📊 Session Info: {data}")
                
        except Exception as e:
            print(f"❌ WebSocket error: {e}")

def main():
    """Main client demo"""
    print("🚀 AI Gateway Client Demo")
    print("=" * 40)
    
    client = AIGatewayClient()
    
    # Test health
    print("\n🔍 Testing health check...")
    try:
        health = client.health_check()
        print(f"✅ Health: {health['status']}")
        print(f"   Active sessions: {health['active_sessions']}")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return
    
    # Test providers
    print("\n🔍 Testing providers...")
    try:
        providers = client.get_providers()
        print(f"✅ Current providers: {providers['current']}")
    except Exception as e:
        print(f"❌ Providers failed: {e}")
    
    # Test image generation
    print("\n🔍 Testing image generation...")
    try:
        image_result = client.generate_image("A beautiful sunset over mountains")
        print(f"✅ Generated {len(image_result['images'])} images")
        print(f"   Provider: {image_result['provider']}")
        print(f"   Sample URL: {image_result['images'][0][:50]}...")
    except Exception as e:
        print(f"❌ Image generation failed: {e}")
    
    # Test LLM (will fail without API key)
    print("\n🔍 Testing LLM...")
    try:
        llm_result = client.generate_text("Hello, how are you?")
        print(f"✅ LLM Response: {llm_result}")
    except Exception as e:
        print(f"❌ LLM failed (expected without API key): {e}")
    
    # Test WebSocket
    print("\n🔍 Testing WebSocket voice chat...")
    try:
        asyncio.run(client.voice_chat())
    except Exception as e:
        print(f"❌ WebSocket failed: {e}")
    
    print("\n✅ Client demo complete!")

if __name__ == "__main__":
    main()