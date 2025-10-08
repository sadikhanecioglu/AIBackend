#!/usr/bin/env python3
"""
Debug script to examine GenerationRequest structure
"""

from llm_provider import GenerationRequest
import inspect

print("🔍 GenerationRequest Analysis")
print("=" * 50)

# Check constructor signature
print("📝 Constructor signature:")
print(inspect.signature(GenerationRequest.__init__))
print()

# Create an instance
print("🔧 Creating GenerationRequest instance...")
req = GenerationRequest(
    prompt="test prompt", history=[{"role": "user", "content": "hello"}]
)
print(f"✅ Created: {req}")
print()

# List all attributes
print("📋 Available attributes:")
for attr in dir(req):
    if not attr.startswith("_"):
        try:
            value = getattr(req, attr)
            if callable(value):
                print(f" - {attr}: <method>")
            else:
                print(f" - {attr}: {value}")
        except Exception as e:
            print(f" - {attr}: <Error: {e}>")

print()

# Check if messages attribute exists
print("🔍 Checking for 'messages' attribute:")
if hasattr(req, "messages"):
    print(f"✅ Has 'messages': {req.messages}")
else:
    print("❌ No 'messages' attribute found")

# Check if prompt attribute exists
print("\n🔍 Checking for 'prompt' attribute:")
if hasattr(req, "prompt"):
    print(f"✅ Has 'prompt': {req.prompt}")
else:
    print("❌ No 'prompt' attribute found")

# Check if history attribute exists
print("\n🔍 Checking for 'history' attribute:")
if hasattr(req, "history"):
    print(f"✅ Has 'history': {req.history}")
else:
    print("❌ No 'history' attribute found")
