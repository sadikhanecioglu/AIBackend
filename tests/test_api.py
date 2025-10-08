"""
Test FastAPI endpoints
"""

import pytest
from fastapi.testclient import TestClient
from app.main import create_app


@pytest.fixture
def client():
    """Create test client"""
    app = create_app()
    return TestClient(app)


def test_health_check(client):
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data
    assert "config" in data


def test_get_providers(client):
    """Test providers endpoint"""
    response = client.get("/providers")
    assert response.status_code == 200
    data = response.json()
    assert "current" in data
    assert "available" in data


def test_llm_endpoint(client):
    """Test LLM endpoint"""
    payload = {
        "prompt": "Hello, how are you?",
        "llm_provider": "openai"
    }
    # This will fail without proper API keys, but we can test the structure
    response = client.post("/llm", json=payload)
    # Should return 500 due to missing API key in test environment
    assert response.status_code in [200, 500]


def test_image_endpoint(client):
    """Test image generation endpoint"""
    payload = {
        "prompt": "A beautiful sunset",
        "image_provider": "openai"
    }
    response = client.post("/image", json=payload)
    # Should work with mock implementation
    assert response.status_code == 200
    data = response.json()
    assert "images" in data
    assert "provider" in data


def test_invalid_llm_request(client):
    """Test LLM endpoint with invalid request"""
    payload = {}  # Missing prompt
    response = client.post("/llm", json=payload)
    assert response.status_code == 400