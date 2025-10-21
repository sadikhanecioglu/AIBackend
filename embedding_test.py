#!/usr/bin/env python3
"""
Test script for Embedding API
Tests all embedding providers and similarity calculations
"""

import requests
import numpy as np
import asyncio
from typing import List, Dict, Any
import json

BASE_URL = "http://localhost:8000"

# ANSI colors for output
class Colors:
    GREEN = '\033[92m'
    BLUE = '\033[94m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def print_header(text: str):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}{Colors.RESET}\n")

def print_success(text: str):
    print(f"{Colors.GREEN}✅ {text}{Colors.RESET}")

def print_error(text: str):
    print(f"{Colors.RED}❌ {text}{Colors.RESET}")

def print_info(text: str):
    print(f"{Colors.YELLOW}ℹ️  {text}{Colors.RESET}")

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors"""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# Test 1: Single text embedding with OpenAI
def test_single_embedding_openai():
    print_header("Test 1: Single Text Embedding (OpenAI)")
    
    payload = {
        "text": "The quick brown fox jumps over the lazy dog",
        "embedding_provider": "openai",
        "model": "text-embedding-3-small"
    }
    
    print_info(f"Payload: {json.dumps(payload, indent=2)}")
    
    try:
        response = requests.post(f"{BASE_URL}/api/v1/llm/embed", json=payload)
        response.raise_for_status()
        
        data = response.json()
        print_success("Embedding generated successfully")
        print_info(f"Provider: {data['provider']}")
        print_info(f"Model: {data['model']}")
        print_info(f"Dimensions: {data['dimensions']}")
        print_info(f"Usage: {data.get('usage', 'N/A')}")
        print_info(f"First 10 values: {data['embeddings'][0][:10]}")
        
        return data
        
    except requests.exceptions.HTTPError as e:
        print_error(f"HTTP Error: {e.response.status_code}")
        print_error(f"Response: {e.response.text}")
    except Exception as e:
        print_error(f"Error: {e}")

# Test 2: Batch embeddings (multiple texts)
def test_batch_embeddings():
    print_header("Test 2: Batch Embeddings (Multiple Texts)")
    
    texts = [
        "The cat sat on the mat",
        "Dogs are loyal companions",
        "Birds can fly high",
        "Fish swim in water"
    ]
    
    payload = {
        "texts": texts,
        "embedding_provider": "openai",
        "model": "text-embedding-3-small"
    }
    
    print_info(f"Number of texts: {len(texts)}")
    
    try:
        response = requests.post(f"{BASE_URL}/api/v1/llm/embed", json=payload)
        response.raise_for_status()
        
        data = response.json()
        print_success(f"Generated {len(data['embeddings'])} embeddings")
        print_info(f"Dimensions: {data['dimensions']}")
        print_info(f"Usage: {data.get('usage', 'N/A')}")
        
        return data, texts
        
    except requests.exceptions.HTTPError as e:
        print_error(f"HTTP Error: {e.response.status_code}")
        print_error(f"Response: {e.response.text}")
    except Exception as e:
        print_error(f"Error: {e}")

# Test 3: Similarity calculation
def test_similarity(data: Dict, texts: List[str]):
    print_header("Test 3: Similarity Calculation")
    
    embeddings = data['embeddings']
    
    print_info("Calculating pairwise similarities...")
    
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            similarity = cosine_similarity(embeddings[i], embeddings[j])
            print(f"  '{texts[i]}' <-> '{texts[j]}'")
            print(f"  Similarity: {similarity:.4f}\n")

# Test 4: Semantic search simulation
def test_semantic_search():
    print_header("Test 4: Semantic Search Simulation")
    
    query = "animals and pets"
    documents = [
        "Dogs are loyal pets that enjoy playing fetch",
        "Cats are independent animals",
        "Birds can fly and sing beautifully",
        "Mathematics is a challenging subject",
        "The weather is sunny today"
    ]
    
    try:
        # Get query embedding
        print_info("Generating query embedding...")
        query_response = requests.post(
            f"{BASE_URL}/api/v1/llm/embed",
            json={
                "text": query,
                "embedding_provider": "openai",
                "model": "text-embedding-3-small"
            }
        )
        query_response.raise_for_status()
        query_embedding = query_response.json()["embeddings"][0]
        print_success("Query embedding generated")
        
        # Get document embeddings
        print_info("Generating document embeddings...")
        docs_response = requests.post(
            f"{BASE_URL}/api/v1/llm/embed",
            json={
                "texts": documents,
                "embedding_provider": "openai",
                "model": "text-embedding-3-small"
            }
        )
        docs_response.raise_for_status()
        doc_embeddings = docs_response.json()["embeddings"]
        print_success(f"Generated {len(doc_embeddings)} document embeddings")
        
        # Calculate similarities
        print_info("\nSimilarity scores:")
        similarities = []
        for doc, emb in zip(documents, doc_embeddings):
            sim = cosine_similarity(query_embedding, emb)
            similarities.append((doc, sim))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        print_info(f"Query: '{query}'\n")
        for i, (doc, sim) in enumerate(similarities, 1):
            print(f"  {i}. {sim:.4f} - {doc}")
        
    except requests.exceptions.HTTPError as e:
        print_error(f"HTTP Error: {e.response.status_code}")
        print_error(f"Response: {e.response.text}")
    except Exception as e:
        print_error(f"Error: {e}")

# Test 5: Different providers
def test_different_providers():
    print_header("Test 5: Different Providers")
    
    text = "Machine learning is revolutionizing technology"
    providers = [
        ("openai", "text-embedding-3-small"),
        ("google", "text-embedding-004"),
    ]
    
    for provider, model in providers:
        print_info(f"Testing {provider.upper()} with {model}...")
        
        try:
            response = requests.post(
                f"{BASE_URL}/api/v1/llm/embed",
                json={
                    "text": text,
                    "embedding_provider": provider,
                    "model": model
                }
            )
            response.raise_for_status()
            
            data = response.json()
            print_success(f"{provider}: {data['dimensions']} dimensions")
            
        except requests.exceptions.HTTPError as e:
            print_error(f"{provider}: HTTP {e.response.status_code}")
        except Exception as e:
            print_error(f"{provider}: {e}")
        
        print()

# Test 6: Error handling
def test_error_handling():
    print_header("Test 6: Error Handling")
    
    test_cases = [
        {
            "name": "Missing text",
            "payload": {"embedding_provider": "openai"}
        },
        {
            "name": "Invalid provider",
            "payload": {
                "text": "test",
                "embedding_provider": "invalid_provider"
            }
        },
    ]
    
    for test in test_cases:
        print_info(f"Testing: {test['name']}")
        
        try:
            response = requests.post(
                f"{BASE_URL}/api/v1/llm/embed",
                json=test['payload']
            )
            
            if response.status_code != 200:
                print_error(f"Expected error - Status {response.status_code}")
                print_info(f"Response: {response.json()}")
            else:
                print_error("Should have failed but didn't!")
                
        except Exception as e:
            print_error(f"Error: {e}")
        
        print()

# Run all tests
def main():
    print(f"\n{Colors.BOLD}{Colors.BLUE}")
    print("╔" + "="*58 + "╗")
    print("║" + " "*58 + "║")
    print("║" + "  EMBEDDING API TEST SUITE".center(58) + "║")
    print("║" + " "*58 + "║")
    print("╚" + "="*58 + "╝")
    print(Colors.RESET)
    
    print_info(f"Base URL: {BASE_URL}")
    
    try:
        # Test 1
        result1 = test_single_embedding_openai()
        
        # Test 2
        result2 = test_batch_embeddings()
        if result2:
            data, texts = result2
            
            # Test 3
            test_similarity(data, texts)
        
        # Test 4
        test_semantic_search()
        
        # Test 5
        test_different_providers()
        
        # Test 6
        test_error_handling()
        
        print_header("All Tests Completed! 🎉")
        
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Tests interrupted by user{Colors.RESET}")
    except Exception as e:
        print_error(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()
