# Embedding API Documentation 📊

## Overview

The Embedding API converts text into vector representations (embeddings) that capture semantic meaning. These embeddings can be used for:
- Semantic search
- Similarity comparison
- Clustering
- Dimensionality reduction
- Vector database storage

## Supported Providers

| Provider | Model | Dimensions | Max Tokens |
|----------|-------|-----------|-----------|
| **OpenAI** | text-embedding-3-small | 1536 | 8191 |
| **OpenAI** | text-embedding-3-large | 3072 | 8191 |
| **OpenAI** | text-embedding-ada-002 | 1536 | 8191 |
| **Google** | text-embedding-004 | 768 | 2048 |
| **Google** | text-embedding-001 | 768 | 3000 |

## API Endpoints

### Generate Single Embedding

```http
POST /api/llm/embed
Content-Type: application/json

{
  "text": "Your text to embed",
  "embedding_provider": "openai",
  "model": "text-embedding-3-small"
}
```

**Response:**
```json
{
  "embeddings": [[0.123, 0.456, ..., 0.789]],
  "provider": "openai",
  "model": "text-embedding-3-small",
  "dimensions": 1536,
  "usage": {
    "prompt_tokens": 5,
    "total_tokens": 5
  }
}
```

### Generate Multiple Embeddings (Batch)

```http
POST /api/llm/embed
Content-Type: application/json

{
  "texts": [
    "First document to embed",
    "Second document to embed",
    "Third document to embed"
  ],
  "embedding_provider": "openai",
  "model": "text-embedding-3-small"
}
```

**Response:**
```json
{
  "embeddings": [
    [0.123, 0.456, ..., 0.789],
    [0.234, 0.567, ..., 0.890],
    [0.345, 0.678, ..., 0.901]
  ],
  "provider": "openai",
  "model": "text-embedding-3-small",
  "dimensions": 1536,
  "usage": {
    "prompt_tokens": 15,
    "total_tokens": 15
  }
}
```

## Usage Examples

### Python Client

```python
import requests
import numpy as np

BASE_URL = "http://localhost:8000"

# Single text embedding
def embed_single_text():
    response = requests.post(
        f"{BASE_URL}/api/llm/embed",
        json={
            "text": "The quick brown fox jumps over the lazy dog",
            "embedding_provider": "openai",
            "model": "text-embedding-3-small"
        }
    )
    return response.json()

# Multiple texts embedding (batch)
def embed_multiple_texts():
    response = requests.post(
        f"{BASE_URL}/api/llm/embed",
        json={
            "texts": [
                "The cat sat on the mat",
                "Dogs are loyal companions",
                "Birds can fly high"
            ],
            "embedding_provider": "openai",
            "model": "text-embedding-3-small"
        }
    )
    return response.json()

# Calculate similarity between embeddings
def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors"""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# Example usage
result = embed_multiple_texts()
embeddings = result["embeddings"]

# Compare first and second text
similarity = cosine_similarity(embeddings[0], embeddings[1])
print(f"Similarity between text 1 and 2: {similarity:.4f}")

# Using different providers
def embed_with_google():
    response = requests.post(
        f"{BASE_URL}/api/llm/embed",
        json={
            "text": "Machine learning is fascinating",
            "embedding_provider": "google",
            "model": "text-embedding-004"
        }
    )
    return response.json()
```

### Node.js/JavaScript Client

```javascript
const axios = require('axios');

const BASE_URL = 'http://localhost:8000';

// Embed single text
async function embedSingleText() {
    try {
        const response = await axios.post(
            `${BASE_URL}/api/llm/embed`,
            {
                text: "The quick brown fox jumps over the lazy dog",
                embedding_provider: "openai",
                model: "text-embedding-3-small"
            }
        );
        console.log("Single embedding:", response.data);
        return response.data;
    } catch (error) {
        console.error("Embedding error:", error.response.data);
    }
}

// Embed multiple texts
async function embedMultipleTexts() {
    try {
        const response = await axios.post(
            `${BASE_URL}/api/llm/embed`,
            {
                texts: [
                    "The cat sat on the mat",
                    "Dogs are loyal companions",
                    "Birds can fly high"
                ],
                embedding_provider: "openai",
                model: "text-embedding-3-small"
            }
        );
        console.log("Batch embeddings:", response.data);
        return response.data;
    } catch (error) {
        console.error("Embedding error:", error.response.data);
    }
}

// Calculate cosine similarity
function cosineSimilarity(vec1, vec2) {
    const dotProduct = vec1.reduce((sum, a, i) => sum + a * vec2[i], 0);
    const mag1 = Math.sqrt(vec1.reduce((sum, a) => sum + a * a, 0));
    const mag2 = Math.sqrt(vec2.reduce((sum, a) => sum + a * a, 0));
    return dotProduct / (mag1 * mag2);
}

// Example: Semantic search
async function semanticSearch(query, documents) {
    // Get query embedding
    const queryResult = await embedSingleText();
    const queryEmbedding = queryResult.embeddings[0];

    // Get document embeddings
    const docsResult = await axios.post(
        `${BASE_URL}/api/llm/embed`,
        {
            texts: documents,
            embedding_provider: "openai",
            model: "text-embedding-3-small"
        }
    );

    // Calculate similarities
    const similarities = docsResult.data.embeddings.map((emb, idx) => ({
        document: documents[idx],
        similarity: cosineSimilarity(queryEmbedding, emb)
    }));

    // Sort by similarity
    return similarities.sort((a, b) => b.similarity - a.similarity);
}
```

### cURL Examples

```bash
# Single text embedding
curl -X POST "http://localhost:8000/api/llm/embed" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The quick brown fox",
    "embedding_provider": "openai",
    "model": "text-embedding-3-small"
  }'

# Batch embeddings
curl -X POST "http://localhost:8000/api/llm/embed" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "Document one",
      "Document two",
      "Document three"
    ],
    "embedding_provider": "openai",
    "model": "text-embedding-3-small"
  }'

# Using Google provider
curl -X POST "http://localhost:8000/api/llm/embed" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Machine learning is fascinating",
    "embedding_provider": "google",
    "model": "text-embedding-004"
  }'
```

## Use Cases

### 1. Semantic Search

```python
async def semantic_search(query: str, documents: List[str]):
    """Find most similar documents to query"""
    # Get query embedding
    query_result = requests.post(f"{BASE_URL}/api/llm/embed", json={
        "text": query,
        "embedding_provider": "openai"
    })
    query_emb = query_result.json()["embeddings"][0]
    
    # Get document embeddings
    docs_result = requests.post(f"{BASE_URL}/api/llm/embed", json={
        "texts": documents,
        "embedding_provider": "openai"
    })
    doc_embs = docs_result.json()["embeddings"]
    
    # Calculate similarities
    similarities = [
        (doc, np.dot(query_emb, doc_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(doc_emb)))
        for doc, doc_emb in zip(documents, doc_embs)
    ]
    
    return sorted(similarities, key=lambda x: x[1], reverse=True)
```

### 2. Text Clustering

```python
from sklearn.cluster import KMeans

async def cluster_documents(documents: List[str], n_clusters: int = 3):
    """Cluster documents based on semantic similarity"""
    # Get embeddings
    result = requests.post(f"{BASE_URL}/api/llm/embed", json={
        "texts": documents,
        "embedding_provider": "openai"
    })
    embeddings = result.json()["embeddings"]
    
    # Cluster
    kmeans = KMeans(n_clusters=n_clusters)
    labels = kmeans.fit_predict(embeddings)
    
    # Group documents by cluster
    clusters = {i: [] for i in range(n_clusters)}
    for doc, label in zip(documents, labels):
        clusters[label].append(doc)
    
    return clusters
```

### 3. Duplicate Detection

```python
async def find_duplicates(documents: List[str], threshold: float = 0.95):
    """Find duplicate or highly similar documents"""
    result = requests.post(f"{BASE_URL}/api/llm/embed", json={
        "texts": documents,
        "embedding_provider": "openai"
    })
    embeddings = result.json()["embeddings"]
    
    duplicates = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            similarity = np.dot(embeddings[i], embeddings[j]) / \
                        (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]))
            if similarity > threshold:
                duplicates.append({
                    "doc1": documents[i],
                    "doc2": documents[j],
                    "similarity": similarity
                })
    
    return duplicates
```

### 4. Vector Database Storage

```python
import pinecone

async def store_embeddings_in_pinecone(documents: List[str]):
    """Store embeddings in Pinecone vector database"""
    # Initialize Pinecone
    pinecone.init(api_key="YOUR_PINECONE_KEY", environment="us-west1-gcp")
    
    # Get embeddings
    result = requests.post(f"{BASE_URL}/api/llm/embed", json={
        "texts": documents,
        "embedding_provider": "openai",
        "model": "text-embedding-3-small"
    })
    embeddings = result.json()["embeddings"]
    
    # Prepare vectors for Pinecone
    vectors = [
        (str(i), emb, {"text": doc})
        for i, (emb, doc) in enumerate(zip(embeddings, documents))
    ]
    
    # Upload to Pinecone
    index = pinecone.Index("documents")
    index.upsert(vectors=vectors)
    
    return len(vectors)
```

## Performance Tips

1. **Batch Processing**: Use `texts` parameter for multiple embeddings instead of individual calls
2. **Model Selection**: Use `text-embedding-3-small` for speed, `text-embedding-3-large` for better quality
3. **Caching**: Store embeddings for frequently used texts
4. **Dimensionality Reduction**: Use PCA or UMAP for visualization
5. **Normalization**: Normalize vectors before storage if using cosine similarity

## Cost Optimization

| Provider | Model | Price (per 1M tokens) |
|----------|-------|----------------------|
| OpenAI | text-embedding-3-small | $0.02 |
| OpenAI | text-embedding-3-large | $0.13 |
| Google | text-embedding-004 | $0.025 (per 1K requests) |

**Tips:**
- Use smaller models for non-critical applications
- Batch requests to reduce API calls
- Cache results for repeated queries
- Consider on-premise solutions for high volume

## Error Handling

```python
try:
    response = requests.post(
        f"{BASE_URL}/api/llm/embed",
        json={"text": query, "embedding_provider": "openai"}
    )
    response.raise_for_status()
    embeddings = response.json()["embeddings"]
except requests.exceptions.HTTPError as e:
    if e.response.status_code == 400:
        print("Invalid request - check your input")
    elif e.response.status_code == 500:
        print("Server error - API key may be invalid")
    else:
        print(f"Error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Environment Configuration

Add to `.env`:
```env
# OpenAI (for embeddings)
OPENAI_API_KEY=sk-proj-YOUR_KEY

# Google (for embeddings)
GOOGLE_API_KEY=YOUR_KEY

# Anthropic (experimental)
ANTHROPIC_API_KEY=YOUR_KEY
```

---

**Last Updated**: October 21, 2025
**API Version**: v1
