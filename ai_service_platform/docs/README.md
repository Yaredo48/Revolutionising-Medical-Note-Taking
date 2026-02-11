# AI Service Platform Documentation

## Overview

AI Service Platform is a multi-tenant SaaS platform for delivering AI services (LLM, RAG, Embeddings, Speech) to customers.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    AI Service Platform                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │   API GW    │  │  Auth Svc   │  │   Usage Tracker     │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                   Services Layer                     │    │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌─────────┐ │    │
│  │  │   RAG    │ │   LLM    │ │ Embedding│ │ Speech  │ │    │
│  │  └──────────┘ └──────────┘ └──────────┘ └─────────┘ │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                Provider Abstraction                  │    │
│  │  ┌─────────────┐ ┌─────────────┐ ┌───────────────┐ │    │
│  │  │ HuggingFace │ │   OpenAI    │ │   Anthropic   │ │    │
│  │  └─────────────┘ └─────────────┘ └───────────────┘ │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                 Vector Databases                     │    │
│  │  ┌─────────────┐ ┌─────────────┐ ┌───────────────┐ │    │
│  │  │  Pinecone   │ │   Qdrant   │ │   Weaviate    │ │    │
│  │  └─────────────┘ └─────────────┘ └───────────────┘ │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Getting Started

### Prerequisites

- Python 3.11+
- PostgreSQL 14+
- Redis 7+
- Docker (optional)

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd ai_service_platform

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -e .

# Copy environment file
cp .env.example .env

# Edit .env with your configuration
nano .env

# Run database migrations
alembic upgrade head

# Start the server
uvicorn src.main:app --reload
```

### Docker Deployment

```bash
# Build and start services
docker-compose up -d

# View logs
docker-compose logs -f api

# Stop services
docker-compose down
```

## API Reference

### Authentication

All API requests require authentication via Bearer token:

```bash
curl -H "Authorization: Bearer <token>" \
     https://api.example.com/api/v1/tenants/me
```

### Rate Limits

| Plan       | RPM  | Tokens/Min | Daily Limit |
|------------|------|------------|------------|
| Free       | 100  | 10,000     | 100,000    |
| Pro        | 1,000| 100,000    | 1,000,000  |
| Enterprise | Custom| Custom    | Custom     |

### Endpoints

#### Authentication

- `POST /api/v1/auth/token` - Get access token
- `POST /api/v1/auth/register` - Register new user

#### Tenants

- `GET /api/v1/tenants/me` - Get current tenant
- `PUT /api/v1/tenants/me` - Update tenant

#### Users

- `GET /api/v1/users/me` - Get current user
- `PUT /api/v1/users/me` - Update current user

#### API Keys

- `POST /api/v1/api-keys` - Create API key
- `GET /api/v1/api-keys` - List API keys
- `DELETE /api/v1/api-keys/{id}` - Delete API key

#### AI Services

- `POST /api/v1/services/rag/search` - Search documents
- `POST /api/v1/services/rag/generate` - Generate with RAG
- `POST /api/v1/services/rag/chat` - Chat with RAG
- `GET /api/v1/services/providers` - List providers

#### Usage

- `GET /api/v1/usage/summary` - Get usage summary
- `GET /api/v1/usage/daily` - Get daily usage breakdown

## Usage Examples

### RAG Search

```python
import httpx

response = httpx.post(
    "https://api.example.com/api/v1/services/rag/search",
    params={"query": "What is machine learning?", "top_k": 5},
    headers={"Authorization": "Bearer <token>"}
)
print(response.json())
```

### Chat with RAG

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain quantum computing."}
]

response = httpx.post(
    "https://api.example.com/api/v1/services/rag/chat",
    json={"messages": messages},
    headers={"Authorization": "Bearer <token>"}
)
print(response.json())
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `AI_PLATFORM_ENV` | Environment (development/production) | development |
| `AI_PLATFORM_DEBUG` | Enable debug mode | false |
| `AI_PLATFORM_DATABASE__HOST` | PostgreSQL host | localhost |
| `AI_PLATFORM_REDIS__HOST` | Redis host | localhost |
| `AI_PLATFORM_JWT__SECRET_KEY` | JWT signing secret | - |
| `HF_API_KEY` | HuggingFace API key | - |
| `OPENAI_API_KEY` | OpenAI API key | - |

## Pricing Plans

| Feature | Free | Pro | Enterprise |
|---------|------|-----|------------|
| API Calls | 100/day | 1000/day | Unlimited |
| Tokens | 100K/day | 1M/day | Unlimited |
| Storage | 1GB | 100GB | Unlimited |
| Support | Community | Email | Dedicated |
| SLA | None | 99% | 99.9% |

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

MIT License - see [LICENSE](../LICENSE) for details.
