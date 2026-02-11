# Migration Guide

## From Medical RAG App to AI Service Platform

This guide outlines the steps to migrate from the original medical RAG application to the new AI Service Platform architecture.

## Overview of Changes

### Before (Monolithic Medical App)
```
单一应用 = WebSocket API + Hardcoded HF + Pinecone
- 单一租户
- 医疗专用
- 无认证
- 无计费
```

### After (Multi-Tenant SaaS Platform)
```
多租户平台 = REST API + 可插拔提供商 + 多向量存储
- 多租户隔离
- 通用AI服务
- JWT + API密钥认证
- 使用跟踪和计费
```

## Step-by-Step Migration

### 1. Database Migration

#### New Tables Required

```sql
-- Tenants table (new)
CREATE TABLE tenants (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    slug VARCHAR(63) UNIQUE NOT NULL,
    plan VARCHAR(50) DEFAULT 'free',
    settings JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW()
);

-- Users now have tenant_id
ALTER TABLE users ADD COLUMN tenant_id INTEGER REFERENCES tenants(id);

-- API Keys table (new)
CREATE TABLE api_keys (
    id SERIAL PRIMARY KEY,
    tenant_id INTEGER REFERENCES tenants(id),
    name VARCHAR(255),
    key_hash VARCHAR(255),
    scopes JSONB,
    rate_limit INTEGER DEFAULT 100,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Usage records (new)
CREATE TABLE usage_records (
    id BIGSERIAL PRIMARY KEY,
    tenant_id INTEGER REFERENCES tenants(id),
    service_type VARCHAR(50),
    metric_type VARCHAR(50),
    value BIGINT,
    created_at TIMESTAMP DEFAULT NOW()
);
```

### 2. Code Migration

#### Old to New Mapping

| Old Component | New Component | Location |
|---------------|---------------|----------|
| `src/main_ws.py` | `src/main.py` | FastAPI app with routes |
| `src/rag/retriever.py` | `src/services/rag.py` | RAG service class |
| `src/embeddings/embed_pinecone_hf.py` | `src/providers/huggingface.py` | Embedding provider |
| `src/rag/summarizer_hf.py` | `src/providers/huggingface.py` | LLM provider |
| `src/speech/stt_hf.py` | `src/providers/huggingface.py` | Speech provider |

#### Key Changes

1. **Authentication**: Add JWT Bearer auth to all endpoints
2. **Tenant Isolation**: Prefix all vector collections with `t_{tenant_id}_`
3. **Provider Abstraction**: Use `provider_registry.get_provider()` instead of direct HF calls
4. **Usage Tracking**: Call `usage_tracker.record_usage()` after each API call

### 3. Environment Variables

#### Old (.env)
```env
HF_API_KEY=...
PINECONE_API_KEY=...
```

#### New (.env)
```env
# All AI_PLATFORM_ prefixed
AI_PLATFORM_DATABASE__HOST=localhost
AI_PLATFORM_JWT__SECRET_KEY=your-secret
HF_API_KEY=...
OPENAI_API_KEY=...
PINECONE_API_KEY=...
```

### 4. API Changes

#### Before (WebSocket only)
```javascript
// Connect to WS
const ws = new WebSocket('ws://localhost:8000/ws/session_id');
ws.send(audioBlob);
ws.onmessage = (event) => {
  console.log(event.data); // {transcription, note}
};
```

#### After (REST API)
```javascript
// Get token first
const token = await login(email, password);

// Make API calls
const response = await fetch('https://api.example.com/api/v1/services/rag/generate', {
  method: 'POST',
  headers: {
    'Authorization': `Bearer ${token}`,
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    query: 'What is machine learning?',
    collection: 'documents',
    max0
  })
_tokens: 100});
```

### 5. Vector Data Migration

#### Old Collection Names
```
medical-rag
```

#### New Collection Names (Tenant-Scoped)
```
t_{tenant_id}_medical-rag
t_{tenant_id}_documents
t_{tenant_id}_knowledge-base
```

#### Migration Script
```python
async def migrate_collections(old_index: str, new_prefix: str):
    """Migrate old collections to tenant-scoped names."""
    pinecone = PineconeClient()
    
    # List all vectors from old index
    vectors = await pinecone.list_all(old_index)
    
    for tenant_id in get_all_tenant_ids():
        new_collection = f"{new_prefix}_{tenant_id}_documents"
        await pinecone.create_collection(new_collection)
        
        # Re-index vectors for each tenant
        tenant_vectors = [v for v in vectors if v.tenant_id == tenant_id]
        await pinecone.upsert(new_collection, tenant_vectors)
```

## Rollback Plan

If migration fails:

1. Keep old database tables
2. Run old and new APIs in parallel for 30 days
3. Use feature flags to control traffic routing
4. Monitor error rates and latency

## Testing Checklist

- [ ] All existing endpoints work with new auth
- [ ] Tenant isolation prevents cross-tenant access
- [ ] Rate limiting applies correctly
- [ ] Usage tracking records all API calls
- [ ] Vector search returns correct results
- [ ] API documentation is complete

## Timeline

| Phase | Duration | Tasks |
|-------|----------|-------|
| Phase 1 | Week 1 | Database schema, new tables |
| Phase 2 | Week 2 | Core services, auth |
| Phase 3 | Week 3 | API endpoints, docs |
| Phase 4 | Week 4 | Testing, migration |
| Phase 5 | Week 5 | Launch, monitoring |
