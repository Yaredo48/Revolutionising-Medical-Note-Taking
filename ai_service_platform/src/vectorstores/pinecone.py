"""
Pinecone vector database implementation.
"""

import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from pinecone import Pinecone, ServerlessSpec

import sys
sys.path.insert(0, '/home/yared/Documents/GenAIProject/Revolutionising-Medical-Note-Taking/ai_service_platform')

from src.vectorstores.base import (
    BaseVectorStore, VectorStore, VectorEntry, SearchResult, CollectionInfo,
    VectorMetric, vector_store_registry
)


@dataclass
class PineconeConfig:
    """Pinecone configuration."""
    api_key: str
    environment: str
    index_name: str = "ai-platform"
    dimension: int = 384
    metric: str = "cosine"


class PineconeVectorStore(BaseVectorStore):
    """Pinecone implementation of vector store."""
    
    provider_name = "pinecone"
    
    def __init__(self, api_key: str = None, environment: str = None, 
                 index_name: str = "ai-platform", **kwargs):
        self.api_key = api_key or os.getenv("PINECONE_API_KEY")
        self.environment = environment or os.getenv("PINECONE_ENV")
        self.index_name = index_name
        self.client = None
        self._index = None
    
    async def initialize(self) -> None:
        """Initialize Pinecone client."""
        self.client = Pinecone(api_key=self.api_key)
        
        # Create index if it doesn't exist
        if self.index_name not in [i.name for i in self.client.list_indexes()]:
            self.client.create_index(
                name=self.index_name,
                dimension=384,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region=self.environment)
            )
        
        self._index = self.client.Index(self.index_name)
    
    async def health_check(self) -> bool:
        """Check if Pinecone is healthy."""
        try:
            stats = self._index.describe_index_stats()
            return "dimension" in stats
        except Exception:
            return False
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List available Pinecone configurations."""
        return [{"name": self.index_name, "dimension": 384, "metric": "cosine"}]
    
    async def create_collection(
        self,
        name: str,
        dimension: int,
        metric: VectorMetric = VectorMetric.COSINE,
        **kwargs
    ) -> bool:
        """Create a Pinecone index (collection)."""
        try:
            self.client.create_index(
                name=name,
                dimension=dimension,
                metric=metric.value,
                spec=ServerlessSpec(cloud="aws", region=self.environment)
            )
            return True
        except Exception:
            return False
    
    async def delete_collection(self, name: str) -> bool:
        """Delete a Pinecone index."""
        try:
            self.client.delete_index(name)
            return True
        except Exception:
            return False
    
    async def get_collection(self, name: str) -> Optional[CollectionInfo]:
        """Get collection information."""
        try:
            stats = self._index.describe_index_stats()
            return CollectionInfo(
                name=name,
                dimension=stats.get("dimension", 384),
                metric=VectorMetric(stats.get("metric", "cosine")),
                count=stats.get("total_vector_count", 0),
                provider="pinecone"
            )
        except Exception:
            return None
    
    async def upsert_vectors(
        self,
        collection: str,
        vectors: List[VectorEntry],
        **kwargs
    ) -> int:
        """Upsert vectors to Pinecone."""
        try:
            index = self.client.Index(collection)
            
            # Convert to Pinecone format
            pinecone_vectors = []
            for v in vectors:
                pinecone_vectors.append({
                    "id": v.id,
                    "values": v.vector,
                    "metadata": v.metadata or {}
                })
            
            index.upsert(vectors=pinecone_vectors)
            return len(pinecone_vectors)
        except Exception:
            return 0
    
    async def delete_vectors(
        self,
        collection: str,
        ids: List[str],
        **kwargs
    ) -> int:
        """Delete vectors by IDs."""
        try:
            index = self.client.Index(collection)
            index.delete(ids=ids)
            return len(ids)
        except Exception:
            return 0
    
    async def search(
        self,
        collection: str,
        query_vector: List[float],
        top_k: int = 10,
        filters: Dict[str, Any] = None,
        **kwargs
    ) -> List[SearchResult]:
        """Search for similar vectors."""
        try:
            index = self.client.Index(collection)
            
            query_params = {
                "vector": query_vector,
                "top_k": top_k,
                "include_metadata": True
            }
            
            if filters:
                query_params["filter"] = filters
            
            results = index.query(**query_params)
            
            search_results = []
            for match in results.get("matches", []):
                search_results.append(SearchResult(
                    id=match["id"],
                    score=match["score"],
                    metadata=match.get("metadata", {}),
                    document=match.get("metadata", {}).get("text", "")
                ))
            
            return search_results
        except Exception:
            return []
    
    async def search_text(
        self,
        collection: str,
        query_text: str,
        embed_model: str = None,
        top_k: int = 10,
        filters: Dict[str, Any] = None,
        **kwargs
    ) -> List[SearchResult]:
        """Search using text (requires embedding generation)."""
        # This requires an embedding provider
        # In practice, you'd inject the embedding provider
        raise NotImplementedError("Use search with pre-computed embeddings")
    
    async def get_vector_count(self, collection: str) -> int:
        """Get vector count."""
        try:
            stats = self._index.describe_index_stats()
            return stats.get("total_vector_count", 0)
        except Exception:
            return 0
    
    async def clear_collection(self, collection: str) -> bool:
        """Clear all vectors from a collection."""
        try:
            index = self.client.Index(collection)
            index.delete(delete_all=True)
            return True
        except Exception:
            return False


# Register Pinecone vector store
vector_store_registry.register(PineconeVectorStore)
