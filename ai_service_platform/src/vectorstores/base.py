"""
Base vector database interface.
Abstracts vector database operations for different backends.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum


class VectorMetric(str, Enum):
    """Similarity metrics for vector search."""
    COSINE = "cosine"
    DOT_PRODUCT = "dot_product"
    EUCLIDEAN = "euclidean"


@dataclass
class VectorEntry:
    """A vector entry for storage."""
    id: str
    vector: List[float]
    metadata: Dict[str, Any] = None
    document: str = None


@dataclass
class SearchResult:
    """Result from vector search."""
    id: str
    score: float
    metadata: Dict[str, Any]
    document: str


@dataclass
class CollectionInfo:
    """Information about a vector collection."""
    name: str
    dimension: int
    metric: VectorMetric
    count: int
    provider: str


class BaseVectorStore(ABC):
    """Abstract base class for vector database backends."""
    
    provider_name: str
    
    @abstractmethod
    async def create_collection(
        self,
        name: str,
        dimension: int,
        metric: VectorMetric = VectorMetric.COSINE,
        **kwargs
    ) -> bool:
        """Create a new vector collection."""
        pass
    
    @abstractmethod
    async def delete_collection(self, name: str) -> bool:
        """Delete a vector collection."""
        pass
    
    @abstractmethod
    async def get_collection(self, name: str) -> Optional[CollectionInfo]:
        """Get collection information."""
        pass
    
    @abstractmethod
    async def upsert_vectors(
        self,
        collection: str,
        vectors: List[VectorEntry],
        **kwargs
    ) -> int:
        """Insert or update vectors."""
        pass
    
    @abstractmethod
    async def delete_vectors(
        self,
        collection: str,
        ids: List[str],
        **kwargs
    ) -> int:
        """Delete vectors by IDs."""
        pass
    
    @abstractmethod
    async def search(
        self,
        collection: str,
        query_vector: List[float],
        top_k: int = 10,
        filters: Dict[str, Any] = None,
        **kwargs
    ) -> List[SearchResult]:
        """Search for similar vectors."""
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
    async def get_vector_count(self, collection: str) -> int:
        """Get the number of vectors in a collection."""
        pass
    
    @abstractmethod
    async def clear_collection(self, collection: str) -> bool:
        """Clear all vectors from a collection."""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the database is healthy."""
        pass


class VectorStoreRegistry:
    """Registry for managing vector database backends."""
    
    def __init__(self):
        self._stores: Dict[str, BaseVectorStore] = {}
        self._store_classes: Dict[str, type] = {}
    
    def register(self, store_class: type, name: str = None) -> None:
        """Register a vector store class."""
        store_name = name or store_class.provider_name
        self._store_classes[store_name] = store_class
    
    def get_store(
        self,
        name: str,
        **config
    ) -> BaseVectorStore:
        """Get or create a vector store instance."""
        if name in self._stores:
            return self._stores[name]
        
        if name not in self._store_classes:
            raise ValueError(f"Vector store '{name}' not registered. Available: {list(self._store_classes.keys())}")
        
        store_class = self._store_classes[name]
        store = store_class(**config)
        self._stores[name] = store
        return store
    
    async def initialize_store(self, name: str) -> None:
        """Initialize a vector store."""
        store = self.get_store(name)
        await store.health_check()  # Just to verify connection
    
    def list_registered(self) -> List[str]:
        """List all registered vector store names."""
        return list(self._store_classes.keys())
    
    def clear_cache(self) -> None:
        """Clear store cache."""
        self._stores.clear()


# Global vector store registry
vector_store_registry = VectorStoreRegistry()


class TenantAwareVectorStore:
    """Wrapper that adds tenant isolation to vector stores."""
    
    def __init__(
        self,
        base_store: BaseVectorStore,
        tenant_id: int,
        prefix: str = "t"
    ):
        self.base_store = base_store
        self.tenant_id = tenant_id
        self.prefix = prefix
    
    def _get_collection_name(self, collection: str) -> str:
        """Get tenant-scoped collection name."""
        return f"{self.prefix}_{self.tenant_id}_{collection}"
    
    async def create_collection(
        self,
        name: str,
        dimension: int,
        metric: VectorMetric = VectorMetric.COSINE,
        **kwargs
    ) -> bool:
        """Create a tenant-scoped collection."""
        scoped_name = self._get_collection_name(name)
        return await self.base_store.create_collection(
            scoped_name, dimension, metric, **kwargs
        )
    
    async def upsert_vectors(
        self,
        collection: str,
        vectors: List[VectorEntry],
        **kwargs
    ) -> int:
        """Upsert vectors to tenant-scoped collection."""
        scoped_name = self._get_collection_name(collection)
        return await self.base_store.upsert_vectors(scoped_name, vectors, **kwargs)
    
    async def search(
        self,
        collection: str,
        query_vector: List[float],
        top_k: int = 10,
        filters: Dict[str, Any] = None,
        **kwargs
    ) -> List[SearchResult]:
        """Search in tenant-scoped collection."""
        scoped_name = self._get_collection_name(collection)
        return await self.base_store.search(
            scoped_name, query_vector, top_k, filters, **kwargs
        )
    
    async def search_text(
        self,
        collection: str,
        query_text: str,
        embed_model: str = None,
        top_k: int = 10,
        filters: Dict[str, Any] = None,
        **kwargs
    ) -> List[SearchResult]:
        """Search text in tenant-scoped collection."""
        scoped_name = self._get_collection_name(collection)
        return await self.base_store.search_text(
            scoped_name, query_text, embed_model, top_k, filters, **kwargs
        )
