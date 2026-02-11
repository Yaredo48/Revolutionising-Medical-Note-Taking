"""
RAG (Retrieval-Augmented Generation) service.
Implements the core RAG pipeline with tenant isolation.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import sys
sys.path.insert(0, '/home/yared/Documents/GenAIProject/Revolutionising-Medical-Note-Taking/ai_service_platform')

from src.providers.base import provider_registry, LLMResult, EmbeddingResult
from src.vectorstores.base import vector_store_registry, TenantAwareVectorStore
from src.core.usage import UsageTracker
from src.models.schemas import ServiceType, UsageMetricType


@dataclass
class RAGConfig:
    """RAG configuration."""
    default_provider: str = "huggingface"
    default_collection: str = "documents"
    top_k: int = 5
    chunk_size: int = 1000
    chunk_overlap: int = 200
    prompt_template: str = None


@dataclass
class RAGResult:
    """Result from RAG pipeline."""
    answer: str
    sources: List[Dict[str, Any]]
    usage: Dict[str, int]


class RAGService:
    """
    RAG (Retrieval-Augmented Generation) service.
    
    Provides a unified interface for:
    - Indexing documents into vector stores
    - Searching for relevant context
    - Generating answers with retrieved context
    """
    
    def __init__(
        self,
        tenant_id: int,
        config: RAGConfig = None,
        usage_tracker: UsageTracker = None
    ):
        self.tenant_id = tenant_id
        self.config = config or RAGConfig()
        self.usage_tracker = usage_tracker
        
        # Initialize embedding provider
        self.embed_provider = provider_registry.get_provider(
            self.config.default_provider,
            provider_type="embedding"
        )
        
        # Initialize LLM provider
        self.llm_provider = provider_registry.get_provider(
            self.config.default_provider
        )
        
        # Initialize tenant-aware vector store
        vector_store = vector_store_registry.get_store("pinecone")
        self.vector_store = TenantAwareVectorStore(vector_store, tenant_id)
    
    async def initialize(self) -> None:
        """Initialize all providers."""
        await self.embed_provider.initialize()
        await self.llm_provider.initialize()
    
    async def index_document(
        self,
        text: str,
        document_id: str,
        collection: str = None,
        metadata: Dict[str, Any] = None
    ) -> int:
        """
        Index a document into the vector store.
        
        Args:
            text: The document text
            document_id: Unique ID for the document
            collection: Vector collection name
            metadata: Additional metadata
        
        Returns:
            Number of chunks indexed
        """
        collection = collection or self.config.default_collection
        
        # Chunk the document
        chunks = self._chunk_text(text)
        
        # Generate embeddings for all chunks
        embedding_result = await self.embed_provider.embed(chunks)
        
        # Create vector entries
        from src.vectorstores.base import VectorEntry
        vectors = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embedding_result.embeddings)):
            vectors.append(VectorEntry(
                id=f"{document_id}_{i}",
                vector=embedding,
                metadata={
                    **(metadata or {}),
                    "document_id": document_id,
                    "chunk_index": i,
                    "text": chunk[:500]  # Store preview
                },
                document=chunk
            ))
        
        # Upsert to vector store
        count = await self.vector_store.upsert_vectors(collection, vectors)
        
        # Record usage
        if self.usage_tracker:
            await self.usage_tracker.record_usage(
                tenant_id=self.tenant_id,
                service_type=ServiceType.RAG,
                metric_type=UsageMetricType.VECTORS,
                value=count,
                metadata={"document_id": document_id, "collection": collection}
            )
        
        return count
    
    async def search(
        self,
        query: str,
        collection: str = None,
        top_k: int = None,
        filters: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant documents.
        
        Args:
            query: Search query
            collection: Vector collection name
            top_k: Number of results
            filters: Metadata filters
        
        Returns:
            List of search results
        """
        collection = collection or self.config.default_collection
        top_k = top_k or self.config.top_k
        
        # Generate query embedding
        embedding_result = await self.embed_provider.embed_one(query)
        
        # Search vector store
        results = await self.vector_store.search(
            collection=collection,
            query_vector=embedding_result,
            top_k=top_k,
            filters=filters
        )
        
        return [
            {
                "id": r.id,
                "score": r.score,
                "text": r.document,
                "metadata": r.metadata
            }
            for r in results
        ]
    
    async def generate_with_context(
        self,
        query: str,
        collection: str = None,
        system_prompt: str = None,
        **llm_kwargs
    ) -> RAGResult:
        """
        Generate an answer with retrieved context.
        
        Args:
            query: The user query
            collection: Vector collection name
            system_prompt: Custom system prompt
            **llm_kwargs: Additional LLM arguments
        
        Returns:
            RAGResult with answer and sources
        """
        # Retrieve relevant context
        context_results = await self.search(query, collection)
        
        # Build context from results
        context = "\n\n".join([
            f"[Source {i+1}]: {r['text']}"
            for i, r in enumerate(context_results[:5])
        ])
        
        # Build prompt
        default_system = """You are a helpful AI assistant. Answer the user's question based on the provided context. 
If the context doesn't contain enough information, say so."""
        
        system_prompt = system_prompt or default_system
        
        full_prompt = f"""{system_prompt}

Context:
{context}

Question: {query}

Answer:"""
        
        # Generate answer
        llm_result = await self.llm_provider.generate(
            prompt=full_prompt,
            **llm_kwargs
        )
        
        # Record usage
        if self.usage_tracker:
            await self.usage_tracker.record_tokens(
                tenant_id=self.tenant_id,
                prompt_tokens=llm_result.usage.get("prompt_tokens", 0),
                completion_tokens=llm_result.usage.get("completion_tokens", 0),
                service_type=ServiceType.RAG
            )
        
        return RAGResult(
            answer=llm_result.text,
            sources=context_results,
            usage=llm_result.usage
        )
    
    async def chat_with_context(
        self,
        messages: List[Dict[str, str]],
        collection: str = None,
        system_prompt: str = None,
        **llm_kwargs
    ) -> RAGResult:
        """
        Chat with retrieved context.
        
        Args:
            messages: Chat message history
            collection: Vector collection name
            system_prompt: Custom system prompt
            **llm_kwargs: Additional LLM arguments
        
        Returns:
            RAGResult with answer and sources
        """
        # Get the last user message for retrieval
        last_message = messages[-1].get("content", "") if messages else ""
        
        # Retrieve context
        context_results = await self.search(last_message, collection)
        
        # Build context from results
        context = "\n\n".join([
            f"[Source {i+1}]: {r['text']}"
            for i, r in enumerate(context_results[:5])
        ])
        
        # Add context to system message
        default_system = f"""You are a helpful AI assistant. Answer the user's questions based on the provided context.

Context from knowledge base:
{context}

"""
        if system_prompt:
            system_prompt = default_system + system_prompt
        else:
            system_prompt = default_system
        
        # Prepare messages with system prompt
        full_messages = [{"role": "system", "content": system_prompt}] + messages
        
        # Generate response
        llm_result = await self.llm_provider.chat(
            messages=full_messages,
            **llm_kwargs
        )
        
        # Record usage
        if self.usage_tracker:
            await self.usage_tracker.record_tokens(
                tenant_id=self.tenant_id,
                prompt_tokens=llm_result.usage.get("prompt_tokens", 0),
                completion_tokens=llm_result.usage.get("completion_tokens", 0),
                service_type=ServiceType.RAG
            )
        
        return RAGResult(
            answer=llm_result.text,
            sources=context_results,
            usage=llm_result.usage
        )
    
    def _chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks.
        
        Uses a simple recursive chunking strategy.
        """
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.config.chunk_size
            
            # Try to split at sentence boundary
            if end < len(text):
                # Look for sentence endings
                for sep in [".\n", ".\n", ".\n", ".\n"]:
                    last_sep = text.rfind(sep, start, end)
                    if last_sep != -1:
                        end = last_sep + len(sep)
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - self.config.chunk_overlap
        
        return chunks
    
    async def delete_document(
        self,
        document_id: str,
        collection: str = None
    ) -> int:
        """
        Delete all vectors for a document.
        
        Args:
            document_id: The document ID
            collection: Vector collection name
        
        Returns:
            Number of vectors deleted
        """
        collection = collection or self.config.default_collection
        
        # Search for document chunks
        results = await self.vector_store.search(
            collection=collection,
            query_vector=[0.0] * 384,  # Dummy vector
            top_k=1000,
            filters={"document_id": document_id}
        )
        
        # Get IDs to delete
        ids = [r.id for r in results]
        
        if ids:
            return await self.vector_store.base_store.delete_vectors(collection, ids)
        
        return 0
    
    async def list_collections(self) -> List[str]:
        """List all collections for the tenant."""
        # In a multi-tenant setup, this would query the database
        return [self.config.default_collection]
