"""
Base provider interface for AI services.
Defines the abstract base classes that all AI providers must implement.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, AsyncIterator
from dataclasses import dataclass
from enum import Enum


class ProviderType(str, Enum):
    """Types of AI providers."""
    LLM = "llm"
    EMBEDDING = "embedding"
    SPEECH_TO_TEXT = "speech_to_text"
    TEXT_TO_SPEECH = "text_to_speech"
    IMAGE = "image"


@dataclass
class ModelInfo:
    """Information about a model."""
    name: str
    provider: str
    display_name: str
    description: str = ""
    max_tokens: Optional[int] = None
    max_input_tokens: Optional[int] = None
    supports_streaming: bool = False
    pricing: Dict[str, float] = None  # {"input": 0.01, "output": 0.03}
    capabilities: List[str] = None


@dataclass
class EmbeddingResult:
    """Result from an embedding model."""
    embeddings: List[List[float]]
    model: str
    usage: Dict[str, int]


@dataclass
class LLMResult:
    """Result from an LLM."""
    text: str
    model: str
    usage: Dict[str, int]
    finish_reason: str


@dataclass
class TranscriptionResult:
    """Result from speech-to-text."""
    text: str
    language: Optional[str] = None
    duration: Optional[float] = None


@dataclass
class TTSResult:
    """Result from text-to-speech."""
    audio: bytes
    format: str
    duration: float


class BaseAIProvider(ABC):
    """Abstract base class for AI providers."""
    
    provider_type: ProviderType
    provider_name: str
    
    def __init__(self, api_key: str = None, **kwargs):
        """Initialize the provider."""
        self.api_key = api_key
        self.config = kwargs
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the provider (load models, etc.)."""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the provider is available."""
        pass
    
    @abstractmethod
    def list_models(self) -> List[ModelInfo]:
        """List available models for this provider."""
        pass


class BaseLLMProvider(BaseAIProvider):
    """Abstract base class for LLM providers."""
    
    provider_type = ProviderType.LLM
    
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        model: str = None,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        stream: bool = False,
        **kwargs
    ) -> LLMResult:
        """Generate text from a prompt."""
        pass
    
    @abstractmethod
    async def stream_generate(
        self,
        prompt: str,
        model: str = None,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs
    ) -> AsyncIterator[str]:
        """Stream text generation from a prompt."""
        pass
    
    @abstractmethod
    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: str = None,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        stream: bool = False,
        **kwargs
    ) -> LLMResult:
        """Chat completion with message history."""
        pass


class BaseEmbeddingProvider(BaseAIProvider):
    """Abstract base class for embedding providers."""
    
    provider_type = ProviderType.EMBEDDING
    
    @abstractmethod
    async def embed(
        self,
        texts: List[str],
        model: str = None,
        **kwargs
    ) -> EmbeddingResult:
        """Generate embeddings for a list of texts."""
        pass
    
    @abstractmethod
    async def embed_one(
        self,
        text: str,
        model: str = None,
        **kwargs
    ) -> List[float]:
        """Generate embedding for a single text."""
        pass


class BaseSpeechToTextProvider(BaseAIProvider):
    """Abstract base class for speech-to-text providers."""
    
    provider_type = ProviderType.SPEECH_TO_TEXT
    
    @abstractmethod
    async def transcribe(
        self,
        audio: bytes,
        language: str = None,
        model: str = None,
        **kwargs
    ) -> TranscriptionResult:
        """Transcribe audio to text."""
        pass
    
    @abstractmethod
    async def transcribe_file(
        self,
        file_path: str,
        language: str = None,
        model: str = None,
        **kwargs
    ) -> TranscriptionResult:
        """Transcribe an audio file to text."""
        pass


class BaseTextToSpeechProvider(BaseAIProvider):
    """Abstract base class for text-to-speech providers."""
    
    provider_type = ProviderType.TEXT_TO_SPEECH
    
    @abstractmethod
    async def synthesize(
        self,
        text: str,
        voice: str = None,
        model: str = None,
        format: str = "mp3",
        **kwargs
    ) -> TTSResult:
        """Synthesize text to speech."""
        pass


class ProviderRegistry:
    """Registry for managing AI providers."""
    
    def __init__(self):
        self._providers: Dict[str, BaseAIProvider] = {}
        self._provider_classes: Dict[str, type] = {}
    
    def register(self, provider_class: type, name: str = None) -> None:
        """Register a provider class."""
        provider_name = name or provider_class.provider_name
        self._provider_classes[provider_name] = provider_class
    
    def get_provider(
        self,
        name: str,
        api_key: str = None,
        **config
    ) -> BaseAIProvider:
        """Get or create a provider instance."""
        if name in self._providers:
            return self._providers[name]
        
        if name not in self._provider_classes:
            raise ValueError(f"Provider '{name}' not registered. Available: {list(self._provider_classes.keys())}")
        
        provider_class = self._provider_classes[name]
        provider = provider_class(api_key=api_key, **config)
        self._providers[name] = provider
        return provider
    
    async def initialize_provider(self, name: str) -> None:
        """Initialize a provider."""
        provider = self.get_provider(name)
        await provider.initialize()
    
    def list_registered(self) -> List[str]:
        """List all registered provider names."""
        return list(self._provider_classes.keys())
    
    def clear_cache(self) -> None:
        """Clear provider cache (for reconfiguration)."""
        self._providers.clear()


# Global provider registry
provider_registry = ProviderRegistry()
