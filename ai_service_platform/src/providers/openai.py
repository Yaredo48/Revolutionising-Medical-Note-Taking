"""
OpenAI provider implementation.
Implements LLM, Embedding, and other services using OpenAI API.
"""

import os
from typing import Dict, List, Optional, AsyncIterator
from dataclasses import dataclass

import openai as openai_client
from openai import AsyncOpenAI

import sys
sys.path.insert(0, '/home/yared/Documents/GenAIProject/Revolutionising-Medical-Note-Taking/ai_service_platform')

from src.providers.base import (
    BaseLLMProvider, BaseEmbeddingProvider, BaseTextToSpeechProvider,
    LLMResult, EmbeddingResult, TTSResult, ModelInfo, ProviderType,
    provider_registry
)


@dataclass
class OpenAIModelInfo:
    """OpenAI model information."""
    model_id: str
    context_window: int
    output_limit: int


class OpenAILLMProvider(BaseLLMProvider):
    """OpenAI provider for LLM services."""
    
    provider_name = "openai"
    provider_type = ProviderType.LLM
    
    # Model configurations
    MODELS = {
        "gpt-4": {"context": 8192, "output": 4096},
        "gpt-4-turbo": {"context": 128000, "output": 4096},
        "gpt-3.5-turbo": {"context": 16385, "output": 4096},
        "gpt-3.5-turbo-16k": {"context": 16385, "output": 4096},
    }
    
    def __init__(self, api_key: str = None, **kwargs):
        super().__init__(api_key, **kwargs)
        self.client = None
        self.base_url = kwargs.get("base_url", None)
    
    async def initialize(self) -> None:
        """Initialize the OpenAI client."""
        if not self.api_key:
            self.api_key = os.getenv("OPENAI_API_KEY")
        
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
    
    async def health_check(self) -> bool:
        """Check if OpenAI API is available."""
        try:
            await self.client.models.list()
            return True
        except Exception:
            return False
    
    def list_models(self) -> List[ModelInfo]:
        """List available OpenAI models."""
        return [
            ModelInfo(
                name="gpt-4-turbo-preview",
                provider="openai",
                display_name="GPT-4 Turbo",
                description="Latest GPT-4 with 128K context",
                max_tokens=4096,
                supports_streaming=True,
                pricing={"input": 0.01, "output": 0.03}
            ),
            ModelInfo(
                name="gpt-4",
                provider="openai",
                display_name="GPT-4",
                description="OpenAI's GPT-4 model",
                max_tokens=4096,
                supports_streaming=True,
                pricing={"input": 0.03, "output": 0.06}
            ),
            ModelInfo(
                name="gpt-3.5-turbo",
                provider="openai",
                display_name="GPT-3.5 Turbo",
                description="Fast and cost-effective GPT-3.5",
                max_tokens=4096,
                supports_streaming=True,
                pricing={"input": 0.0005, "output": 0.0015}
            ),
        ]
    
    async def generate(
        self,
        prompt: str,
        model: str = "gpt-3.5-turbo",
        max_tokens: int = 1000,
        temperature: float = 0.7,
        stream: bool = False,
        **kwargs
    ) -> LLMResult:
        """Generate text using OpenAI."""
        try:
            response = await self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                stream=stream,
                **kwargs
            )
            
            choice = response.choices[0]
            usage = response.usage
            
            return LLMResult(
                text=choice.message.content,
                model=model,
                usage={
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "total_tokens": usage.total_tokens
                },
                finish_reason=choice.finish_reason or "stop"
            )
        except Exception as e:
            raise RuntimeError(f"OpenAI generation failed: {str(e)}")
    
    async def stream_generate(
        self,
        prompt: str,
        model: str = "gpt-3.5-turbo",
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs
    ) -> AsyncIterator[str]:
        """Stream text generation from OpenAI."""
        response = await self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
            **kwargs
        )
        
        async for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    
    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: str = "gpt-3.5-turbo",
        max_tokens: int = 1000,
        temperature: float = 0.7,
        stream: bool = False,
        **kwargs
    ) -> LLMResult:
        """Chat completion using OpenAI."""
        try:
            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=stream,
                **kwargs
            )
            
            choice = response.choices[0]
            usage = response.usage
            
            return LLMResult(
                text=choice.message.content,
                model=model,
                usage={
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "total_tokens": usage.total_tokens
                },
                finish_reason=choice.finish_reason or "stop"
            )
        except Exception as e:
            raise RuntimeError(f"OpenAI chat failed: {str(e)}")


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """OpenAI provider for embedding services."""
    
    provider_name = "openai"
    provider_type = ProviderType.EMBEDDING
    
    MODELS = {
        "text-embedding-3-small": {"dimensions": 1536},
        "text-embedding-3-large": {"dimensions": 3072},
        "text-embedding-ada-002": {"dimensions": 1536},
    }
    
    def __init__(self, api_key: str = None, **kwargs):
        super().__init__(api_key, **kwargs)
        self.client = None
    
    async def initialize(self) -> None:
        """Initialize the OpenAI client."""
        if not self.api_key:
            self.api_key = os.getenv("OPENAI_API_KEY")
        
        self.client = AsyncOpenAI(api_key=self.api_key)
    
    async def health_check(self) -> bool:
        """Check if OpenAI API is available."""
        try:
            await self.client.embeddings.list()
            return True
        except Exception:
            return False
    
    def list_models(self) -> List[ModelInfo]:
        """List available embedding models."""
        return [
            ModelInfo(
                name="text-embedding-3-small",
                provider="openai",
                display_name="Text Embedding 3 Small",
                description="Small, efficient embedding model",
                max_tokens=8191,
                pricing={"input": 0.00002, "output": 0.0}
            ),
            ModelInfo(
                name="text-embedding-3-large",
                provider="openai",
                display_name="Text Embedding 3 Large",
                description="Large, high-quality embedding model",
                max_tokens=8191,
                pricing={"input": 00013, "output": 0.0}
            ),
        ]
    
    async def embed(
        self,
        texts: List[str],
        model: str = "text-embedding-3-small",
        **kwargs
    ) -> EmbeddingResult:
        """Generate embeddings for multiple texts."""
        try:
            response = await self.client.embeddings.create(
                model=model,
                input=texts
            )
            
            embeddings = [data.embedding for data in response.data]
            usage = response.usage
            
            return EmbeddingResult(
                embeddings=embeddings,
                model=model,
                usage={
                    "prompt_tokens": usage.prompt_tokens,
                    "total_tokens": usage.total_tokens
                }
            )
        except Exception as e:
            raise RuntimeError(f"OpenAI embedding failed: {str(e)}")
    
    async def embed_one(self, text: str, model: str = "text-embedding-3-small") -> List[float]:
        """Generate embedding for a single text."""
        result = await self.embed([text], model)
        return result.embeddings[0]


class OpenAITTSProvider(BaseTextToSpeechProvider):
    """OpenAI provider for text-to-speech services."""
    
    provider_name = "openai"
    provider_type = ProviderType.TEXT_TO_SPEECH
    
    MODELS = {
        "tts-1": {"quality": "standard"},
        "tts-1-hd": {"quality": "high"},
    }
    
    VOICES = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
    
    def __init__(self, api_key: str = None, **kwargs):
        super().__init__(api_key, **kwargs)
        self.client = None
    
    async def initialize(self) -> None:
        """Initialize the OpenAI client."""
        if not self.api_key:
            self.api_key = os.getenv("OPENAI_API_KEY")
        
        self.client = AsyncOpenAI(api_key=self.api_key)
    
    async def health_check(self) -> bool:
        """Check if OpenAI API is available."""
        try:
            await self.client.models.list()
            return True
        except Exception:
            return False
    
    def list_models(self) -> List[ModelInfo]:
        """List available TTS models."""
        return [
            ModelInfo(
                name="tts-1",
                provider="openai",
                display_name="TTS 1",
                description="Standard quality text-to-speech",
                pricing={"input": 0.015, "output": 0.0}
            ),
            ModelInfo(
                name="tts-1-hd",
                provider="openai",
                display_name="TTS 1 HD",
                description="High quality text-to-speech",
                pricing={"input": 0.030, "output": 0.0}
            ),
        ]
    
    async def synthesize(
        self,
        text: str,
        voice: str = "alloy",
        model: str = "tts-1",
        format: str = "mp3",
        **kwargs
    ) -> TTSResult:
        """Synthesize text to speech."""
        try:
            response = await self.client.audio.speech.create(
                model=model,
                voice=voice,
                input=text,
                response_format=format
            )
            
            audio = await response.read()
            
            return TTSResult(
                audio=audio,
                format=format,
                duration=len(audio) / 48000 / 4  # Rough estimate
            )
        except Exception as e:
            raise RuntimeError(f"OpenAI TTS failed: {str(e)}")


# Register OpenAI providers
provider_registry.register(OpenAILLMProvider)
provider_registry.register(OpenAIEmbeddingProvider)
provider_registry.register(OpenAITTSProvider)
