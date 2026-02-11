"""
HuggingFace provider implementation.
Implements all AI service types using HuggingFace Inference API.
"""

import os
from typing import Dict, List, Optional, AsyncIterator
from dataclasses import dataclass

import httpx
from huggingface_hub import InferenceClient

import sys
sys.path.insert(0, '/home/yared/Documents/GenAIProject/Revolutionising-Medical-Note-Taking/ai_service_platform')

from src.providers.base import (
    BaseLLMProvider, BaseEmbeddingProvider, BaseSpeechToTextProvider,
    BaseTextToSpeechProvider, LLMResult, EmbeddingResult, TranscriptionResult, TTSResult,
    ModelInfo, ProviderType, provider_registry
)


@dataclass
class HuggingFaceModelInfo:
    """HuggingFace model information."""
    id: str
    task: str
    pipeline_tag: str


class HuggingFaceLLMProvider(BaseLLMProvider):
    """HuggingFace provider for LLM services."""
    
    provider_name = "huggingface"
    provider_type = ProviderType.LLM
    
    # Default models for different tasks
    DEFAULT_MODELS = {
        "text-generation": "HuggingFaceH4/zephyr-7b-beta",
        "chat-completion": "HuggingFaceH4/zephyr-7b-beta",
        "summarization": "facebook/bart-large-cnn",
        "text2text-generation": "google/flan-t5-large",
    }
    
    def __init__(self, api_key: str = None, **kwargs):
        super().__init__(api_key, **kwargs)
        self.client = None
        self._model_cache = {}
    
    async def initialize(self) -> None:
        """Initialize the HuggingFace client."""
        if not self.api_key:
            self.api_key = os.getenv("HF_API_KEY")
        
        self.client = InferenceClient(token=self.api_key)
        
        # Pre-load available models
        await self._load_models()
    
    async def _load_models(self) -> None:
        """Load available text-generation models."""
        self._model_cache = {
            "text-generation": [
                ModelInfo(
                    name="HuggingFaceH4/zephyr-7b-beta",
                    provider="huggingface",
                    display_name="Zephyr 7B",
                    description="A fine-tuned version of Mistral 7B for chat",
                    max_tokens=4096,
                    supports_streaming=True,
                    pricing={"input": 0.0, "output": 0.0}
                ),
                ModelInfo(
                    name="meta-llama/Llama-2-7b-chat-hf",
                    provider="huggingface",
                    display_name="Llama 2 7B Chat",
                    description="Meta's Llama 2 7B chat model",
                    max_tokens=4096,
                    supports_streaming=True,
                    pricing={"input": 0.0, "output": 0.0}
                ),
            ],
            "chat-completion": [
                ModelInfo(
                    name="HuggingFaceH4/zephyr-7b-beta",
                    provider="huggingface",
                    display_name="Zephyr 7B",
                    description="A fine-tuned version of Mistral 7B for chat",
                    max_tokens=4096,
                    supports_streaming=True,
                    pricing={"input": 0.0, "output": 0.0}
                ),
            ]
        }
    
    async def health_check(self) -> bool:
        """Check if HuggingFace API is available."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "https://huggingface.co/api/whoami",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    timeout=5.0
                )
                return response.status_code == 200
        except Exception:
            return False
    
    def list_models(self) -> List[ModelInfo]:
        """List all available HuggingFace models."""
        models = []
        for task_models in self._model_cache.values():
            models.extend(task_models)
        return models
    
    async def generate(
        self,
        prompt: str,
        model: str = None,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        stream: bool = False,
        **kwargs
    ) -> LLMResult:
        """Generate text using HuggingFace text-generation."""
        model = model or self.DEFAULT_MODELS["text-generation"]
        
        try:
            response = self.client.text_generation(
                model=model,
                inputs=prompt,
                parameters={
                    "max_new_tokens": max_tokens,
                    "temperature": temperature,
                    "do_sample": temperature > 0.0,
                    "stream": stream,
                }
            )
            
            generated_text = response.generated_text if hasattr(response, 'generated_text') else response
            
            return LLMResult(
                text=generated_text,
                model=model,
                usage={"prompt_tokens": len(prompt), "completion_tokens": len(generated_text)},
                finish_reason="stop"
            )
        except Exception as e:
            raise RuntimeError(f"HuggingFace generation failed: {str(e)}")
    
    async def stream_generate(
        self,
        prompt: str,
        model: str = None,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs
    ) -> AsyncIterator[str]:
        """Stream text generation from HuggingFace."""
        model = model or self.DEFAULT_MODELS["text-generation"]
        
        async for chunk in self.client.text_generation(
            model=model,
            inputs=prompt,
            parameters={
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "do_sample": temperature > 0.0,
                "stream": True,
            }
        ):
            yield chunk
    
    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: str = None,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        stream: bool = False,
        **kwargs
    ) -> LLMResult:
        """Chat completion using HuggingFace."""
        # Convert messages to prompt format
        prompt = self._messages_to_prompt(messages)
        return await self.generate(
            prompt=prompt,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=stream,
            **kwargs
        )
    
    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert chat messages to prompt format."""
        prompt = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            prompt += f"<|{role}|>\n{content}\n"
        prompt += "<|assistant|>\n"
        return prompt


class HuggingFaceEmbeddingProvider(BaseEmbeddingProvider):
    """HuggingFace provider for embedding services."""
    
    provider_name = "huggingface"
    provider_type = ProviderType.EMBEDDING
    
    DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    DIMENSION = 384
    
    def __init__(self, api_key: str = None, **kwargs):
        super().__init__(api_key, **kwargs)
        self.client = None
    
    async def initialize(self) -> None:
        """Initialize the HuggingFace client."""
        if not self.api_key:
            self.api_key = os.getenv("HF_API_KEY")
        
        self.client = InferenceClient(token=self.api_key)
    
    async def health_check(self) -> bool:
        """Check if HuggingFace API is available."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "https://huggingface.co/api/whoami",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    timeout=5.0
                )
                return response.status_code == 200
        except Exception:
            return False
    
    def list_models(self) -> List[ModelInfo]:
        """List available embedding models."""
        return [
            ModelInfo(
                name=self.DEFAULT_MODEL,
                provider="huggingface",
                display_name="All-MiniLM-L6-v2",
                description="Fast, high-quality sentence embeddings",
                max_tokens=256,
                pricing={"input": 0.0, "output": 0.0}
            ),
            ModelInfo(
                name="sentence-transformers/all-mpnet-base-v2",
                provider="huggingface",
                display_name="MPNet Base",
                description="High-quality sentence embeddings",
                max_tokens=384,
                pricing={"input": 0.0, "output": 0.0}
            ),
        ]
    
    async def embed(
        self,
        texts: List[str],
        model: str = None,
        **kwargs
    ) -> EmbeddingResult:
        """Generate embeddings for multiple texts."""
        model = model or self.DEFAULT_MODEL
        
        embeddings = []
        for text in texts:
            emb = await self.embed_one(text, model)
            embeddings.append(emb)
        
        return EmbeddingResult(
            embeddings=embeddings,
            model=model,
            usage={"total_tokens": sum(len(t.split()) for t in texts)}
        )
    
    async def embed_one(self, text: str, model: str = None) -> List[float]:
        """Generate embedding for a single text."""
        model = model or self.DEFAULT_MODEL
        
        try:
            response = self.client.text_embedding(
                model=model,
                inputs=text
            )
            return response
        except Exception as e:
            raise RuntimeError(f"HuggingFace embedding failed: {str(e)}")


class HuggingFaceSpeechToTextProvider(BaseSpeechToTextProvider):
    """HuggingFace provider for speech-to-text (Whisper)."""
    
    provider_name = "huggingface"
    provider_type = ProviderType.SPEECH_TO_TEXT
    
    DEFAULT_MODEL = "openai/whisper-small"
    
    def __init__(self, api_key: str = None, **kwargs):
        super().__init__(api_key, **kwargs)
        self.client = None
    
    async def initialize(self) -> None:
        """Initialize the HuggingFace client."""
        if not self.api_key:
            self.api_key = os.getenv("HF_API_KEY")
        
        self.client = InferenceClient(token=self.api_key)
    
    async def health_check(self) -> bool:
        """Check if HuggingFace API is available."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "https://huggingface.co/api/whoami",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    timeout=5.0
                )
                return response.status_code == 200
        except Exception:
            return False
    
    def list_models(self) -> List[ModelInfo]:
        """List available speech-to-text models."""
        return [
            ModelInfo(
                name="openai/whisper-base",
                provider="huggingface",
                display_name="Whisper Base",
                description="OpenAI Whisper base model",
                pricing={"input": 0.0, "output": 0.0}
            ),
            ModelInfo(
                name="openai/whisper-small",
                provider="huggingface",
                display_name="Whisper Small",
                description="OpenAI Whisper small model",
                pricing={"input": 0.0, "output": 0.0}
            ),
            ModelInfo(
                name="openai/whisper-medium",
                provider="huggingface",
                display_name="Whisper Medium",
                description="OpenAI Whisper medium model",
                pricing={"input": 0.0, "output": 0.0}
            ),
        ]
    
    async def transcribe(
        self,
        audio: bytes,
        language: str = None,
        model: str = None,
        **kwargs
    ) -> TranscriptionResult:
        """Transcribe audio using Whisper."""
        model = model or self.DEFAULT_MODEL
        
        try:
            result = self.client.audio_to_text(
                model=model,
                audio=audio
            )
            
            return TranscriptionResult(
                text=result.get("text", ""),
                language=result.get("language"),
            )
        except Exception as e:
            raise RuntimeError(f"HuggingFace transcription failed: {str(e)}")
    
    async def transcribe_file(
        self,
        file_path: str,
        language: str = None,
        model: str = None,
        **kwargs
    ) -> TranscriptionResult:
        """Transcribe an audio file."""
        with open(file_path, "rb") as f:
            audio = f.read()
        
        return await self.transcribe(audio, language, model, **kwargs)


# Register HuggingFace providers
provider_registry.register(HuggingFaceLLMProvider)
provider_registry.register(HuggingFaceEmbeddingProvider)
provider_registry.register(HuggingFaceSpeechToTextProvider)
