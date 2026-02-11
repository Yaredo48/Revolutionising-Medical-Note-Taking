"""
Pydantic models for AI Service Platform.
Defines data structures for tenants, users, API keys, and usage tracking.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum

from pydantic import BaseModel, Field, EmailStr
from pydantic import ConfigDict


class UserRole(str, Enum):
    """User roles within a tenant."""
    OWNER = "owner"
    ADMIN = "admin"
    DEVELOPER = "developer"
    END_USER = "end_user"


class APIKeyType(str, Enum):
    """Types of API keys."""
    PRODUCTION = "production"
    DEVELOPMENT = "development"
    TEST = "test"


class ServiceType(str, Enum):
    """AI service types available on the platform."""
    LLM = "llm"
    RAG = "rag"
    EMBEDDING = "embedding"
    SPEECH_TO_TEXT = "speech_to_text"
    TEXT_TO_SPEECH = "text_to_speech"
    IMAGE_GENERATION = "image_generation"
    MODERATION = "moderation"


class DocumentStatus(str, Enum):
    """Document processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class UsageMetricType(str, Enum):
    """Types of usage metrics."""
    API_CALLS = "api_calls"
    TOKENS_PROMPT = "tokens_prompt"
    TOKENS_COMPLETION = "tokens_completion"
    TOKENS_TOTAL = "tokens_total"
    CHARACTERS = "characters"
    SECONDS = "seconds"
    DOCUMENTS = "documents"
    VECTORS = "vectors"


# ============================================================================
# Tenant Models
# ============================================================================

class TenantBase(BaseModel):
    """Base tenant model."""
    name: str = Field(..., min_length=1, max_length=255, description="Organization name")
    slug: str = Field(..., min_length=1, max_length=63, pattern="^[a-z][a-z0-9-]*[a-z0-9]$", 
                      description="URL-safe identifier")
    description: Optional[str] = Field(default=None, max_length=1000, description="Organization description")
    plan: str = Field(default="free", description="Pricing plan")


class TenantCreate(TenantBase):
    """Tenant creation model."""
    owner_email: EmailStr = Field(..., description="Owner email address")
    owner_name: str = Field(..., min_length=1, max_length=255, description="Owner full name")


class TenantUpdate(BaseModel):
    """Tenant update model."""
    name: Optional[str] = Field(default=None, min_length=1, max_length=255)
    description: Optional[str] = Field(default=None, max_length=1000)
    plan: Optional[str] = None
    settings: Optional[Dict[str, Any]] = Field(default=None, description="Tenant-specific settings")


class TenantInDB(TenantBase):
    """Tenant model for database storage."""
    model_config = ConfigDict(from_attributes=True)
    
    id: int
    owner_id: int
    is_active: bool = True
    settings: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime


class TenantResponse(TenantBase):
    """Tenant response model."""
    id: int
    is_active: bool
    settings: Dict[str, Any]
    created_at: datetime
    updated_at: datetime


# ============================================================================
# User Models
# ============================================================================

class UserBase(BaseModel):
    """Base user model."""
    email: EmailStr
    name: str = Field(..., min_length=1, max_length=255)


class UserCreate(UserBase):
    """User creation model."""
    password: str = Field(..., min_length=8, max_length=128)
    role: UserRole = UserRole.DEVELOPER


class UserUpdate(BaseModel):
    """User update model."""
    name: Optional[str] = Field(default=None, min_length=1, max_length=255)
    email: Optional[EmailStr] = None
    password: Optional[str] = Field(default=None, min_length=8, max_length=128)
    role: Optional[UserRole] = None
    is_active: Optional[bool] = None


class UserInDB(UserBase):
    """User model for database storage."""
    model_config = ConfigDict(from_attributes=True)
    
    id: int
    tenant_id: int
    role: UserRole
    hashed_password: str
    is_active: bool = True
    is_verified: bool = False
    last_login: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime


class UserResponse(UserBase):
    """User response model."""
    id: int
    tenant_id: int
    role: UserRole
    is_active: bool
    is_verified: bool
    last_login: Optional[datetime]
    created_at: datetime
    updated_at: datetime


# ============================================================================
# API Key Models
# ============================================================================

class APIKeyBase(BaseModel):
    """Base API key model."""
    name: str = Field(..., min_length=1, max_length=255, description="API key name")
    key_type: APIKeyType = APIKeyType.DEVELOPMENT
    scopes: List[str] = Field(default_factory=list, description="Allowed service scopes")
    rate_limit: int = Field(default=100, description="Requests per minute")


class APIKeyCreate(APIKeyBase):
    """API key creation model."""
    expires_at: Optional[datetime] = Field(default=None, description="Expiration datetime")


class APIKeyUpdate(BaseModel):
    """API key update model."""
    name: Optional[str] = Field(default=None, min_length=1, max_length=255)
    scopes: Optional[List[str]] = None
    rate_limit: Optional[int] = None
    is_active: Optional[bool] = None
    expires_at: Optional[datetime] = None


class APIKeyInDB(APIKeyBase):
    """API key model for database storage."""
    model_config = ConfigDict(from_attributes=True)
    
    id: int
    tenant_id: int
    key_hash: str
    key_prefix: str = Field(..., description="First 8 chars for identification")
    is_active: bool = True
    last_used_at: Optional[datetime] = None
    created_by_id: int
    expires_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime


class APIKeyResponse(APIKeyBase):
    """API key response model (includes the actual key)."""
    id: int
    tenant_id: int
    key: str = Field(..., description="The full API key (only shown on creation)")
    key_prefix: str
    is_active: bool
    last_used_at: Optional[datetime]
    created_by_id: int
    expires_at: Optional[datetime]
    created_at: datetime
    updated_at: datetime


class APIKeyListItem(BaseModel):
    """API key list item (without the key)."""
    id: int
    name: str
    key_type: APIKeyType
    key_prefix: str
    scopes: List[str]
    rate_limit: int
    is_active: bool
    last_used_at: Optional[datetime]
    expires_at: Optional[datetime]
    created_at: datetime


# ============================================================================
# Usage Models
# ============================================================================

class UsageRecord(BaseModel):
    """Usage record for a single API call."""
    tenant_id: int
    user_id: Optional[int] = None
    api_key_id: Optional[int] = None
    service_type: ServiceType
    metric_type: UsageMetricType
    value: int
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class UsageSummary(BaseModel):
    """Usage summary for a tenant."""
    tenant_id: int
    period_start: datetime
    period_end: datetime
    total_api_calls: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    breakdown: Dict[str, Dict[str, int]] = Field(default_factory=dict)


class UsageLimit(BaseModel):
    """Usage limits for a plan."""
    plan: str
    max_api_calls_per_day: int
    max_tokens_per_day: int
    max_storage_gb: float
    max_vector_count: int
    rate_limit_rpm: int
    concurrent_requests: int


# ============================================================================
# Service Configuration Models
# ============================================================================

class ProviderConfig(BaseModel):
    """AI provider configuration for a tenant."""
    provider: str = Field(..., description="Provider name (openai, anthropic, huggingface)")
    api_key: Optional[str] = Field(default=None, description="API key (can be encrypted)")
    models: Dict[str, str] = Field(default_factory=dict, description="Model mappings")
    is_default: bool = Field(default=False, description="Is the default provider for this type")


class TenantServiceConfig(BaseModel):
    """Service configuration for a tenant."""
    tenant_id: int
    service_type: ServiceType
    providers: List[ProviderConfig] = Field(default_factory=list)
    settings: Dict[str, Any] = Field(default_factory=dict)


# ============================================================================
# Document Models
# ============================================================================

class DocumentBase(BaseModel):
    """Base document model."""
    filename: str = Field(..., max_length=255)
    file_type: str = Field(..., description="MIME type")
    file_size: int = Field(..., description="File size in bytes")


class DocumentCreate(DocumentBase):
    """Document creation model."""
    collection_name: Optional[str] = Field(default=None, description="Vector collection to add to")
    chunk_strategy: str = Field(default="recursive", description="Chunking strategy")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DocumentInDB(DocumentBase):
    """Document model for database storage."""
    model_config = ConfigDict(from_attributes=True)
    
    id: int
    tenant_id: int
    collection_name: str
    file_path: str
    status: DocumentStatus
    chunk_count: int = 0
    vector_count: int = 0
    metadata: Dict[str, Any]
    processing_error: Optional[str] = None
    created_by_id: int
    created_at: datetime
    updated_at: datetime


class DocumentResponse(DocumentBase):
    """Document response model."""
    id: int
    tenant_id: int
    collection_name: str
    status: DocumentStatus
    chunk_count: int
    vector_count: int
    metadata: Dict[str, Any]
    created_by_id: int
    created_at: datetime
    updated_at: datetime


# ============================================================================
# Authentication Models
# ============================================================================

class Token(BaseModel):
    """Authentication token response."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    refresh_token: Optional[str] = None


class TokenData(BaseModel):
    """Token payload data."""
    tenant_id: int
    user_id: int
    api_key_id: Optional[int] = None
    role: UserRole
    scopes: List[str] = []


class LoginRequest(BaseModel):
    """Login request model."""
    email: EmailStr
    password: str


class RefreshTokenRequest(BaseModel):
    """Refresh token request model."""
    refresh_token: str


# ============================================================================
# Error Models
# ============================================================================

class ErrorDetail(BaseModel):
    """Error detail model."""
    code: str
    message: str
    field: Optional[str] = None


class APIError(BaseModel):
    """API error response model."""
    error: str
    message: str
    details: Optional[List[ErrorDetail]] = None
    request_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
