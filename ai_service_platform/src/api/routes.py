"""
REST API routes for AI Service Platform.
Provides endpoints for authentication, tenants, services, and usage.
"""

from datetime import datetime
from typing import List, Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Query
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession
import hashlib

import sys
sys.path.insert(0, '/home/yared/Documents/GenAIProject/Revolutionising-Medical-Note-Taking/ai_service_platform')

from src.db.database import get_db, Tenant, User, APIKey, generate_api_key, hash_api_key
from src.core.auth import (
    get_current_user, get_current_tenant, create_access_token, create_refresh_token,
    verify_password, get_password_hash, authenticate_user
)
from src.core.usage import UsageTracker
from src.services.rag import RAGService, RAGConfig
from src.providers.base import provider_registry
from src.models.schemas import (
    Token, UserCreate, UserUpdate, UserResponse, TenantCreate, TenantUpdate, TenantResponse,
    APIKeyCreate, APIKeyUpdate, APIKeyResponse, APIKeyListItem,
    UsageSummary, ServiceType
)

# Create routers
auth_router = APIRouter(prefix="/auth", tags=["Authentication"])
tenant_router = APIRouter(prefix="/tenants", tags=["Tenants"])
user_router = APIRouter(prefix="/users", tags=["Users"])
apikey_router = APIRouter(prefix="/api-keys", tags=["API Keys"])
service_router = APIRouter(prefix="/services", tags=["AI Services"])
usage_router = APIRouter(prefix="/usage", tags=["Usage"])


# ============================================================================
# Authentication Routes
# ============================================================================

@auth_router.post("/token", response_model=Token)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db)
):
    """
    OAuth2 compatible token login.
    Authenticates user and returns JWT token.
    """
    # Find user by email within tenant
    result = await db.execute(
        User.__table__.select().where(
            User.email == form_data.username,
            User.tenant_id == 1  # Default tenant for demo
        )
    )
    user = result.scalar_one_or_none()
    
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User is inactive"
        )
    
    # Generate tokens
    access_token = create_access_token(
        tenant_id=user.tenant_id,
        user_id=user.id,
        role=user.role,
        scopes=[]
    )
    
    refresh_token = create_refresh_token(
        tenant_id=user.tenant_id,
        user_id=user.id
    )
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_in=30 * 60,  # 30 minutes
        refresh_token=refresh_token
    )


@auth_router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(
    user_data: UserCreate,
    db: AsyncSession = Depends(get_db)
):
    """
    Register a new user within a tenant.
    """
    # Check if email already exists in tenant
    result = await db.execute(
        User.__table__.select().where(
            User.email == user_data.email,
            User.tenant_id == 1  # Default tenant for demo
        )
    )
    
    if result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Create user
    user = User(
        tenant_id=1,  # Default tenant
        email=user_data.email,
        name=user_data.name,
        hashed_password=get_password_hash(user_data.password),
        role=user_data.role,
        is_active=True,
        is_verified=False
    )
    
    db.add(user)
    await db.commit()
    await db.refresh(user)
    
    return UserResponse(
        id=user.id,
        tenant_id=user.tenant_id,
        email=user.email,
        name=user.name,
        role=user.role,
        is_active=user.is_active,
        is_verified=user.is_verified,
        last_login=user.last_login,
        created_at=user.created_at,
        updated_at=user.updated_at
    )


# ============================================================================
# Tenant Routes
# ============================================================================

@tenant_router.get("/me", response_model=TenantResponse)
async def get_current_tenant(
    tenant: Tenant = Depends(get_current_tenant)
):
    """Get current tenant information."""
    return TenantResponse(
        id=tenant.id,
        name=tenant.name,
        slug=tenant.slug,
        description=tenant.description,
        plan=tenant.plan,
        is_active=tenant.is_active,
        settings=tenant.settings,
        created_at=tenant.created_at,
        updated_at=tenant.updated_at
    )


@tenant_router.put("/me", response_model=TenantResponse)
async def update_tenant(
    update_data: TenantUpdate,
    tenant: Tenant = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db)
):
    """Update current tenant."""
    if update_data.name:
        tenant.name = update_data.name
    if update_data.description is not None:
        tenant.description = update_data.description
    if update_data.plan:
        tenant.plan = update_data.plan
    if update_data.settings:
        tenant.settings = update_data.settings
    
    await db.commit()
    await db.refresh(tenant)
    
    return TenantResponse(
        id=tenant.id,
        name=tenant.name,
        slug=tenant.slug,
        description=tenant.description,
        plan=tenant.plan,
        is_active=tenant.is_active,
        settings=tenant.settings,
        created_at=tenant.created_at,
        updated_at=tenant.updated_at
    )


# ============================================================================
# User Routes
# ============================================================================

@user_router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    user: User = Depends(get_current_user)
):
    """Get current user information."""
    return UserResponse(
        id=user.id,
        tenant_id=user.tenant_id,
        email=user.email,
        name=user.name,
        role=user.role,
        is_active=user.is_active,
        is_verified=user.is_verified,
        last_login=user.last_login,
        created_at=user.created_at,
        updated_at=user.updated_at
    )


@user_router.put("/me", response_model=UserResponse)
async def update_current_user(
    update_data: UserUpdate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Update current user."""
    if update_data.name:
        user.name = update_data.name
    if update_data.email:
        user.email = update_data.email
    if update_data.password:
        user.hashed_password = get_password_hash(update_data.password)
    if update_data.role:
        user.role = update_data.role
    if update_data.is_active is not None:
        user.is_active = update_data.is_active
    
    await db.commit()
    await db.refresh(user)
    
    return UserResponse(
        id=user.id,
        tenant_id=user.tenant_id,
        email=user.email,
        name=user.name,
        role=user.role,
        is_active=user.is_active,
        is_verified=user.is_verified,
        last_login=user.last_login,
        created_at=user.created_at,
        updated_at=user.updated_at
    )


# ============================================================================
# API Key Routes
# ============================================================================

@apikey_router.post("", response_model=APIKeyResponse, status_code=status.HTTP_201_CREATED)
async def create_api_key(
    key_data: APIKeyCreate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Create a new API key."""
    # Generate key
    api_key = generate_api_key()
    key_hash = hash_api_key(api_key)
    key_prefix = api_key[:8]
    
    # Create API key record
    api_key_obj = APIKey(
        tenant_id=user.tenant_id,
        name=key_data.name,
        key_type=key_data.key_type,
        key_hash=key_hash,
        key_prefix=key_prefix,
        scopes=key_data.scopes,
        rate_limit=key_data.rate_limit,
        created_by_id=user.id,
        expires_at=key_data.expires_at
    )
    
    db.add(api_key_obj)
    await db.commit()
    await db.refresh(api_key_obj)
    
    return APIKeyResponse(
        id=api_key_obj.id,
        tenant_id=api_key_obj.tenant_id,
        name=api_key_obj.name,
        key_type=api_key_obj.key_type,
        key=api_key,  # Only shown on creation!
        key_prefix=api_key_obj.key_prefix,
        scopes=api_key_obj.scopes,
        rate_limit=api_key_obj.rate_limit,
        is_active=api_key_obj.is_active,
        last_used_at=api_key_obj.last_used_at,
        created_by_id=api_key_obj.created_by_id,
        expires_at=api_key_obj.expires_at,
        created_at=api_key_obj.created_at,
        updated_at=api_key_obj.updated_at
    )


@apikey_router.get("", response_model=List[APIKeyListItem])
async def list_api_keys(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """List all API keys for the tenant."""
    result = await db.execute(
        APIKey.__table__.select().where(
            APIKey.tenant_id == user.tenant_id
        ).order_by(APIKey.created_at.desc())
    )
    
    keys = []
    for row in result.all():
        keys.append(APIKeyListItem(
            id=row.id,
            name=row.name,
            key_type=row.key_type,
            key_prefix=row.key_prefix,
            scopes=row.scopes,
            rate_limit=row.rate_limit,
            is_active=row.is_active,
            last_used_at=row.last_used_at,
            expires_at=row.expires_at,
            created_at=row.created_at
        ))
    
    return keys


@apikey_router.delete("/{key_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_api_key(
    key_id: int,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Delete an API key."""
    result = await db.execute(
        APIKey.__table__.select().where(
            APIKey.id == key_id,
            APIKey.tenant_id == user.tenant_id
        )
    )
    key = result.scalar_one_or_none()
    
    if not key:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found"
        )
    
    await db.delete(key)
    await db.commit()


# ============================================================================
# AI Service Routes
# ============================================================================

@service_router.post("/rag/search")
async def rag_search(
    query: str = Query(..., description="Search query"),
    collection: Optional[str] = Query(None, description="Collection name"),
    top_k: int = Query(5, ge=1, le=20),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Search documents using RAG.
    """
    usage_tracker = UsageTracker(db)
    rag_service = RAGService(
        tenant_id=user.tenant_id,
        config=RAGConfig(default_collection=collection),
        usage_tracker=usage_tracker
    )
    await rag_service.initialize()
    
    # Record API call
    await usage_tracker.record_api_call(
        tenant_id=user.tenant_id,
        service_type=ServiceType.RAG,
        user_id=user.id
    )
    
    results = await rag_service.search(query, collection, top_k)
    
    return {"query": query, "results": results}


@service_router.post("/rag/generate")
async def rag_generate(
    query: str = Query(..., description="Question to answer"),
    collection: Optional[str] = Query(None, description="Collection name"),
    system_prompt: Optional[str] = Query(None, description="Custom system prompt"),
    max_tokens: int = Query(1000, ge=1, le=4000),
    temperature: float = Query(0.7, ge=0.0, ge=2.0),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Generate an answer with RAG context.
    """
    usage_tracker = UsageTracker(db)
    rag_service = RAGService(
        tenant_id=user.tenant_id,
        config=RAGConfig(default_collection=collection),
        usage_tracker=usage_tracker
    )
    await rag_service.initialize()
    
    # Record API call
    await usage_tracker.record_api_call(
        tenant_id=user.tenant_id,
        service_type=ServiceType.RAG,
        user_id=user.id
    )
    
    result = await rag_service.generate_with_context(
        query=query,
        collection=collection,
        system_prompt=system_prompt,
        max_tokens=max_tokens,
        temperature=temperature
    )
    
    return {
        "query": query,
        "answer": result.answer,
        "sources": result.sources,
        "usage": result.usage
    }


@service_router.post("/rag/chat")
async def rag_chat(
    messages: List[dict],
    collection: Optional[str] = Query(None, description="Collection name"),
    system_prompt: Optional[str] = Query(None, description="Custom system prompt"),
    max_tokens: int = Query(1000, ge=1, le=4000),
    temperature: float = Query(0.7, ge=0.0, le=2.0),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Chat with RAG context.
    """
    usage_tracker = UsageTracker(db)
    rag_service = RAGService(
        tenant_id=user.tenant_id,
        config=RAGConfig(default_collection=collection),
        usage_tracker=usage_tracker
    )
    await rag_service.initialize()
    
    # Record API call
    await usage_tracker.record_api_call(
        tenant_id=user.tenant_id,
        service_type=ServiceType.RAG,
        user_id=user.id
    )
    
    result = await rag_service.chat_with_context(
        messages=messages,
        collection=collection,
        system_prompt=system_prompt,
        max_tokens=max_tokens,
        temperature=temperature
    )
    
    return {
        "answer": result.answer,
        "sources": result.sources,
        "usage": result.usage
    }


@service_router.get("/providers")
async def list_providers():
    """List available AI providers."""
    return {
        "providers": provider_registry.list_registered(),
        "embedding_providers": ["huggingface", "openai"],
        "llm_providers": ["huggingface", "openai"],
        "vector_stores": vector_store_registry.list_registered()
    }


# ============================================================================
# Usage Routes
# ============================================================================

@usage_router.get("/summary", response_model=UsageSummary)
async def get_usage_summary(
    days: int = Query(30, ge=1, le=365),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get usage summary for the tenant."""
    usage_tracker = UsageTracker(db)
    return await usage_tracker.get_usage_summary(
        tenant_id=user.tenant_id,
        days=days
    )


@usage_router.get("/daily")
async def get_daily_usage(
    days: int = Query(30, ge=1, le=365),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get daily usage breakdown."""
    usage_tracker = UsageTracker(db)
    return await usage_tracker.get_daily_usage(
        tenant_id=user.tenant_id,
        days=days
    )
