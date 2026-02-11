"""
Database models for AI Service Platform using SQLAlchemy.
Defines the ORM models for tenants, users, API keys, and usage tracking.
"""

from datetime import datetime
from typing import Optional
from uuid import uuid4

from sqlalchemy import (
    Column, Integer, String, Boolean, DateTime, Text, 
    ForeignKey, JSON, BigInteger, Float, Index, UniqueConstraint,
    Enum as SQLEnum
)
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func

import sys
sys.path.insert(0, str(__file__).replace('/src/db/database.py', ''))

from src.models.schemas import UserRole, APIKeyType, DocumentStatus

Base = declarative_base()


class Tenant(Base):
    """Tenant/Organization model."""
    __tablename__ = "tenants"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    slug = Column(String(63), unique=True, nullable=False, index=True)
    description = Column(Text, nullable=True)
    plan = Column(String(50), default="free")
    owner_id = Column(Integer, nullable=False)
    is_active = Column(Boolean, default=True)
    settings = Column(JSON, default=dict)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    users = relationship("User", back_populates="tenant", cascade="all, delete-orphan")
    api_keys = relationship("APIKey", back_populates="tenant", cascade="all, delete-orphan")
    documents = relationship("Document", back_populates="tenant", cascade="all, delete-orphan")
    usage_records = relationship("UsageRecord", back_populates="tenant", cascade="all, delete-orphan")
    service_configs = relationship("TenantServiceConfig", back_populates="tenant", cascade="all, delete-orphan")


class User(Base):
    """User model within a tenant."""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    tenant_id = Column(Integer, ForeignKey("tenants.id"), nullable=False, index=True)
    email = Column(String(255), nullable=False, index=True)
    name = Column(String(255), nullable=False)
    hashed_password = Column(String(255), nullable=False)
    role = Column(SQLEnum(UserRole), default=UserRole.DEVELOPER)
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    last_login = Column(DateTime(timezone=True), nullable=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    tenant = relationship("Tenant", back_populates="users")
    api_keys = relationship("APIKey", back_populates="created_by")
    documents = relationship("Document", back_populates="created_by")
    usage_records = relationship("UsageRecord", back_populates="user")
    
    __table_args__ = (
        UniqueConstraint('tenant_id', 'email', name='uq_tenant_user_email'),
    )


class APIKey(Base):
    """API Key model for tenant authentication."""
    __tablename__ = "api_keys"
    
    id = Column(Integer, primary_key=True, index=True)
    tenant_id = Column(Integer, ForeignKey("tenants.id"), nullable=False, index=True)
    name = Column(String(255), nullable=False)
    key_type = Column(SQLEnum(APIKeyType), default=APIKeyType.DEVELOPMENT)
    key_hash = Column(String(255), nullable=False)
    key_prefix = Column(String(8), nullable=False, index=True)
    scopes = Column(JSON, default=list)
    rate_limit = Column(Integer, default=100)
    is_active = Column(Boolean, default=True)
    last_used_at = Column(DateTime(timezone=True), nullable=True)
    created_by_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    tenant = relationship("Tenant", back_populates="api_keys")
    created_by = relationship("User", back_populates="api_keys")
    usage_records = relationship("UsageRecord", back_populates="api_key")


class Document(Base):
    """Document model for uploaded files."""
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True, index=True)
    tenant_id = Column(Integer, ForeignKey("tenants.id"), nullable=False, index=True)
    filename = Column(String(255), nullable=False)
    file_type = Column(String(100), nullable=False)
    file_size = Column(BigInteger, nullable=False)
    collection_name = Column(String(255), nullable=False)
    file_path = Column(String(1024), nullable=False)
    status = Column(SQLEnum(DocumentStatus), default=DocumentStatus.PENDING)
    chunk_count = Column(Integer, default=0)
    vector_count = Column(Integer, default=0)
    metadata = Column(JSON, default=dict)
    processing_error = Column(Text, nullable=True)
    created_by_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    tenant = relationship("Tenant", back_populates="documents")
    created_by = relationship("User", back_populates="documents")


class UsageRecord(Base):
    """Usage tracking record for billing and analytics."""
    __tablename__ = "usage_records"
    
    id = Column(BigInteger, primary_key=True, index=True)
    tenant_id = Column(Integer, ForeignKey("tenants.id"), nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    api_key_id = Column(Integer, ForeignKey("api_keys.id"), nullable=True, index=True)
    service_type = Column(String(50), nullable=False, index=True)
    metric_type = Column(String(50), nullable=False)
    value = Column(BigInteger, nullable=False)
    metadata = Column(JSON, default=dict)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    
    # Relationships
    tenant = relationship("Tenant", back_populates="usage_records")
    user = relationship("User", back_populates="usage_records")
    api_key = relationship("APIKey", back_populates="usage_records")
    
    __table_args__ = (
        Index('ix_usage_tenant_timestamp', 'tenant_id', 'timestamp'),
        Index('ix_usage_tenant_service', 'tenant_id', 'service_type'),
    )


class TenantServiceConfig(Base):
    """Service configuration per tenant."""
    __tablename__ = "tenant_service_configs"
    
    id = Column(Integer, primary_key=True, index=True)
    tenant_id = Column(Integer, ForeignKey("tenants.id"), nullable=False, index=True)
    service_type = Column(String(50), nullable=False)
    provider = Column(String(50), nullable=False)
    models = Column(JSON, default=dict)
    settings = Column(JSON, default=dict)
    is_default = Column(Boolean, default=False)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    tenant = relationship("Tenant", back_populates="service_configs")
    
    __table_args__ = (
        UniqueConstraint('tenant_id', 'service_type', 'provider', name='uq_tenant_service_provider'),
    )


class VectorCollection(Base):
    """Vector collection metadata."""
    __tablename__ = "vector_collections"
    
    id = Column(Integer, primary_key=True, index=True)
    tenant_id = Column(Integer, ForeignKey("tenants.id"), nullable=False, index=True)
    name = Column(String(255), nullable=False)
    provider = Column(String(50), default="pinecone")
    dimension = Column(Integer, default=384)
    metric = Column(String(50), default="cosine")
    metadata = Column(JSON, default=dict)
    document_count = Column(Integer, default=0)
    vector_count = Column(BigInteger, default=0)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    __table_args__ = (
        UniqueConstraint('tenant_id', 'name', name='uq_tenant_collection'),
    )


# Helper function to generate API key
def generate_api_key():
    """Generate a new API key."""
    return f"sk_{uuid4().hex}_{uuid4().hex}"


def hash_api_key(key: str) -> str:
    """Hash an API key for storage."""
    import hashlib
    return hashlib.sha256(key.encode()).hexdigest()
