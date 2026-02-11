"""
Authentication service for AI Service Platform.
Handles JWT token generation, validation, and API key authentication.
"""

from datetime import datetime, timedelta
from typing import Optional, List

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

import sys
sys.path.insert(0, '/home/yared/Documents/GenAIProject/Revolutionising-Medical-Note-Taking/ai_service_platform')

from src.core.config import settings
from src.db.database import get_db, User, Tenant, APIKey
from src.models.schemas import TokenData, UserRole

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# HTTP Bearer token scheme
security = HTTPBearer()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash a password."""
    return pwd_context.hash(password)


def create_access_token(
    tenant_id: int,
    user_id: int,
    role: UserRole,
    scopes: List[str] = None,
    expires_delta: Optional[timedelta] = None
) -> str:
    """Create a JWT access token."""
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(
            minutes=settings.jwt.access_token_expire_minutes
        )
    
    to_encode = {
        "sub": str(user_id),
        "tenant_id": tenant_id,
        "role": role.value,
        "scopes": scopes or [],
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "access"
    }
    
    encoded_jwt = jwt.encode(
        to_encode, 
        settings.jwt.secret_key, 
        algorithm=settings.jwt.algorithm
    )
    return encoded_jwt


def create_refresh_token(tenant_id: int, user_id: int) -> str:
    """Create a JWT refresh token."""
    expire = datetime.utcnow() + timedelta(days=settings.jwt.refresh_token_expire_days)
    
    to_encode = {
        "sub": str(user_id),
        "tenant_id": tenant_id,
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "refresh"
    }
    
    encoded_jwt = jwt.encode(
        to_encode,
        settings.jwt.secret_key,
        algorithm=settings.jwt.algorithm
    )
    return encoded_jwt


def decode_token(token: str) -> TokenData:
    """Decode and validate a JWT token."""
    try:
        payload = jwt.decode(
            token,
            settings.jwt.secret_key,
            algorithms=[settings.jwt.algorithm]
        )
        
        user_id = int(payload.get("sub"))
        tenant_id = int(payload.get("tenant_id"))
        role = payload.get("role")
        scopes = payload.get("scopes", [])
        
        if role:
            role = UserRole(role)
        
        return TokenData(
            tenant_id=tenant_id,
            user_id=user_id,
            role=role,
            scopes=scopes
        )
    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"}
        )


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db)
) -> User:
    """Get the current authenticated user from JWT token."""
    token = credentials.credentials
    token_data = decode_token(token)
    
    result = await db.execute(
        select(User).where(
            User.id == token_data.user_id,
            User.tenant_id == token_data.tenant_id,
            User.is_active == True
        )
    )
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    return user


async def get_current_tenant(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> Tenant:
    """Get the current user's tenant."""
    result = await db.execute(
        select(Tenant).where(
            Tenant.id == current_user.tenant_id,
            Tenant.is_active == True
        )
    )
    tenant = result.scalar_one_or_none()
    
    if not tenant:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Tenant not found or inactive"
        )
    
    return tenant


async def authenticate_api_key(
    api_key: str,
    db: AsyncSession
) -> Optional[APIKey]:
    """Authenticate an API key and return the associated APIKey object."""
    import hashlib
    
    # Hash the provided key for comparison
    key_hash = hashlib.sha256(api_key.encode()).hexdigest()
    key_prefix = api_key[:8]
    
    result = await db.execute(
        select(APIKey).where(
            APIKey.key_hash == key_hash,
            APIKey.is_active == True
        )
    )
    api_key_obj = result.scalar_one_or_none()
    
    if api_key_obj:
        # Check expiration
        if api_key_obj.expires_at and api_key_obj.expires_at < datetime.utcnow():
            return None
    
    return api_key_obj


async def get_api_key_tenant(
    api_key_obj: APIKey,
    db: AsyncSession
) -> Tenant:
    """Get the tenant associated with an API key."""
    result = await db.execute(
        select(Tenant).where(
            Tenant.id == api_key_obj.tenant_id,
            Tenant.is_active == True
        )
    )
    return result.scalar_one()


class RoleChecker:
    """Dependency class for checking user roles."""
    
    def __init__(self, allowed_roles: List[UserRole]):
        self.allowed_roles = allowed_roles
    
    async def __call__(
        self,
        current_user: User = Depends(get_current_user)
    ) -> User:
        if current_user.role not in self.allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role {current_user.role} not allowed. Required: {self.allowed_roles}"
            )
        return current_user


# Pre-configured role checkers
require_owner = RoleChecker([UserRole.OWNER])
require_admin = RoleChecker([UserRole.OWNER, UserRole.ADMIN])
require_developer = RoleChecker([UserRole.OWNER, UserRole.ADMIN, UserRole.DEVELOPER])
require_end_user = RoleChecker([UserRole.OWNER, UserRole.ADMIN, UserRole.DEVELOPER, UserRole.END_USER])


async def authenticate_user(
    db: AsyncSession,
    tenant_id: int,
    email: str,
    password: str
) -> Optional[User]:
    """Authenticate a user by email and password."""
    result = await db.execute(
        select(User).where(
            User.tenant_id == tenant_id,
            User.email == email,
            User.is_active == True
        )
    )
    user = result.scalar_one_or_none()
    
    if not user:
        return None
    
    if not verify_password(password, user.hashed_password):
        return None
    
    return user
