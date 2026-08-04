"""Authentication router."""

from fastapi import APIRouter, Depends, HTTPException, status

from medvision.application.services.auth_service import (
    create_access_token,
    create_refresh_token,
    hash_password,
    verify_password,
)
from medvision.domain.entities.user import User
from medvision.infrastructure.db.repositories import SQLUserRepository
from medvision.presentation.api.dependencies import get_current_user, get_user_repo
from medvision.presentation.api.schemas import LoginRequest, TokenResponse, UserCreate, UserResponse

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(user_data: UserCreate, user_repo: SQLUserRepository = Depends(get_user_repo)):
    existing = await user_repo.get_by_email(user_data.email)
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    existing = await user_repo.get_by_username(user_data.username)
    if existing:
        raise HTTPException(status_code=400, detail="Username already taken")

    user = User(
        email=user_data.email,
        username=user_data.username,
        hashed_password=hash_password(user_data.password),
        role=user_data.role,
    )
    created = await user_repo.create(user)
    return UserResponse(
        id=created.id,
        email=created.email,
        username=created.username,
        role=created.role,
        is_active=created.is_active,
        created_at=created.created_at,
    )


@router.post("/login", response_model=TokenResponse)
async def login(credentials: LoginRequest, user_repo: SQLUserRepository = Depends(get_user_repo)):
    user = await user_repo.get_by_username(credentials.username)
    if not user or not verify_password(credentials.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account disabled")

    return TokenResponse(
        access_token=create_access_token(user.id, user.role),
        refresh_token=create_refresh_token(user.id),
    )


@router.get("/me", response_model=UserResponse)
async def get_me(current_user: User = Depends(get_current_user)):
    return UserResponse(
        id=current_user.id,
        email=current_user.email,
        username=current_user.username,
        role=current_user.role,
        is_active=current_user.is_active,
        created_at=current_user.created_at,
    )
