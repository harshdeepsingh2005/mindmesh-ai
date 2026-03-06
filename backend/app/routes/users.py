"""MindMesh AI — User Management Routes (Admin)."""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from ..database.db import get_db
from ..database.schemas import UserResponse, UserListResponse
from ..models.user import User
from ..dependencies import require_roles
from ..logging_config import logger

router = APIRouter()


@router.get(
    "/",
    response_model=UserListResponse,
    summary="List all users (admin only)",
)
async def list_users(
    role: str = None,
    skip: int = 0,
    limit: int = 50,
    _current_user: User = Depends(require_roles(["admin"])),
    db: AsyncSession = Depends(get_db),
) -> UserListResponse:
    """List all users with optional role filtering. Admin only.

    Args:
        role: Optional role filter (student, teacher, admin).
        skip: Pagination offset.
        limit: Pagination limit.
        _current_user: Injected admin user.
        db: Database session.

    Returns:
        List of users and total count.
    """
    query = select(User)
    count_query = select(func.count()).select_from(User)

    if role:
        query = query.where(User.role == role)
        count_query = count_query.where(User.role == role)

    query = query.offset(skip).limit(limit)

    result = await db.execute(query)
    users = result.scalars().all()

    count_result = await db.execute(count_query)
    total = count_result.scalar()

    logger.info(f"Admin listed users: total={total}, role_filter={role}")
    return UserListResponse(
        users=[UserResponse.model_validate(u) for u in users],
        total=total,
    )


@router.get(
    "/{user_id}",
    response_model=UserResponse,
    summary="Get user by ID (admin only)",
)
async def get_user(
    user_id: str,
    _current_user: User = Depends(require_roles(["admin"])),
    db: AsyncSession = Depends(get_db),
) -> UserResponse:
    """Retrieve a specific user by ID. Admin only.

    Args:
        user_id: Target user ID.
        _current_user: Injected admin user.
        db: Database session.

    Returns:
        The user's profile.

    Raises:
        HTTPException: If user not found.
    """
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with id={user_id} not found.",
        )

    return UserResponse.model_validate(user)


@router.delete(
    "/{user_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete user by ID (admin only)",
)
async def delete_user(
    user_id: str,
    _current_user: User = Depends(require_roles(["admin"])),
    db: AsyncSession = Depends(get_db),
) -> None:
    """Delete a user account. Admin only.

    Args:
        user_id: Target user ID.
        _current_user: Injected admin user.
        db: Database session.

    Raises:
        HTTPException: If user not found.
    """
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with id={user_id} not found.",
        )

    await db.delete(user)
    await db.commit()

    logger.info(f"Admin deleted user: id={user_id}")
