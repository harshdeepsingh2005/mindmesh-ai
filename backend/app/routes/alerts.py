"""MindMesh AI — Alert API Routes.

Endpoints for:
  • Listing / filtering alerts
  • Viewing alert details
  • Updating alert status (acknowledge, resolve, dismiss)
  • Bulk acknowledge
  • Retrieving alerts for a specific student
  • Open alert count (for dashboard badge)
  • Manual alert creation
  • Triggering teacher notifications
"""

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from ..database.db import get_db
from ..database.schemas import (
    AlertCreate,
    AlertResponse,
    AlertListResponse,
    AlertStatusUpdate,
    AlertBulkAcknowledge,
    AlertCountResponse,
    AlertNotificationResponse,
)
from ..models.user import User
from ..dependencies import require_roles
from ..services.alert_service import (
    generate_alert,
    get_alert_by_id,
    list_alerts,
    get_open_alerts_for_student,
    count_open_alerts,
    update_alert_status,
    bulk_acknowledge,
    notify_teachers,
)
from ..services.student_service import StudentService
from ..logging_config import logger

router = APIRouter()


# ── List all alerts ──────────────────────────────────────────────


@router.get(
    "/",
    response_model=AlertListResponse,
    summary="List alerts with filters",
)
async def list_all_alerts(
    student_id: Optional[str] = Query(None, description="Filter by student"),
    status_filter: Optional[str] = Query(
        None,
        alias="status",
        description="Filter by status: open, acknowledged, resolved, dismissed",
    ),
    alert_type: Optional[str] = Query(
        None, description="Filter by type: high_risk, info"
    ),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_roles(["admin", "teacher"])),
) -> AlertListResponse:
    """List all alerts with optional filtering and pagination.

    Teacher/admin only.
    """
    alerts, total = await list_alerts(
        db,
        student_id=student_id,
        status_filter=status_filter,
        alert_type=alert_type,
        skip=skip,
        limit=limit,
    )

    return AlertListResponse(
        alerts=[AlertResponse.model_validate(a) for a in alerts],
        total=total,
    )


# ── Open alert count (dashboard badge) ──────────────────────────


@router.get(
    "/count",
    response_model=AlertCountResponse,
    summary="Count open alerts",
)
async def open_alert_count(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_roles(["admin", "teacher"])),
) -> AlertCountResponse:
    """Return the total number of open alerts.

    Useful for dashboard notification badges.
    """
    count = await count_open_alerts(db)
    return AlertCountResponse(open_count=count)


# ── Bulk acknowledge ─────────────────────────────────────────────


@router.post(
    "/bulk-acknowledge",
    summary="Acknowledge multiple alerts at once",
)
async def bulk_acknowledge_alerts(
    body: AlertBulkAcknowledge,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_roles(["admin", "teacher"])),
):
    """Acknowledge multiple open alerts in a single request.

    Only alerts with status 'open' will be transitioned.
    """
    count = await bulk_acknowledge(db, body.alert_ids)

    logger.info(
        f"Bulk acknowledge by user={current_user.id}: "
        f"{count}/{len(body.alert_ids)} alerts acknowledged"
    )

    return {
        "acknowledged": count,
        "requested": len(body.alert_ids),
    }


# ── Manual alert creation ───────────────────────────────────────


@router.post(
    "/",
    response_model=AlertResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create an alert manually",
)
async def create_alert(
    body: AlertCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_roles(["admin", "teacher"])),
) -> AlertResponse:
    """Manually create an alert for a student.

    Teachers/admins can create informational or high-risk alerts
    based on observations outside the automated pipeline.
    """
    # Verify student exists
    svc = StudentService(db)
    await svc._get_student_or_404(body.student_id)

    alert = await generate_alert(
        db,
        student_id=body.student_id,
        risk_score=body.risk_score,
        alert_type=body.alert_type,
        message=body.message,
        deduplicate_minutes=5,  # shorter window for manual alerts
    )

    if alert is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A similar alert was recently created for this student.",
        )

    logger.info(
        f"Manual alert created by user={current_user.id}: "
        f"alert_id={alert.id}, student_id={body.student_id}"
    )

    return AlertResponse.model_validate(alert)


# ── Alerts for a student ────────────────────────────────────────


@router.get(
    "/student/{student_id}",
    response_model=AlertListResponse,
    summary="Get alerts for a student",
)
async def get_student_alerts(
    student_id: str,
    status_filter: Optional[str] = Query(None, alias="status"),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_roles(["admin", "teacher", "student"])),
) -> AlertListResponse:
    """Get all alerts for a specific student.

    Students can only view their own alerts.
    """
    # Access control for students
    if current_user.role == "student":
        svc = StudentService(db)
        student = await svc._get_student_or_404(student_id)
        if not student.user_id or student.user_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You can only view your own alerts.",
            )

    alerts, total = await list_alerts(
        db,
        student_id=student_id,
        status_filter=status_filter,
        skip=skip,
        limit=limit,
    )

    return AlertListResponse(
        alerts=[AlertResponse.model_validate(a) for a in alerts],
        total=total,
    )


# ── Get single alert ────────────────────────────────────────────


@router.get(
    "/{alert_id}",
    response_model=AlertResponse,
    summary="Get alert by ID",
)
async def get_alert(
    alert_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_roles(["admin", "teacher"])),
) -> AlertResponse:
    """Retrieve a single alert by its ID."""
    alert = await get_alert_by_id(db, alert_id)
    if not alert:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Alert with id={alert_id} not found.",
        )
    return AlertResponse.model_validate(alert)


# ── Update alert status ─────────────────────────────────────────


@router.patch(
    "/{alert_id}/status",
    response_model=AlertResponse,
    summary="Update alert status",
)
async def change_alert_status(
    alert_id: str,
    body: AlertStatusUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_roles(["admin", "teacher"])),
) -> AlertResponse:
    """Transition an alert to a new status.

    Valid transitions:
      - open → acknowledged
      - open → dismissed
      - acknowledged → resolved
      - acknowledged → dismissed

    Teacher/admin only.
    """
    alert = await update_alert_status(
        db,
        alert_id,
        body.status,
        updated_by=current_user.id,
    )

    logger.info(
        f"Alert {alert_id} status changed to '{body.status}' "
        f"by user={current_user.id}"
    )

    return AlertResponse.model_validate(alert)


# ── Notify teachers for an alert ─────────────────────────────────


@router.post(
    "/{alert_id}/notify",
    response_model=AlertNotificationResponse,
    summary="Send notifications for an alert",
)
async def send_alert_notifications(
    alert_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_roles(["admin"])),
) -> AlertNotificationResponse:
    """Dispatch notifications to teachers for a specific alert.

    Admin only.  Currently sends in-app notifications.
    Future: email, SMS, push notifications.
    """
    alert = await get_alert_by_id(db, alert_id)
    if not alert:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Alert with id={alert_id} not found.",
        )

    notifications = await notify_teachers(db, alert)

    return AlertNotificationResponse(
        alert_id=alert.id,
        notifications_sent=len(notifications),
        notifications=notifications,
    )
