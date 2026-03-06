"""MindMesh AI — Alert & Notification Service.

Manages the full lifecycle of alerts:
  • Generation when risk thresholds are exceeded
  • Storage in the database
  • Retrieval with filtering / pagination
  • Status transitions (open → acknowledged → resolved → dismissed)
  • Teacher / counselor notification dispatch

An alert is the primary mechanism for escalating high-risk students
to human counselors.  The system NEVER acts autonomously on clinical
decisions — alerts exist to surface information for qualified humans.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

from fastapi import HTTPException, status
from sqlalchemy import select, func, and_, or_, update, desc
from sqlalchemy.ext.asyncio import AsyncSession

from ..models.alert import Alert
from ..models.student import Student
from ..models.user import User
from ..logging_config import logger


# ── Alert status state machine ───────────────────────────────────
# open → acknowledged → resolved
# open → dismissed
VALID_TRANSITIONS: Dict[str, List[str]] = {
    "open": ["acknowledged", "dismissed"],
    "acknowledged": ["resolved", "dismissed"],
    "resolved": [],
    "dismissed": [],
}

# Risk score that triggers automatic alert generation
ALERT_RISK_THRESHOLD = 70


# ── Alert Generation ────────────────────────────────────────────


async def generate_alert(
    db: AsyncSession,
    *,
    student_id: str,
    risk_score: int,
    alert_type: str = "high_risk",
    message: Optional[str] = None,
    deduplicate_minutes: int = 60,
) -> Optional[Alert]:
    """Create an alert for a student if one doesn't already exist recently.

    Deduplication: if an open/acknowledged alert of the same type was
    created within ``deduplicate_minutes`` for this student, a new one
    is NOT created (prevents alert fatigue).

    Args:
        db: Async database session.
        student_id: The student to alert on.
        risk_score: The triggering risk score (0-100).
        alert_type: ``"high_risk"`` or ``"info"``.
        message: Human-readable alert message.  Auto-generated if None.
        deduplicate_minutes: Window in which duplicate alerts are suppressed.

    Returns:
        The created Alert, or None if deduplicated away.
    """
    # ── Deduplication check ──────────────────────────────────────
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=deduplicate_minutes)
    existing = await db.execute(
        select(Alert).where(
            and_(
                Alert.student_id == student_id,
                Alert.alert_type == alert_type,
                Alert.status.in_(["open", "acknowledged"]),
                Alert.created_at >= cutoff,
            )
        )
    )
    if existing.scalar_one_or_none() is not None:
        logger.debug(
            f"Alert suppressed (dedup): student_id={student_id}, "
            f"type={alert_type}, window={deduplicate_minutes}m"
        )
        return None

    # ── Build message ────────────────────────────────────────────
    if message is None:
        if alert_type == "high_risk":
            message = (
                f"High risk detected — risk score {risk_score}/100.  "
                "Immediate counselor review recommended."
            )
        else:
            message = f"Informational alert — risk score {risk_score}/100."

    alert = Alert(
        id=str(uuid.uuid4()),
        student_id=student_id,
        risk_score=risk_score,
        alert_type=alert_type,
        message=message,
        created_at=datetime.now(timezone.utc),
        status="open",
    )
    db.add(alert)
    await db.commit()
    await db.refresh(alert)

    logger.warning(
        f"ALERT CREATED: id={alert.id}, student_id={student_id}, "
        f"type={alert_type}, score={risk_score}, status=open"
    )

    return alert


async def generate_alert_from_risk(
    db: AsyncSession,
    student_id: str,
    risk_score: int,
    risk_level: str,
) -> Optional[Alert]:
    """Convenience wrapper called by the risk-scoring pipeline.

    Only generates an alert when risk_level == 'high' or
    risk_score >= ALERT_RISK_THRESHOLD.

    Args:
        db: Async database session.
        student_id: Student ID.
        risk_score: Composite risk score 0-100.
        risk_level: 'low', 'medium', or 'high'.

    Returns:
        Alert if created, else None.
    """
    if risk_level != "high" and risk_score < ALERT_RISK_THRESHOLD:
        return None

    return await generate_alert(
        db,
        student_id=student_id,
        risk_score=risk_score,
        alert_type="high_risk",
    )


# ── Alert Queries ────────────────────────────────────────────────


async def get_alert_by_id(
    db: AsyncSession,
    alert_id: str,
) -> Optional[Alert]:
    """Fetch a single alert by ID."""
    result = await db.execute(select(Alert).where(Alert.id == alert_id))
    return result.scalar_one_or_none()


async def list_alerts(
    db: AsyncSession,
    *,
    student_id: Optional[str] = None,
    status_filter: Optional[str] = None,
    alert_type: Optional[str] = None,
    skip: int = 0,
    limit: int = 50,
) -> Tuple[List[Alert], int]:
    """List alerts with optional filters and pagination.

    Args:
        db: Async database session.
        student_id: Filter by student.
        status_filter: Filter by status (open, acknowledged, resolved, dismissed).
        alert_type: Filter by type (high_risk, info).
        skip: Pagination offset.
        limit: Pagination limit.

    Returns:
        Tuple of (alert list, total count).
    """
    conditions = []
    if student_id:
        conditions.append(Alert.student_id == student_id)
    if status_filter:
        conditions.append(Alert.status == status_filter)
    if alert_type:
        conditions.append(Alert.alert_type == alert_type)

    where_clause = and_(*conditions) if conditions else True

    # Count
    count_result = await db.execute(
        select(func.count(Alert.id)).where(where_clause)
    )
    total = count_result.scalar() or 0

    # Fetch
    result = await db.execute(
        select(Alert)
        .where(where_clause)
        .order_by(desc(Alert.created_at))
        .offset(skip)
        .limit(limit)
    )
    alerts = list(result.scalars().all())

    return alerts, total


async def get_open_alerts_for_student(
    db: AsyncSession,
    student_id: str,
) -> List[Alert]:
    """Get all open or acknowledged alerts for a student."""
    result = await db.execute(
        select(Alert)
        .where(
            and_(
                Alert.student_id == student_id,
                Alert.status.in_(["open", "acknowledged"]),
            )
        )
        .order_by(desc(Alert.created_at))
    )
    return list(result.scalars().all())


async def count_open_alerts(db: AsyncSession) -> int:
    """Count all alerts with status 'open'."""
    result = await db.execute(
        select(func.count(Alert.id)).where(Alert.status == "open")
    )
    return result.scalar() or 0


# ── Alert Status Transitions ────────────────────────────────────


async def update_alert_status(
    db: AsyncSession,
    alert_id: str,
    new_status: str,
    *,
    updated_by: Optional[str] = None,
) -> Alert:
    """Transition an alert to a new status.

    Validates the transition against the state machine.

    Args:
        db: Async database session.
        alert_id: The alert to update.
        new_status: Target status.
        updated_by: User ID performing the transition (for audit).

    Returns:
        The updated Alert.

    Raises:
        HTTPException 404: Alert not found.
        HTTPException 422: Invalid status transition.
    """
    alert = await get_alert_by_id(db, alert_id)
    if not alert:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Alert with id={alert_id} not found.",
        )

    allowed = VALID_TRANSITIONS.get(alert.status, [])
    if new_status not in allowed:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Cannot transition alert from '{alert.status}' to '{new_status}'. "
                f"Allowed transitions: {allowed}"
            ),
        )

    old_status = alert.status
    alert.status = new_status
    await db.commit()
    await db.refresh(alert)

    logger.info(
        f"Alert status updated: id={alert_id}, "
        f"{old_status} → {new_status}, by={updated_by}"
    )

    return alert


async def bulk_acknowledge(
    db: AsyncSession,
    alert_ids: List[str],
) -> int:
    """Acknowledge multiple alerts at once.

    Only transitions alerts that are currently 'open'.

    Args:
        db: Async database session.
        alert_ids: List of alert IDs to acknowledge.

    Returns:
        Number of alerts actually updated.
    """
    result = await db.execute(
        update(Alert)
        .where(
            and_(
                Alert.id.in_(alert_ids),
                Alert.status == "open",
            )
        )
        .values(status="acknowledged")
    )
    await db.commit()
    count = result.rowcount  # type: ignore[union-attr]
    logger.info(f"Bulk acknowledge: {count}/{len(alert_ids)} alerts updated")
    return count


# ── Teacher Notification ─────────────────────────────────────────


async def get_teachers_for_student(
    db: AsyncSession,
    student_id: str,
) -> List[User]:
    """Find teacher/admin users who should be notified about a student.

    Current strategy: notify all users with role 'teacher' or 'admin'.
    Future: scope to teachers assigned to the student's school/class.

    Args:
        db: Async database session.
        student_id: The student ID (used for future school-scoping).

    Returns:
        List of teacher/admin User objects.
    """
    result = await db.execute(
        select(User).where(User.role.in_(["teacher", "admin"]))
    )
    return list(result.scalars().all())


async def notify_teachers(
    db: AsyncSession,
    alert: Alert,
) -> List[Dict]:
    """Dispatch notifications to teachers for a given alert.

    Current implementation: creates in-app notification records
    (logged + returned).  Future phases will add email, SMS,
    push notifications, and webhook integrations.

    Args:
        db: Async database session.
        alert: The alert to notify about.

    Returns:
        List of notification dicts with delivery status.
    """
    teachers = await get_teachers_for_student(db, alert.student_id)

    notifications = []
    for teacher in teachers:
        notification = {
            "teacher_id": teacher.id,
            "teacher_name": teacher.name,
            "teacher_email": teacher.email,
            "alert_id": alert.id,
            "student_id": alert.student_id,
            "alert_type": alert.alert_type,
            "risk_score": alert.risk_score,
            "message": alert.message,
            "channel": "in_app",
            "status": "delivered",
            "delivered_at": datetime.now(timezone.utc).isoformat(),
        }
        notifications.append(notification)

        logger.info(
            f"Notification sent: teacher={teacher.id} ({teacher.email}), "
            f"alert={alert.id}, channel=in_app"
        )

    # Future: persist Notification records to a notifications table
    # Future: dispatch via email/SMS/push in background task

    return notifications


async def generate_and_notify(
    db: AsyncSession,
    student_id: str,
    risk_score: int,
    risk_level: str,
) -> Optional[Dict]:
    """One-call convenience: generate alert + notify teachers.

    Called by the risk-scoring pipeline after each assessment.

    Args:
        db: Async database session.
        student_id: Student ID.
        risk_score: Composite risk score.
        risk_level: Risk level string.

    Returns:
        Dict with alert and notification info, or None if no alert needed.
    """
    alert = await generate_alert_from_risk(db, student_id, risk_score, risk_level)
    if alert is None:
        return None

    notifications = await notify_teachers(db, alert)

    return {
        "alert_id": alert.id,
        "student_id": student_id,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "alert_status": alert.status,
        "notifications_sent": len(notifications),
        "notifications": notifications,
    }
