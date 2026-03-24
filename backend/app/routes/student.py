"""Student monitoring routes — CRUD, mood check-ins, journal entries, history."""

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional

from ..database.db import get_db
from ..database.schemas import (
    StudentCreate,
    StudentUpdate,
    StudentResponse,
    StudentDetailResponse,
    StudentListResponse,
    MoodCheckinRequest,
    JournalEntryRequest,
    BehavioralRecordResponse,
    BehavioralRecordListResponse,
    SOSRequest,
    PeerReportRequest,
    AlertResponse,
)
from ..dependencies import get_current_user, require_roles
from ..models.user import User
from ..services.student_service import StudentService

router = APIRouter()


# ─── Student Profile CRUD ───────────────────────────────────


@router.post(
    "/profiles",
    response_model=StudentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create student profile",
)
async def create_student_profile(
    payload: StudentCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_roles(["admin", "teacher"])),
):
    """Create a new student profile (admin/teacher only)."""
    svc = StudentService(db)
    student = await svc.create_profile(payload, user_id=current_user.id)
    return student


@router.get(
    "/profiles",
    response_model=StudentListResponse,
    summary="List student profiles",
)
async def list_student_profiles(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    school: Optional[str] = None,
    grade: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_roles(["admin", "teacher"])),
):
    """List all student profiles with optional filters (admin/teacher only)."""
    svc = StudentService(db)
    students, total = await svc.list_profiles(
        skip=skip, limit=limit, school=school, grade=grade
    )
    return StudentListResponse(students=students, total=total)


@router.get(
    "/profiles/{student_id}",
    response_model=StudentDetailResponse,
    summary="Get student profile detail",
)
async def get_student_profile(
    student_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_roles(["admin", "teacher", "student"])),
):
    """Get detailed student profile. Students can only view their own profile."""
    svc = StudentService(db)
    student = await svc.get_profile_detail(student_id, current_user)
    return student


@router.put(
    "/profiles/{student_id}",
    response_model=StudentResponse,
    summary="Update student profile",
)
async def update_student_profile(
    student_id: str,
    payload: StudentUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_roles(["admin", "teacher"])),
):
    """Update an existing student profile (admin/teacher only)."""
    svc = StudentService(db)
    student = await svc.update_profile(student_id, payload)
    return student


@router.delete(
    "/profiles/{student_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete student profile",
)
async def delete_student_profile(
    student_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_roles(["admin"])),
):
    """Delete a student profile and all related records (admin only)."""
    svc = StudentService(db)
    await svc.delete_profile(student_id)
    return None


# ─── Mood Check-in ──────────────────────────────────────────


@router.post(
    "/checkin",
    response_model=BehavioralRecordResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Submit mood check-in",
)
async def submit_mood_checkin(
    payload: MoodCheckinRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_roles(["student"])),
):
    """Student submits a mood check-in (creates a behavioral record)."""
    svc = StudentService(db)
    record = await svc.create_mood_checkin(current_user, payload)
    return record


# ─── Journal Entry ──────────────────────────────────────────


@router.post(
    "/journal",
    response_model=BehavioralRecordResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Submit journal entry",
)
async def submit_journal_entry(
    payload: JournalEntryRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_roles(["student"])),
):
    """Student submits a journal entry (creates a behavioral record)."""
    svc = StudentService(db)
    record = await svc.create_journal_entry(current_user, payload)
    return record


# ─── Behavioral Records ────────────────────────────────────


@router.get(
    "/records/{student_id}",
    response_model=BehavioralRecordListResponse,
    summary="Get behavioral records for a student",
)
async def get_behavioral_records(
    student_id: str,
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    activity_type: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_roles(["admin", "teacher", "student"])),
):
    """Retrieve behavioral records for a student. Students can only view their own."""
    svc = StudentService(db)
    records, total = await svc.get_records(
        student_id, current_user, skip=skip, limit=limit, activity_type=activity_type
    )
    return BehavioralRecordListResponse(records=records, total=total)


# ─── Wellbeing History ──────────────────────────────────────


@router.post(
    "/sos",
    response_model=AlertResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Trigger SOS alert",
)
async def submit_sos(
    payload: SOSRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_roles(["student"])),
):
    """Student triggers an immediate SOS alert for counseling."""
    svc = StudentService(db)
    alert = await svc.trigger_sos(current_user, payload)
    return alert


@router.post(
    "/report_peer",
    response_model=AlertResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Anonymous peer intervention",
)
async def report_peer_concern(
    payload: PeerReportRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_roles(["student"])),
):
    """Student anonymously reports a concern regarding a peer."""
    svc = StudentService(db)
    alert = await svc.report_peer(current_user, payload)
    return alert


@router.get(
    "/history/{student_id}",
    summary="Retrieve wellbeing history",
)
async def retrieve_wellbeing_history(
    student_id: str,
    days: int = Query(30, ge=1, le=365),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_roles(["admin", "teacher", "student"])),
):
    """Retrieve aggregated wellbeing history for a student."""
    svc = StudentService(db)
    history = await svc.get_wellbeing_history(student_id, current_user, days=days)
    return history
