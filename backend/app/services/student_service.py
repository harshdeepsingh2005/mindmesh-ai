"""Student service — business logic for student profiles and behavioral records."""

import uuid
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

from fastapi import HTTPException, status
from sqlalchemy import select, func, and_
from sqlalchemy.ext.asyncio import AsyncSession

from ..models.student import Student
from ..models.behavioral_record import BehavioralRecord
from ..models.user import User
from ..database.schemas import (
    StudentCreate,
    StudentUpdate,
    StudentDetailResponse,
    MoodCheckinRequest,
    JournalEntryRequest,
)
from ..utils.sanitize import sanitize_text, sanitize_identifier, sanitize_name
from ..logging_config import logger


class StudentService:
    """Service layer for student-related operations."""

    def __init__(self, db: AsyncSession):
        self.db = db

    # ─── Profile CRUD ───────────────────────────────────────

    async def create_profile(self, payload: StudentCreate, user_id: str = None) -> Student:
        """Create a new student profile."""
        # Check for duplicate student_identifier
        existing = await self.db.execute(
            select(Student).where(
                Student.student_identifier == sanitize_identifier(payload.student_identifier)
            )
        )
        if existing.scalar_one_or_none():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Student identifier '{payload.student_identifier}' already exists.",
            )

        student = Student(
            id=str(uuid.uuid4()),
            user_id=user_id,
            student_identifier=sanitize_identifier(payload.student_identifier),
            age=payload.age,
            school=sanitize_name(payload.school),
            grade=sanitize_name(payload.grade),
            guardian_contact=payload.guardian_contact,
        )
        self.db.add(student)
        await self.db.commit()
        await self.db.refresh(student)
        return student

    async def list_profiles(
        self,
        skip: int = 0,
        limit: int = 20,
        school: Optional[str] = None,
        grade: Optional[str] = None,
    ) -> Tuple[List[Student], int]:
        """List student profiles with optional filters."""
        query = select(Student)
        count_query = select(func.count(Student.id))

        if school:
            query = query.where(Student.school.ilike(f"%{sanitize_name(school)}%"))
            count_query = count_query.where(Student.school.ilike(f"%{sanitize_name(school)}%"))
        if grade:
            query = query.where(Student.grade == sanitize_name(grade))
            count_query = count_query.where(Student.grade == sanitize_name(grade))

        total_result = await self.db.execute(count_query)
        total = total_result.scalar() or 0

        result = await self.db.execute(query.offset(skip).limit(limit))
        students = list(result.scalars().all())
        return students, total

    async def get_profile_detail(
        self, student_id: str, current_user: User
    ) -> StudentDetailResponse:
        """Get detailed student profile with aggregated info."""
        student = await self._get_student_or_404(student_id)

        # Students can only view their own profile
        if current_user.role == "student":
            if not student.user_id or student.user_id != current_user.id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="You can only view your own profile.",
                )

        # Count behavioral records
        record_count_result = await self.db.execute(
            select(func.count(BehavioralRecord.id)).where(
                BehavioralRecord.student_id == student_id
            )
        )
        record_count = record_count_result.scalar() or 0

        # Get latest risk level (if any risk scores exist)
        latest_risk_level = None
        if student.risk_scores:
            latest_risk = max(student.risk_scores, key=lambda r: r.calculated_at)
            latest_risk_level = latest_risk.risk_level

        # Get user info if linked
        user_name = None
        user_email = None
        if student.user_id:
            user_result = await self.db.execute(
                select(User).where(User.id == student.user_id)
            )
            user = user_result.scalar_one_or_none()
            if user:
                user_name = user.name
                user_email = user.email

        return StudentDetailResponse(
            id=student.id,
            user_id=student.user_id or "",
            student_identifier=student.student_identifier,
            age=student.age,
            school=student.school,
            grade=student.grade,
            guardian_contact=student.guardian_contact,
            user_name=user_name,
            user_email=user_email,
            behavioral_record_count=record_count,
            latest_risk_level=latest_risk_level,
        )

    async def update_profile(
        self, student_id: str, payload: StudentUpdate
    ) -> Student:
        """Update a student profile."""
        student = await self._get_student_or_404(student_id)

        update_data = payload.model_dump(exclude_unset=True)
        if "school" in update_data and update_data["school"]:
            update_data["school"] = sanitize_name(update_data["school"])
        if "grade" in update_data and update_data["grade"]:
            update_data["grade"] = sanitize_name(update_data["grade"])

        for field, value in update_data.items():
            setattr(student, field, value)

        await self.db.commit()
        await self.db.refresh(student)
        return student

    async def delete_profile(self, student_id: str) -> None:
        """Delete a student profile and all related records (cascade)."""
        student = await self._get_student_or_404(student_id)
        await self.db.delete(student)
        await self.db.commit()

    # ─── Mood Check-in ──────────────────────────────────────

    async def create_mood_checkin(
        self, current_user: User, payload: MoodCheckinRequest
    ) -> BehavioralRecord:
        """Create a behavioral record from a mood check-in."""
        student = await self._get_student_for_user(current_user)

        # Normalize mood rating to 0-1 emotion score
        emotion_score = payload.mood_rating / 10.0

        record = BehavioralRecord(
            id=str(uuid.uuid4()),
            student_id=student.id,
            text_input=sanitize_text(payload.notes) if payload.notes else None,
            activity_type="checkin",
            emotion_score=emotion_score,
        )
        self.db.add(record)
        await self.db.commit()
        await self.db.refresh(record)

        # Trigger AI analysis if text is present
        if record.text_input:
            await self._run_ai_analysis(record)

        return record

    # ─── Journal Entry ──────────────────────────────────────

    async def create_journal_entry(
        self, current_user: User, payload: JournalEntryRequest
    ) -> BehavioralRecord:
        """Create a behavioral record from a journal entry."""
        student = await self._get_student_for_user(current_user)

        record = BehavioralRecord(
            id=str(uuid.uuid4()),
            student_id=student.id,
            text_input=sanitize_text(payload.text),
            activity_type="journal",
        )
        self.db.add(record)
        await self.db.commit()
        await self.db.refresh(record)

        # Trigger AI analysis on journal text
        if record.text_input:
            await self._run_ai_analysis(record)

        return record

    # ─── Behavioral Records ─────────────────────────────────

    async def get_records(
        self,
        student_id: str,
        current_user: User,
        skip: int = 0,
        limit: int = 20,
        activity_type: Optional[str] = None,
    ) -> Tuple[List[BehavioralRecord], int]:
        """Get behavioral records for a student."""
        student = await self._get_student_or_404(student_id)

        # Students can only view their own records
        if current_user.role == "student":
            if not student.user_id or student.user_id != current_user.id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="You can only view your own records.",
                )

        query = select(BehavioralRecord).where(
            BehavioralRecord.student_id == student_id
        )
        count_query = select(func.count(BehavioralRecord.id)).where(
            BehavioralRecord.student_id == student_id
        )

        if activity_type:
            query = query.where(BehavioralRecord.activity_type == activity_type)
            count_query = count_query.where(
                BehavioralRecord.activity_type == activity_type
            )

        query = query.order_by(BehavioralRecord.timestamp.desc())

        total_result = await self.db.execute(count_query)
        total = total_result.scalar() or 0

        result = await self.db.execute(query.offset(skip).limit(limit))
        records = list(result.scalars().all())
        return records, total

    # ─── Wellbeing History ──────────────────────────────────

    async def get_wellbeing_history(
        self, student_id: str, current_user: User, days: int = 30
    ) -> dict:
        """Get aggregated wellbeing history for a student."""
        student = await self._get_student_or_404(student_id)

        # Students can only view their own history
        if current_user.role == "student":
            if not student.user_id or student.user_id != current_user.id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="You can only view your own history.",
                )

        since = datetime.utcnow() - timedelta(days=days)

        # Fetch records in the time window
        result = await self.db.execute(
            select(BehavioralRecord)
            .where(
                and_(
                    BehavioralRecord.student_id == student_id,
                    BehavioralRecord.timestamp >= since,
                )
            )
            .order_by(BehavioralRecord.timestamp.asc())
        )
        records = list(result.scalars().all())

        # Aggregate
        checkins = [r for r in records if r.activity_type == "checkin"]
        journals = [r for r in records if r.activity_type == "journal"]

        avg_emotion = None
        if checkins:
            scores = [r.emotion_score for r in checkins if r.emotion_score is not None]
            avg_emotion = round(sum(scores) / len(scores), 3) if scores else None

        avg_sentiment = None
        sentiment_scores = [
            r.sentiment_score for r in records if r.sentiment_score is not None
        ]
        if sentiment_scores:
            avg_sentiment = round(
                sum(sentiment_scores) / len(sentiment_scores), 3
            )

        analyzed_count = sum(
            1 for r in records if r.emotion_analysis is not None
        )

        return {
            "student_id": student_id,
            "period_days": days,
            "total_records": len(records),
            "total_checkins": len(checkins),
            "total_journals": len(journals),
            "analyzed_records": analyzed_count,
            "average_emotion_score": avg_emotion,
            "average_sentiment_score": avg_sentiment,
            "latest_risk_level": (
                student.risk_scores[-1].risk_level if student.risk_scores else None
            ),
            "active_alerts": len(
                [a for a in student.alerts if a.status == "active"]
            ),
        }

    # ─── AI Analysis Integration ────────────────────────────

    async def _run_ai_analysis(self, record: BehavioralRecord) -> None:
        """Run the AI analysis pipeline on a behavioral record.

        Catches and logs errors to avoid breaking the main submission flow.
        """
        try:
            from .ai_analysis import analyze_record
            await analyze_record(self.db, record)
        except Exception as e:
            logger.error(
                f"AI analysis failed for record_id={record.id}: {e}",
                exc_info=True,
            )

    # ─── Helpers ────────────────────────────────────────────

    async def _get_student_or_404(self, student_id: str) -> Student:
        """Fetch a student by ID or raise 404."""
        result = await self.db.execute(
            select(Student).where(Student.id == student_id)
        )
        student = result.scalar_one_or_none()
        if not student:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Student with id '{student_id}' not found.",
            )
        return student

    async def _get_student_for_user(self, user: User) -> Student:
        """Get the student profile linked to a user account."""
        result = await self.db.execute(
            select(Student).where(Student.user_id == user.id)
        )
        student = result.scalar_one_or_none()
        if not student:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No student profile linked to your account. Contact an admin.",
            )
        return student
