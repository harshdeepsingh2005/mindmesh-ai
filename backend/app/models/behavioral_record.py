"""Behavioral and emotional data record model."""

import uuid
from datetime import datetime
from typing import Optional, TYPE_CHECKING

from sqlalchemy import String, DateTime, Text, Float, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..database.db import Base

if TYPE_CHECKING:
    from .student import Student
    from .emotion_analysis import EmotionAnalysis


class BehavioralRecord(Base):
    """Behavioral and emotional data record for a student."""

    __tablename__ = "behavioral_records"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    student_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("students.id"), nullable=False, index=True
    )
    timestamp: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False, index=True
    )
    text_input: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    activity_type: Mapped[str] = mapped_column(String(100), nullable=False)
    emotion_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    sentiment_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Relationships
    student: Mapped["Student"] = relationship(
        "Student", back_populates="behavioral_records"
    )
    emotion_analysis: Mapped[Optional["EmotionAnalysis"]] = relationship(
        "EmotionAnalysis", back_populates="record", uselist=False, lazy="selectin"
    )

    def __repr__(self) -> str:
        return f"<BehavioralRecord(id={self.id}, student_id={self.student_id}, type={self.activity_type})>"
