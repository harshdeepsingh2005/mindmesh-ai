"""Student profile model."""

import uuid
from typing import List, Optional, TYPE_CHECKING

from sqlalchemy import String, Integer, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..database.db import Base

if TYPE_CHECKING:
    from .user import User
    from .behavioral_record import BehavioralRecord
    from .risk_score import RiskScore
    from .alert import Alert


class Student(Base):
    """Student profile model linked to a user account."""

    __tablename__ = "students"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id"), unique=True, nullable=False, index=True
    )
    student_identifier: Mapped[str] = mapped_column(
        String(255), unique=True, nullable=False, index=True
    )
    age: Mapped[int] = mapped_column(Integer, nullable=False)
    school: Mapped[str] = mapped_column(String(255), nullable=False)
    grade: Mapped[str] = mapped_column(String(50), nullable=False)
    guardian_contact: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="student_profile")
    behavioral_records: Mapped[List["BehavioralRecord"]] = relationship(
        "BehavioralRecord",
        back_populates="student",
        lazy="selectin",
        cascade="all, delete-orphan",
    )
    risk_scores: Mapped[List["RiskScore"]] = relationship(
        "RiskScore",
        back_populates="student",
        lazy="selectin",
        cascade="all, delete-orphan",
    )
    alerts: Mapped[List["Alert"]] = relationship(
        "Alert",
        back_populates="student",
        lazy="selectin",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<Student(id={self.id}, user_id={self.user_id}, identifier={self.student_identifier})>"
