"""Alert model."""

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import String, Integer, Text, DateTime, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..database.db import Base

if TYPE_CHECKING:
    from .student import Student


class Alert(Base):
    """Alert generated when a student's risk exceeds thresholds."""

    __tablename__ = "alerts"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    student_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("students.id"), nullable=False, index=True
    )
    risk_score: Mapped[int] = mapped_column(Integer, nullable=False)
    alert_type: Mapped[str] = mapped_column(String(50), nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )
    status: Mapped[str] = mapped_column(
        String(50), default="open", nullable=False
    )

    # Relationships
    student: Mapped["Student"] = relationship(
        "Student", back_populates="alerts"
    )

    def __repr__(self) -> str:
        return f"<Alert(id={self.id}, student_id={self.student_id}, type={self.alert_type})>"
