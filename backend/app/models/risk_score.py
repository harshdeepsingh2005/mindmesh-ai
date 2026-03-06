"""Risk score model."""

import uuid
from datetime import datetime
from typing import Optional, TYPE_CHECKING

from sqlalchemy import String, Integer, DateTime, ForeignKey, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..database.db import Base

if TYPE_CHECKING:
    from .student import Student


class RiskScore(Base):
    """Calculated risk score for a student."""

    __tablename__ = "risk_scores"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    student_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("students.id"), nullable=False, index=True
    )
    risk_score: Mapped[int] = mapped_column(Integer, nullable=False)
    risk_level: Mapped[str] = mapped_column(String(50), nullable=False)
    contributing_factors: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    calculated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False, index=True
    )

    # Relationships
    student: Mapped["Student"] = relationship(
        "Student", back_populates="risk_scores"
    )

    def __repr__(self) -> str:
        return f"<RiskScore(id={self.id}, student_id={self.student_id}, score={self.risk_score})>"
