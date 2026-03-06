"""Emotion analysis result model."""

import uuid
from typing import TYPE_CHECKING

from sqlalchemy import String, Float, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..database.db import Base

if TYPE_CHECKING:
    from .behavioral_record import BehavioralRecord


class EmotionAnalysis(Base):
    """AI emotion analysis result for a behavioral record."""

    __tablename__ = "emotion_analysis"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    record_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("behavioral_records.id"),
        unique=True,
        nullable=False,
        index=True,
    )
    predicted_emotion: Mapped[str] = mapped_column(String(100), nullable=False)
    confidence_score: Mapped[float] = mapped_column(Float, nullable=False)
    model_version: Mapped[str] = mapped_column(String(50), nullable=False)

    # Relationships
    record: Mapped["BehavioralRecord"] = relationship(
        "BehavioralRecord", back_populates="emotion_analysis"
    )

    def __repr__(self) -> str:
        return f"<EmotionAnalysis(id={self.id}, emotion={self.predicted_emotion})>"
