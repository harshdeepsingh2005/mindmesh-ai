"""Tests for alert system endpoints."""

import pytest
from httpx import AsyncClient

from tests.conftest import auth_header


class TestAlertList:
    """GET /alerts/"""

    async def test_list_empty(self, client: AsyncClient, admin_user):
        resp = await client.get("/alerts/", headers=auth_header(admin_user))
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0
        assert data["alerts"] == []

    async def test_student_cannot_list_all(
        self, client: AsyncClient, student_user
    ):
        resp = await client.get("/alerts/", headers=auth_header(student_user))
        assert resp.status_code == 403


class TestAlertCount:
    """GET /alerts/count"""

    async def test_count_open_alerts(self, client: AsyncClient, admin_user):
        resp = await client.get("/alerts/count", headers=auth_header(admin_user))
        assert resp.status_code == 200
        data = resp.json()
        assert "open_count" in data
        assert data["open_count"] >= 0


class TestManualAlert:
    """POST /alerts/"""

    async def test_create_manual_alert(
        self, client: AsyncClient, admin_user, db_session
    ):
        # Create a student first
        from app.models.student import Student

        student = Student(
            student_identifier="STU-ALERT",
            age=15,
            school="Alert School",
            grade="10th",
        )
        db_session.add(student)
        await db_session.commit()
        await db_session.refresh(student)

        resp = await client.post(
            "/alerts/",
            json={
                "student_id": student.id,
                "risk_score": 85,
                "alert_type": "high_risk",
                "message": "Test alert for high risk student",
            },
            headers=auth_header(admin_user),
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["alert_type"] == "high_risk"
        assert data["risk_score"] == 85
        assert data["status"] == "open"


class TestBulkAcknowledge:
    """POST /alerts/bulk-acknowledge"""

    async def test_bulk_ack_empty_list(self, client: AsyncClient, admin_user):
        resp = await client.post(
            "/alerts/bulk-acknowledge",
            json={"alert_ids": ["nonexistent-id-1"]},
            headers=auth_header(admin_user),
        )
        # Should succeed but acknowledge 0
        assert resp.status_code == 200
        data = resp.json()
        assert data["acknowledged"] == 0

    async def test_bulk_ack_with_real_alerts(
        self, client: AsyncClient, admin_user, db_session
    ):
        from app.models.student import Student
        from app.models.alert import Alert

        student = Student(
            student_identifier="STU-BULK",
            age=14,
            school="Bulk School",
            grade="9th",
        )
        db_session.add(student)
        await db_session.commit()
        await db_session.refresh(student)

        alert = Alert(
            student_id=student.id,
            risk_score=75,
            alert_type="high_risk",
            message="Bulk test alert",
            status="open",
        )
        db_session.add(alert)
        await db_session.commit()
        await db_session.refresh(alert)

        resp = await client.post(
            "/alerts/bulk-acknowledge",
            json={"alert_ids": [alert.id]},
            headers=auth_header(admin_user),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["acknowledged"] == 1
