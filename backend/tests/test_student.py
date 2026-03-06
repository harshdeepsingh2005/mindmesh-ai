"""Tests for student monitoring endpoints."""

import pytest
from httpx import AsyncClient

from tests.conftest import auth_header


class TestStudentProfiles:
    """POST / GET /student/profiles"""

    async def test_create_profile(self, client: AsyncClient, admin_user):
        resp = await client.post(
            "/student/profiles",
            json={
                "student_identifier": "STU-001",
                "age": 14,
                "school": "Test High School",
                "grade": "9th",
            },
            headers=auth_header(admin_user),
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["student_identifier"] == "STU-001"
        assert data["school"] == "Test High School"

    async def test_create_duplicate_profile(self, client: AsyncClient, admin_user):
        payload = {
            "student_identifier": "STU-DUP",
            "age": 15,
            "school": "School",
            "grade": "10th",
        }
        headers = auth_header(admin_user)
        await client.post("/student/profiles", json=payload, headers=headers)
        resp = await client.post("/student/profiles", json=payload, headers=headers)
        # Should get a conflict or validation error
        assert resp.status_code in (409, 400, 422)

    async def test_list_profiles(self, client: AsyncClient, admin_user):
        headers = auth_header(admin_user)
        await client.post(
            "/student/profiles",
            json={
                "student_identifier": "STU-LST",
                "age": 16,
                "school": "List School",
                "grade": "11th",
            },
            headers=headers,
        )
        resp = await client.get("/student/profiles", headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "students" in data
        assert data["total"] >= 1

    async def test_student_cannot_create_profile(
        self, client: AsyncClient, student_user
    ):
        resp = await client.post(
            "/student/profiles",
            json={
                "student_identifier": "STU-NO",
                "age": 14,
                "school": "Nope",
                "grade": "9th",
            },
            headers=auth_header(student_user),
        )
        assert resp.status_code == 403


class TestMoodCheckin:
    """POST /student/checkin — student only."""

    async def test_mood_checkin(self, client: AsyncClient, student_user, db_session):
        # Need a student profile linked to the student_user
        from app.models.student import Student

        student = Student(
            student_identifier="STU-MOOD",
            user_id=student_user["id"],
            age=15,
            school="Mood School",
            grade="10th",
        )
        db_session.add(student)
        await db_session.commit()

        resp = await client.post(
            "/student/checkin",
            json={"mood_rating": 7, "notes": "Feeling pretty good today"},
            headers=auth_header(student_user),
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["activity_type"] == "checkin"


class TestJournalEntry:
    """POST /student/journal — student only."""

    async def test_journal_entry(self, client: AsyncClient, student_user, db_session):
        from app.models.student import Student

        student = Student(
            student_identifier="STU-JRN",
            user_id=student_user["id"],
            age=16,
            school="Journal School",
            grade="11th",
        )
        db_session.add(student)
        await db_session.commit()

        resp = await client.post(
            "/student/journal",
            json={
                "text": "Today I learned about async programming and it was fun!",
                "mood_tag": "happy",
            },
            headers=auth_header(student_user),
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["activity_type"] == "journal"

    async def test_journal_empty_text_rejected(
        self, client: AsyncClient, student_user, db_session
    ):
        from app.models.student import Student

        student = Student(
            student_identifier="STU-EMPTY",
            user_id=student_user["id"],
            age=14,
            school="Empty School",
            grade="9th",
        )
        db_session.add(student)
        await db_session.commit()

        resp = await client.post(
            "/student/journal",
            json={"text": "", "mood_tag": "neutral"},
            headers=auth_header(student_user),
        )
        assert resp.status_code == 422
