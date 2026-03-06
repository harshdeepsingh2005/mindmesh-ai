# MindMesh AI API Specification

## 1. Introduction

The MindMesh AI API provides secure, scalable endpoints for collecting, analyzing, and reporting on behavioral and emotional signals from students in educational environments. The API enables integration with web/mobile apps, dashboards, and third-party systems, supporting early risk detection and intervention for student mental health.

## 2. Base API Configuration

- **Base URL:** `https://api.mindmesh.ai/v1/`
- **Authentication:** JWT (JSON Web Token) in `Authorization: Bearer <token>` header
- **Request Format:** JSON
- **Response Format:** JSON

## 3. Core API Endpoints

### User Management

| Endpoint                | Method | Description                |
|------------------------|--------|----------------------------|
| /auth/register         | POST   | Register a new user        |
| /auth/login            | POST   | User login                 |
| /users/me              | GET    | Get current user profile   |
| /users/me              | PUT    | Update user profile        |

### Student Monitoring

| Endpoint                        | Method | Description                        |
|----------------------------------|--------|------------------------------------|
| /students                       | POST   | Create student profile             |
| /students/{student_id}          | GET    | Retrieve student data              |
| /students/{student_id}/records  | PUT    | Update student behavioral records  |

### Emotion Analysis

| Endpoint                | Method | Description                |
|------------------------|--------|----------------------------|
| /emotion/analyze       | POST   | Analyze text/behavioral input for emotion |

#### Example Request
```json
{
	"student_id": "abc123",
	"text_input": "I feel anxious about exams.",
	"timestamp": "2026-03-06T09:30:00Z"
}
```

#### Example Response
```json
{
	"detected_emotion": "anxiety",
	"emotion_confidence": 0.92,
	"analysis_metadata": {
		"model_version": "1.2.0",
		"processing_time_ms": 45
	}
}
```

### Risk Scoring

| Endpoint                | Method | Description                |
|------------------------|--------|----------------------------|
| /risk/score            | POST   | Calculate risk score for student |

#### Example Request
```json
{
	"student_id": "abc123"
}
```

#### Example Response
```json
{
	"risk_score": 78,
	"risk_category": "high",
	"explanation_factors": [
		"Low mood trend",
		"High absence rate",
		"Negative sentiment in journal"
	]
}
```

### Alert System

| Endpoint                | Method | Description                |
|------------------------|--------|----------------------------|
| /alerts/generate        | POST   | Generate alert if risk exceeds threshold |

#### Example Request
```json
{
	"student_id": "abc123",
	"risk_score": 78,
	"alert_type": "high_risk"
}
```

#### Example Response
```json
{
	"alert_id": "alert789",
	"status": "created",
	"message": "High risk detected for student abc123. Counselor notified."
}
```

### Dashboard Analytics

| Endpoint                | Method | Description                |
|------------------------|--------|----------------------------|
| /dashboard/analytics    | GET    | Retrieve aggregated analytics for teachers/admins |

#### Example Response
```json
{
	"total_students": 500,
	"students_at_risk": 23,
	"average_risk_score": 42,
	"risk_distribution": {
		"low": 350,
		"medium": 127,
		"high": 23
	},
	"top_concerns": ["anxiety", "isolation", "stress"]
}
```

## 4. Request and Response Examples

(See above for endpoint-specific examples.)

## 5. Error Handling

All error responses use the following format:
```json
{
	"error": {
		"code": 400,
		"message": "Bad Request: Missing required field 'student_id'"
	}
}
```

| Code | Meaning                |
|------|------------------------|
| 400  | Bad Request            |
| 401  | Unauthorized           |
| 404  | Not Found              |
| 500  | Internal Server Error  |

## 6. Security Considerations

- All endpoints require JWT authentication except /auth/register and /auth/login.
- User roles (student, teacher, admin) restrict access to sensitive endpoints.
- All data in transit is encrypted via HTTPS.
- Sensitive data (PII, risk scores) is never exposed to unauthorized users.
- Audit logs are maintained for all access to student data.
