# MindMesh AI Data Schema

## 1. Overview

The MindMesh AI database is designed to securely store user, student, behavioral, emotional, and risk assessment data for the platform. The schema supports scalable analytics, machine learning, and privacy requirements for educational environments.

## 2. Core Tables

### Users
| Field          | Type         | Description                       |
|----------------|--------------|-----------------------------------|
| id             | UUID (PK)    | Unique user ID                    |
| name           | VARCHAR      | Full name                         |
| email          | VARCHAR      | Email address (unique)            |
| role           | VARCHAR      | student, teacher, admin           |
| password_hash  | VARCHAR      | Hashed password                   |
| created_at     | TIMESTAMP    | Account creation time             |

### Students
| Field             | Type         | Description                       |
|-------------------|--------------|-----------------------------------|
| id                | UUID (PK)    | Unique student record ID          |
| student_identifier| VARCHAR      | School-issued student ID (unique) |
| age               | INTEGER      | Age of student                    |
| school            | VARCHAR      | School name                       |
| grade             | VARCHAR      | Grade or class                    |
| guardian_contact  | VARCHAR      | Guardian contact info             |

### BehavioralRecords
| Field          | Type         | Description                       |
|----------------|--------------|-----------------------------------|
| id             | UUID (PK)    | Unique record ID                  |
| student_id     | UUID (FK)    | Linked student                    |
| timestamp      | TIMESTAMP    | Time of record                    |
| text_input     | TEXT         | Student's journal or input        |
| activity_type  | VARCHAR      | Type of activity (survey, journal, etc.) |
| emotion_score  | FLOAT        | Numeric emotion score              |
| sentiment_score| FLOAT        | Sentiment analysis score           |

### EmotionAnalysis
| Field            | Type         | Description                       |
|------------------|--------------|-----------------------------------|
| id               | UUID (PK)    | Unique analysis ID                |
| record_id        | UUID (FK)    | Linked behavioral record          |
| predicted_emotion| VARCHAR      | Predicted emotion label           |
| confidence_score | FLOAT        | Model confidence                  |
| model_version    | VARCHAR      | ML model version                  |

### RiskScores
| Field               | Type         | Description                       |
|---------------------|--------------|-----------------------------------|
| id                  | UUID (PK)    | Unique risk score ID              |
| student_id          | UUID (FK)    | Linked student                    |
| risk_score          | INTEGER      | Calculated risk score (0-100)     |
| risk_level          | VARCHAR      | low, medium, high                 |
| contributing_factors| JSONB        | List of factors                   |
| calculated_at       | TIMESTAMP    | Time of calculation               |

### Alerts
| Field          | Type         | Description                       |
|----------------|--------------|-----------------------------------|
| id             | UUID (PK)    | Unique alert ID                   |
| student_id     | UUID (FK)    | Linked student                    |
| risk_score     | INTEGER      | Risk score at alert time          |
| alert_type     | VARCHAR      | high_risk, info, etc.             |
| message        | TEXT         | Alert message                     |
| created_at     | TIMESTAMP    | Alert creation time               |
| status         | VARCHAR      | open, resolved, dismissed         |

## 3. Relationships

- **Students → BehavioralRecords:** One-to-many (a student has many records)
- **BehavioralRecords → EmotionAnalysis:** One-to-one (each record has one analysis)
- **Students → RiskScores:** One-to-many (a student has many risk scores)
- **Students → Alerts:** One-to-many (a student has many alerts)

## 4. Indexing Strategy

- Unique indexes on `Users.email` and `Students.student_identifier` for fast lookups.
- Indexes on `BehavioralRecords.student_id`, `RiskScores.student_id`, and `Alerts.student_id` for efficient joins and queries.
- Timestamp indexes on `BehavioralRecords.timestamp` and `RiskScores.calculated_at` for time-based analytics.

## 5. Data Retention and Privacy

- All sensitive fields (PII, guardian_contact, password_hash) are encrypted at rest.
- Access to student and risk data is role-restricted (teachers/admins only).
- Data retention policies comply with educational privacy laws (e.g., FERPA, GDPR).
- Audit logs are maintained for all access and changes to sensitive data.
