import asyncio
import random
from datetime import datetime, timedelta
from app.database.db import AsyncSessionLocal, init_db
from app.models.user import User
from app.models.student import Student
from app.models.risk_score import RiskScore
from app.models.alert import Alert

async def seed():
    # Ensure tables exist
    await init_db()

    async with AsyncSessionLocal() as session:
        print("Seeding database with 100 synthetic students and risk data...")
        
        # Create 100 students
        for i in range(1, 101):
            email = f"synth_student{i}@mindmesh.edu"
            
            # User
            user = User(
                name=f"Student {i}",
                email=email,
                role="student",
                password_hash="fakehash"
            )
            session.add(user)
            await session.flush()
            
            # Student Profile
            student = Student(
                user_id=user.id,
                student_identifier=f"STU-{i:04d}",
                age=random.randint(14, 18),
                school="MindMesh High",
                grade=str(random.randint(9, 12))
            )
            session.add(student)
            await session.flush()
            
            # Simulated Risk Distribution (10% High, 20% Med, 70% Low)
            rand = random.random()
            is_high_risk = rand < 0.1
            is_med_risk = 0.1 <= rand < 0.3
            risk_level = "HIGH" if is_high_risk else ("MEDIUM" if is_med_risk else "LOW")
            risk_val = random.randint(75, 100) if is_high_risk else (random.randint(40, 74) if is_med_risk else random.randint(0, 39))
            
            risk = RiskScore(
                student_id=student.id,
                risk_score=risk_val,
                risk_level=risk_level,
            )
            session.add(risk)
            
            # Create Alerts for High Risk students
            if is_high_risk:
                alert = Alert(
                    student_id=student.id,
                    risk_score=risk_val,
                    alert_type="SYSTEM",
                    message="Anomaly score indicates significantly elevated unsupervised distress.",
                    status="open"
                )
                session.add(alert)
                
        await session.commit()
        print("Database seeded completely! Dashboard will now reflect Live Analytics.")

if __name__ == "__main__":
    asyncio.run(seed())
