"""MindMesh AI — Database Connection & Session Management."""

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase

from ..config import settings
from ..logging_config import logger


# Async engine
engine = create_async_engine(
    settings.effective_database_url,
    echo=settings.DEBUG,
    future=True,
)

# Async session factory
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    """Base class for all database models."""
    pass


async def get_db() -> AsyncSession:
    """Dependency that provides an async database session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


async def init_db() -> None:
    """Create all database tables and seed initial demo accounts."""
    from sqlalchemy import select
    from ..models.user import User
    from ..services.auth import hash_password
    
    logger.info("Initializing database tables...")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables created successfully.")
    
    # Run user data seeding
    async with AsyncSessionLocal() as session:
        try:
            # Check if users already exist
            result = await session.execute(select(User).limit(1))
            if result.scalar_one_or_none() is None:
                logger.info("Database empty, seeding 5 admin and 5 student accounts...")
                
                users_to_add = []
                # 5 Admin accounts
                for i in range(1, 6):
                    admin = User(
                        id=f"admin-demo-uuid-{i}",
                        email=f"admin{i}@mindmesh.ai",
                        password_hash=hash_password("admin123"),
                        full_name=f"Demo Admin {i}",
                        role="admin",
                        is_active=True
                    )
                    users_to_add.append(admin)
                
                # 5 Student accounts
                for i in range(1, 6):
                    student = User(
                        id=f"student-demo-uuid-{i}",
                        email=f"student{i}@mindmesh.ai",
                        password_hash=hash_password("mindmesh2026!"),
                        full_name=f"Demo Student {i}",
                        role="student",
                        is_active=True
                    )
                    users_to_add.append(student)
                
                session.add_all(users_to_add)
                await session.commit()
                logger.info("Seeded 10 user accounts (5 admin, 5 student).")
            else:
                logger.info("Database already seeded. Skipping initial seeding.")
        except Exception as e:
            logger.error(f"Failed to seed users: {e}")

async def close_db() -> None:
    """Close the database engine."""
    logger.info("Closing database connection...")
    await engine.dispose()
    logger.info("Database connection closed.")
