"""Async database engine and session for the ML module.

Independent of the backend module — connects directly to the same PostgreSQL
database using the same connection URL. This allows ML scripts to run
standalone without importing from backend.
"""

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from ml.src.config import ml_settings

engine = create_async_engine(
    ml_settings.database_url,
    pool_size=5,
    max_overflow=10,
    echo=False,
)

async_session_factory = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_session() -> AsyncSession:
    """Create a new async database session."""
    async with async_session_factory() as session:
        try:
            yield session
        finally:
            await session.close()
