"""
Database Connection and Session Management.
Uses SQLAlchemy with async SQLite for hackathon simplicity.
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy.pool import StaticPool

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Create declarative base
Base = declarative_base()

# Database engine (will be initialized on startup)
_engine = None
_async_session_factory = None


async def init_db():
    """Initialize the database connection and create tables."""
    global _engine, _async_session_factory
    
    # Ensure data directory exists
    Path("./data").mkdir(parents=True, exist_ok=True)
    
    # Create async engine
    # Using StaticPool for SQLite to handle async properly
    _engine = create_async_engine(
        settings.DATABASE_URL,
        echo=settings.DATABASE_ECHO,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool
    )
    
    # Create session factory
    _async_session_factory = async_sessionmaker(
        bind=_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autocommit=False,
        autoflush=False
    )
    
    # Import models to register them with Base
    from app.db import models  # noqa
    
    # Create all tables
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    logger.info(f"Database initialized: {settings.DATABASE_URL}")


async def close_db():
    """Close database connections."""
    global _engine
    
    if _engine:
        await _engine.dispose()
        logger.info("Database connections closed")


@asynccontextmanager
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Get a database session."""
    if not _async_session_factory:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    
    session = _async_session_factory()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


async def get_session() -> AsyncSession:
    """Get a database session (alternative interface)."""
    if not _async_session_factory:
        raise RuntimeError("Database not initialized")
    
    return _async_session_factory()
