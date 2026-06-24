"""
ClinIQ — Database connection pool and session factory.
SQLite for dev/demo; PostgreSQL for production (set DATABASE_URL env var).
"""
import os
from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import sessionmaker, Session
from cliniq.db.models import Base

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./cliniq_dev.db")

# SQLite-specific: enable WAL mode and foreign key enforcement
def _configure_sqlite(dbapi_conn, connection_record):
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


def build_engine(url: str = DATABASE_URL, echo: bool = False):
    kwargs = {"echo": echo}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
    engine = create_engine(url, **kwargs)
    if url.startswith("sqlite"):
        event.listen(engine, "connect", _configure_sqlite)
    return engine


def build_session_factory(engine) -> sessionmaker:
    return sessionmaker(bind=engine, autocommit=False, autoflush=False)


# Module-level defaults (overridden in tests via conftest fixtures)
engine = build_engine()
SessionLocal = build_session_factory(engine)


def init_db(eng=None):
    """Create all tables. Safe to call multiple times (CREATE IF NOT EXISTS)."""
    target = eng or engine
    Base.metadata.create_all(bind=target)


def get_db() -> Session:
    """FastAPI dependency: yields a DB session and closes it after the request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
