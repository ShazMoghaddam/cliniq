"""
ClinIQ — FastAPI application factory.
Run locally: uvicorn cliniq.api.app:app --reload
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from cliniq.config.settings import get_settings
from cliniq.db.database import init_db
from cliniq.api.routers import auth, trials, sites, portfolio

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create all tables on startup (idempotent)."""
    init_db()
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_TITLE,
        version=settings.APP_VERSION,
        description=(
            "ClinIQ — Clinical Trial Site Performance Intelligence API. "
            "All endpoints require a valid JWT Bearer token."
        ),
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],          # Tighten in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(auth.router)
    app.include_router(trials.router)
    app.include_router(sites.router)
    app.include_router(portfolio.router)

    @app.get("/health", tags=["health"])
    def health():
        return {"status": "ok", "version": settings.APP_VERSION}

    return app


app = create_app()
