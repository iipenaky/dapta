"""
main.py
-------
DAPTA API — FastAPI application entry point.
All configuration via core/config.py — never hardcode values here.
"""

from __future__ import annotations
import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.config import settings
from core.middleware import ErrorHandlerMiddleware
from database import init_db
from routers import auth, assessment, recommendations, sessions
from services.dapta_service import dapta_service

logging.basicConfig(
    level=logging.DEBUG if not settings.is_production else logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger("dapta.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting DAPTA API (environment=%s)", settings.environment)
    await init_db()
    logger.info("Loading DAPTA ML components…")
    dapta_service.load()
    logger.info("DAPTA API ready.")
    yield
    logger.info("DAPTA API shutting down.")


def create_app() -> FastAPI:
    app = FastAPI(
        title="DAPTA API",
        description="Discourse-Aware Personalised Therapy Agent",
        version="1.0.0",
        lifespan=lifespan,
        docs_url=None if settings.is_production else "/docs",
        redoc_url=None if settings.is_production else "/redoc",
    )

    # Middleware — outermost runs first
    app.add_middleware(ErrorHandlerMiddleware, debug=not settings.is_production)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routers
    app.include_router(auth.router,            prefix="/api/auth",            tags=["Auth"])
    app.include_router(sessions.router,        prefix="/api/sessions",        tags=["Sessions"])
    app.include_router(assessment.router,      prefix="/api/assessment",      tags=["Assessment"])
    app.include_router(recommendations.router, prefix="/api/recommendations", tags=["Recommendations"])

    @app.get("/api/health", tags=["Meta"])
    async def health():
        return {"status": "ok", "service": "DAPTA API", "environment": settings.environment}

    return app


app = create_app()

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=not settings.is_production,
    )
