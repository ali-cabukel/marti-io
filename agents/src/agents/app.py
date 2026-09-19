"""FastAPI application factory."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from agents.api.middleware.logging import PayloadLoggingMiddleware
from agents.api.routers import agents, chat, dashboard, health, ws
from agents.checkpointer import init_checkpointer, shutdown_checkpointer
from agents.limiter import limiter
from agents.logging_setup import configure_logging, get_logger, log_payload
from agents.redis_client import init_redis, shutdown_redis
from agents.settings import Settings, apply_settings_to_env, get_settings
from agents.ws.manager import init_ws_manager

logger = get_logger("agents.app")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    apply_settings_to_env(settings)
    configure_logging(settings)

    static_dir = settings.static_dir
    static_dir.mkdir(parents=True, exist_ok=True)
    index = static_dir / "index.html"
    if not index.exists():
        index.write_text(_fallback_index(), encoding="utf-8")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        log_payload(
            logger,
            "app.startup",
            {
                "env": settings.app_env,
                "checkpointer": settings.checkpointer,
                "log_format": settings.log_format,
                "static_dir": str(settings.static_dir),
            },
        )
        await init_checkpointer(settings)
        redis = await init_redis(settings)
        init_ws_manager(redis)
        log_payload(
            logger,
            "ws.backend",
            {"mode": "redis" if redis is not None else "local"},
        )
        yield
        await shutdown_redis()
        await shutdown_checkpointer()
        log_payload(logger, "app.shutdown", {})

    app = FastAPI(
        title=settings.app_name,
        description="Multi-agent daily assistant API — LangGraph, MCP, and platform integrations.",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs" if settings.expose_docs else None,
        redoc_url="/redoc" if settings.expose_docs else None,
        openapi_url="/openapi.json" if settings.expose_docs else None,
    )

    app.state.limiter = limiter
    app.state.settings = settings
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.add_middleware(SlowAPIMiddleware)
    app.add_middleware(PayloadLoggingMiddleware)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    prefix = settings.api_prefix.rstrip("/")
    app.include_router(health.router, prefix=prefix)
    app.include_router(dashboard.router, prefix=prefix)
    app.include_router(agents.router, prefix=prefix)
    app.include_router(chat.router, prefix=prefix)
    app.include_router(ws.router, prefix=prefix)

    if static_dir.exists():
        assets_dir = static_dir / "assets"
        if assets_dir.is_dir():
            app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

        @app.get("/", include_in_schema=False)
        async def spa_index():
            return FileResponse(static_dir / "index.html")

        @app.get("/{full_path:path}", include_in_schema=False)
        async def spa_fallback(full_path: str):
            if full_path.startswith("api") or full_path in ("docs", "openapi.json", "redoc"):
                raise HTTPException(status_code=404, detail="Not found")
            candidate = static_dir / full_path
            if candidate.is_file():
                return FileResponse(candidate)
            return FileResponse(static_dir / "index.html")

    return app


def _fallback_index() -> str:
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Daily Agent Hub</title>
  <style>
    body { font-family: system-ui, sans-serif; max-width: 40rem; margin: 4rem auto; padding: 0 1rem; }
    code { background: #f4f4f5; padding: 0.1rem 0.3rem; border-radius: 4px; }
  </style>
</head>
<body>
  <h1>Daily Agent Hub</h1>
  <p>API is running. Build the web UI with <code>cd web && npm install && npm run build</code>.</p>
  <p><a href="/api/health">/api/health</a> · <a href="/docs">/docs</a> (dev only)</p>
</body>
</html>
"""
