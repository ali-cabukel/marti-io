"""Shared LangGraph checkpointer lifecycle (Postgres, SQLite, or memory)."""

from __future__ import annotations

from typing import Any

import aiosqlite
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from psycopg import AsyncConnection
from psycopg.rows import dict_row
from psycopg.sql import SQL, Identifier

from agents.settings import Settings

_checkpointer: Any | None = None
_pg_conn: AsyncConnection | None = None
_sqlite_conn: aiosqlite.Connection | None = None


async def _prepare_postgres_schema(conn: AsyncConnection, schema: str) -> None:
    ident = Identifier(schema)
    await conn.execute(SQL("CREATE SCHEMA IF NOT EXISTS {}").format(ident))
    await conn.execute(SQL("SET search_path TO {}, public").format(ident))


async def init_checkpointer(settings: Settings):
    """Create and configure the process-wide async checkpointer."""
    global _checkpointer, _pg_conn, _sqlite_conn

    if settings.checkpointer == "postgres":
        _pg_conn = await AsyncConnection.connect(
            settings.database_url,
            autocommit=True,
            prepare_threshold=0,
            row_factory=dict_row,
        )
        schema = settings.effective_database_schema
        if schema:
            await _prepare_postgres_schema(_pg_conn, schema)
        saver = AsyncPostgresSaver(conn=_pg_conn)
        await saver.setup()
        _checkpointer = saver
        return _checkpointer

    if settings.checkpointer == "sqlite":
        settings.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        _sqlite_conn = await aiosqlite.connect(str(settings.sqlite_path))
        saver = AsyncSqliteSaver(_sqlite_conn)
        await saver.setup()
        _checkpointer = saver
        return _checkpointer

    _checkpointer = MemorySaver()
    return _checkpointer


def get_checkpointer():
    if _checkpointer is None:
        raise RuntimeError(
            "Checkpointer not initialized — call init_checkpointer() at startup"
        )
    return _checkpointer


async def shutdown_checkpointer() -> None:
    global _checkpointer, _pg_conn, _sqlite_conn

    _checkpointer = None
    if _pg_conn is not None:
        await _pg_conn.close()
        _pg_conn = None
    if _sqlite_conn is not None:
        await _sqlite_conn.close()
        _sqlite_conn = None
