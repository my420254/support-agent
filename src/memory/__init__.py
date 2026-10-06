from contextlib import AbstractAsyncContextManager

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from memory.sqlite import get_sqlite_saver, get_sqlite_store


def initialize_database() -> AbstractAsyncContextManager[AsyncSqliteSaver]:
    """Return the checkpointer context manager for the configured database.

    P0-P4 use SQLite. P5 switches to Postgres (see memory/postgres.py, restored then).
    """
    return get_sqlite_saver()


def initialize_store():
    """Return the long-term memory store context manager."""
    return get_sqlite_store()


__all__ = ["initialize_database", "initialize_store"]
