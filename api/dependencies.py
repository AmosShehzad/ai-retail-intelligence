"""
Shared dependencies injected into FastAPI endpoints.
Using FastAPI's dependency injection system means:
- DB connection is opened per request
- Always properly closed after response (even on errors)
- Easy to swap (e.g., switch from SQLite to PostgreSQL later)
"""

import sqlite3
from typing import Generator
from database.db_manager import get_connection

def get_db() -> Generator:
    """
    Dependency that provides a DB connection per request.
    
    'yield' makes this a context manager:
    - Code before yield = setup (open connection)
    - Code after yield = teardown (close connection)
    FastAPI handles calling both sides automatically.
    """
    conn = get_connection()
    try:
        yield conn
    finally:
        conn.close()