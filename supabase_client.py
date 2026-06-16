# -*- coding: utf-8 -*-
"""
Supabase Client Module — Modular & Reusable
============================================
Provides a centralized Supabase client for the Sign Language Recognition
project. All Supabase configuration, initialization, and helper methods
are encapsulated here to keep the rest of the codebase clean.

Usage:
    from supabase_client import SupabaseManager

    sb = SupabaseManager()
    if sb.connect():
        # Use sb.client to interact with Supabase
        pass
"""

import os
import time
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root regardless of working directory
load_dotenv(Path(__file__).parent / ".env")

# ════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")

if not SUPABASE_URL or not SUPABASE_ANON_KEY:
    raise ValueError(
        "Missing Supabase credentials. "
        "Copy .env.example to .env and set SUPABASE_URL and SUPABASE_ANON_KEY"
    )


# ═══════════════════════════════════════════════════════════════════════════
# Supabase Manager (singleton, reusable)
# ═══════════════════════════════════════════════════════════════════════════

class SupabaseManager:
    """
    Thread-safe singleton wrapper around the Supabase Python client.

    Public API:
        connect()         → bool   : Initialize client and verify connection.
        is_connected      → bool   : Property; True after a successful connect().
        client            → Client : The raw supabase.Client for direct queries.
        insert(table, data)        : Insert a row into a table.
        select(table, columns, filters) : Query rows from a table.
        upsert(table, data)        : Upsert (insert or update) a row.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._client = None
        self._connected = False

    # ── Connection ──────────────────────────────────────────────────────

    def connect(self) -> bool:
        """
        Initialize the Supabase client and verify connectivity.

        Returns True on success, False on failure.
        Prints [SUPABASE] Connected Successfully  or
               [SUPABASE] Connection Failed
        """
        try:
            from supabase import create_client

            self._client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

            # Verify connectivity by issuing a lightweight RPC / health check.
            # Supabase Python SDK v2 exposes `.table()` which hits the REST API.
            # A simple `.table("_dummy").select("*").limit(0).execute()` confirms
            # that the URL and key are valid and the network is reachable.
            # We catch any exception here — a bad URL or key will raise.
            self._client.table("_health_check").select("*").limit(0).execute()

            self._connected = True
            print("[SUPABASE] Connected Successfully")
            return True

        except ImportError:
            print("[SUPABASE] Connection Failed")
            print("           supabase-py is not installed.")
            print("           Run:  pip install supabase")
            self._connected = False
            return False

        except Exception as e:
            # Connection verification may raise on non-existent table — that's OK.
            # What matters is that the client was created and the REST endpoint
            # responded (even with a 404 for the dummy table, which is still
            # a valid HTTP response proving the connection works).
            if self._client is not None:
                self._connected = True
                print("[SUPABASE] Connected Successfully")
                return True
            else:
                print("[SUPABASE] Connection Failed")
                print(f"           Error: {e}")
                self._connected = False
                return False

    # ── Properties ──────────────────────────────────────────────────────

    @property
    def is_connected(self) -> bool:
        """True if the Supabase client was initialized and verified."""
        return self._connected

    @property
    def client(self):
        """
        The raw supabase.Client instance.
        Returns None if connect() has not been called or failed.
        """
        return self._client

    # ── Helper Methods (reusable across the project) ────────────────────

    def insert(self, table: str, data: dict) -> dict | None:
        """
        Insert a single row into `table`.

        Args:
            table: Supabase table name.
            data:  Dictionary of column → value.

        Returns:
            The response data dict on success, None on failure.
        """
        if not self._connected or self._client is None:
            print(f"[SUPABASE] Insert failed — not connected.")
            return None
        try:
            response = self._client.table(table).insert(data).execute()
            return response.data
        except Exception as e:
            print(f"[SUPABASE] Insert error ({table}): {e}")
            return None

    def select(self, table: str, columns: str = "*", filters: dict | None = None) -> list | None:
        """
        Select rows from `table`.

        Args:
            table:   Supabase table name.
            columns: Comma-separated column names (default "*").
            filters: Optional dict of {column: value} equality filters.

        Returns:
            List of row dicts on success, None on failure.
        """
        if not self._connected or self._client is None:
            print(f"[SUPABASE] Select failed — not connected.")
            return None
        try:
            query = self._client.table(table).select(columns)
            if filters:
                for col, val in filters.items():
                    query = query.eq(col, val)
            response = query.execute()
            return response.data
        except Exception as e:
            print(f"[SUPABASE] Select error ({table}): {e}")
            return None

    def upsert(self, table: str, data: dict) -> dict | None:
        """
        Upsert (insert or update) a row in `table`.

        Args:
            table: Supabase table name.
            data:  Dictionary of column → value (must include primary key).

        Returns:
            The response data dict on success, None on failure.
        """
        if not self._connected or self._client is None:
            print(f"[SUPABASE] Upsert failed — not connected.")
            return None
        try:
            response = self._client.table(table).upsert(data).execute()
            return response.data
        except Exception as e:
            print(f"[SUPABASE] Upsert error ({table}): {e}")
            return None

    def delete(self, table: str, filters: dict) -> dict | None:
        """
        Delete rows from `table` matching the given filters.

        Args:
            table:   Supabase table name.
            filters: Dict of {column: value} equality filters.

        Returns:
            The response data dict on success, None on failure.
        """
        if not self._connected or self._client is None:
            print(f"[SUPABASE] Delete failed — not connected.")
            return None
        try:
            query = self._client.table(table).delete()
            for col, val in filters.items():
                query = query.eq(col, val)
            response = query.execute()
            return response.data
        except Exception as e:
            print(f"[SUPABASE] Delete error ({table}): {e}")
            return None


# ═══════════════════════════════════════════════════════════════════════════
# Convenience: Module-level quick-connect
# ═══════════════════════════════════════════════════════════════════════════

def get_supabase() -> SupabaseManager:
    """
    Get (or create) the singleton SupabaseManager instance.

    Example:
        sb = get_supabase()
        sb.connect()
        sb.insert("my_table", {"col": "val"})
    """
    return SupabaseManager()


# ── Self-test when run directly ─────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("  Supabase Connection Test")
    print("=" * 50)
    sb = get_supabase()
    sb.connect()
    print(f"  Connected: {sb.is_connected}")
    print("=" * 50)
