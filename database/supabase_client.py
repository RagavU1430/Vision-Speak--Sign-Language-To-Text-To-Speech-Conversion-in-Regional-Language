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
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

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

            # Verify connectivity by checking that the client was created.
            # Using a health-check endpoint is more reliable than querying
            # a non-existent table (which always 404s).
            try:
                self._client.auth.get_session()
            except Exception:
                pass

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

    # ── User Authentication & Profile Caching ───────────────────────────

    def sign_up_user(self, email, password, name, age, phone_number, emergency_contact, preferred_language) -> dict:
        """
        Signs up a new user via Supabase Auth and registers their profile details in user_profiles table.
        """
        if not self._connected or self._client is None:
            return {"success": False, "error": "Not connected to Supabase"}
        try:
            # Sign up in Auth
            res = self._client.auth.sign_up({
                "email": email,
                "password": password
            })
            if not res or not res.user:
                return {"success": False, "error": "Sign up failed (no user returned)"}
            
            auth_user_id = res.user.id
            
            # Create profile data
            profile_data = {
                "auth_user_id": auth_user_id,
                "name": name,
                "age": int(age) if age else None,
                "email": email,
                "phone_number": phone_number,
                "emergency_contact": emergency_contact
            }
            
            # Insert into user_profiles table
            profile_res = self._client.table("user_profiles").insert(profile_data).execute()
            
            # Check if session is active (if email verification is disabled)
            has_session = res.session is not None
            if has_session:
                self.save_session_locally(res.session)
                
            return {
                "success": True,
                "user_id": auth_user_id,
                "email_confirm_required": not has_session,
                "profile": profile_res.data[0] if (profile_res and profile_res.data) else profile_data
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def login_user(self, email, password) -> dict:
        """
        Logs in the user with email/password and caches the session locally.
        """
        if not self._connected or self._client is None:
            return {"success": False, "error": "Not connected to Supabase"}
        try:
            res = self._client.auth.sign_in_with_password({
                "email": email,
                "password": password
            })
            if not res or not res.session or not res.user:
                return {"success": False, "error": "Login failed (no session/user returned)"}
                
            self.save_session_locally(res.session)
            
            profile = self.get_user_profile(res.user.id)
            return {
                "success": True,
                "user": res.user,
                "session": res.session,
                "profile": profile
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def logout_user(self):
        """
        Logs out of Supabase and clears the local session cache.
        """
        try:
            if self._connected and self._client:
                self._client.auth.sign_out()
        except Exception as e:
            print(f"[SUPABASE] Sign out error: {e}")
            
        try:
            from pathlib import Path
            session_file = Path(__file__).parent / "session.json"
            if session_file.exists():
                session_file.unlink()
                print("[SUPABASE] Local session cleared.")
        except Exception as e:
            print(f"[SUPABASE] Error removing local session file: {e}")

    def save_session_locally(self, session):
        """
        Caches access and refresh tokens locally in session.json.
        """
        try:
            import json
            from pathlib import Path
            session_file = Path(__file__).parent / "session.json"
            data = {
                "access_token": session.access_token,
                "refresh_token": session.refresh_token
            }
            with open(session_file, "w", encoding="utf-8") as f:
                json.dump(data, f)
            print("[SUPABASE] Session saved locally.")
        except Exception as e:
            print(f"[SUPABASE] Error saving session: {e}")

    def get_user_profile(self, auth_user_id) -> dict | None:
        """
        Fetches the user profile details from the user_profiles table.
        """
        if not self._connected or self._client is None:
            return None
        try:
            res = self._client.table("user_profiles").select("*").eq("auth_user_id", auth_user_id).execute()
            if res.data:
                return res.data[0]
            return None
        except Exception as e:
            print(f"[SUPABASE] Error fetching user profile: {e}")
            return None

    def update_user_profile(self, auth_user_id, profile_data) -> bool:
        """
        Updates the user's profile details.
        """
        if not self._connected or self._client is None:
            return False
        try:
            from datetime import datetime, timezone
            profile_data["updated_at"] = datetime.now(timezone.utc).isoformat()
            res = self._client.table("user_profiles").update(profile_data).eq("auth_user_id", auth_user_id).execute()
            return len(res.data) > 0
        except Exception as e:
            print(f"[SUPABASE] Error updating user profile: {e}")
            return False

    def reset_password_for_email(self, email: str) -> bool:
        """
        Sends a password reset email via Supabase Auth.
        """
        if not self._connected or self._client is None:
            return False
        try:
            self._client.auth.reset_password_for_email(email)
            return True
        except Exception as e:
            print(f"[SUPABASE] Reset password error: {e}")
            return False

    def auto_login(self) -> dict | None:
        """
        Attempts to perform auto-login using local cached credentials.
        Returns user info and profile if successful, otherwise None.
        """
        if not self._connected or self._client is None:
            return None
        try:
            import json
            from pathlib import Path
            session_file = Path(__file__).parent / "session.json"
            if not session_file.exists():
                return None
                
            with open(session_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            access_token = data.get("access_token")
            refresh_token = data.get("refresh_token")
            
            if not access_token or not refresh_token:
                return None
                
            # Set session
            res = self._client.auth.set_session(access_token, refresh_token)
            if not res or not res.session or not res.user:
                return None
                
            # Cache the refreshed session
            self.save_session_locally(res.session)
            
            # Fetch profile
            profile = self.get_user_profile(res.user.id)
            if profile:
                return {
                    "user": res.user,
                    "session": res.session,
                    "profile": profile
                }
            return None
        except Exception as e:
            print(f"[SUPABASE] Auto-login failed: {e}")
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
