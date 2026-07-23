"""
sara/core/memory.py
SQLite-backed persistent storage for Sara AI.

NOTE (structure fix): this used to exist as two near-duplicate copies —
an older draft here and the actual one in use at sara/tools/database.py.
This file is now the single canonical PreferencesDB (content taken from
the newer, in-use version); sara/tools/database.py has been removed.
"""

from __future__ import annotations

import logging
import os
import queue
import sqlite3
import threading
from concurrent.futures import Future, TimeoutError as FutureTimeoutError
from datetime import datetime
from typing import Callable, Optional

from config import Config

logger = logging.getLogger(__name__)

# PRODUCTION-AUDIT FIX (DB split-brain): this used to be computed as
# os.path.join(os.path.dirname(__file__), "sara_data.db"), i.e. relative
# to THIS file's folder (sara/tools/). Meanwhile config.py independently
# computes its own canonical Config.DB_PATH (project root), and
# reminders.py used to default to a bare "sara_data.db" (relative to
# CWD). All three defaults disagreed, so — depending on where the app
# was launched from — preferences/conversation history, reminders, and
# the "canonical" path in config.py could all silently point at THREE
# DIFFERENT physical .db files. Now this module's default is Config.DB_PATH
# itself, so there is exactly one canonical, CWD-independent database
# file for the whole app, matching reminders.py's fix.
_DEFAULT_DB_PATH = Config.DB_PATH

_VALID_ROLES = frozenset({"user", "assistant", "system"})


class PreferencesDB:
    """Manages persistent user preferences and conversation logs via SQLite."""

    # See sara/core/llm/engine.py (SaraLLM._serializable) -- self.db is
    # exposed directly off the Api object, so this stops pywebview's js_api
    # bridge from recursing into the live sqlite3 connections/locks.
    _serializable = False

    def __init__(self, db_path: str = _DEFAULT_DB_PATH) -> None:
        self.db_path = db_path
        self._closed = False
        self._close_lock = threading.Lock()

        self._local = threading.local()
        self._read_conns_lock = threading.Lock()
        self._read_conns: list[sqlite3.Connection] = []

        self._write_conn: Optional[sqlite3.Connection] = None
        self._queue: queue.Queue = queue.Queue()
        self._writer_thread: Optional[threading.Thread] = None

        try:
            self._write_conn = self._open_connection()
            self._create_tables(self._write_conn)
            logger.debug("PreferencesDB initialized at '%s'.", self.db_path)
            if Config.DEBUG_MODE:
                print(f"[Debug] PreferencesDB initialized at '{self.db_path}'.")
        except sqlite3.Error as e:
            logger.error("Failed to initialize preferences database: %s", e)
            print(f"[Error] Failed to initialize preferences database: {e}")
            self._write_conn = None
            self._closed = True
            return

        self._writer_thread = threading.Thread(
            target=self._writer_loop,
            name="PreferencesDB-Writer",
            daemon=True,
        )
        self._writer_thread.start()

    def __repr__(self) -> str:
        status = "closed" if self._closed else "open"
        return f"PreferencesDB(db_path={self.db_path!r}, status={status!r})"

    # ── Context manager ───────────────────────────────────────────────────

    def __enter__(self) -> "PreferencesDB":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # ── Connection setup ──────────────────────────────────────────────────

    def _open_connection(self) -> sqlite3.Connection:
        """Opens a connection with pragmas tuned for low-latency local use."""
        conn = sqlite3.connect(
            self.db_path,
            check_same_thread=False,
            timeout=5.0,
        )
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        conn.execute("PRAGMA temp_store=MEMORY;")
        conn.execute("PRAGMA cache_size=-8000;")
        conn.execute("PRAGMA mmap_size=268435456;")
        conn.execute("PRAGMA wal_autocheckpoint=1000;")
        return conn

    def _create_tables(self, conn: sqlite3.Connection) -> None:
        # executescript issues an implicit COMMIT; no explicit conn.commit() needed.
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS preferences (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS conversation_log (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                role      TEXT    NOT NULL,
                message   TEXT    NOT NULL,
                timestamp TEXT    NOT NULL
            );
            """)

    def _get_read_conn(self) -> sqlite3.Connection:
        """Lazily creates and caches one connection per calling thread."""
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = self._open_connection()
            self._local.conn = conn
            with self._read_conns_lock:
                self._read_conns.append(conn)
        return conn

    # ── Writer thread ─────────────────────────────────────────────────────

    def _writer_loop(self) -> None:
        while True:
            job = self._queue.get()
            if job is None:
                break
            fn, future = job
            write_conn = self._write_conn
            if write_conn is None:
                if future is not None:
                    future.set_exception(RuntimeError("Write connection is closed."))
                continue
            try:
                result = fn(write_conn)
                if future is not None:
                    future.set_result(result)
            except Exception as e:
                if future is not None:
                    future.set_exception(e)
                else:
                    logger.error("Background DB write failed: %s", e)
                    print(f"[Error] Background DB write failed: {e}")

    def _submit_write(
        self,
        fn: Callable[[sqlite3.Connection], bool],
        wait: bool = True,
        timeout: float = 5.0,
    ) -> bool:
        """
        Queues fn(write_conn) to run on the writer thread.
        If wait=True, blocks until it finishes and returns its result.
        If wait=False, returns True immediately (fire-and-forget).
        """
        with self._close_lock:
            if self._closed or self._write_conn is None:
                return False
            future: Optional[Future] = Future() if wait else None
            self._queue.put((fn, future))

        if wait and future is not None:
            try:
                return bool(future.result(timeout=timeout))
            except FutureTimeoutError:
                logger.error("DB write timed out after %.1fs.", timeout)
                print(f"[Error] DB write did not complete in time ({timeout}s).")
                return False
            except Exception as e:
                logger.error("DB write raised an exception: %s", e)
                print(f"[Error] DB write raised an exception: {e}")
                return False
        return True

    # ── Input validation helpers ──────────────────────────────────────────

    @staticmethod
    def _validate_key(key: str) -> None:
        """Raises ValueError if key is not a non-empty, non-whitespace string."""
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"Preference key must be a non-empty string; got {key!r}.")

    @staticmethod
    def _validate_nonempty(value: str, name: str) -> str:
        """Strips value and raises ValueError if the result is empty."""
        stripped = value.strip()
        if not stripped:
            raise ValueError(f"{name} must not be empty or whitespace-only.")
        return stripped

    # ── Generic key-value ─────────────────────────────────────────────────

    def set_preference(self, key: str, value: str, wait: bool = True) -> bool:
        """Inserts or updates a preference by key."""
        self._validate_key(key)
        if not isinstance(value, str):
            raise TypeError(
                f"Preference value must be a str; got {type(value).__name__!r}."
            )

        def _do(conn: sqlite3.Connection) -> bool:
            try:
                conn.execute(
                    """
                    INSERT INTO preferences (key, value) VALUES (?, ?)
                    ON CONFLICT(key) DO UPDATE SET value = excluded.value
                    """,
                    (key, value),
                )
                conn.commit()
                return True
            except sqlite3.Error as e:
                conn.rollback()
                logger.error("set_preference('%s'): %s", key, e)
                print(f"[Error] set_preference('{key}'): {e}")
                return False

        return self._submit_write(_do, wait=wait)

    def get_preference(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Returns the stored value for key, or default if not found."""
        self._validate_key(key)
        if self._closed:
            return default
        try:
            conn = self._get_read_conn()
            cursor = conn.execute("SELECT value FROM preferences WHERE key = ?", (key,))
            row = cursor.fetchone()
            return row[0] if row else default
        except sqlite3.Error as e:
            logger.error("get_preference('%s'): %s", key, e)
            print(f"[Error] get_preference('{key}'): {e}")
            return default

    def delete_preference(self, key: str, wait: bool = True) -> bool:
        """Deletes a preference by key. Returns True if a row was deleted."""
        self._validate_key(key)

        def _do(conn: sqlite3.Connection) -> bool:
            try:
                cursor = conn.execute("DELETE FROM preferences WHERE key = ?", (key,))
                conn.commit()
                return cursor.rowcount > 0
            except sqlite3.Error as e:
                conn.rollback()
                logger.error("delete_preference('%s'): %s", key, e)
                print(f"[Error] delete_preference('{key}'): {e}")
                return False

        return self._submit_write(_do, wait=wait)

    def get_all_preferences(self) -> dict[str, str]:
        """Returns all stored preferences as a key-value dict."""
        if self._closed:
            return {}
        try:
            conn = self._get_read_conn()
            cursor = conn.execute("SELECT key, value FROM preferences")
            return dict(cursor.fetchall())
        except sqlite3.Error as e:
            logger.error("get_all_preferences: %s", e)
            print(f"[Error] get_all_preferences: {e}")
            return {}

    # ── Typed helpers ─────────────────────────────────────────────────────

    def get_wake_word(self) -> str:
        """Returns the stored wake word, falling back to Config.WAKE_WORD."""
        return self.get_preference("wake_word", default=Config.WAKE_WORD)

    def set_wake_word(self, wake_word: str) -> bool:
        """Persists a new wake word (lowercased and stripped)."""
        cleaned = self._validate_nonempty(wake_word, "wake_word")
        return self.set_preference("wake_word", cleaned.lower())

    def get_user_name(self) -> Optional[str]:
        """Returns the stored user name, or None if not set."""
        return self.get_preference("user_name", default=None)

    def set_user_name(self, name: str) -> bool:
        """Persists the user name (stripped)."""
        cleaned = self._validate_nonempty(name, "user name")
        ok = self.set_preference("user_name", cleaned)
        # BUGFIX: force this specific write out of the WAL and into the
        # main db file immediately. user_name is written rarely (once per
        # onboarding) but must survive even an abrupt process kill —
        # unlike high-frequency prefs (slider drags) where this would be
        # wasteful, a single explicit checkpoint here is cheap and safe.
        if ok and self._write_conn is not None:
            try:
                self._write_conn.execute("PRAGMA wal_checkpoint(FULL);")
            except sqlite3.Error as e:
                logger.warning("wal_checkpoint after set_user_name failed: %s", e)
        return ok

    # ── Conversation log ──────────────────────────────────────────────────

    def log_message(self, role: str, message: str, wait: bool = False) -> bool:
        """
        Appends a message to the persistent conversation log.

        Defaults to fire-and-forget (wait=False). Pass wait=True for a hard
        commit guarantee before continuing (e.g. before clearing in-memory history).
        """
        if role not in _VALID_ROLES:
            raise ValueError(
                f"role must be one of {sorted(_VALID_ROLES)}; got {role!r}."
            )
        if not message or not message.strip():
            raise ValueError("message must not be empty or whitespace-only.")

        def _do(conn: sqlite3.Connection) -> bool:
            try:
                conn.execute(
                    "INSERT INTO conversation_log (role, message, timestamp) VALUES (?, ?, ?)",
                    (role, message, datetime.now().isoformat()),
                )
                conn.commit()
                return True
            except sqlite3.Error as e:
                conn.rollback()
                logger.error("log_message: %s", e)
                print(f"[Error] log_message: {e}")
                return False

        return self._submit_write(_do, wait=wait)

    def get_recent_messages(self, limit: int = 20) -> list[dict[str, str]]:
        """Returns the most recent `limit` messages in chronological order."""
        if not isinstance(limit, int) or limit <= 0:
            return []
        if self._closed:
            return []
        try:
            conn = self._get_read_conn()
            cursor = conn.execute(
                """
                SELECT role, message, timestamp
                FROM (
                    SELECT role, message, timestamp, id
                    FROM conversation_log
                    ORDER BY id DESC
                    LIMIT ?
                ) ORDER BY id ASC
                """,
                (limit,),
            )
            return [
                {"role": r[0], "message": r[1], "timestamp": r[2]}
                for r in cursor.fetchall()
            ]
        except sqlite3.Error as e:
            logger.error("get_recent_messages: %s", e)
            print(f"[Error] get_recent_messages: {e}")
            return []

    def clear_conversation_log(self, wait: bool = True) -> bool:
        """Wipes all rows from the conversation log (for 'forget history' feature)."""

        def _do(conn: sqlite3.Connection) -> bool:
            try:
                conn.execute("DELETE FROM conversation_log")
                conn.commit()
                return True
            except sqlite3.Error as e:
                conn.rollback()
                logger.error("clear_conversation_log: %s", e)
                print(f"[Error] clear_conversation_log: {e}")
                return False

        return self._submit_write(_do, wait=wait)

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def close(self) -> None:
        """Closes all connections and stops the writer thread. Safe to call multiple times."""
        with self._close_lock:
            if self._closed:
                return
            self._closed = True

        if self._writer_thread is not None:
            self._queue.put(None)
            self._writer_thread.join(timeout=5.0)
            self._writer_thread = None

        if self._write_conn is not None:
            try:
                self._write_conn.close()
            except sqlite3.Error as e:
                logger.warning("Error closing write connection: %s", e)
            self._write_conn = None

        with self._read_conns_lock:
            for conn in self._read_conns:
                try:
                    conn.close()
                except sqlite3.Error as e:
                    logger.warning("Error closing read connection: %s", e)
            self._read_conns.clear()

        logger.debug("PreferencesDB closed.")
        if Config.DEBUG_MODE:
            print("[Debug] PreferencesDB connection closed.")

    def __del__(self) -> None:
        # Only catch Exception, not BaseException — SystemExit/KeyboardInterrupt must propagate.
        try:
            self.close()
        except Exception:
            pass
