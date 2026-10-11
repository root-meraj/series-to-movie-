"""Persistent accounts, revocable sessions and project ownership for one server."""

from __future__ import annotations

import hashlib
import hmac
import secrets
import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from starlette.exceptions import HTTPException

from sts.schemas import WebUserDoc

SESSION_SECONDS = 24 * 60 * 60


def _password_hash(password: str, salt: bytes) -> bytes:
    return hashlib.scrypt(
        password.encode(), salt=salt, n=2**17, r=8, p=1, maxmem=192 * 1024**2, dklen=32
    )


class AccountStore:
    """Keep salted scrypt hashes and hashed session tokens outside job artifacts."""

    def __init__(self, path: Path, max_accounts: int = 100) -> None:
        """Initialize a private SQLite database in the persistent data volume."""
        self.path = path
        self.max_accounts = max_accounts
        path.parent.mkdir(parents=True, exist_ok=True)
        path.parent.chmod(0o700)
        with self._connection() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY, email TEXT UNIQUE NOT NULL,
                    salt BLOB NOT NULL, password_hash BLOB NOT NULL
                );
                CREATE TABLE IF NOT EXISTS sessions (
                    digest TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(id),
                    expires INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS owners (
                    job_id TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(id)
                );
                CREATE TABLE IF NOT EXISTS attempts (
                    key TEXT PRIMARY KEY, count INTEGER NOT NULL, expires INTEGER NOT NULL
                );
            """)
        path.chmod(0o600)

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys = ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    def throttle(self, source: str, email: str) -> None:
        """Bound password work per source and email before running memory-hard hashing."""
        now = int(time.time())
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("DELETE FROM attempts WHERE expires <= ?", (now,))
            for raw, limit in (("source:" + source, 20), ("email:" + email, 8)):
                key = hashlib.sha256(raw.encode()).hexdigest()
                row = db.execute("SELECT count FROM attempts WHERE key = ?", (key,)).fetchone()
                if row and row["count"] >= limit:
                    raise HTTPException(429, "Too many attempts. Wait a minute and try again.")
                db.execute(
                    "INSERT INTO attempts VALUES (?, 1, ?) "
                    "ON CONFLICT(key) DO UPDATE SET count=count+1",
                    (key, now + 60),
                )

    def register(self, email: str, password: str) -> WebUserDoc:
        """Create a unique account while preserving the configured demo account cap."""
        salt = secrets.token_bytes(16)
        digest = _password_hash(password, salt)
        user = WebUserDoc(id=secrets.token_hex(16), email=email)
        try:
            with self._connection() as db:
                db.execute("BEGIN IMMEDIATE")
                if db.execute("SELECT COUNT(*) FROM users").fetchone()[0] >= self.max_accounts:
                    raise HTTPException(503, "New accounts are temporarily full. Try again later.")
                db.execute(
                    "INSERT INTO users VALUES (?, ?, ?, ?)", (user.id, user.email, salt, digest)
                )
        except sqlite3.IntegrityError:
            raise HTTPException(409, "Could not create an account with these details.") from None
        return user

    def login(self, email: str, password: str) -> WebUserDoc:
        """Use identical password work for unknown users and incorrect credentials."""
        with self._connection() as db:
            row = db.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        digest = _password_hash(password, row["salt"] if row else bytes(16))
        expected = row["password_hash"] if row else bytes(32)
        if not hmac.compare_digest(digest, expected) or row is None:
            raise HTTPException(401, "Email or password is incorrect.")
        return WebUserDoc(id=row["id"], email=row["email"])

    def issue_session(self, user_id: str) -> str:
        """Return a random bearer once; persist only its digest with a fixed expiry."""
        token = secrets.token_urlsafe(48)
        now = int(time.time())
        with self._connection() as db:
            db.execute("DELETE FROM sessions WHERE expires <= ?", (now,))
            db.execute(
                "DELETE FROM sessions WHERE digest IN "
                "(SELECT digest FROM sessions WHERE user_id=? ORDER BY expires DESC "
                "LIMIT -1 OFFSET 9)",
                (user_id,),
            )
            db.execute(
                "INSERT INTO sessions VALUES (?, ?, ?)",
                (self.session_digest(token), user_id, now + SESSION_SECONDS),
            )
        return token

    @staticmethod
    def session_digest(token: str) -> str:
        """Compute the database-safe digest of a high-entropy bearer token."""
        return hashlib.sha256(token.encode()).hexdigest()

    def resolve_digest(self, digest: str) -> WebUserDoc | None:
        """Resolve only unexpired sessions, including those bound to video tickets."""
        with self._connection() as db:
            row = db.execute(
                "SELECT u.id, u.email FROM users u JOIN sessions s "
                "ON s.user_id=u.id WHERE s.digest=? AND s.expires>?",
                (digest, int(time.time())),
            ).fetchone()
        return WebUserDoc(id=row["id"], email=row["email"]) if row else None

    def resolve(self, token: str) -> WebUserDoc | None:
        """Authenticate a bearer without storing or logging its plaintext."""
        return self.resolve_digest(self.session_digest(token)) if len(token) <= 256 else None

    def logout(self, token: str) -> None:
        """Revoke the session and every media ticket derived from it immediately."""
        with self._connection() as db:
            db.execute("DELETE FROM sessions WHERE digest=?", (self.session_digest(token),))

    def assign(self, job_id: str, user_id: str) -> None:
        """Bind a newly created project to its authenticated owner before processing."""
        with self._connection() as db:
            db.execute("INSERT INTO owners VALUES (?, ?)", (job_id, user_id))

    def owns(self, job_id: str, user_id: str) -> bool:
        """Deny access to unclaimed legacy jobs and jobs owned by another account."""
        with self._connection() as db:
            return (
                db.execute(
                    "SELECT 1 FROM owners WHERE job_id=? AND user_id=?", (job_id, user_id)
                ).fetchone()
                is not None
            )
