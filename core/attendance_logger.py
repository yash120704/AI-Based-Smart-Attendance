"""
Attendance logging with SQLite local mode and PostgreSQL cloud mode.
"""
import logging
import os
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from config import ATTENDANCE_COOLDOWN_SECONDS, ALLOW_MULTIPLE_ATTENDANCE_PER_DAY, DB_PATH

logger = logging.getLogger(__name__)


class AttendanceLogger:
    """
    Logs attendance events to the configured database.

    Local development keeps the original SQLite behavior. Cloud deployments set
    DATABASE_URL and use Supabase PostgreSQL with the same table shape.
    """

    def __init__(self, db_path=DB_PATH):
        """
        Initialize attendance logger and create tables if needed.

        Args:
            db_path: Path to SQLite database file in local mode
        """
        self.database_url = os.environ.get("DATABASE_URL")
        self.use_postgres = bool(self.database_url)
        self.db_path = Path(db_path)
        self.conn = None
        self.last_logged = {}

        if not self.use_postgres:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.connect()
        self.create_tables()
        backend = "Supabase PostgreSQL" if self.use_postgres else f"SQLite database: {db_path}"
        logger.info(f"AttendanceLogger initialized with {backend}")

    def connect(self):
        """
        Connect to the configured database backend.
        """
        try:
            if self.use_postgres:
                try:
                    import psycopg2
                except ImportError as exc:
                    raise RuntimeError("psycopg2 is required when DATABASE_URL is set") from exc

                self.conn = psycopg2.connect(
                    self.database_url,
                    sslmode=os.environ.get("DATABASE_SSLMODE", "require"),
                )
                logger.info("Connected to PostgreSQL database")
            else:
                self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
                self.conn.row_factory = sqlite3.Row
                logger.info("Connected to SQLite database")
        except Exception as exc:
            logger.error(f"Database connection error: {exc}")
            raise

    def _placeholder(self):
        return "%s" if self.use_postgres else "?"

    def _today_clause(self):
        if self.use_postgres:
            return "DATE(timestamp) = CURRENT_DATE"
        return "DATE(timestamp) = DATE('now', 'localtime')"

    def _proxy_clause(self):
        if self.use_postgres:
            return "is_proxy = TRUE OR status = 'PROXY'"
        return "is_proxy = 1 OR status = 'PROXY'"

    def _boolean_value(self, value):
        return bool(value) if self.use_postgres else int(bool(value))

    def _read_sql(self, query, params=None):
        return pd.read_sql_query(query, self.conn, params=params)

    def create_tables(self):
        """
        Create tables if they don't exist.
        """
        cursor = self.conn.cursor()

        if self.use_postgres:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS persons (
                    id SERIAL PRIMARY KEY,
                    name TEXT UNIQUE NOT NULL,
                    registered_at TIMESTAMP DEFAULT NOW(),
                    total_attendances INTEGER DEFAULT 0,
                    blocked BOOLEAN DEFAULT FALSE,
                    blocked_until TIMESTAMP
                )
                """
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS attendance (
                    id SERIAL PRIMARY KEY,
                    person_name TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT NOW(),
                    face_confidence REAL,
                    behavior_confidence REAL,
                    is_proxy BOOLEAN DEFAULT FALSE,
                    alert_message TEXT,
                    attempts INTEGER DEFAULT 1,
                    status TEXT,
                    blink_detected BOOLEAN DEFAULT FALSE
                )
                """
            )
        else:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS attendance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_name TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    face_confidence REAL,
                    behavior_confidence REAL,
                    is_proxy INTEGER DEFAULT 0,
                    alert_message TEXT,
                    attempts INTEGER DEFAULT 0,
                    status TEXT,
                    blink_detected INTEGER DEFAULT 0
                )
                """
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS persons (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    registered_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    total_attendances INTEGER DEFAULT 0,
                    blocked INTEGER DEFAULT 0,
                    blocked_until DATETIME
                )
                """
            )

        self.conn.commit()
        logger.info("Database tables created/verified")

    def log(self, person_name, face_conf, behavior_conf, is_proxy, alert_message, attempts=0, status="PENDING", blink_detected=0):
        """
        Log attendance event with cooldown check.

        Args:
            person_name: Name of person
            face_conf: Face recognition confidence
            behavior_conf: Behavior model confidence
            is_proxy: Whether proxy was detected
            alert_message: Alert message if any
            attempts: Number of attempts
            status: Attendance status (SUCCESS, PROXY, BLOCKED, RETRY, ALREADY_MARKED)
            blink_detected: Whether blink was detected (0 or 1)

        Returns:
            dict: Result with status and message
        """
        if not ALLOW_MULTIPLE_ATTENDANCE_PER_DAY:
            if self.has_marked_today(person_name):
                logger.info(f"{person_name} already marked attendance today")
                return {
                    "status": "ALREADY_MARKED",
                    "message": f"{person_name} already marked attendance today",
                }

        now = datetime.now()
        last_log_time = self.last_logged.get(person_name)

        if last_log_time:
            time_diff = (now - last_log_time).total_seconds()
            if time_diff < ATTENDANCE_COOLDOWN_SECONDS:
                logger.debug(f"Cooldown active for {person_name}")
                return {
                    "status": "COOLDOWN",
                    "message": f"Cooldown active for {person_name}",
                }

        cursor = self.conn.cursor()
        placeholder = self._placeholder()
        cursor.execute(
            f"""
            INSERT INTO attendance
            (person_name, face_confidence, behavior_confidence, is_proxy, alert_message, attempts, status, blink_detected)
            VALUES ({placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder})
            """,
            (
                person_name,
                face_conf,
                behavior_conf,
                self._boolean_value(is_proxy),
                alert_message,
                attempts,
                status,
                self._boolean_value(blink_detected),
            ),
        )
        self.conn.commit()
        self.last_logged[person_name] = now

        logger.info(f"Logged attendance for {person_name}: {status}")
        return {"status": "SUCCESS", "message": f"Attendance logged for {person_name}"}

    def log_attendance(self, *args, **kwargs):
        """
        Compatibility alias for API-facing code.
        """
        return self.log(*args, **kwargs)

    def has_marked_today(self, person_name):
        """
        Check if person already marked attendance today.

        Args:
            person_name: Name of person

        Returns:
            bool: True if already marked today
        """
        cursor = self.conn.cursor()
        placeholder = self._placeholder()
        cursor.execute(
            f"""
            SELECT COUNT(*) FROM attendance
            WHERE person_name = {placeholder}
            AND {self._today_clause()}
            AND status = 'SUCCESS'
            """,
            (person_name,),
        )
        result = cursor.fetchone()[0]
        return result > 0

    def mark_person_for_reattendance(self, person_name):
        """
        Mark a person as allowed for re-attendance.

        Args:
            person_name: Name of person

        Returns:
            dict: Result of operation
        """
        cursor = self.conn.cursor()
        placeholder = self._placeholder()
        cursor.execute(
            f"UPDATE persons SET blocked = {placeholder}, blocked_until = NULL WHERE name = {placeholder}",
            (self._boolean_value(False), person_name),
        )
        self.conn.commit()
        logger.info(f"Person {person_name} marked for re-attendance")
        return {"status": "SUCCESS", "message": f"{person_name} can now mark attendance again"}

    def unblock_person(self, person_name):
        """
        Unblock a person from attendance restrictions.
        """
        return self.mark_person_for_reattendance(person_name)

    def block_person(self, person_name, duration_minutes=0):
        """
        Block a person from marking attendance.

        Args:
            person_name: Name of person
            duration_minutes: Duration of block in minutes (0 = indefinite)

        Returns:
            dict: Result of operation
        """
        cursor = self.conn.cursor()
        placeholder = self._placeholder()
        blocked_until = None
        if duration_minutes > 0:
            blocked_until = datetime.now() + timedelta(minutes=duration_minutes)

        cursor.execute(
            f"UPDATE persons SET blocked = {placeholder}, blocked_until = {placeholder} WHERE name = {placeholder}",
            (self._boolean_value(True), blocked_until, person_name),
        )
        self.conn.commit()
        logger.info(f"Person {person_name} blocked")
        return {"status": "SUCCESS", "message": f"{person_name} blocked from attendance"}

    def is_person_blocked(self, person_name):
        """
        Check if a person is blocked.

        Args:
            person_name: Name of person

        Returns:
            bool: True if blocked
        """
        cursor = self.conn.cursor()
        placeholder = self._placeholder()
        cursor.execute(
            f"""
            SELECT blocked, blocked_until FROM persons
            WHERE name = {placeholder}
            """,
            (person_name,),
        )
        result = cursor.fetchone()
        if not result:
            return False

        blocked, blocked_until = result
        if not blocked:
            return False

        if blocked_until:
            if isinstance(blocked_until, datetime):
                blocked_until_dt = blocked_until
            else:
                blocked_until_dt = datetime.fromisoformat(str(blocked_until))

            if blocked_until_dt > datetime.now():
                return True
            self.mark_person_for_reattendance(person_name)
            return False

        return True

    def get_today_attendance(self):
        """
        Get today's attendance records.

        Returns:
            DataFrame: Today's attendance records
        """
        query = f"""
            SELECT * FROM attendance
            WHERE {self._today_clause()}
            ORDER BY timestamp DESC
        """
        return self._read_sql(query)

    def get_today_records(self):
        """
        Compatibility alias for API-facing code.
        """
        return self.get_today_attendance()

    def get_today_attendance_count(self):
        """
        Get today's attendance count without building a DataFrame.

        Returns:
            int: Number of attendance rows for today
        """
        cursor = self.conn.cursor()
        cursor.execute(
            f"""
            SELECT COUNT(*) FROM attendance
            WHERE {self._today_clause()}
            """
        )
        return int(cursor.fetchone()[0])

    def get_all_attendance(self):
        """
        Get all attendance records.

        Returns:
            DataFrame: All attendance records
        """
        query = "SELECT * FROM attendance ORDER BY timestamp DESC"
        return self._read_sql(query)

    def get_attendance_range(self, start_date, end_date):
        """
        Get attendance records for date range.

        Args:
            start_date: Start date (datetime or string YYYY-MM-DD)
            end_date: End date (datetime or string YYYY-MM-DD)

        Returns:
            DataFrame: Attendance records in range
        """
        if isinstance(start_date, datetime):
            start_date = start_date.strftime("%Y-%m-%d")
        if isinstance(end_date, datetime):
            end_date = end_date.strftime("%Y-%m-%d")

        placeholder = self._placeholder()
        query = f"""
            SELECT * FROM attendance
            WHERE DATE(timestamp) BETWEEN {placeholder} AND {placeholder}
            ORDER BY timestamp DESC
        """
        return self._read_sql(query, params=(start_date, end_date))

    def get_proxy_alerts(self):
        """
        Get all proxy alert records.

        Returns:
            DataFrame: Proxy alert records
        """
        query = f"""
            SELECT * FROM attendance
            WHERE {self._proxy_clause()}
            ORDER BY timestamp DESC
        """
        return self._read_sql(query)

    def get_proxy_alert_count(self):
        """
        Get proxy alert count without building a DataFrame.

        Returns:
            int: Number of proxy alert rows
        """
        cursor = self.conn.cursor()
        cursor.execute(
            f"""
            SELECT COUNT(*) FROM attendance
            WHERE {self._proxy_clause()}
            """
        )
        return int(cursor.fetchone()[0])

    def get_person_attendance(self, person_name):
        """
        Get all attendance records for a person.

        Args:
            person_name: Name of person

        Returns:
            DataFrame: Person's attendance records
        """
        placeholder = self._placeholder()
        query = f"""
            SELECT * FROM attendance
            WHERE person_name = {placeholder}
            ORDER BY timestamp DESC
        """
        return self._read_sql(query, params=(person_name,))

    def register_person(self, person_name):
        """
        Register a new person in the database.

        Args:
            person_name: Name of person
        """
        cursor = self.conn.cursor()
        placeholder = self._placeholder()
        try:
            if self.use_postgres:
                cursor.execute(
                    """
                    INSERT INTO persons (name, blocked)
                    VALUES (%s, FALSE)
                    ON CONFLICT (name) DO NOTHING
                    """,
                    (person_name,),
                )
            else:
                cursor.execute(
                    f"INSERT INTO persons (name, blocked) VALUES ({placeholder}, 0)",
                    (person_name,),
                )
            self.conn.commit()
            logger.info(f"Person {person_name} registered in database")
        except sqlite3.IntegrityError:
            logger.debug(f"Person {person_name} already registered")

    def update_person_attendance_count(self, person_name):
        """
        Update total attendance count for a person.

        Args:
            person_name: Name of person
        """
        cursor = self.conn.cursor()
        placeholder = self._placeholder()
        cursor.execute(
            f"""
            UPDATE persons
            SET total_attendances = (
                SELECT COUNT(*) FROM attendance
                WHERE person_name = {placeholder} AND status = 'SUCCESS'
            )
            WHERE name = {placeholder}
            """,
            (person_name, person_name),
        )
        self.conn.commit()

    def get_persons_stats(self):
        """
        Get statistics for all registered persons.

        Returns:
            DataFrame: Person statistics
        """
        if self.use_postgres:
            query = """
                SELECT
                    p.name,
                    p.registered_at,
                    p.total_attendances,
                    p.blocked,
                    COUNT(a.id) as total_records,
                    COALESCE(SUM(CASE WHEN a.status = 'PROXY' THEN 1 ELSE 0 END), 0) as proxy_count
                FROM persons p
                LEFT JOIN attendance a ON p.name = a.person_name
                GROUP BY p.id, p.name, p.registered_at, p.total_attendances, p.blocked
                ORDER BY p.name
            """
        else:
            query = """
                SELECT
                    p.name,
                    p.registered_at,
                    p.total_attendances,
                    p.blocked,
                    COUNT(a.id) as total_records,
                    SUM(CASE WHEN a.status = 'PROXY' THEN 1 ELSE 0 END) as proxy_count
                FROM persons p
                LEFT JOIN attendance a ON p.name = a.person_name
                GROUP BY p.name
                ORDER BY p.name
            """
        return self._read_sql(query)

    def close(self):
        """
        Close database connection.
        """
        if self.conn:
            self.conn.close()
            self.conn = None
            logger.info("Database connection closed")

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
