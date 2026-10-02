# db_manager.py
"""
Database manager for the OBE Syllabus Generator (Milestone 2).

Responsibilities
----------------
- init_db()                    : create the SQLite file + tables from schema.sql
- insert_syllabus()            : ingest one SyllabusSchema-shaped dict
- get_syllabus_by_course_code(): read one course back as a SyllabusSchema-shaped dict
- update_course_outcome()      : human-in-the-loop edit of one CLO's description
- delete_syllabus()            : delete a course; FKs cascade to all child rows

Design notes
------------
- Every connection sets PRAGMA foreign_keys = ON so ON DELETE CASCADE fires.
- All writes go through a single transaction (BEGIN ... COMMIT) so partial
  failures leave the DB in its previous consistent state.
- Insert order respects FK dependency: courses -> course_outcomes ->
  weekly_schedules -> lesson_outcomes -> weekly_schedule_clo.
- Row objects use sqlite3.Row so column access is by name.
"""

import argparse
import json
import sqlite3
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
DEFAULT_DB = HERE / "obe_syllabus.db"
SCHEMA_SQL = HERE / "schema.sql"


# ---------------------------------------------------------------------------
# Connection helper
# ---------------------------------------------------------------------------
def _connect(db_path: Path) -> sqlite3.Connection:
    """Open a connection with foreign keys enforced and Row row_factory."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.row_factory = sqlite3.Row
    return conn


# ---------------------------------------------------------------------------
# init_db
# ---------------------------------------------------------------------------
def init_db(db_path: Path = DEFAULT_DB) -> None:
    """Create the DB and apply schema.sql. Idempotent: drops and recreates.

    Raises:
        FileNotFoundError: if schema.sql is missing.
    """
    if not SCHEMA_SQL.exists():
        raise FileNotFoundError(f"schema.sql not found at {SCHEMA_SQL}")

    # If the file already exists and has data, warn. We still recreate.
    existed = db_path.exists()

    conn = _connect(db_path)
    try:
        sql = SCHEMA_SQL.read_text(encoding="utf-8")
        conn.executescript(sql)
        conn.commit()
    finally:
        conn.close()

    if existed:
        print(f"[init_db] Recreated schema in existing DB: {db_path.name}")
    else:
        print(f"[init_db] Created new DB: {db_path.name}")


# ---------------------------------------------------------------------------
# insert_syllabus
# ---------------------------------------------------------------------------
def insert_syllabus(payload: dict, db_path: Path = DEFAULT_DB) -> None:
    """Insert one complete syllabus payload (SyllabusSchema.model_dump() shape).

    Uses a single transaction; on any error, rolls back and re-raises.
    Re-inserting a course_code that already exists will fail on the primary
    key (SQLite will raise IntegrityError). Delete first to replace.

    Args:
        payload: dict with keys course_metadata, course_outcomes, weekly_schedule.
        db_path: path to the SQLite file.
    """
    meta = payload["course_metadata"]
    cos = payload["course_outcomes"]
    weeks = payload["weekly_schedule"]

    conn = _connect(db_path)
    try:
        with conn:  # wraps in a transaction; auto-rollback on exception
            # 1) courses
            conn.execute(
                """
                INSERT INTO courses
                    (course_code, course_title, course_description,
                     credits, prerequisites)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    meta["course_code"],
                    meta["course_title"],
                    meta["course_description"],
                    int(meta["credits"]),
                    ",".join(meta.get("prerequisites", [])),
                ),
            )

            # 2) course_outcomes
            conn.executemany(
                """
                INSERT INTO course_outcomes
                    (clo_id, course_code, description,
                     bloom_level, ksa_category, mapped_plo)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        co["clo_id"],
                        meta["course_code"],
                        co["description"],
                        co["bloom_level"],
                        co["ksa_category"],
                        int(co["mapped_plo"]),
                    )
                    for co in cos
                ],
            )

            # 3) weekly_schedules
            conn.executemany(
                """
                INSERT INTO weekly_schedules
                    (course_code, week_number, topic, teaching_activities,
                     assessment, evidence)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        meta["course_code"],
                        int(w["week_number"]),
                        w["topic"],
                        w["teaching_activities"],
                        w["assessment"],
                        w["evidence"],
                    )
                    for w in weeks
                ],
            )

            # 4) lesson_outcomes
            llo_rows = []
            for w in weeks:
                for llo in w["lesson_outcomes"]:
                    llo_rows.append(
                        (
                            llo["llo_id"],
                            meta["course_code"],
                            int(w["week_number"]),
                            llo["description"],
                            llo["ksa_category"],
                        )
                    )
            conn.executemany(
                """
                INSERT INTO lesson_outcomes
                    (llo_id, course_code, week_number, description, ksa_category)
                VALUES (?, ?, ?, ?, ?)
                """,
                llo_rows,
            )

            # 5) weekly_schedule_clo (many-to-many)
            align_rows = []
            for w in weeks:
                for clo_id in w["aligned_clo"]:
                    align_rows.append(
                        (meta["course_code"], int(w["week_number"]), clo_id)
                    )
            conn.executemany(
                """
                INSERT INTO weekly_schedule_clo
                    (course_code, week_number, clo_id)
                VALUES (?, ?, ?)
                """,
                align_rows,
            )

        # Success — print a compact summary.
        print(
            f"[insert_syllabus] {meta['course_code']}: "
            f"{len(cos)} CLOs, {len(weeks)} weeks, "
            f"{len(llo_rows)} LLOs, {len(align_rows)} CLO-week links"
        )
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# get_syllabus_by_course_code
# ---------------------------------------------------------------------------
def get_syllabus_by_course_code(
    course_code: str, db_path: Path = DEFAULT_DB
) -> dict | None:
    """Read one course back as a dict in SyllabusSchema.shape.

    Returns None if the course does not exist.

    The returned dict is exactly what SyllabusSchema(**result) would accept:
        {
          "course_metadata": {...},
          "course_outcomes": [...],
          "weekly_schedule": [...]
        }
    """
    conn = _connect(db_path)
    try:
        # 1) courses
        row = conn.execute(
            "SELECT * FROM courses WHERE course_code = ?", (course_code,)
        ).fetchone()
        if row is None:
            return None

        prereq_str = row["prerequisites"] or ""
        prerequisites = [p for p in prereq_str.split(",") if p]

        course_metadata = {
            "course_code": row["course_code"],
            "course_title": row["course_title"],
            "course_description": row["course_description"],
            "credits": row["credits"],
            "prerequisites": prerequisites,
        }

        # 2) course_outcomes (ordered by clo_id for stable output)
        co_rows = conn.execute(
            """
            SELECT clo_id, description, bloom_level, ksa_category, mapped_plo
            FROM course_outcomes
            WHERE course_code = ?
            ORDER BY clo_id
            """,
            (course_code,),
        ).fetchall()
        course_outcomes = [
            {
                "clo_id": r["clo_id"],
                "description": r["description"],
                "bloom_level": r["bloom_level"],
                "ksa_category": r["ksa_category"],
                "mapped_plo": r["mapped_plo"],
            }
            for r in co_rows
        ]

        # 3) weekly_schedules + their LLOs + their aligned_clo
        week_rows = conn.execute(
            """
            SELECT week_number, topic, teaching_activities, assessment, evidence
            FROM weekly_schedules
            WHERE course_code = ?
            ORDER BY week_number
            """,
            (course_code,),
        ).fetchall()

        weekly_schedule = []
        for w in week_rows:
            wk_num = w["week_number"]

            llo_rows = conn.execute(
                """
                SELECT llo_id, description, ksa_category
                FROM lesson_outcomes
                WHERE course_code = ? AND week_number = ?
                ORDER BY llo_id
                """,
                (course_code, wk_num),
            ).fetchall()
            lesson_outcomes = [
                {
                    "llo_id": r["llo_id"],
                    "description": r["description"],
                    "ksa_category": r["ksa_category"],
                }
                for r in llo_rows
            ]

            clo_rows = conn.execute(
                """
                SELECT clo_id
                FROM weekly_schedule_clo
                WHERE course_code = ? AND week_number = ?
                ORDER BY clo_id
                """,
                (course_code, wk_num),
            ).fetchall()
            aligned_clo = [r["clo_id"] for r in clo_rows]

            weekly_schedule.append(
                {
                    "week_number": wk_num,
                    "topic": w["topic"],
                    "lesson_outcomes": lesson_outcomes,
                    "teaching_activities": w["teaching_activities"],
                    "assessment": w["assessment"],
                    "evidence": w["evidence"],
                    "aligned_clo": aligned_clo,
                }
            )

        return {
            "course_metadata": course_metadata,
            "course_outcomes": course_outcomes,
            "weekly_schedule": weekly_schedule,
        }
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# update_course_outcome
# ---------------------------------------------------------------------------
def update_course_outcome(
    course_code: str,
    clo_id: str,
    new_description: str,
    db_path: Path = DEFAULT_DB,
) -> bool:
    """Update one CLO's description. Returns True if a row was changed.

    This is the human-in-the-loop edit demonstrated in the video. Only the
    description changes; bloom_level, ksa_category, and mapped_plo are left
    alone (a proper edit UI would let the user change those too, but that is
    out of scope for a single update function).
    """
    conn = _connect(db_path)
    try:
        with conn:
            cur = conn.execute(
                """
                UPDATE course_outcomes
                SET description = ?
                WHERE course_code = ? AND clo_id = ?
                """,
                (new_description, course_code, clo_id),
            )
            changed = cur.rowcount
        if changed:
            print(
                f"[update_course_outcome] {course_code} / {clo_id}: "
                f"description updated ({len(new_description)} chars)"
            )
        else:
            print(
                f"[update_course_outcome] {course_code} / {clo_id}: "
                f"no matching row (nothing changed)"
            )
        return changed > 0
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# delete_syllabus
# ---------------------------------------------------------------------------
def delete_syllabus(course_code: str, db_path: Path = DEFAULT_DB) -> dict:
    """Delete a course row. FKs cascade to all child tables.

    Returns a dict with before/after counts per table so the caller can
    verify cascade behavior.
    """
    tables = (
        "courses",
        "course_outcomes",
        "weekly_schedules",
        "lesson_outcomes",
        "weekly_schedule_clo",
    )
    conn = _connect(db_path)
    try:
        before = {
            t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            for t in tables
        }
        with conn:
            cur = conn.execute(
                "DELETE FROM courses WHERE course_code = ?", (course_code,)
            )
            deleted = cur.rowcount
        after = {
            t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            for t in tables
        }
        diff = {t: before[t] - after[t] for t in tables}
        print(f"[delete_syllabus] {course_code}: courses deleted = {deleted}")
        for t in tables:
            print(f"  {t:22s} before={before[t]:4d}  after={after[t]:4d}  delta=-{diff[t]}")
        return {"deleted_courses": deleted, "before": before, "after": after, "diff": diff}
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------
def _cli() -> int:
    parser = argparse.ArgumentParser(
        description="DB manager CLI: init, insert, get, update, delete, count."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init", help="Create DB from schema.sql (drops existing).")
    p_init.add_argument("--db", default=str(DEFAULT_DB))

    p_insert = sub.add_parser("insert", help="Insert a syllabus JSON file.")
    p_insert.add_argument("--json", required=True)
    p_insert.add_argument("--db", default=str(DEFAULT_DB))

    p_get = sub.add_parser("get", help="Print one course as JSON.")
    p_get.add_argument("--code", required=True)
    p_get.add_argument("--db", default=str(DEFAULT_DB))

    p_upd = sub.add_parser("update", help="Update one CLO's description.")
    p_upd.add_argument("--code", required=True)
    p_upd.add_argument("--clo", required=True)
    p_upd.add_argument("--desc", required=True)
    p_upd.add_argument("--db", default=str(DEFAULT_DB))

    p_del = sub.add_parser("delete", help="Delete a course (cascades).")
    p_del.add_argument("--code", required=True)
    p_del.add_argument("--db", default=str(DEFAULT_DB))

    p_count = sub.add_parser("count", help="Print row counts for all tables.")
    p_count.add_argument("--db", default=str(DEFAULT_DB))

    args = parser.parse_args()
    db = Path(args.db)

    if args.cmd == "init":
        init_db(db)
        return 0

    if args.cmd == "insert":
        payload = json.loads(Path(args.json).read_text(encoding="utf-8"))
        insert_syllabus(payload, db)
        return 0

    if args.cmd == "get":
        result = get_syllabus_by_course_code(args.code, db)
        if result is None:
            print(f"[get] {args.code}: NOT FOUND", file=sys.stderr)
            return 1
        print(json.dumps(result, indent=2))
        return 0

    if args.cmd == "update":
        ok = update_course_outcome(args.code, args.clo, args.desc, db)
        return 0 if ok else 2

    if args.cmd == "delete":
        delete_syllabus(args.code, db)
        return 0

    if args.cmd == "count":
        conn = _connect(db)
        try:
            for t in ("courses", "course_outcomes", "weekly_schedules",
                      "lesson_outcomes", "weekly_schedule_clo"):
                n = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                print(f"{t:22s} {n:4d}")
        finally:
            conn.close()
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())