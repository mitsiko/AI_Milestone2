-- schema.sql
-- Milestone 2: normalized SQLite schema for the OBE Syllabus Generator.
--
-- Design notes
-- ------------
-- 1. Five tables, no JSON blobs. Every value that Pydantic validates also
--    lives in a typed column, so a human-in-the-loop editor (DB Browser,
--    sqlite3 CLI) can read and modify a single outcome without parsing JSON.
-- 2. Foreign keys with ON DELETE CASCADE. Deleting a course cleans up every
--    course_outcome, weekly_schedule, lesson_outcome, and CLO mapping row.
-- 3. "aligned_clo" is a many-to-many relation (one week can cover 1-3 CLOs,
--    one CLO can be covered by many weeks). It lives in its own mapping
--    table, weekly_schedule_clo, with a composite primary key.
-- 4. Cascade behavior requires PRAGMA foreign_keys = ON; on every connection.
--    db_manager.py sets it explicitly; see Stage 3.
--
-- Field names mirror the Milestone 1 Pydantic field names so that a JSON
-- payload from llm_engine.py maps onto these tables without translation.

-- Turn on foreign-key enforcement for THIS session (schema-time only;
-- db_manager.py must do the same at runtime on every new connection).
PRAGMA foreign_keys = ON;

-- ---------------------------------------------------------------------------
-- Drop tables in dependency order (child first), so re-running schema.sql
-- from scratch is safe. In a real deployment this would be a migration.
-- ---------------------------------------------------------------------------
DROP TABLE IF EXISTS weekly_schedule_clo;
DROP TABLE IF EXISTS lesson_outcomes;
DROP TABLE IF EXISTS weekly_schedules;
DROP TABLE IF EXISTS course_outcomes;
DROP TABLE IF EXISTS courses;

-- ---------------------------------------------------------------------------
-- courses
-- One row per syllabus. PK is the human-readable course code.
-- ---------------------------------------------------------------------------
CREATE TABLE courses (
    course_code         TEXT    NOT NULL,
    course_title        TEXT    NOT NULL,
    course_description  TEXT    NOT NULL,
    credits             INTEGER NOT NULL CHECK (credits BETWEEN 1 AND 6),
    -- prerequisites is a list in JSON; we store it as a comma-separated
    -- string for simplicity. A production system would use a separate
    -- course_prerequisites join table, but that is out of scope here.
    prerequisites       TEXT    NOT NULL DEFAULT '',
    PRIMARY KEY (course_code)
);

-- ---------------------------------------------------------------------------
-- course_outcomes (CLOs)
-- Child of courses. Each CLO belongs to exactly one course.
-- ---------------------------------------------------------------------------
CREATE TABLE course_outcomes (
    clo_id          TEXT    NOT NULL,               -- e.g. 'CLO1'
    course_code     TEXT    NOT NULL,
    description     TEXT    NOT NULL,
    bloom_level     TEXT    NOT NULL
                    CHECK (bloom_level IN
                        ('Remember', 'Understand', 'Apply',
                         'Analyze', 'Evaluate', 'Create')),
    ksa_category    TEXT    NOT NULL CHECK (ksa_category IN ('K', 'S', 'A')),
    mapped_plo      INTEGER NOT NULL CHECK (mapped_plo BETWEEN 1 AND 6),
    PRIMARY KEY (clo_id, course_code),
    FOREIGN KEY (course_code)
        REFERENCES courses (course_code)
        ON DELETE CASCADE
        ON UPDATE CASCADE
);

-- ---------------------------------------------------------------------------
-- weekly_schedules
-- Child of courses. One row per week per course.
-- ---------------------------------------------------------------------------
CREATE TABLE weekly_schedules (
    course_code         TEXT    NOT NULL,
    week_number         INTEGER NOT NULL CHECK (week_number BETWEEN 1 AND 18),
    topic               TEXT    NOT NULL,
    teaching_activities TEXT    NOT NULL,
    assessment          TEXT    NOT NULL,
    evidence            TEXT    NOT NULL,
    PRIMARY KEY (course_code, week_number),
    FOREIGN KEY (course_code)
        REFERENCES courses (course_code)
        ON DELETE CASCADE
        ON UPDATE CASCADE
);

-- ---------------------------------------------------------------------------
-- lesson_outcomes (LLOs)
-- Grandchild of courses (child of weekly_schedules). One row per LLO.
-- ---------------------------------------------------------------------------
CREATE TABLE lesson_outcomes (
    llo_id          TEXT    NOT NULL,               -- e.g. 'LLO1.1'
    course_code     TEXT    NOT NULL,
    week_number     INTEGER NOT NULL,
    description     TEXT    NOT NULL,
    ksa_category    TEXT    NOT NULL CHECK (ksa_category IN ('K', 'S', 'A')),
    PRIMARY KEY (llo_id, course_code, week_number),
    FOREIGN KEY (course_code, week_number)
        REFERENCES weekly_schedules (course_code, week_number)
        ON DELETE CASCADE
        ON UPDATE CASCADE
);

-- ---------------------------------------------------------------------------
-- weekly_schedule_clo
-- Many-to-many mapping: which CLOs are covered by which week.
-- Both FKs cascade: deleting a course deletes its weekly_schedule_clo rows
-- via TWO paths (through weekly_schedules and through course_outcomes);
-- SQLite handles this fine because both parent rows are deleted.
-- ---------------------------------------------------------------------------
CREATE TABLE weekly_schedule_clo (
    course_code     TEXT    NOT NULL,
    week_number     INTEGER NOT NULL,
    clo_id          TEXT    NOT NULL,
    PRIMARY KEY (course_code, week_number, clo_id),
    FOREIGN KEY (course_code, week_number)
        REFERENCES weekly_schedules (course_code, week_number)
        ON DELETE CASCADE
        ON UPDATE CASCADE,
    FOREIGN KEY (clo_id, course_code)
        REFERENCES course_outcomes (clo_id, course_code)
        ON DELETE CASCADE
        ON UPDATE CASCADE
);

-- ---------------------------------------------------------------------------
-- Indexes
-- The PKs already index the main access patterns. These two extras speed up
-- the two most common lookups: "give me all weeks for a course" and
-- "give me all LLOs for a week".
-- ---------------------------------------------------------------------------
CREATE INDEX idx_weekly_schedules_course
    ON weekly_schedules (course_code);

CREATE INDEX idx_lesson_outcomes_week
    ON lesson_outcomes (course_code, week_number);