# export_engine.py
"""
Export engine for the OBE Syllabus Generator (Milestone 2).

Pipeline (SQLite -> Jinja2 -> HTML):

  1. (optional) ensure the DB is populated. The Stage-3 cascade demo emptied
     the DB on purpose; --ensure-populated re-inserts sample_validated_output.json
     so the demo flow is one command.
  2. Read one course from SQLite via db_manager.get_syllabus_by_course_code().
  3. Validate the readback against SyllabusSchema (defensive: proves the DB
     round-trip did not corrupt data).
  4. Render templates/uphsd_ccs_template.html with that dict via Jinja2.
  5. Write obe_ms2/<COURSE_CODE>_syllabus.html (browser-ready).

This module NEVER calls the LLM. It is purely a presentation layer on top of
already-validated data.
"""

import argparse
import json
import sys
import webbrowser
from datetime import datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from db_manager import (
    DEFAULT_DB,
    get_syllabus_by_course_code,
    init_db,
    insert_syllabus,
)
from obe_schemas import SyllabusSchema

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
TEMPLATES_DIR = HERE / "templates"
TEMPLATE_NAME = "uphsd_ccs_template.html"
SAMPLE_JSON = HERE / "sample_validated_output.json"

MODEL_NAME = "qwen2.5:3b"


# ---------------------------------------------------------------------------
# Ensure DB is populated
# ---------------------------------------------------------------------------
def ensure_populated(course_code: str, db_path: Path, json_path: Path) -> None:
    """If the course is not in the DB, re-init and insert the sample JSON.

    Rationale: the Stage-3 demo cascades-deletes the course, leaving an empty
    DB. Rather than asking the user to re-run init+insert by hand, we do it
    here in one step for a smooth demo.
    """
    existing = get_syllabus_by_course_code(course_code, db_path)
    if existing is not None:
        print(f"[export] {course_code} already in DB; skipping re-insert.")
        return

    if not json_path.exists():
        raise FileNotFoundError(
            f"{json_path.name} not found and {course_code} not in DB. "
            "Run llm_engine.py first or pass --json."
        )

    print(f"[export] {course_code} not in DB; re-initializing and inserting from {json_path.name}.")
    init_db(db_path)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    insert_syllabus(payload, db_path)


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------
def render_syllabus_html(
    payload: dict,
    output_path: Path,
    template_name: str = TEMPLATE_NAME,
    model_name: str = MODEL_NAME,
) -> Path:
    """Render a SyllabusSchema-shaped dict into a standalone HTML document.

    The template lives in templates/ and is loaded from disk; it is NOT
    embedded here so the layout can be edited without touching Python.
    """
    if not TEMPLATES_DIR.exists():
        raise FileNotFoundError(f"templates/ folder not found at {TEMPLATES_DIR}")

    env = Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=select_autoescape(["html", "xml"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = env.get_template(template_name)

    html = template.render(
        payload=payload,
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
        model_name=model_name,
    )
    output_path.write_text(html, encoding="utf-8")
    return output_path


# ---------------------------------------------------------------------------
# End-to-end: SQLite -> Jinja2 -> HTML
# ---------------------------------------------------------------------------
def export(
    course_code: str,
    db_path: Path = DEFAULT_DB,
    json_path: Path = SAMPLE_JSON,
    ensure: bool = True,
    open_browser: bool = False,
    output_path: Path | None = None,
) -> Path:
    """Read a course from SQLite, render it, and write an HTML file.

    Args:
        course_code: e.g. "CS201"
        db_path:     path to SQLite file (default: obe_syllabus.db)
        json_path:   fallback JSON if the course is missing from the DB
        ensure:      if True, re-insert from json_path when the DB is empty
        open_browser: if True, open the output in the default browser
        output_path:  override the default <COURSE_CODE>_syllabus.html

    Returns:
        The absolute path of the rendered HTML file.
    """
    if ensure:
        ensure_populated(course_code, db_path, json_path)

    print(f"[export] Reading {course_code} from {db_path.name}...")
    payload = get_syllabus_by_course_code(course_code, db_path)
    if payload is None:
        raise RuntimeError(
            f"{course_code} not found in DB and ensure=False. "
            "Run with --ensure-populated or insert it first."
        )

    # Defensive re-validation: proves the DB round-trip is lossless.
    SyllabusSchema(**payload)
    print(
        f"[export] Readback validated: "
        f"{len(payload['course_outcomes'])} CLOs, "
        f"{len(payload['weekly_schedule'])} weeks."
    )

    if output_path is None:
        safe_code = course_code.replace(" ", "_")
        output_path = HERE / f"{safe_code}_syllabus.html"

    print(f"[export] Rendering template {TEMPLATE_NAME}...")
    render_syllabus_html(payload, output_path)

    size_kb = output_path.stat().st_size / 1024
    print(f"[export] Wrote {output_path.name} ({size_kb:.1f} KB).")

    if open_browser:
        webbrowser.open(output_path.as_uri())
        print(f"[export] Opened in browser.")

    return output_path.resolve()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export an OBE syllabus from SQLite to HTML via Jinja2."
    )
    parser.add_argument(
        "--code",
        default="CS201",
        help="Course code to export (default: CS201).",
    )
    parser.add_argument(
        "--db",
        default=str(DEFAULT_DB),
        help="Path to SQLite file (default: obe_syllabus.db).",
    )
    parser.add_argument(
        "--json",
        default=str(SAMPLE_JSON),
        help="Fallback JSON to re-insert if the course is not in the DB.",
    )
    parser.add_argument(
        "--no-ensure",
        action="store_true",
        help="Do not re-populate the DB if the course is missing; fail instead.",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open the rendered HTML in the default browser.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Override the output HTML path.",
    )
    args = parser.parse_args()

    try:
        path = export(
            course_code=args.code,
            db_path=Path(args.db),
            json_path=Path(args.json),
            ensure=not args.no_ensure,
            open_browser=args.open,
            output_path=Path(args.out) if args.out else None,
        )
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    print(f"\nDone. Open in a browser: file:///{path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())