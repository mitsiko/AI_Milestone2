"""
_verify_imports.py
Throwaway smoke test for Stage 0.5. Verifies that:
  1. obe_schemas.py imports cleanly.
  2. llm_engine.py imports cleanly (Ollama client loads without contacting the server).
  3. sample_validated_output.json exists and re-validates against SyllabusSchema.

Safe to delete after Stage 0.5.
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# --- Check 1: schemas import ------------------------------------------------
try:
    from obe_schemas import (
        CourseMetadataSchema,
        CourseOutcomeSchema,
        LessonOutcomeSchema,
        WeeklyScheduleSchema,
        SyllabusSchema,
    )
    print("[OK] obe_schemas.py imported.")
except Exception as exc:
    print(f"[FAIL] obe_schemas.py import failed: {exc!r}")
    sys.exit(1)

# --- Check 2: llm_engine imports (does NOT call Ollama) ---------------------
try:
    import llm_engine
    _ = llm_engine.MODEL_NAME
    _ = llm_engine.MAX_RETRIES
    _ = llm_engine.SYSTEM_PROMPT_PASS1
    _ = llm_engine.SYSTEM_PROMPT_PASS2
    _ = llm_engine.SYSTEM_PROMPT_PASS3
    _ = llm_engine.generate_syllabus
    print(f"[OK] llm_engine.py imported. MODEL_NAME={llm_engine.MODEL_NAME}, MAX_RETRIES={llm_engine.MAX_RETRIES}")
except Exception as exc:
    print(f"[FAIL] llm_engine.py import failed: {exc!r}")
    sys.exit(1)

# --- Check 3: baseline JSON exists and re-validates -------------------------
json_path = HERE / "sample_validated_output.json"
if not json_path.exists():
    print(f"[FAIL] {json_path.name} not found in {HERE}")
    sys.exit(1)

try:
    raw = json.loads(json_path.read_text(encoding="utf-8"))
    payload = SyllabusSchema(**raw)
    print(
        f"[OK] {json_path.name} re-validated. "
        f"CLOs={len(payload.course_outcomes)}, "
        f"weeks={len(payload.weekly_schedule)}."
    )
except Exception as exc:
    print(f"[FAIL] sample_validated_output.json failed validation: {exc!r}")
    sys.exit(1)

print("\nStage 0.5 verification PASSED. Ready for Stage 1.")