# obe_schemas.py
"""
Pydantic data contracts for the OBE Syllabus Generator (Milestone 2).

Extends Milestone 1 with:
  - WeeklyScheduleSchema.evidence  (artifact that proves the assessment happened)
  - A model_validator that enforces the aligned-CLO focus rule:
      * Weeks 9 and 18 -> all CLO ids referenced by the syllabus
      * Every other week -> 1 to 3 CLO ids

Retains all Milestone 1 rules: measurable Bloom's verbs, K/S/A coverage,
unique CLO/LLO ids, referential integrity, week-number ordering, and the
week 9 / week 18 exam-week topic pinning.
"""

from typing import List, Literal
from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Verb sets
# ---------------------------------------------------------------------------
BANNED_VERBS = {
    "understand", "know", "learn", "be familiar with", "study",
    "be exposed to",
}

ACTIVE_VERBS = {
    "identify", "define", "describe", "explain", "recall", "list", "recognize",
    "apply", "implement", "configure", "calculate", "analyze", "differentiate",
    "compare", "solve", "demonstrate", "use", "execute", "reflect",
    "evaluate", "design", "develop", "integrate", "construct", "synthesize",
    "formulate", "create", "defend", "justify", "propose", "build",
    "collaborate", "coordinate", "practice", "uphold", "advocate", "commit",
    "respect", "value", "exhibit", "show", "appreciate",
}


import re as _re

_LLO_PREFIX = _re.compile(r"^(LLO|CLO)\s*\d+(\.\d+)?\s*[:.\-]?\s*", _re.IGNORECASE)


def _check_measurable_verb(text: str) -> str:
    stripped = text.strip()
    stripped = _LLO_PREFIX.sub("", stripped).strip()
    if not stripped:
        raise ValueError("Description cannot be empty.")
    first_word = stripped.split()[0].lower().strip(",.:;")
    if first_word in BANNED_VERBS:
        raise ValueError(
            f"Description must not start with a vague verb ('{first_word}')."
        )
    if first_word not in ACTIVE_VERBS:
        raise ValueError(
            f"Description must start with a recognized active verb. Got '{first_word}'."
        )
    return stripped


# ---------------------------------------------------------------------------
# KSA normalizer
# ---------------------------------------------------------------------------
_KSA_ALIASES = {
    "K": "K", "KNOWLEDGE": "K", "COGNITIVE": "K", "C": "K",
    "S": "S", "SKILLS": "S", "SKILL": "S", "PSYCHOMOTOR": "S",
    "A": "A", "ATTITUDE": "A", "AFFECTIVE": "A", "E": "A",
}


def _normalize_ksa(value: str) -> str:
    key = str(value).strip().upper()
    if key in _KSA_ALIASES:
        return _KSA_ALIASES[key]
    raise ValueError(f"ksa_category must be K, S, or A. Got '{value}'.")


# ---------------------------------------------------------------------------
# Course metadata
# ---------------------------------------------------------------------------
class CourseMetadataSchema(BaseModel):
    course_code: str = Field(..., min_length=2)
    course_title: str = Field(..., min_length=3)
    course_description: str = Field(..., min_length=20)
    credits: int = Field(..., ge=1, le=6)
    prerequisites: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Course Learning Outcomes
# ---------------------------------------------------------------------------
class CourseOutcomeSchema(BaseModel):
    clo_id: str = Field(..., description="e.g. CLO1")
    description: str = Field(..., min_length=10)
    bloom_level: Literal[
        "Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"
    ]
    ksa_category: Literal["K", "S", "A"]
    mapped_plo: int = Field(default=1, ge=1, le=6)

    @field_validator("clo_id")
    @classmethod
    def clo_id_format(cls, v: str) -> str:
        v = v.strip().upper()
        if not v.startswith("CLO") or not v[3:].isdigit():
            raise ValueError("clo_id must be like 'CLO1'.")
        return v

    @field_validator("description")
    @classmethod
    def description_has_verb(cls, v: str) -> str:
        return _check_measurable_verb(v)

    @field_validator("ksa_category", mode="before")
    @classmethod
    def ksa_normalize(cls, v):
        return _normalize_ksa(v)


# ---------------------------------------------------------------------------
# Lesson Learning Outcomes
# ---------------------------------------------------------------------------
class LessonOutcomeSchema(BaseModel):
    llo_id: str
    description: str = Field(..., min_length=8)
    ksa_category: Literal["K", "S", "A"] = Field(default=None)  # inferred if missing

    @model_validator(mode="before")
    @classmethod
    def infer_ksa_from_llo_id(cls, data):
        """If the model omitted ksa_category, infer it from the LLO id suffix.

        Convention enforced by the prompt:
            LLO<n>.1 -> K
            LLO<n>.2 -> S
            LLO<n>.3 -> A
        """
        if not isinstance(data, dict):
            return data
        llo_id = str(data.get("llo_id", "")).strip().upper()
        if data.get("ksa_category") in (None, "", "null"):
            if llo_id.endswith(".1"):
                data["ksa_category"] = "K"
            elif llo_id.endswith(".2"):
                data["ksa_category"] = "S"
            elif llo_id.endswith(".3"):
                data["ksa_category"] = "A"
        return data

    @field_validator("llo_id")
    @classmethod
    def llo_id_format(cls, v: str) -> str:
        v = v.strip().upper()
        if not v.startswith("LLO") or "." not in v:
            raise ValueError("llo_id must be like 'LLO1.1'.")
        return v

    @field_validator("description")
    @classmethod
    def description_has_verb(cls, v: str) -> str:
        return _check_measurable_verb(v)

    @field_validator("ksa_category", mode="before")
    @classmethod
    def ksa_normalize(cls, v):
        if v is None:
            return v  # let infer_ksa_from_llo_id run first
        return _normalize_ksa(v)


# ---------------------------------------------------------------------------
# Weekly schedule
# ---------------------------------------------------------------------------
class WeeklyScheduleSchema(BaseModel):
    week_number: int = Field(..., ge=1, le=18)
    topic: str = Field(..., min_length=2)
    lesson_outcomes: List[LessonOutcomeSchema] = Field(..., min_length=3)
    teaching_activities: str = Field(..., min_length=5)
    assessment: str = Field(..., min_length=2)
    evidence: str = Field(..., min_length=2)  # NEW in Milestone 2
    aligned_clo: List[str] = Field(..., min_length=1)

    @field_validator("lesson_outcomes")
    @classmethod
    def must_have_all_ksa(cls, llos: List[LessonOutcomeSchema]) -> List[LessonOutcomeSchema]:
        categories = {llo.ksa_category for llo in llos}
        missing = {"K", "S", "A"} - categories
        if missing:
            raise ValueError(
                f"Week must have at least one K, one S, and one A. Missing: {sorted(missing)}"
            )
        return llos

    @model_validator(mode="after")
    def exam_week_business_rules(self) -> "WeeklyScheduleSchema":
        if self.week_number == 9:
            self.topic = "Midterm Assessment and Review"
            self.assessment = "Midterm Exam"
        elif self.week_number == 18:
            self.topic = "Final Project Defense and Comprehensive Review"
            self.assessment = "Final Project Defense"
        return self


# ---------------------------------------------------------------------------
# Root payload
# ---------------------------------------------------------------------------
class SyllabusSchema(BaseModel):
    course_metadata: CourseMetadataSchema
    course_outcomes: List[CourseOutcomeSchema] = Field(..., min_length=3, max_length=6)
    weekly_schedule: List[WeeklyScheduleSchema] = Field(..., min_length=18, max_length=18)

    @model_validator(mode="after")
    def integrity_checks(self) -> "SyllabusSchema":
        weeks = [w.week_number for w in self.weekly_schedule]
        if weeks != list(range(1, 19)):
            raise ValueError(f"Week numbers must be 1..18 in order. Got {weeks}.")

        clo_ids = [c.clo_id for c in self.course_outcomes]
        if len(clo_ids) != len(set(clo_ids)):
            raise ValueError("Duplicate clo_id values.")

        referenced = {clo for w in self.weekly_schedule for clo in w.aligned_clo}
        missing = set(clo_ids) - referenced
        if missing:
            raise ValueError(f"CLOs never referenced: {sorted(missing)}")

        unknown = referenced - set(clo_ids)
        if unknown:
            raise ValueError(f"Weeks reference unknown CLOs: {sorted(unknown)}")

        second_half = [w.topic for w in self.weekly_schedule if w.week_number >= 8]
        if len(second_half) != len(set(second_half)):
            raise ValueError("Weeks 8-18 must have distinct topics.")

        # ---- NEW: aligned-CLO focus rule ----------------------------------
        # Weeks 9 and 18 must cover every CLO. Non-exam weeks must cover 1-3.
        for w in self.weekly_schedule:
            if w.week_number in (9, 18):
                if set(w.aligned_clo) != set(clo_ids):
                    raise ValueError(
                        f"Week {w.week_number} (exam) must reference all CLOs "
                        f"{sorted(clo_ids)}. Got {sorted(w.aligned_clo)}."
                    )
            else:
                n = len(set(w.aligned_clo))
                if not (1 <= n <= 3):
                    raise ValueError(
                        f"Week {w.week_number} must reference 1-3 CLOs. Got {n}: "
                        f"{sorted(w.aligned_clo)}."
                    )
        return self