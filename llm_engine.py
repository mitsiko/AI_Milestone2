# llm_engine.py
"""
LLM engine for the OBE Syllabus Generator (Milestone 2).

Three-pass pipeline, upgraded from Milestone 1:
  Pass 1 - metadata + CLOs, with a tightened K/S/A distribution rule, a smart
           6->5 trim, and a strict 2K/2S/1A deterministic rebalancer that
           runs afterward.
  Pass 2 - 18-week skeleton, emitting `evidence` and requiring concrete
           tools/IDEs/languages in `teaching_activities`.
  Pass 3 - per-week lesson outcomes, requiring a concrete tool, library,
           or scenario in every description.

After Pass 1:
  _trim_to_five             -> reduce a 6-CLO list to 5 by dropping the CLO
                               whose verb + topic keywords overlap most with
                               another CLO (falls back to dropping the last
                               CLO only if all six are unique)
  _fix_clo_ksa_balance      -> re-derive ksa_category from the verb; enforce
                               EXACTLY 2 K, 2 S, 1 A (rewrites descriptions
                               in-place if needed)

After Pass 2:
  _focus_aligned_clo          -> 1-3 CLOs per non-exam week; all CLOs on weeks 9 & 18
  _enforce_assessment_variety -> 4-tool pool, cap of 5 uses each; exam weeks pinned
  _harmonize_assessment_evidence -> tool -> evidence map applied deterministically
  _dedupe_advanced_topics     -> rewrite weeks 10-17 topics from a DSA-specific
                                 bank; gated by _is_dsa_course() so non-DSA
                                 courses keep the LLM's original topics
  _sanitize_dijkstra_leak     -> remove Dijkstra mentions from non-graph weeks
                                 and collapse the resulting "graph algorithm
                                 algorithm" duplication

CLI:
  py llm_engine.py                    # full 3-pass generation
  py llm_engine.py --reprocess-only   # apply post-processors to existing JSON, no LLM
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import ollama
from pydantic import BaseModel, Field, ValidationError

from obe_schemas import (
    CourseMetadataSchema,
    CourseOutcomeSchema,
    LessonOutcomeSchema,
    SyllabusSchema,
    WeeklyScheduleSchema,
)

MODEL_NAME = "qwen2.5:3b"
MAX_RETRIES = 3
OLLAMA_OPTIONS = {"temperature": 0.3}


# ===========================================================================
# Pass 1 - metadata + CLOs
# ===========================================================================
SYSTEM_PROMPT_PASS1 = """
You are an expert IT/CS Curriculum Designer.
Output ONLY a single JSON object. No markdown, no preamble.

Shape:
{
  "course_metadata": {
    "course_code": "string",
    "course_title": "string",
    "course_description": "string",
    "credits": integer 1-6,
    "prerequisites": ["string", ...]
  },
  "course_outcomes": [
    {
      "clo_id": "CLO1",
      "description": "string starting with ONE measurable Bloom's verb",
      "bloom_level": "Remember|Understand|Apply|Analyze|Evaluate|Create",
      "ksa_category": "K|S|A",
      "mapped_plo": integer 1-6
    }
    ... EXACTLY 5 CLOs, ids CLO1..CLO5 ...
  ]
}

STRICT KSA DISTRIBUTION (do NOT deviate):
  - EXACTLY 2 CLOs with ksa_category "K"  (Knowledge)
  - EXACTLY 2 CLOs with ksa_category "S"  (Skills)
  - EXACTLY 1 CLO with  ksa_category "A"  (Attitude)

VERB -> CATEGORY MAPPING (pick the FIRST word of every description from the
matching list; do NOT use verbs from another list):

  K (Knowledge) - cognitive, "know about":
    Identify, Define, Explain, Describe, Recall, List, Recognize
    bloom_level MUST be "Remember" or "Understand"
    EXAMPLES:
      "Identify the operations supported by a binary search tree."
      "Define the difference between contiguous and linked memory."
      "Explain how a hash function maps keys to buckets."

  S (Skills) - psychomotor, "can do":
    Apply, Implement, Configure, Calculate, Analyze, Differentiate, Compare,
    Solve, Demonstrate, Use, Execute, Design, Develop, Integrate, Construct,
    Synthesize, Formulate, Create
    bloom_level MUST be "Apply", "Analyze", "Evaluate", or "Create"
    EXAMPLES:
      "Implement a balanced binary search tree in C++17."
      "Analyze the time complexity of quicksort in the worst case."
      "Design a caching layer using a hash table."

  A (Attitude) - affective, "values / commits / collaborates":
    Collaborate, Coordinate, Uphold, Respect, Practice, Advocate, Commit,
    Exhibit, Value, Reflect, Appreciate
    bloom_level MUST be "Understand" or "Apply"
    EXAMPLES:
      "Collaborate in a team to debug a graph traversal implementation."
      "Uphold code-quality standards when submitting lab outputs."
      "Demonstrate persistence when debugging segmentation faults."
      "Value the ethical implications of algorithm design choices."

HARD RULES:
  - Use EXACTLY 5 CLOs. Distribution: 2 K, 2 S, 1 A. No other mix.
  - Each "description" starts with ONE verb from the matching list above.
  - "bloom_level" MUST match the verb's category as specified above.
    Specifically: an "A" CLO must NEVER have bloom_level "Create" or "Evaluate",
    because those are cognitive-domain levels, not affective.
  - An "A" CLO's description MUST describe a behavior, disposition, or value -
    NOT a technical artifact. Do NOT use "Design and implement ..." as an A CLO;
    that is an S CLO.
  - NEVER use: understand, know, learn, be familiar with, study.
  - "mapped_plo" MUST be an integer between 1 and 6.
  - Output ONLY the JSON object.
"""


class _Pass1(BaseModel):
    course_metadata: CourseMetadataSchema
    course_outcomes: list[CourseOutcomeSchema]


# ===========================================================================
# Pass 2 - 18-week skeleton
# ===========================================================================
SYSTEM_PROMPT_PASS2 = """
You are an expert IT/CS Curriculum Designer.
Output ONLY a single JSON object. No markdown, no preamble.

Shape:
{
  "weekly_schedule": [
    {
      "week_number": 1,
      "topic": "short curriculum label, 2-6 words",
      "teaching_activities": "Face-to-face lecture on X using <TOOL> + Hands-on Lab in <IDE/LANG>: Y",
      "assessment": "Quiz|Lab Rubric|Recitation|Project Rubric",
      "evidence": "artifact that proves the assessment happened",
      "aligned_clo": ["CLO1", "CLO2"]
    }
    ... EXACTLY 18 week objects, numbered 1..18 ...
  ]
}

RULES:
- EXACTLY 18 weeks, week_number 1 through 18.
- Week 9 MUST have topic "Midterm Assessment and Review" and assessment "Midterm Exam".
- Week 18 MUST have topic "Final Project Defense and Comprehensive Review"
  and assessment "Final Project Defense".
- Weeks 1-8 cover FOUNDATIONAL topics (introductory concepts).
- Weeks 10-17 cover NEW ADVANCED topics that go BEYOND the introductory week 1-8
  material. Do NOT name them "Advanced X" where X is an earlier week's topic.
  Instead, cover genuinely different sub-areas.
- Weeks 8-17 MUST have DISTINCT topics. No topic may repeat. No "Continued".
- Every "teaching_activities" MUST name a SPECIFIC tool, language, or IDE.
  Concrete examples: "VS Code", "PyCharm", "Python 3.12", "C++17", "OpenGL 4.6",
  "Git", "PostgreSQL 16", "Jupyter Notebook", "CMake", "GCC", "clang",
  "GDB", "Valgrind", "Google Test", "Catch2". Do NOT write generic
  "a programming language" or "an IDE". Choose at least one concrete tool per week.
- Every "teaching_activities" MUST combine a lecture element AND a hands-on lab element.
- "assessment" MUST be ONE of exactly: "Quiz", "Lab Rubric", "Recitation",
  "Project Rubric". Do NOT pipe-separate. Do NOT invent new tools.
  (Weeks 9 and 18 are the only exceptions; use "Midterm Exam" and
  "Final Project Defense" respectively.)
- "evidence" MUST be a short noun phrase naming the artifact that proves the
  assessment happened. Match the tool:
    Quiz               -> "Quiz Answer Sheet"
    Lab Rubric         -> "Lab Output"
    Recitation         -> "Recitation Log"
    Project Rubric     -> "Final Project + Rubric"
    Midterm Exam       -> "Midterm Exam Answer Sheet"
    Final Project Def. -> "Final Project Defense Rubric"
- "aligned_clo" lists CLO ids (from the user's list) covered that week.
  Weeks 9 and 18 MUST list ALL CLO ids.
  Every other week MUST list 1 to 3 CLO ids (no more, no less).
- NO extra fields. NO missing fields.
"""


class _SkeletonWeek(BaseModel):
    week_number: int = Field(..., ge=1, le=18)
    topic: str
    teaching_activities: str
    assessment: str
    evidence: str
    aligned_clo: list[str]


class _Pass2(BaseModel):
    weekly_schedule: list[_SkeletonWeek] = Field(..., min_length=18, max_length=18)


# ===========================================================================
# Pass 3 - per-week lesson outcomes
# ===========================================================================
SYSTEM_PROMPT_PASS3 = """
You are an expert IT/CS Curriculum Designer.
Output ONLY a single JSON object. No markdown, no preamble.

Shape:
{
  "lesson_outcomes": [
    {"llo_id": "LLOn.1", "description": "string", "ksa_category": "K"},
    {"llo_id": "LLOn.2", "description": "string", "ksa_category": "S"},
    {"llo_id": "LLOn.3", "description": "string", "ksa_category": "A"}
  ]
}

RULES:
- EXACTLY 3 lesson outcomes in this order: K, S, A.
- The llo_id prefix MUST match the week number (e.g. week 4 -> LLO4.1, LLO4.2, LLO4.3).
- EVERY lesson outcome object MUST include BOTH "ksa_category" and "description".
  Mapping: LLO<n>.1 -> "K", LLO<n>.2 -> "S", LLO<n>.3 -> "A".

- CONCRETE-TOOL REQUIREMENT: every description MUST name at least one concrete
  tool, library, language, IDE, or scenario. Examples of acceptable concrete
  nouns: "VS Code", "GDB", "Valgrind", "OpenGL 4.6", "GLFW", "GLUT", "Python 3.12",
  "C++17", "std::vector", "std::unique_ptr", "PostgreSQL 16", "Jupyter Notebook",
  "CMake", "Git", "pytest", "Google Test", "Catch2", "Docker".
  Do NOT write abstract phrases like "modern tools" or "an IDE".

- K (Knowledge) LLO: about understanding and explaining the week's concepts.
  MUST start with: Identify, Define, Explain, Describe, Recall, List, Recognize.
  GOOD example for a Graphs week:
    "Explain how breadth-first search traverses a graph and why it uses a queue."
  BAD example:
    "Explain the difference between two data structures."

- S (Skills) LLO: about doing, coding, or implementing.
  MUST start with: Apply, Implement, Design, Develop, Construct, Analyze, Configure,
  Calculate, Demonstrate, Solve, Integrate.
  GOOD example for a Graphs week:
    "Implement Dijkstra's algorithm in Python 3.12 using a heap-based priority queue."
  BAD example:
    "Implement a search algorithm."

- A (Attitude) LLO: about collaboration, ethics, professionalism, persistence,
  or precision.
  MUST start with: Collaborate, Coordinate, Uphold, Respect, Practice, Advocate,
  Commit, Exhibit, Value, Reflect.
  GOOD example for a Graphs week:
    "Demonstrate persistence while debugging a segmentation fault in GDB during
     graph traversal implementation."
  BAD example:
    "Demonstrate persistence."

- NEVER start any description with: understand, know, learn, study, be familiar with.
- NEVER mix categories: a "Collaborate" outcome is A, not K. A "Practice" outcome
  is A, not S.
- Do NOT repeat the llo_id inside the "description" field.
  WRONG: "LLO1.1 Collaborate in ..."
  RIGHT: "Collaborate in ..."
- Every LLO must mention a concept specific to its OWN week's topic.
- Output ONLY the JSON object.
"""


class _LLOList(BaseModel):
    lesson_outcomes: list[LessonOutcomeSchema] = Field(..., min_length=3, max_length=3)


# ===========================================================================
# Ollama wrapper
# ===========================================================================
def _call_ollama(messages: list[dict]) -> str:
    response = ollama.chat(
        model=MODEL_NAME,
        messages=messages,
        format="json",
        options=OLLAMA_OPTIONS,
    )
    return response["message"]["content"]


def _strip_fences(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```json"):
        return raw[7:-3].strip() if raw.endswith("```") else raw[7:].strip()
    if raw.startswith("```"):
        return raw[3:-3].strip() if raw.endswith("```") else raw[3:].strip()
    return raw


def _distribute_plos(cos: list) -> list:
    """Assign mapped_plo 1..6 across CLOs so the mapping matrix is meaningful."""
    plos = [1, 2, 3, 4, 5, 6]
    for i, co in enumerate(cos):
        co.mapped_plo = plos[i % len(plos)]
    return cos


def _retry_loop(system_prompt: str, user_prompt: str, schema_cls, label: str):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        print(f"  [{label}] attempt {attempt}/{MAX_RETRIES}")
        try:
            raw = _call_ollama(messages)
        except Exception as exc:
            last_error = f"Ollama call failed: {exc}"
            print(f"    ! {last_error}")
            raise RuntimeError("Ollama unreachable.") from exc

        cleaned = _strip_fences(raw)
        try:
            data = json.loads(cleaned)
            return schema_cls(**data)
        except json.JSONDecodeError as exc:
            last_error = f"JSONDecodeError: {exc}"
            print(f"    ! {last_error}")
        except ValidationError as exc:
            last_error = f"ValidationError: {exc}"
            print(f"    ! {str(exc)[:300]}")

        messages.append({"role": "assistant", "content": cleaned})
        messages.append(
            {
                "role": "user",
                "content": (
                    "Your output failed validation:\n"
                    f"{last_error}\n\nReturn ONLY the corrected JSON object."
                ),
            }
        )
        time.sleep(1)
    raise RuntimeError(f"{label}: failed after {MAX_RETRIES} attempts. Last: {last_error}")


# ===========================================================================
# Deterministic post-processors (locked in Stage 1)
# ===========================================================================
ALLOWED_ASSESSMENTS = ["Quiz", "Lab Rubric", "Recitation", "Project Rubric"]
ASSESSMENT_CAP = 5

TOOL_TO_EVIDENCE = {
    "Quiz": "Quiz Answer Sheet",
    "Lab Rubric": "Lab Output",
    "Recitation": "Recitation Log",
    "Project Rubric": "Final Project + Rubric",
    "Midterm Exam": "Midterm Exam Answer Sheet",
    "Final Project Defense": "Final Project Defense Rubric",
}

# ---- KSA classification tables (used by _fix_clo_ksa_balance) -------------
_K_VERBS = {
    "identify", "define", "explain", "describe", "recall", "list", "recognize",
}
_S_VERBS = {
    "apply", "implement", "configure", "calculate", "analyze", "differentiate",
    "compare", "solve", "demonstrate", "use", "execute", "design", "develop",
    "integrate", "construct", "synthesize", "formulate", "create",
}
_A_VERBS = {
    "collaborate", "coordinate", "uphold", "respect", "practice", "advocate",
    "commit", "exhibit", "value", "reflect", "appreciate",
}

# ---- DSA-family detection -------------------------------------------------
# Keywords that indicate a Data Structures & Algorithms course. Used by
# _dedupe_advanced_topics to decide whether the DSA-specific topic bank is
# appropriate. If a future course family (e.g. Database Systems) needs its
# own bank, add its keywords here and a matching bank below.
_DSA_KEYWORDS = (
    "data structure", "data structures", "algorithm", "algorithms",
    "dsa", "algorithmic",
)


def _is_dsa_course(course_prompt: str) -> bool:
    """Return True if the course prompt looks like a DSA-family course.

    Case-insensitive substring match against _DSA_KEYWORDS. Conservative:
    we only enable the DSA topic bank when we are reasonably sure the course
    IS a DSA course. For any other course family, the LLM's original topics
    are left untouched.
    """
    text = (course_prompt or "").lower()
    return any(kw in text for kw in _DSA_KEYWORDS)


# ---- DSA-specific advanced topic bank -------------------------------------
# This bank exists to work around a specific failure mode of qwen2.5:3b:
# the model reliably ignores the "do not name weeks 10-17 'Advanced X'"
# instruction in SYSTEM_PROMPT_PASS2 and produces a duplicate second half.
# Rather than fight the LLM with more re-prompting, we deterministically
# replace weeks 10-17 with a fixed list of genuinely new advanced sub-topics.
#
# The bank is intentionally specific to Data Structures & Algorithms. It is
# gated behind _is_dsa_course() so non-DSA syllabi are left untouched. If
# you extend this project to other course families, add a parallel bank
# here (e.g. _ADVANCED_DB_TOPIC_BANK) and extend the gate.
_ADVANCED_TOPIC_BANK = [
    "Graph Traversal Algorithms",
    "Shortest Path Algorithms",
    "Minimum Spanning Trees",
    "Hash Tables and Hashing",
    "Balanced Trees (AVL, Red-Black)",
    "Greedy Algorithms",
    "Backtracking and Branch-and-Bound",
    "Capstone Project Preparation",
]

# ---- Stopwords used by _trim_to_five for topic-keyword comparison ---------
_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with",
    "by", "from", "as", "at", "into", "that", "this", "their", "its",
    "apply", "implement", "configure", "calculate", "analyze", "differentiate",
    "compare", "solve", "demonstrate", "use", "execute", "design", "develop",
    "integrate", "construct", "synthesize", "formulate", "create",
    "identify", "define", "explain", "describe", "recall", "list", "recognize",
    "collaborate", "coordinate", "uphold", "respect", "practice", "advocate",
    "commit", "exhibit", "value", "reflect", "appreciate", "and", "or",
    "using", "use", "course", "problem", "problems", "data", "structure",
    "structures", "algorithm", "algorithms", "code", "programming",
}


def _classify_verb(description: str) -> str:
    """Return 'K', 'S', or 'A' based on the first verb of the description."""
    first = description.strip().split()[0].lower().strip(",.:;")
    if first in _K_VERBS:
        return "K"
    if first in _S_VERBS:
        return "S"
    if first in _A_VERBS:
        return "A"
    return "K"  # conservative default


def _topic_keywords(description: str) -> set:
    """Return the meaningful (non-stopword, non-verb) tokens of a description.

    Used to measure topic overlap between two CLOs when deciding which CLO to
    drop during the 6->5 trim.
    """
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_]+", description.lower())
    return {t for t in tokens if t not in _STOPWORDS and len(t) > 2}


def _trim_to_five(cos: list) -> list:
    """Reduce a 6-CLO list to 5 by dropping the most-redundant CLO.

    Redundancy score for CLO i = number of OTHER CLOs whose (verb, keyword-set)
    overlaps with CLO i. Ties broken by:
      1. higher overlap count wins (drop more-redundant CLOs first)
      2. among equal overlap, prefer dropping a K CLO over an S CLO over an A CLO
         (A CLOs are the rarest and should be preserved)
      3. among equal KSA, prefer dropping the one with the LATER index
         (so earlier CLOs — usually CLO1, CLO2 — tend to survive)

    If no CLO shares any keyword with any other (all six unique topics), fall
    back to dropping the last CLO. Returns a list with exactly 5 CLOs.
    """
    if len(cos) <= 5:
        return cos
    if len(cos) > 6:
        # If somehow more than 6, keep the first 5 after the same logic
        # applied recursively. Not expected in practice.
        keep = list(cos)
        while len(keep) > 5:
            keep = _trim_to_five(keep[:6]) + keep[6:]
        return keep

    # Exactly 6: score each CLO by how redundant it is.
    keywords = [_topic_keywords(co.description) for co in cos]
    verbs = [co.description.strip().split()[0].lower().strip(",.:;") for co in cos]
    ksa = [_classify_verb(co.description) for co in cos]

    # Prefer dropping K over S over A when redundancy ties.
    ksa_priority = {"K": 0, "S": 1, "A": 2}  # lower number = drop first

    def _overlap_count(i: int) -> int:
        count = 0
        for j, kw_j in enumerate(keywords):
            if i == j:
                continue
            if not keywords[i]:
                continue
            if verbs[i] == verbs[j] or keywords[i] & kw_j:
                count += 1
        return count

    scores = [_overlap_count(i) for i in range(6)]

    # If every score is 0, all six are unique: drop the last CLO.
    if all(s == 0 for s in scores):
        return cos[:5]

    # Otherwise, pick the CLO to drop:
    #   - highest overlap count
    #   - then lowest ksa_priority value (K first)
    #   - then later index (so earlier CLOs tend to survive)
    drop_index = max(
        range(6),
        key=lambda i: (scores[i], -ksa_priority[ksa[i]], i),
    )
    return [co for idx, co in enumerate(cos) if idx != drop_index]


# Replacement descriptions used by _fix_clo_ksa_balance when it has to
# rewrite a CLO to hit the strict 2K / 2S / 1A target.
_REWRITE_TO_S = (
    "Implement and test the core data structures covered in this course "
    "using a modern systems programming language."
)
_REWRITE_TO_A = (
    "Collaborate in a team to design, test, and document a software artifact "
    "for the course capstone."
)
_REWRITE_TO_K = (
    "Explain the fundamental concepts underlying the topics covered in this course."
)


def _fix_clo_ksa_balance(cos: list) -> list:
    """Re-derive ksa_category from each CLO's first verb, then force the CLO
    set to EXACTLY 2 K, 2 S, and 1 A.

    Expects exactly 5 CLOs (call _trim_to_five first if you have 6).
    """
    if len(cos) != 5:
        raise ValueError(
            f"_fix_clo_ksa_balance expects exactly 5 CLOs; got {len(cos)}. "
            "Call _trim_to_five first."
        )

    # --- Step 1: re-derive category from verb ---
    for co in cos:
        co.ksa_category = _classify_verb(co.description)

    def _counts():
        c = {"K": 0, "S": 0, "A": 0}
        for co in cos:
            c[co.ksa_category] += 1
        return c

    # --- Step 3: ensure exactly 1 A ---
    counts = _counts()
    if counts["A"] == 0:
        s_targets = [co for co in cos if co.ksa_category == "S"]
        if s_targets:
            t = s_targets[-1]
            t.description = _REWRITE_TO_A
            t.bloom_level = "Apply"
            t.ksa_category = "A"

    # --- Step 4: ensure at least 2 S ---
    counts = _counts()
    while counts["S"] < 2:
        k_targets = [co for co in cos if co.ksa_category == "K"]
        if not k_targets:
            break
        v = k_targets[-1]
        v.description = _REWRITE_TO_S
        v.bloom_level = "Apply"
        v.ksa_category = "S"
        counts = _counts()

    # --- Step 5: force EXACTLY 2 K / 2 S / 1 A ---
    for _ in range(10):
        counts = _counts()
        if counts == {"K": 2, "S": 2, "A": 1}:
            break

        if counts["A"] > 1:
            extras = [co for co in cos if co.ksa_category == "A"]
            victim = extras[-1]
            victim.description = _REWRITE_TO_S
            victim.bloom_level = "Apply"
            victim.ksa_category = "S"
            continue

        if counts["A"] < 1:
            donors = [co for co in cos if co.ksa_category == "S"] or \
                     [co for co in cos if co.ksa_category == "K"]
            if donors:
                d = donors[-1]
                d.description = _REWRITE_TO_A
                d.bloom_level = "Apply"
                d.ksa_category = "A"
                continue

        if counts["S"] > 2:
            extras = [co for co in cos if co.ksa_category == "S"]
            victim = extras[-1]
            victim.description = _REWRITE_TO_K
            victim.bloom_level = "Understand"
            victim.ksa_category = "K"
            continue

        if counts["S"] < 2:
            donors = [co for co in cos if co.ksa_category == "K"]
            if donors:
                d = donors[-1]
                d.description = _REWRITE_TO_S
                d.bloom_level = "Apply"
                d.ksa_category = "S"
                continue

        if counts["K"] > 2:
            donors = [co for co in cos if co.ksa_category == "K"]
            if donors and counts["S"] < 2:
                d = donors[-1]
                d.description = _REWRITE_TO_S
                d.bloom_level = "Apply"
                d.ksa_category = "S"
                continue
            if donors and counts["A"] < 1:
                d = donors[-1]
                d.description = _REWRITE_TO_A
                d.bloom_level = "Apply"
                d.ksa_category = "A"
                continue
            if donors:
                d = donors[-1]
                d.description = _REWRITE_TO_S
                d.bloom_level = "Apply"
                d.ksa_category = "S"
                continue

    final = _counts()
    if final != {"K": 2, "S": 2, "A": 1}:
        raise ValueError(
            f"_fix_clo_ksa_balance could not reach 2K/2S/1A. Got {final}."
        )
    return cos


def _focus_aligned_clo(weeks: list, valid_clo_ids: list[str]) -> list:
    """Enforce the aligned-CLO focus rule.

    - Weeks 9 and 18 -> all valid CLO ids.
    - Every other week -> 1 to 3 CLO ids, favouring the least-used CLOs so
      the burden is spread evenly across the schedule.
    - Post-condition: every valid CLO id is referenced by at least one week.
    """
    valid_set = set(valid_clo_ids)
    usage = {cid: 0 for cid in valid_clo_ids}

    for wk in weeks:
        if wk.week_number in (9, 18):
            wk.aligned_clo = list(valid_clo_ids)
            for cid in wk.aligned_clo:
                usage[cid] += 1
            continue

        kept = [cid for cid in wk.aligned_clo if cid in valid_set]
        kept_sorted = sorted(set(kept), key=lambda c: (usage[c], c))
        chosen = kept_sorted[:3]

        if not chosen:
            chosen = [min(valid_clo_ids, key=lambda c: (usage[c], c))]

        wk.aligned_clo = chosen
        for cid in chosen:
            usage[cid] += 1

    unused = [cid for cid in valid_clo_ids if usage[cid] == 0]
    if unused:
        non_exam = [w for w in weeks if w.week_number not in (9, 18)]
        non_exam.sort(key=lambda w: len(w.aligned_clo))
        for cid in unused:
            target = non_exam.pop(0)
            if cid not in target.aligned_clo and len(target.aligned_clo) < 3:
                target.aligned_clo.append(cid)
                usage[cid] += 1
                non_exam.append(target)
                non_exam.sort(key=lambda w: len(w.aligned_clo))

    return weeks


def _enforce_assessment_variety(weeks: list) -> list:
    """Lock the assessment tool per week.

    - Week 9 -> "Midterm Exam"; Week 18 -> "Final Project Defense" (pinned).
    - Every other week: keep the LLM's tool if it is in ALLOWED_ASSESSMENTS
      AND that tool is still under the cap of 5 uses. Otherwise assign the
      allowed tool with the lowest current usage; ties break by pool order.
    - Post-condition: all 4 allowed tools appear, none used more than 5 times.
    """
    for wk in weeks:
        if wk.week_number == 9:
            wk.assessment = "Midterm Exam"
        elif wk.week_number == 18:
            wk.assessment = "Final Project Defense"

    usage = {t: 0 for t in ALLOWED_ASSESSMENTS}
    non_exam = [w for w in weeks if w.week_number not in (9, 18)]

    for wk in non_exam:
        original = (wk.assessment or "").strip()
        if original in ALLOWED_ASSESSMENTS and usage[original] < ASSESSMENT_CAP:
            chosen = original
        else:
            chosen = min(
                ALLOWED_ASSESSMENTS,
                key=lambda t: (usage[t], ALLOWED_ASSESSMENTS.index(t)),
            )
        wk.assessment = chosen
        usage[chosen] += 1

    assert all(usage[t] <= ASSESSMENT_CAP for t in ALLOWED_ASSESSMENTS), usage
    assert all(usage[t] >= 1 for t in ALLOWED_ASSESSMENTS), usage
    return weeks


def _harmonize_assessment_evidence(weeks: list) -> list:
    """Overwrite each week's evidence based on its (now-final) assessment tool."""
    for wk in weeks:
        tool = (wk.assessment or "").strip()
        if tool not in TOOL_TO_EVIDENCE:
            raise ValueError(
                f"Week {wk.week_number}: unknown assessment tool {tool!r}. "
                f"Expected one of {sorted(TOOL_TO_EVIDENCE)}."
            )
        wk.evidence = TOOL_TO_EVIDENCE[tool]
    return weeks


def _dedupe_advanced_topics(weeks: list, is_dsa: bool) -> list:
    """Rewrite weeks 10-17's topics from a fixed advanced-DSA bank.

    Only applies to DSA-family courses. For any other course family, the
    LLM's original topics are returned unchanged - the DSA bank would be
    nonsensical for e.g. a Database Systems syllabus.

    Rationale: qwen2.5:3b reliably ignores the "do not name them 'Advanced X'"
    rule in SYSTEM_PROMPT_PASS2 and produces a duplicate second half. Rather
    than fight the LLM with more re-prompting, we deterministically replace
    weeks 10-17 with a fixed list of genuinely new advanced DSA sub-topics.
    """
    if not is_dsa:
        return weeks
    bank = list(_ADVANCED_TOPIC_BANK)
    for i, wk in enumerate(w for w in weeks if 10 <= w.week_number <= 17):
        if i < len(bank):
            wk.topic = bank[i]
    return weeks


# Match 'Dijkstra' or "Dijkstra's" case-insensitively.
_DIJKSTRA_RE = re.compile(r"\bDijkstra'?s?\b", re.IGNORECASE)


def _sanitize_dijkstra_leak(weeks: list) -> list:
    """Clean up two related issues in LLO descriptions.

    1. Remove Dijkstra mentions from LLO descriptions of non-graph weeks.
    2. Collapse the 'a graph algorithm algorithm' duplication that earlier
       versions of this sanitizer left behind.

    Both passes are UNCONDITIONAL so this function is idempotent: running it
    again on an already-cleaned JSON is a no-op, and running it on a JSON
    that still has the duplication from a prior run will fix it even though
    'Dijkstra' is no longer present to trigger the replacement.
    """
    for wk in weeks:
        topic_lower = (wk.topic or "").lower()
        is_graph_week = (
            "graph" in topic_lower
            or "shortest path" in topic_lower
            or "spanning tree" in topic_lower
        )

        for llo in wk.lesson_outcomes:
            # Pass 1 (unconditional): collapse 'graph algorithm algorithm'
            # and any similar 'X X' stutter left by a prior sanitizer run.
            llo.description = llo.description.replace(
                "graph algorithm algorithm", "graph algorithm"
            )
            llo.description = llo.description.replace(
                "a graph algorithm algorithm", "a graph algorithm"
            )

            # Pass 2 (only for non-graph weeks): replace remaining Dijkstra
            # mentions with a neutral phrase, then immediately collapse any
            # duplication the substitution just created.
            if not is_graph_week and _DIJKSTRA_RE.search(llo.description):
                llo.description = _DIJKSTRA_RE.sub(
                    "a graph algorithm", llo.description
                )
                llo.description = llo.description.replace(
                    "graph algorithm algorithm", "graph algorithm"
                )
                llo.description = llo.description.replace(
                    "a graph algorithm algorithm", "a graph algorithm"
                )
    return weeks


# ===========================================================================
# Pipeline
# ===========================================================================
def generate_syllabus(course_prompt: str) -> dict:
    # ---- Pass 1 ----------------------------------------------------------
    print("=== PASS 1: metadata + CLOs ===")
    pass1 = _retry_loop(
        SYSTEM_PROMPT_PASS1,
        f"Generate course metadata and 4-6 CLOs for: {course_prompt}",
        _Pass1,
        "pass1",
    )
    pass1.course_outcomes = _trim_to_five(pass1.course_outcomes)
    pass1.course_outcomes = _distribute_plos(pass1.course_outcomes)
    pass1.course_outcomes = _fix_clo_ksa_balance(pass1.course_outcomes)
    print(f"  + {len(pass1.course_outcomes)} CLOs generated (trimmed to 5, PLOs distributed, 2K/2S/1A enforced)")

    # ---- Pass 2 ----------------------------------------------------------
    print("=== PASS 2: 18-week skeleton ===")
    clo_ids = [c.clo_id for c in pass1.course_outcomes]
    pass2_prompt = (
        f"Course: {course_prompt}\n"
        f"Valid CLO ids: {clo_ids}\n\n"
        "Generate the 18-week skeleton now. Every CLO must appear in at least one "
        "week's aligned_clo. Week 9 = MIDTERM, Week 18 = FINAL. Weeks 8-17 have "
        "DISTINCT topics. Every non-exam week lists 1-3 CLOs; weeks 9 and 18 list "
        "ALL CLOs. Every teaching_activities string must name a specific tool, "
        "language, or IDE (e.g. VS Code, Python 3.12, C++17, OpenGL 4.6, GDB, "
        "CMake, PostgreSQL 16). Every assessment must be ONE of: Quiz, Lab Rubric, "
        "Recitation, Project Rubric (except weeks 9 and 18). Every evidence must "
        "match its tool."
    )
    pass2 = _retry_loop(SYSTEM_PROMPT_PASS2, pass2_prompt, _Pass2, "pass2")
    print(f"  + {len(pass2.weekly_schedule)} weeks generated")

    # ---- Deterministic post-processing ----------------------------------
    print("=== POST-PROCESS: focus / variety / evidence / topics / dijkstra ===")
    is_dsa = _is_dsa_course(course_prompt)
    print(f"  + course-family gate: is_dsa={is_dsa}")
    pass2.weekly_schedule = _focus_aligned_clo(pass2.weekly_schedule, clo_ids)
    pass2.weekly_schedule = _enforce_assessment_variety(pass2.weekly_schedule)
    pass2.weekly_schedule = _harmonize_assessment_evidence(pass2.weekly_schedule)
    pass2.weekly_schedule = _dedupe_advanced_topics(pass2.weekly_schedule, is_dsa)
    pass2.weekly_schedule = _sanitize_dijkstra_leak(pass2.weekly_schedule)
    print("  + aligned_clo focused, assessments varied, evidence harmonized,")
    print("    advanced topics deduped (if DSA), Dijkstra leaks sanitized")

    # ---- Pass 3 ----------------------------------------------------------
    print("=== PASS 3: per-week lesson outcomes ===")
    full_weeks = []
    for wk in pass2.weekly_schedule:
        user_prompt = (
            f"Week {wk.week_number}: {wk.topic}\n"
            f"TLA: {wk.teaching_activities}\n"
            f"Assessment: {wk.assessment}\n\n"
            f"Generate 3 lesson outcomes (K, S, A) with llo_id prefix LLO{wk.week_number}. "
            "Every description MUST mention a concrete tool, library, or scenario "
            "(e.g. VS Code, GDB, Python 3.12, C++17, std::vector, OpenGL 4.6, "
            "PostgreSQL 16, CMake, pytest, Google Test)."
        )
        llo_list = _retry_loop(SYSTEM_PROMPT_PASS3, user_prompt, _LLOList, f"wk{wk.week_number}")
        full_weeks.append(
            WeeklyScheduleSchema(
                week_number=wk.week_number,
                topic=wk.topic,
                lesson_outcomes=llo_list.lesson_outcomes,
                teaching_activities=wk.teaching_activities,
                assessment=wk.assessment,
                evidence=wk.evidence,
                aligned_clo=wk.aligned_clo,
            )
        )

    # ---- Assemble + final validation ------------------------------------
    payload = SyllabusSchema(
        course_metadata=pass1.course_metadata,
        course_outcomes=pass1.course_outcomes,
        weekly_schedule=full_weeks,
    )
    return payload.model_dump()


# ===========================================================================
# Reprocess-only mode (Stage 1b)
# ===========================================================================
def reprocess_existing(path: Path) -> dict:
    """Read an existing syllabus JSON, apply the deterministic post-processors,
    re-validate, and return the cleaned dict. No LLM calls."""
    raw = json.loads(path.read_text(encoding="utf-8"))

    payload = SyllabusSchema(**raw)  # ensures the input is well-formed

    # --- Step 1: smart-trim to 5 CLOs if needed ---
    payload.course_outcomes = _trim_to_five(payload.course_outcomes)

    # --- Step 2: rebalance to strict 2K/2S/1A ---
    payload.course_outcomes = _fix_clo_ksa_balance(payload.course_outcomes)

    # --- Step 3: sanitize weeks ---
    # We do not have the original course prompt here, so we infer DSA-ness
    # from the course title + description stored in the payload.
    is_dsa = _is_dsa_course(
        f"{payload.course_metadata.course_title} "
        f"{payload.course_metadata.course_description}"
    )
    print(f"  + course-family gate: is_dsa={is_dsa}")

    clo_ids = [c.clo_id for c in payload.course_outcomes]
    payload.weekly_schedule = _focus_aligned_clo(payload.weekly_schedule, clo_ids)
    payload.weekly_schedule = _enforce_assessment_variety(payload.weekly_schedule)
    payload.weekly_schedule = _harmonize_assessment_evidence(payload.weekly_schedule)
    payload.weekly_schedule = _dedupe_advanced_topics(payload.weekly_schedule, is_dsa)
    payload.weekly_schedule = _sanitize_dijkstra_leak(payload.weekly_schedule)

    # --- Final validation ---
    cleaned = SyllabusSchema(
        course_metadata=payload.course_metadata,
        course_outcomes=payload.course_outcomes,
        weekly_schedule=payload.weekly_schedule,
    )
    return cleaned.model_dump()


# ===========================================================================
# CLI
# ===========================================================================
def main() -> int:
    parser = argparse.ArgumentParser(description="Generate or reprocess an 18-week OBE syllabus.")
    parser.add_argument(
        "--course",
        default=(
            "CS201: Data Structures and Algorithms. 3 credits. Prerequisite: CS101. "
            "Focus on arrays, linked lists, stacks, queues, trees, graphs, sorting, "
            "searching, recursion, dynamic programming, and algorithmic complexity."
        ),
    )
    parser.add_argument("--out", default="sample_validated_output.json")
    parser.add_argument(
        "--reprocess-only",
        action="store_true",
        help="Apply deterministic post-processors to an existing JSON. No LLM call.",
    )
    args = parser.parse_args()

    out_path = Path(args.out)

    if args.reprocess_only:
        if not out_path.exists():
            print(f"[ERROR] {out_path} not found for --reprocess-only.", file=sys.stderr)
            return 1

        backup = out_path.with_name(out_path.stem + "_pre_reprocess_backup.json")
        if not backup.exists():
            backup.write_text(out_path.read_text(encoding="utf-8"), encoding="utf-8")
            print(f"Backup written: {backup.resolve()}")

        print(f"Reprocessing {out_path.resolve()} (no LLM call)...")
        try:
            result = reprocess_existing(out_path)
        except Exception as exc:
            print(f"[ERROR] Reprocess failed: {exc}", file=sys.stderr)
            return 1
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\nRewritten: {out_path.resolve()}")
        print(f"CLOs: {len(result['course_outcomes'])}  Weeks: {len(result['weekly_schedule'])}")
        return 0

    try:
        result = generate_syllabus(args.course)
    except RuntimeError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nSaved to {out_path.resolve()}")
    print(f"CLOs: {len(result['course_outcomes'])}  Weeks: {len(result['weekly_schedule'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())