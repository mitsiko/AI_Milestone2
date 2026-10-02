# llm_engine.py
"""
LLM engine for the OBE Syllabus Generator (Milestone 2).

Three-pass pipeline, upgraded from Milestone 1 and hardened in Stage 7:
  Pass 1 - metadata + CLOs, with a tightened K/S/A distribution rule, a smart
           6->5 trim, and a strict 2K/2S/1A deterministic rebalancer.
  Pass 2 - 18-week skeleton, emitting `evidence` and requiring concrete
           tools/IDEs/languages in `teaching_activities`.
  Pass 3 - per-week lesson outcomes, requiring a concrete tool, library,
           or scenario AND a concept directly from the week's topic.

Stage 7 hardening (in this file):
  - _topic_keywords + _TOPIC_SYNONYMS + _llos_match_topic: after Pass 3 for
    a week, verify at least one LLO mentions a topic keyword or synonym.
  - _TOPIC_LLO_FALLBACK: a deterministic K/S/A triple per known topic, used
    to replace a drifted week's LLOs.
  - A tighter Pass 3 user prompt and stricter SYSTEM_PROMPT_PASS3 rules
    ("topic is FINAL; every LLO must reference it").

Post-processing order (unchanged):
  After Pass 1:
    _trim_to_five
    _distribute_plos
    _fix_clo_ksa_balance
  After Pass 2 (before Pass 3):
    _focus_aligned_clo
    _enforce_assessment_variety
    _harmonize_assessment_evidence
    _dedupe_advanced_topics   (gated by _is_dsa_course)
    _sanitize_dijkstra_leak
  After Pass 3, per week:
    _llos_match_topic -> maybe _TOPIC_LLO_FALLBACK replacement

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

- TOPIC FIDELITY (this rule outranks everything else):
  * The topic provided in the user prompt is FINAL. Do not substitute, invent,
    or recall topics from earlier weeks.
  * Every lesson outcome MUST mention a concept from the topic above.
  * If the topic is "Minimum Spanning Trees", every LLO must be about MSTs
    (Prim, Kruskal, union-find, weighted graphs). It must NOT mention sorting,
    Big O, arrays, recursion, or any other subject.
  * Do NOT mention Dijkstra unless the topic contains 'graph' or 'shortest path'.
  * Do NOT mention sorting algorithms unless the topic contains 'sort'.
  * Do NOT mention recursion unless the topic contains 'recursion'.
  * Do NOT mention Big O notation unless the topic contains 'complexity' or 'analysis'.

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
# Deterministic post-processors
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
_DSA_KEYWORDS = (
    "data structure", "data structures", "algorithm", "algorithms",
    "dsa", "algorithmic",
)


def _is_dsa_course(course_prompt: str) -> bool:
    """Return True if the course prompt looks like a DSA-family course."""
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
# gated behind _is_dsa_course() so non-DSA syllabi are left untouched.
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

# ---- Stopwords used by _trim_to_five and _topic_keywords ------------------
_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with",
    "by", "from", "as", "at", "into", "that", "this", "their", "its",
    "apply", "implement", "configure", "calculate", "analyze", "differentiate",
    "compare", "solve", "demonstrate", "use", "execute", "design", "develop",
    "integrate", "construct", "synthesize", "formulate", "create",
    "identify", "define", "explain", "describe", "recall", "list", "recognize",
    "collaborate", "coordinate", "uphold", "respect", "practice", "advocate",
    "commit", "exhibit", "value", "reflect", "appreciate", "using", "use",
    "course", "problem", "problems", "code", "programming",
}

# ---- Topic keyword extraction (Stage 7) -----------------------------------
def _topic_keywords(topic: str) -> set[str]:
    """Extract meaningful lowercase keywords from a topic string.

    Filters out stopwords and single-character tokens. Used by the topic-match
    validator to decide whether an LLO mentions the week's subject.
    """
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_\-]+", topic.lower())
    return {t for t in tokens if t not in _STOPWORDS and len(t) > 2}


# ---- Topic synonym/related-term map (Stage 7) -----------------------------
# Maps a topic substring (lowercased) to a set of related terms that should
# also count as "matching" the topic. Keys are substring tests, so a topic
# like "Graph Traversal Algorithms" matches the "graph" entry.
_TOPIC_SYNONYMS: dict[str, set[str]] = {
    "graph": {
        "graph", "graphs", "bfs", "dfs", "breadth", "depth", "visited",
        "adjacency", "queue", "stack", "traversal", "traverse", "vertex",
        "vertices", "edge", "edges",
    },
    "shortest path": {
        "shortest", "path", "paths", "dijkstra", "bellman", "ford",
        "priority", "heap", "relaxation", "weighted",
    },
    "spanning tree": {
        "spanning", "tree", "trees", "prim", "kruskal", "union", "find",
        "disjoint", "weighted", "mst",
    },
    "hash": {
        "hash", "hashes", "hashing", "bucket", "collision", "chaining",
        "probing", "load", "factor", "key", "value",
    },
    "balanced tree": {
        "balanced", "avl", "red", "black", "rotation", "height", "balance",
        "tree", "trees", "insertion", "deletion",
    },
    "greedy": {
        "greedy", "local", "optimal", "exchange", "argument", "activity",
        "selection", "huffman", "fractional", "knapsack", "interval",
        "scheduling",
    },
    "backtracking": {
        "backtracking", "branch", "bound", "state", "space", "pruning",
        "constraint", "satisfaction", "queens", "sudoku", "n-queens",
    },
    "capstone": {
        "capstone", "project", "scoping", "documentation", "version",
        "control", "git", "readme", "repository", "deliverable",
    },
    "midterm": {
        "midterm", "review", "synthesis", "self", "assessment", "reflection",
        "recap", "consolidation",
    },
    "final project defense": {
        "final", "defense", "presentation", "integration", "review",
        "portfolio", "demonstration", "capstone",
    },
    "sort": {
        "sort", "sorting", "quicksort", "mergesort", "heapsort", "bubble",
        "insertion", "selection", "cocktail", "shaker", "merge", "quick",
        "partition", "pivot",
    },
    "search": {
        "search", "searching", "binary", "linear", "a*", "heuristic",
        "open", "closed",
    },
    "recursion": {
        "recursion", "recursive", "tail", "base", "case", "call", "stack",
        "factorial", "fibonacci", "hanoi",
    },
    "dynamic programming": {
        "dynamic", "programming", "memoization", "memoize", "subproblem",
        "subproblems", "overlapping", "table", "bottom-up", "top-down",
        "knapsack", "fibonacci",
    },
    "complexity": {
        "complexity", "big", "notation", "asymptotic", "time", "space",
        "amortized", "worst", "average", "best",
    },
    "array": {
        "array", "arrays", "linked", "list", "lists", "contiguous",
        "non-contiguous", "memory", "allocation", "pointer",
    },
    "stack": {
        "stack", "stacks", "queue", "queues", "lifo", "fifo", "push", "pop",
        "enqueue", "dequeue",
    },
    "tree": {
        "tree", "trees", "binary", "search", "bst", "node", "leaf", "root",
        "traversal", "insertion", "deletion",
    },
}

# ---- Topic required-concept map (Stage 10) --------------------------------
# For some topics, matching a topic keyword is too weak: the LLM can mention
# the topic by name while actually teaching a different concept. This map
# adds a STRICTER concept check that runs after _llos_match_topic.
#
# Each key is a substring of a topic (lowercased). Each value is a set of
# CONCRETE concept terms; at least one must appear in the week's LLO
# descriptions for the week to be accepted. If none does, the week is
# replaced with the fallback for its topic.
#
# Note: meta mentions like "greedy algorithm" or "greedy strategy" are NOT
# in the required set, because they can appear in LLOs whose content is
# actually about a different topic (e.g. LCS, DP).
_TOPIC_REQUIRED_CONCEPTS: dict[str, set[str]] = {
    "greedy": {
        "huffman",
        "activity selection",
        "activity-selection",
        "fractional knapsack",
        "fractional-knapsack",
        "interval scheduling",
        "interval-scheduling",
        "coin change",
        "coin-change",
        "exchange argument",
        "exchange-argument",
        "greedy choice property",
        "minimum spanning tree",  # Prim/Kruskal are greedy algorithms
        "prim",
        "kruskal",
    },
    "dynamic programming": {
        "memoization",
        "memoize",
        "tabulation",
        "overlapping subproblem",
        "overlapping-subproblem",
        "optimal substructure",
        "optimal-substructure",
        "bottom-up",
        "top-down",
        "knapsack",
        "longest common subsequence",
        "longest-common-subsequence",
        "lcs",
    },
    "backtracking": {
        "state space",
        "state-space",
        "pruning",
        "constraint satisfaction",
        "constraint-satisfaction",
        "n-queens",
        "eight queens",
        "eight-queens",
        "branch and bound",
        "branch-and-bound",
        "sudoku",
    },
}


def _llos_match_topic(topic: str, llos: list) -> bool:
    """Return True if at least one LLO mentions a topic keyword or synonym."""
    keywords = _topic_keywords(topic)
    synonyms: set[str] = set()
    topic_lower = topic.lower()
    for needle, syns in _TOPIC_SYNONYMS.items():
        if needle in topic_lower:
            synonyms |= syns
    acceptable = keywords | synonyms
    if not acceptable:
        # No keywords and no synonyms -> cannot judge; accept.
        return True

    for llo in llos:
        text = llo.description.lower()
        if any(term in text for term in acceptable):
            return True
    return False

def _llos_satisfy_required_concepts(topic: str, llos: list) -> bool:
    """Return True if the week's LLOs satisfy the topic's required-concept check."""
    topic_lower = (topic or "").lower()
    required: set[str] = set()
    for needle, concepts in _TOPIC_REQUIRED_CONCEPTS.items():
        if needle in topic_lower:
            required |= concepts

    if not required:
        return True

    joined = " ".join(llo.description.lower() for llo in llos)
    return any(concept in joined for concept in required)

# ---- Deterministic topic -> LLO fallback table (Stage 7) ------------------
# Each entry is a K/S/A triple. Every text:
#   (a) starts with a Bloom's verb from the matching category,
#   (b) names a concrete tool / language / IDE,
#   (c) mentions a concept from the topic.


def _fallback_llos_for_topic(topic: str, week_number: int) -> list:
    """Return a K/S/A triple whose text references the week's topic by name.

    This fallback is intentionally topic-agnostic so the pipeline works for
    any course (DSA, Database Systems, Networking, Graphics, etc.). The
    topic string is the only input needed; every description mentions it
    verbatim, which guarantees the fallback would itself pass the
    _llos_match_topic check (idempotent replacement).
    """
    return [
        LessonOutcomeSchema(
            llo_id=f"LLO{week_number}.1",
            description=(
                f"Explain the core concepts, terminology, and purpose of {topic}."
            ),
            ksa_category="K",
        ),
        LessonOutcomeSchema(
            llo_id=f"LLO{week_number}.2",
            description=(
                f"Apply the principles of {topic} to solve a small, worked "
                f"example in a lab setting."
            ),
            ksa_category="S",
        ),
        LessonOutcomeSchema(
            llo_id=f"LLO{week_number}.3",
            description=(
                f"Collaborate with peers to review, critique, and improve an "
                f"implementation of {topic}."
            ),
            ksa_category="A",
        ),
    ]

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


def _trim_to_five(cos: list) -> list:
    """Reduce a 6-CLO list to 5 by dropping the most-redundant CLO.

    Redundancy score for CLO i = number of OTHER CLOs whose (verb, keyword-set)
    overlaps with CLO i. Ties broken by:
      1. higher overlap count wins
      2. among equal overlap, prefer dropping a K CLO over an S CLO over an A CLO
      3. among equal KSA, prefer dropping the one with the LATER index
    """
    if len(cos) <= 5:
        return cos
    if len(cos) > 6:
        keep = list(cos)
        while len(keep) > 5:
            keep = _trim_to_five(keep[:6]) + keep[6:]
        return keep

    keywords = [
        {t for t in re.findall(r"[A-Za-z][A-Za-z0-9_]+", co.description.lower())
         if t not in _STOPWORDS and len(t) > 2}
        for co in cos
    ]
    verbs = [co.description.strip().split()[0].lower().strip(",.:;") for co in cos]
    ksa = [_classify_verb(co.description) for co in cos]
    ksa_priority = {"K": 0, "S": 1, "A": 2}

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
    if all(s == 0 for s in scores):
        return cos[:5]

    drop_index = max(
        range(6),
        key=lambda i: (scores[i], -ksa_priority[ksa[i]], i),
    )
    return [co for idx, co in enumerate(cos) if idx != drop_index]

def _renumber_clo_ids(cos: list, weeks: list | None = None):
    """Renumber CLOs to CLO1..CLOn in list order and remap aligned_clo refs.

    Called after _trim_to_five so the syllabus never ships with a gap in
    the CLO id sequence (e.g. CLO1, CLO2, CLO4, CLO5, CLO6 -> CLO1..CLO5).

    If `weeks` is provided, every aligned_clo entry is remapped through the
    same permutation. References to ids that no longer exist are dropped
    (should not happen given upstream integrity checks).

    Returns (cos, weeks, old_to_new).
    """
    old_to_new = {}
    for i, co in enumerate(cos, start=1):
        old_to_new[co.clo_id] = f"CLO{i}"

    for co in cos:
        co.clo_id = old_to_new[co.clo_id]

    if weeks is not None:
        for wk in weeks:
            wk.aligned_clo = [
                old_to_new[ref] for ref in wk.aligned_clo if ref in old_to_new
            ]

    return cos, weeks, old_to_new

# Replacement descriptions used by _fix_clo_ksa_balance when it has to
# rewrite a CLO to hit the strict 2K / 2S / 1A target.
_REWRITE_TO_S = (
    "Apply the fundamental techniques covered in this course to solve "
    "representative problems in a lab setting."
)
_REWRITE_TO_A = (
    "Collaborate with a team to design, test, and document a course deliverable."
)
_REWRITE_TO_K = (
    "Explain the fundamental concepts, terminology, and purpose of the topics "
    "covered in this course."
)


def _fix_clo_ksa_balance(cos: list) -> list:
    """Re-derive ksa_category from each CLO's first verb, then force the CLO
    set to EXACTLY 2 K, 2 S, and 1 A."""
    if len(cos) != 5:
        raise ValueError(
            f"_fix_clo_ksa_balance expects exactly 5 CLOs; got {len(cos)}. "
            "Call _trim_to_five first."
        )

    for co in cos:
        co.ksa_category = _classify_verb(co.description)

    def _counts():
        c = {"K": 0, "S": 0, "A": 0}
        for co in cos:
            c[co.ksa_category] += 1
        return c

    counts = _counts()
    if counts["A"] == 0:
        s_targets = [co for co in cos if co.ksa_category == "S"]
        if s_targets:
            t = s_targets[-1]
            t.description = _REWRITE_TO_A
            t.bloom_level = "Apply"
            t.ksa_category = "A"

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
    - Every other week -> 1 to 3 CLO ids, favouring the least-used CLOs.
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
    """Lock the assessment tool per week with a balanced distribution.

    - Week 9 -> "Midterm Exam"; Week 18 -> "Final Project Defense" (pinned).
    - Every other week: keep the LLM's tool when it is in ALLOWED_ASSESSMENTS
      AND that tool is still under ASSESSMENT_CAP.
    - Then rebalance: any tool used more than MAX_USES times has its LATEST
      occurrences reassigned to the least-used allowed tool.
    - Post-condition: every allowed tool appears between MIN_USES and
      MAX_USES times inclusive across the 16 non-exam weeks.

    For 16 weeks / 4 tools, MIN_USES=3 and MAX_USES=5 give balanced
    distributions like 4/4/4/4, 3/4/4/5, or 3/3/5/5.
    """
    MIN_USES = 3
    MAX_USES = 5

    # --- Step 1: pin exam weeks ------------------------------------------
    for wk in weeks:
        if wk.week_number == 9:
            wk.assessment = "Midterm Exam"
        elif wk.week_number == 18:
            wk.assessment = "Final Project Defense"

    non_exam = [w for w in weeks if w.week_number not in (9, 18)]

    # --- Step 2: keep the LLM's tool while under cap ---------------------
    usage = {t: 0 for t in ALLOWED_ASSESSMENTS}
    for wk in non_exam:
        original = (wk.assessment or "").strip()
        if original in ALLOWED_ASSESSMENTS and usage[original] < ASSESSMENT_CAP:
            wk.assessment = original
        else:
            wk.assessment = min(
                ALLOWED_ASSESSMENTS,
                key=lambda t: (usage[t], ALLOWED_ASSESSMENTS.index(t)),
            )
        usage[wk.assessment] += 1

    # --- Step 3: rebalance overused tools --------------------------------
    # Steal the LATEST occurrence of the most-overused tool and give it to
    # the least-used tool. Repeating converges in 1-3 iterations.
    for _ in range(10):
        counts = {t: 0 for t in ALLOWED_ASSESSMENTS}
        for wk in non_exam:
            counts[wk.assessment] += 1

        over = [t for t in ALLOWED_ASSESSMENTS if counts[t] > MAX_USES]
        if not over:
            break

        donor = max(over, key=lambda t: counts[t])
        recipient = min(
            ALLOWED_ASSESSMENTS,
            key=lambda t: (counts[t], ALLOWED_ASSESSMENTS.index(t)),
        )
        donor_weeks = [w for w in non_exam if w.assessment == donor]
        if not donor_weeks:
            break
        # Reassign the latest occurrence so earlier weeks keep their choice.
        donor_weeks[-1].assessment = recipient

    # --- Step 4: sanity check only (no [3, 5] assertion here) ------------
    # The final [MIN_USES, MAX_USES] balance is asserted LATER, after
    # _project_tool_only_for_project_weeks runs. Asserting here would be
    # checking the wrong moment.
    for wk in non_exam:
        if wk.assessment not in ALLOWED_ASSESSMENTS:
            raise ValueError(
                f"Week {wk.week_number}: invalid assessment {wk.assessment!r}. "
                f"Expected one of {ALLOWED_ASSESSMENTS}."
            )

    return weeks

# ---- Project-flavored topics that justify a "Project Rubric" assessment ---
_PROJECT_TOPIC_HINTS = (
    "capstone",
    "project",
    "defense",
    "portfolio",
    "checkpoint",
    "milestone",
    "integration",
)


def _project_tool_only_for_project_weeks(weeks: list) -> list:
    """Reassign 'Project Rubric' away from weeks whose topic isn't project-like.

    A non-exam week whose assessment is 'Project Rubric' but whose topic does
    NOT contain any of _PROJECT_TOPIC_HINTS is reassigned to the least-used
    of the three remaining allowed tools. This keeps tool counts balanced
    because it runs after _enforce_assessment_variety.

    Exam weeks are pinned:
      - Week 9  -> 'Midterm Exam'          (unaffected)
      - Week 18 -> 'Final Project Defense' (unaffected)

    The evidence field is re-derived later by _harmonize_assessment_evidence,
    so no evidence fixups are needed here.
    """
    non_exam = [w for w in weeks if w.week_number not in (9, 18)]
    other_tools = [t for t in ALLOWED_ASSESSMENTS if t != "Project Rubric"]

    def _is_project_topic(topic: str) -> bool:
        t = (topic or "").lower()
        return any(hint in t for hint in _PROJECT_TOPIC_HINTS)

    for wk in non_exam:
        if wk.assessment != "Project Rubric":
            continue
        if _is_project_topic(wk.topic):
            continue
        # Reassign to the least-used of the remaining three tools, counting
        # current assignments across all non-exam weeks.
        counts = {t: 0 for t in other_tools}
        for w in non_exam:
            if w.assessment in counts:
                counts[w.assessment] += 1
        recipient = min(other_tools, key=lambda t: (counts[t], other_tools.index(t)))
        wk.assessment = recipient

    return weeks

def _assert_balanced_distribution(weeks: list) -> list:
    """Final sanity check on the assessment distribution.

    Rules (varying per tool):
      - Quiz, Lab Rubric, Recitation: each between 3 and 5 uses across the 16
        non-exam weeks.
      - Project Rubric: between 1 and 5 uses, and only on weeks whose topic
        contains a project-flavored hint (enforced earlier by
        _project_tool_only_for_project_weeks; here we only bound the count).

    Runs AFTER _enforce_assessment_variety and
    _project_tool_only_for_project_weeks so the bounds are checked on the
    FINAL state.
    """
    CONTENT_TOOLS = {"Quiz", "Lab Rubric", "Recitation"}
    CONTENT_MIN = 3
    CONTENT_MAX = 5
    PROJECT_MIN = 1
    PROJECT_MAX = 5

    non_exam = [w for w in weeks if w.week_number not in (9, 18)]
    counts = {t: 0 for t in ALLOWED_ASSESSMENTS}
    for w in non_exam:
        if w.assessment not in ALLOWED_ASSESSMENTS:
            raise ValueError(
                f"Week {w.week_number}: invalid assessment {w.assessment!r}."
            )
        counts[w.assessment] += 1

    total = sum(counts.values())
    if total != len(non_exam):
        raise ValueError(
            f"Assessment count mismatch: {total} != {len(non_exam)}."
        )

    for tool in CONTENT_TOOLS:
        if counts[tool] > CONTENT_MAX:
            raise ValueError(
                f"{tool} overused: {counts[tool]} > {CONTENT_MAX}. Full: {counts}."
            )
        if counts[tool] < CONTENT_MIN:
            raise ValueError(
                f"{tool} underused: {counts[tool]} < {CONTENT_MIN}. Full: {counts}."
            )

    if counts["Project Rubric"] > PROJECT_MAX:
        raise ValueError(
            f"Project Rubric overused: {counts['Project Rubric']} > {PROJECT_MAX}. "
            f"Full: {counts}."
        )
    if counts["Project Rubric"] < PROJECT_MIN:
        raise ValueError(
            f"Project Rubric underused: {counts['Project Rubric']} < {PROJECT_MIN}. "
            f"Full: {counts}."
        )

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



def _dedupe_advanced_topics_if_redundant(weeks: list, is_dsa: bool) -> list:
    """Replace weeks 10-17 topics with the DSA bank ONLY when they are
    redundant with weeks 1-8.

    Two redundancy patterns are detected:
      1. Exact duplicate (case-insensitive substring): a week 10-17 topic
         contains a week 1-8 topic (or vice versa).
      2. "Advanced X" pattern where X is a week 1-8 topic, e.g. week 4 is
         "Sorting Algorithms" and week 12 is "Advanced Sorting Algorithms".

    If any redundant week is found, ALL weeks 10-17 are replaced from the
    bank (in order), because partial replacement would leave the schedule
    inconsistent. If no redundancy is found, the LLM's original topics are
    preserved verbatim.

    For non-DSA courses (is_dsa=False), this function is a no-op.
    """
    if not is_dsa:
        return weeks

    # Collect weeks 1-8 topic strings, lowercased, for the redundancy test.
    weeks_1_8_topics = [
        (w.topic or "").strip().lower()
        for w in weeks
        if 1 <= w.week_number <= 8 and (w.topic or "").strip()
    ]

    def _is_redundant(topic: str) -> bool:
        t = (topic or "").strip().lower()
        if not t:
            return False
        t_stripped = t
        if t_stripped.startswith("advanced "):
            t_stripped = t_stripped[len("advanced "):].strip()
        for earlier in weeks_1_8_topics:
            if not earlier:
                continue
            if earlier in t or t in earlier:
                return True
            if earlier in t_stripped or t_stripped in earlier:
                return True
        return False

    weeks_10_17 = [w for w in weeks if 10 <= w.week_number <= 17]
    has_redundancy = any(_is_redundant(w.topic) for w in weeks_10_17)

    if not has_redundancy:
        return weeks

    bank = list(_ADVANCED_TOPIC_BANK)
    for i, wk in enumerate(weeks_10_17):
        if i < len(bank):
            wk.topic = bank[i]
    return weeks


def _regenerate_tla_from_topic(weeks: list) -> list:
    """Rebuild teaching_activities from the FINAL topic so the TLA always matches.

    Runs after any topic rewriting (e.g. the DSA bank). Uses the same
    lecture+lab structure the spec requires, and names a concrete tool per
    week using a small deterministic tool rotation.
    """
    tools = [
        "VS Code",
        "PyCharm",
        "C++17 with GDB",
        "Python 3.12",
        "Jupyter Notebook",
        "CMake with Google Test",
        "PostgreSQL 16",
        "Git",
        "Valgrind",
        "Catch2",
    ]
    for i, wk in enumerate(weeks):
        topic = (wk.topic or "").strip()
        tool = tools[i % len(tools)]
        wk.teaching_activities = (
            f"Face-to-face lecture on {topic} + Hands-on Lab in {tool}: "
            f"applying {topic} to a worked problem"
        )
    return weeks

# Match 'Dijkstra' or "Dijkstra's" case-insensitively.
_DIJKSTRA_RE = re.compile(r"\bDijkstra'?s?\b", re.IGNORECASE)


# Match 'Dijkstra' or "Dijkstra's" case-insensitively.
_DIJKSTRA_RE = re.compile(r"\bDijkstra'?s?\b", re.IGNORECASE)


def _sanitize_dijkstra_leak(weeks: list) -> list:
    """Clean up LLO descriptions with Dijkstra in non-graph weeks.

    Detection: any 'Dijkstra' mention in a week whose topic does not contain
    'graph', 'shortest path', or 'spanning tree' is drift. Substitution
    produces incoherent text (e.g. 'Apply a graph algorithm in a scenario
    where AVL trees are used...'), so we replace the entire LLO with the
    topic-appropriate fallback for its KSA category.

    Also collapses the legacy 'graph algorithm algorithm' duplication,
    which is unconditional and idempotent.
    """
    for wk in weeks:
        topic_lower = (wk.topic or "").lower()
        is_graph_week = (
            "graph" in topic_lower
            or "shortest path" in topic_lower
            or "spanning tree" in topic_lower
        )

        fallback_by_ksa = {
            llo.ksa_category: llo
            for llo in _fallback_llos_for_topic(wk.topic, wk.week_number)
        }

        for llo in wk.lesson_outcomes:
            # Unconditional: collapse any legacy duplication.
            llo.description = llo.description.replace(
                "graph algorithm algorithm", "graph algorithm"
            )
            llo.description = llo.description.replace(
                "a graph algorithm algorithm", "a graph algorithm"
            )

            # Dijkstra in a non-graph week = drift. Replace the LLO.
            if is_graph_week or not _DIJKSTRA_RE.search(llo.description):
                continue

            replacement = fallback_by_ksa.get(llo.ksa_category)
            if replacement is not None:
                llo.description = replacement.description
                print(
                    f"    [wk{wk.week_number} LLO{llo.llo_id} replaced: "
                    f"Dijkstra in a non-graph week]"
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
    pass1.course_outcomes, _ = _renumber_clo_ids(pass1.course_outcomes)
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

    # ---- Pre-Pass-3 deterministic post-processing -----------------------
    # These operate on _SkeletonWeek fields only (topic, assessment, evidence,
    # aligned_clo). Do NOT call _sanitize_dijkstra_leak here: it needs LLOs.
    print("=== POST-PROCESS (pre-Pass-3): focus / variety / evidence / topics ===")
    is_dsa = _is_dsa_course(course_prompt)
    print(f"  + course-family gate: is_dsa={is_dsa}")
    pass2.weekly_schedule = _focus_aligned_clo(pass2.weekly_schedule, clo_ids)
    pass2.weekly_schedule = _enforce_assessment_variety(pass2.weekly_schedule)
    pass2.weekly_schedule = _project_tool_only_for_project_weeks(pass2.weekly_schedule)
    pass2.weekly_schedule = _assert_balanced_distribution(pass2.weekly_schedule)
    pass2.weekly_schedule = _harmonize_assessment_evidence(pass2.weekly_schedule)
    pass2.weekly_schedule = _dedupe_advanced_topics_if_redundant(pass2.weekly_schedule, is_dsa)
    pass2.weekly_schedule = _regenerate_tla_from_topic(pass2.weekly_schedule)
    # NOTE: _sanitize_dijkstra_leak runs AFTER Pass 3, because it operates on
    # lesson_outcomes, which do not exist on _SkeletonWeek objects yet.
    print("  + aligned_clo focused, assessments varied, evidence harmonized,")
    print("    advanced topics deduped (if DSA), TLAs regenerated from final topics")

    # ---- Pass 3 ----------------------------------------------------------
    print("=== PASS 3: per-week lesson outcomes ===")
    full_weeks = []
    replacements = 0
    for wk in pass2.weekly_schedule:
        user_prompt = (
            f'Generate exactly 3 lesson outcomes for a class week whose ONLY topic is: '
            f'"{wk.topic}".\n'
            f'Every outcome MUST mention a concept directly related to "{wk.topic}".\n'
            f'Do NOT mention any other DSA topic (no Dijkstra unless the topic contains '
            f"'graph' or 'shortest path'; no sorting unless the topic contains 'sort'; "
            f"no recursion unless the topic contains 'recursion'; etc.).\n"
            f"The topic is FINAL and cannot be changed.\n\n"
            f"Assessment for the week: {wk.assessment}\n"
            f"Teaching activities: {wk.teaching_activities}\n\n"
            f"Every description MUST also mention a concrete tool, library, or scenario "
            f"(e.g. VS Code, GDB, Python 3.12, C++17, std::vector, OpenGL 4.6, "
            f"PostgreSQL 16, CMake, pytest, Google Test).\n"
            f"Use llo_id prefix LLO{wk.week_number} and return a JSON object with a "
            f"single 'lesson_outcomes' array of exactly 3 items (K, S, A)."
        )
        llo_list = _retry_loop(
            SYSTEM_PROMPT_PASS3, user_prompt, _LLOList, f"wk{wk.week_number}"
        )

        # ---- Stage 7: topic-match validation -------------------------
        # Two checks, in order: (1) does the week mention its topic at all,
        # and (2) for topics with required concepts, does it actually teach
        # those concepts? Both must pass.
        if not _llos_match_topic(wk.topic, llo_list.lesson_outcomes):
            fallback = _fallback_llos_for_topic(wk.topic, wk.week_number)
            llo_list.lesson_outcomes = fallback
            replacements += 1
            print(
                f"    [replaced wk{wk.week_number} with fallback LLOs: topic mismatch]"
            )
        elif not _llos_satisfy_required_concepts(wk.topic, llo_list.lesson_outcomes):
            fallback = _fallback_llos_for_topic(wk.topic, wk.week_number)
            llo_list.lesson_outcomes = fallback
            replacements += 1
            print(
                f"    [replaced wk{wk.week_number} with fallback LLOs: "
                f"required-concept mismatch]"
            )

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

    if replacements:
        print(f"  + Pass 3 topic check: {replacements} week(s) replaced with fallback LLOs.")
    else:
        print("  + Pass 3 topic check: all 18 weeks passed.")

    # ---- Post-Pass-3 cleanup: Dijkstra leak sanitization ----------------
    # NOW the weeks are WeeklyScheduleSchema objects with lesson_outcomes,
    # so _sanitize_dijkstra_leak can safely iterate them.
    full_weeks = _sanitize_dijkstra_leak(full_weeks)
    print("  + Dijkstra leak sanitization applied.")

    # ---- Assemble + final validation ------------------------------------
    payload = SyllabusSchema(
        course_metadata=pass1.course_metadata,
        course_outcomes=pass1.course_outcomes,
        weekly_schedule=full_weeks,
    )
    return payload.model_dump()


# ===========================================================================
# Reprocess-only mode
# ===========================================================================
def reprocess_existing(path: Path) -> dict:
    """Apply deterministic post-processors to an existing JSON. No LLM calls.

    Note: the Stage 7 topic-match validator is intentionally NOT applied here,
    because reprocessing does not regenerate LLOs. Use the full pipeline to
    apply the fallback table.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    payload = SyllabusSchema(**raw)

    payload.course_outcomes = _trim_to_five(payload.course_outcomes)
    payload.course_outcomes, payload.weekly_schedule, _ = _renumber_clo_ids(
        payload.course_outcomes, payload.weekly_schedule
    )
    payload.course_outcomes = _fix_clo_ksa_balance(payload.course_outcomes)

    is_dsa = _is_dsa_course(
        f"{payload.course_metadata.course_title} "
        f"{payload.course_metadata.course_description}"
    )
    print(f"  + course-family gate: is_dsa={is_dsa}")

    clo_ids = [c.clo_id for c in payload.course_outcomes]
    payload.weekly_schedule = _focus_aligned_clo(payload.weekly_schedule, clo_ids)
    payload.weekly_schedule = _enforce_assessment_variety(payload.weekly_schedule)
    payload.weekly_schedule = _project_tool_only_for_project_weeks(payload.weekly_schedule)
    payload.weekly_schedule = _assert_balanced_distribution(payload.weekly_schedule)
    payload.weekly_schedule = _harmonize_assessment_evidence(payload.weekly_schedule)
    payload.weekly_schedule = _dedupe_advanced_topics_if_redundant(payload.weekly_schedule, is_dsa)
    payload.weekly_schedule = _regenerate_tla_from_topic(payload.weekly_schedule)

    # Stage 10: apply the required-concept check on the reprocessed weeks so
    # `--reprocess-only` produces the same quality as a full pipeline run.
    for wk in payload.weekly_schedule:
        if not _llos_satisfy_required_concepts(wk.topic, wk.lesson_outcomes):
            fallback = _fallback_llos_for_topic(wk.topic, wk.week_number)
            wk.lesson_outcomes = fallback
            print(
                f"  + reprocess: wk{wk.week_number} replaced "
                f"(required-concept mismatch)"
            )

    payload.weekly_schedule = _sanitize_dijkstra_leak(payload.weekly_schedule)

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
    parser = argparse.ArgumentParser(
        description="Generate or reprocess an 18-week OBE syllabus."
    )
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
        print(
            f"CLOs: {len(result['course_outcomes'])}  "
            f"Weeks: {len(result['weekly_schedule'])}"
        )
        return 0

    try:
        result = generate_syllabus(args.course)
    except RuntimeError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nSaved to {out_path.resolve()}")
    print(
        f"CLOs: {len(result['course_outcomes'])}  "
        f"Weeks: {len(result['weekly_schedule'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())