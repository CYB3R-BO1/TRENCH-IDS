"""Cross-dataset label harmonization and continual-learning task assignment.

This module is the single source of truth for turning the four datasets' raw,
inconsistent ``Attack`` strings (lowercase in ToN-IoT, verbose/misspelled in
CSE-CIC, PascalCase elsewhere) into one canonical taxonomy, and for mapping each
canonical class onto its continual-learning task.

Task assignment is **similarity-driven**, not attack-family-driven (superseding
the original Plan v3 family grouping, e.g. DoS+DDoS in one task). Rationale: if
two similar attacks are trained jointly in the same task, the model learns both
directly from labels and there is nothing left to test transferability on — the
professor's objection to the original design. Per the professor's spec: (1)
sample ~5,000 rows per attack class (14-class candidate pool, every canonical
class with >=10,000 combined samples across the four datasets — see
``docs/attack-class-counts.md``), (2) compute each class's mean feature vector
over the 37 flow-statistic features, then standardize those per-class means
(z-score across the 14 class means, not the underlying samples), (3) cosine
similarity between the standardized means (``trench_ids.similarity``). Task
pairing (``trench_ids.task_design``) then pairs the two classes that are *most
dissimilar*, subject to never co-locating any pair above a similarity
threshold. The resulting pairing was verified stable across thresholds 0.3-0.7.

If the candidate pool or pairing changes, change it here too — nothing else
should hardcode label strings or task numbers.
"""

from __future__ import annotations

BENIGN = "Benign"

# Raw ``Attack`` string (exactly as it appears in each CSV) -> canonical class.
# Every distinct raw string measured across the four datasets on 2026-07-10 is
# listed here; an unmapped string is treated as an error (see canonical_label).
RAW_TO_CANONICAL: dict[str, str] = {
    "Benign": BENIGN,
    # --- NF-UNSW-NB15-v2 (PascalCase) ---
    "Exploits": "Exploits",
    "Fuzzers": "Fuzzers",
    "Generic": "Generic",
    "Reconnaissance": "Reconnaissance",
    "DoS": "DoS",
    "Analysis": "Analysis",
    "Backdoor": "Backdoor",
    "Shellcode": "Shellcode",
    "Worms": "Worms",
    # --- NF-ToN-IoT-v2 (lowercase) ---
    "scanning": "Scanning",
    "xss": "XSS",
    "ddos": "DDoS",
    "password": "Password",
    "dos": "DoS",
    "injection": "Injection",
    "backdoor": "Backdoor",
    "mitm": "MITM",
    "ransomware": "Ransomware",
    # --- NF-BoT-IoT-v2 (PascalCase) ---
    "DDoS": "DDoS",
    "Theft": "Theft",
    # --- NF-CSE-CIC-IDS2018-v2 (verbose / inconsistent / misspelled) ---
    "DDOS attack-HOIC": "DDoS",
    "DoS attacks-Hulk": "DoS",
    "DDoS attacks-LOIC-HTTP": "DDoS",
    "Bot": "Bot",
    "Infilteration": "Infiltration",  # sic — misspelled in the raw data
    "SSH-Bruteforce": "BruteForce",
    "DoS attacks-GoldenEye": "DoS",
    "FTP-BruteForce": "BruteForce",
    "DoS attacks-SlowHTTPTest": "DoS",
    "DoS attacks-Slowloris": "DoS",
    "Brute Force -Web": "Web Attacks",
    "DDOS attack-LOIC-UDP": "DDoS",
    "Brute Force -XSS": "Web Attacks",
    "SQL Injection": "Injection",
}

# Canonical attack classes dropped entirely before task assignment. Their raw
# strings still map in RAW_TO_CANONICAL (so mapping never fails), but rows are
# filtered out during preprocessing and the class is assigned to no task.
#   Worms: only 164 samples total (NB15 only) — too few for a reliable held-out
#   test split.
#   MITM, Ransomware, Web Attacks, Theft, Analysis, Shellcode: below the
#   10,000-combined-sample threshold used to build the similarity-analysis
#   candidate pool (docs/attack-class-counts.md, "Selected candidate pool: 14
#   classes at a 10K-sample threshold") — not enough signal to trust a
#   500-sample centroid, so excluded from task assignment rather than forced in.
EXCLUDED_CLASSES: frozenset[str] = frozenset(
    {"Worms", "MITM", "Ransomware", "Web Attacks", "Theft", "Analysis", "Shellcode"}
)

# Canonical ATTACK class -> continual-learning task id (T1..T7). Similarity-
# driven pairing (docs/attack-class-counts.md, trench_ids.task_design): each
# task pairs the two classes that are most dissimilar by cosine similarity of
# their per-class mean feature vectors (standardized across the 14-class
# means, over the 37 flow-statistic features -- see trench_ids.similarity).
# This is the global minimum-total-similarity perfect pairing and was verified
# stable across similarity thresholds 0.3-0.7 — no pair here exceeds -0.0
# similarity, let alone the professor's "too similar" concern that motivated
# this redesign (the original Plan v3 grouped DoS+DDoS, Reconnaissance+
# Scanning, and BruteForce+Injection together, all of which scored well above
# 0 similarity and are deliberately split apart below).
# Benign is deliberately NOT assigned a task: it is the shared negative/background
# class injected fresh into EVERY task, not a task of its own. A benign-only task
# is degenerate for a classifier (single class = no decision boundary, nothing to
# store in the relation memory) and redundant with the per-task benign. Excluded
# classes (see EXCLUDED_CLASSES) also have no task.
CANONICAL_TO_TASK: dict[str, int] = {
    "DDoS": 1,
    "XSS": 1,
    "DoS": 2,
    "Bot": 2,
    "Scanning": 3,
    "BruteForce": 3,
    "Reconnaissance": 4,
    "Exploits": 4,
    "Password": 5,
    "Generic": 5,
    "Injection": 6,
    "Backdoor": 6,
    "Infiltration": 7,
    "Fuzzers": 7,
}

# Human-readable task descriptions — the class pair itself, since these tasks
# are similarity-driven pairings rather than attack-family themes.
TASK_THEMES: dict[int, str] = {
    1: "DDoS + XSS",
    2: "DoS + Bot",
    3: "Scanning + BruteForce",
    4: "Reconnaissance + Exploits",
    5: "Password + Generic",
    6: "Injection + Backdoor",
    7: "Infiltration + Fuzzers",
}

# Datasets that contribute each task's *attack* classes, and therefore the
# datasets a task draws its fresh benign subset from. Derived from the
# per-class dataset provenance in docs/datasets.md §4.
TASK_DATASETS: dict[int, tuple[str, ...]] = {
    1: ("ToN", "BoT", "CSE"),
    2: ("UNSW", "ToN", "BoT", "CSE"),
    3: ("ToN", "CSE"),
    4: ("UNSW", "BoT"),
    5: ("ToN", "UNSW"),
    6: ("ToN", "CSE", "UNSW"),
    7: ("CSE", "UNSW"),
}

NUM_TASKS = len(TASK_THEMES)


class UnknownAttackLabel(KeyError):
    """Raised when a raw ``Attack`` string has no canonical mapping."""


def canonical_label(raw: str) -> str:
    """Map a raw ``Attack`` string to its canonical class.

    Raises UnknownAttackLabel on anything not in RAW_TO_CANONICAL, so an
    unexpected/new label fails loudly rather than being silently dropped.
    """
    try:
        return RAW_TO_CANONICAL[raw]
    except KeyError as exc:
        raise UnknownAttackLabel(
            f"Unmapped Attack label {raw!r}. Add it to RAW_TO_CANONICAL in "
            f"labels.py after confirming its attack family."
        ) from exc


def task_of(canonical: str) -> int:
    """Return the task id (1..7) for a canonical class."""
    try:
        return CANONICAL_TO_TASK[canonical]
    except KeyError as exc:
        raise UnknownAttackLabel(f"Canonical class {canonical!r} has no task.") from exc


def is_excluded(canonical: str) -> bool:
    """Whether a canonical class is dropped before task assignment (e.g. Worms)."""
    return canonical in EXCLUDED_CLASSES


def canonical_classes() -> list[str]:
    """All canonical class names — Benign (shared negative) then attack classes."""
    return [BENIGN, *CANONICAL_TO_TASK.keys()]


def attack_classes_for_task(task: int) -> list[str]:
    """Canonical attack classes assigned to a task (excludes Benign)."""
    return [c for c, t in CANONICAL_TO_TASK.items() if t == task and c != BENIGN]
