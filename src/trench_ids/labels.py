"""Cross-dataset label harmonization and continual-learning task assignment.

This module is the single source of truth for turning the four datasets' raw,
inconsistent ``Attack`` strings (lowercase in ToN-IoT, verbose/misspelled in
CSE-CIC, PascalCase elsewhere) into one canonical taxonomy, and for mapping each
canonical class onto its continual-learning task.

Task assignment is **similarity-driven**, not attack-family-driven (superseding
the original Plan v3 family grouping, e.g. DoS+DDoS in one task). Rationale: if
two similar attacks are trained jointly in the same task, the model learns both
directly from labels and there is nothing left to test transferability on — the
professor's objection to the original design.

Method (``trench_ids.similarity`` / ``trench_ids.task_design``): (1) sample
~5,000 rows per attack class (10-class candidate pool, every canonical class
with >=100,000 combined samples across the four datasets — see
``docs/attack-class-counts.md``), (2) compute each class's mean feature vector
over the 37 flow-statistic features, then standardize those per-class means
(z-score across the 10 class means, not the underlying samples), (3) cosine
similarity between the standardized means.

Task assignment is **isolate-and-bundle**, not minimum-total-similarity
pairing: classes that are all pairwise above a similarity threshold (a
"conflict clique") cannot avoid a conflicting co-location no matter how
they're split, so every member of the largest such clique gets its own
singleton task; the remaining classes (each with at least one compatible
partner) are paired off via minimum-weight matching, still respecting the
threshold. For this 10-class pool the clique is {DDoS, Reconnaissance} (plus
overlapping near-clique members at nearby thresholds), yielding **6 tasks**,
not 7 — verified stable across similarity thresholds 0.50-0.52; forcing a 7th
task would mean inventing a split with no similarity basis.

If the candidate pool or task assignment changes, change it here too —
nothing else should hardcode label strings or task numbers.
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
#   MITM, Ransomware, Web Attacks, Theft, Analysis, Shellcode, Exploits,
#   Fuzzers, Backdoor, Generic: below the 100,000-combined-sample threshold
#   used to build the similarity-analysis candidate pool
#   (docs/attack-class-counts.md, "Selected candidate pool: 10 classes at a
#   100K-sample threshold") — chosen over the looser 14-class/10K pool for
#   simpler task construction, per professor's guidance.
EXCLUDED_CLASSES: frozenset[str] = frozenset(
    {
        "Worms",
        "MITM",
        "Ransomware",
        "Web Attacks",
        "Theft",
        "Analysis",
        "Shellcode",
        "Exploits",
        "Fuzzers",
        "Backdoor",
        "Generic",
    }
)

# Canonical ATTACK class -> continual-learning task id (T1..T6). Isolate-and-
# bundle task assignment (docs/attack-class-counts.md, trench_ids.task_design):
# classes forming a mutual "conflict clique" (every pair above the similarity
# threshold) cannot avoid a conflict no matter how they're split, so each gets
# its own singleton task (DDoS, Reconnaissance); the remaining classes are
# paired via minimum-weight matching, still respecting the threshold. Verified
# stable across similarity thresholds 0.50-0.52 — this yields 6 tasks, not 7;
# the 10-class pool's clique structure doesn't support a 7th task without
# inventing a split with no similarity basis. This deliberately splits apart
# the pairs the professor flagged as too similar under the original family
# grouping (DoS/DDoS, Reconnaissance/Scanning, BruteForce/Injection all scored
# well above 0 similarity and never co-occur in a task here).
# Benign is deliberately NOT assigned a task: it is the shared negative/background
# class injected fresh into EVERY task, not a task of its own. A benign-only task
# is degenerate for a classifier (single class = no decision boundary, nothing to
# store in the relation memory) and redundant with the per-task benign. Excluded
# classes (see EXCLUDED_CLASSES) also have no task.
CANONICAL_TO_TASK: dict[str, int] = {
    "DDoS": 1,
    "Reconnaissance": 2,
    "DoS": 3,
    "Injection": 3,
    "Scanning": 4,
    "BruteForce": 4,
    "XSS": 5,
    "Bot": 5,
    "Password": 6,
    "Infiltration": 6,
}

# Human-readable task descriptions — the class(es) itself, since these tasks
# are similarity-driven groupings rather than attack-family themes.
TASK_THEMES: dict[int, str] = {
    1: "DDoS",
    2: "Reconnaissance",
    3: "DoS + Injection",
    4: "Scanning + BruteForce",
    5: "XSS + Bot",
    6: "Password + Infiltration",
}

# Datasets that contribute each task's *attack* classes, and therefore the
# datasets a task draws its fresh benign subset from. Derived from the
# per-class dataset provenance in docs/datasets.md §4.
TASK_DATASETS: dict[int, tuple[str, ...]] = {
    1: ("ToN", "BoT", "CSE"),
    2: ("UNSW", "BoT"),
    3: ("UNSW", "ToN", "BoT", "CSE"),
    4: ("ToN", "CSE"),
    5: ("ToN", "CSE"),
    6: ("ToN", "CSE"),
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
