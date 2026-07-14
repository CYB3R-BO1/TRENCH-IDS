"""Cross-dataset label harmonization and continual-learning task assignment.

This module is the single source of truth for turning the two datasets' raw,
inconsistent ``Attack`` strings (lowercase in ToN-IoT, verbose/misspelled in
CSE-CIC) into one canonical taxonomy, and for mapping each canonical class
onto its continual-learning task.

**Scope: 2 datasets (NF-ToN-IoT-v2, NF-CSE-CIC-IDS2018-v2), 6 attack classes,
4 tasks.** NF-UNSW-NB15-v2 and NF-BoT-IoT-v2 are dropped entirely (professor's
guidance to reduce implementation complexity — see docs/dataset-plan.md).
Reconnaissance (previously UNSW+BoT-IoT, isolated as T2) no longer exists in
either remaining dataset; Bot, BruteForce, and Infiltration (CSE-only,
116K-143K combined) fall well below the ~685K-3.8M range of the six kept
classes and are excluded rather than forced into the pool.

Task assignment is **similarity-driven**, not attack-family-driven: if two
similar attacks were trained jointly in the same task, the model would learn
both directly from labels and there would be nothing left to test
transferability on — the professor's objection to the original family-based
design (e.g. DoS+DDoS together).

Method (``trench_ids.similarity`` / ``trench_ids.task_design``): (1) sample
~5,000 rows per attack class (6-class candidate pool, every canonical class
with a combined count in the ~685K-3.8M range once pooled across ToN-IoT and
CSE-CIC-IDS2018 — see ``docs/attack-class-counts.md``), (2) compute each
class's mean feature vector over the 37 flow-statistic features, then
standardize those per-class means (z-score across the 6 class means, not the
underlying samples), (3) cosine similarity between the standardized means.

Task assignment is **isolate-and-bundle**, not minimum-total-similarity
pairing: classes that are all pairwise above a similarity threshold (a
"conflict clique") cannot avoid a conflicting co-location no matter how
they're split, so every member of the largest such clique gets its own
singleton task; the remaining classes (each with at least one compatible
partner) are paired off via minimum-weight matching, still respecting the
threshold. For this 6-class pool the only pair above threshold is
{Password, Injection} (cosine 0.420), so both are isolated as singleton
tasks; the remaining four classes pair off into DDoS+XSS and DoS+Scanning.
This grouping is stable across similarity thresholds 0.30-0.40 (the next-
highest pair, Scanning/XSS, sits at 0.288 — well outside that range) —
yielding **4 tasks**.

If the candidate pool or task assignment changes, change it here too --
nothing else should hardcode label strings or task numbers.
"""

from __future__ import annotations

BENIGN = "Benign"

# Raw ``Attack`` string (exactly as it appears in each CSV) -> canonical class.
# Every distinct raw string measured across ToN-IoT and CSE-CIC-IDS2018 is
# listed here; an unmapped string is treated as an error (see canonical_label).
RAW_TO_CANONICAL: dict[str, str] = {
    "Benign": BENIGN,
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
#   Backdoor, MITM, Ransomware: ToN-IoT classes below the 6-class candidate-
#   pool threshold.
#   Web Attacks: CSE-CIC class below threshold.
#   Bot, BruteForce, Infiltration: CSE-CIC classes at 116K-143K combined --
#   well below the ~685K-3.8M range of the six kept classes (see
#   docs/attack-class-counts.md).
EXCLUDED_CLASSES: frozenset[str] = frozenset(
    {
        "Backdoor",
        "MITM",
        "Ransomware",
        "Web Attacks",
        "Bot",
        "BruteForce",
        "Infiltration",
    }
)

# Canonical ATTACK class -> continual-learning task id (T1..T4). Isolate-and-
# bundle task assignment (docs/attack-class-counts.md, trench_ids.task_design):
# {Password, Injection} is the only pair above the 0.35 similarity threshold,
# so both are isolated as singleton tasks; the remaining four classes pair off
# via minimum-weight matching, still respecting the threshold. Stable across
# thresholds 0.30-0.40.
# Benign is deliberately NOT assigned a task: it is the shared negative/background
# class injected fresh into EVERY task, not a task of its own. A benign-only task
# is degenerate for a classifier (single class = no decision boundary, nothing to
# store in the relation memory) and redundant with the per-task benign. Excluded
# classes (see EXCLUDED_CLASSES) also have no task.
CANONICAL_TO_TASK: dict[str, int] = {
    "Password": 1,
    "Injection": 2,
    "DDoS": 3,
    "XSS": 3,
    "DoS": 4,
    "Scanning": 4,
}

# Human-readable task descriptions — the class(es) itself, since these tasks
# are similarity-driven groupings rather than attack-family themes.
TASK_THEMES: dict[int, str] = {
    1: "Password",
    2: "Injection",
    3: "DDoS + XSS",
    4: "DoS + Scanning",
}

# Datasets that contribute each task's *attack* classes, and therefore the
# datasets a task draws its fresh benign subset from. Derived from the
# per-class dataset provenance in docs/datasets.md §4. Password is ToN-IoT
# only (Password does not appear in CSE-CIC), so T1 is single-dataset; every
# other task spans both datasets.
TASK_DATASETS: dict[int, tuple[str, ...]] = {
    1: ("ToN",),
    2: ("ToN", "CSE"),
    3: ("ToN", "CSE"),
    4: ("ToN", "CSE"),
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
    """Return the task id (1..4) for a canonical class."""
    try:
        return CANONICAL_TO_TASK[canonical]
    except KeyError as exc:
        raise UnknownAttackLabel(f"Canonical class {canonical!r} has no task.") from exc


def is_excluded(canonical: str) -> bool:
    """Whether a canonical class is dropped before task assignment."""
    return canonical in EXCLUDED_CLASSES


def canonical_classes() -> list[str]:
    """All canonical class names — Benign (shared negative) then attack classes."""
    return [BENIGN, *CANONICAL_TO_TASK.keys()]


def attack_classes_for_task(task: int) -> list[str]:
    """Canonical attack classes assigned to a task (excludes Benign)."""
    return [c for c, t in CANONICAL_TO_TASK.items() if t == task and c != BENIGN]
