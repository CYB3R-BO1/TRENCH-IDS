"""Cross-dataset label harmonization and continual-learning task assignment.

This module is the single source of truth for turning the three datasets'
raw, inconsistent ``Attack`` strings into one canonical taxonomy, for
restricting each canonical class to its sanctioned source dataset(s), and
for mapping each canonical class onto its continual-learning task.

**Scope: 3 datasets (NF-ToN-IoT-v2, NF-CSE-CIC-IDS2018-v2, NF-BoT-IoT-v2),
10 attack classes, 6 tasks.** NF-UNSW-NB15-v2 stays excluded — no class
survives dropping it once BoT-IoT is back in (its only relevant
contribution, Reconnaissance, is 99.5% supplied by BoT-IoT anyway). This
supersedes the prior 2-dataset/6-class/4-task reduction: the professor's
guidance changed from minimizing complexity to building a richer benchmark
(more tasks, more classes, no pre-graph sampling cap). See
``docs/superpowers/specs/2026-07-13-dataset-task-respec-design.md`` for the
full design and ``docs/dataset-plan.md`` for the authoritative write-up.

**Per-dataset class-source restriction (``CLASS_DATASETS``):** a canonical
class's rows are only ever drawn from the dataset(s) listed for it here,
even if the same raw label also appears in another dataset. This matters
because BoT-IoT independently contains rows labeled ``DDoS``/``DoS`` — but
BoT-IoT's only sanctioned contribution to this benchmark is Reconnaissance;
its DDoS/DoS rows must not leak into those classes and inflate/bias them.
``preprocess.py`` and ``similarity.py`` both thread this restriction through
their per-dataset streaming passes.

Task assignment is **similarity-driven**, not attack-family-driven: if two
similar attacks were trained jointly in the same task, the model would learn
both directly from labels and there would be nothing left to test
transferability on.

Method (``trench_ids.similarity`` / ``trench_ids.task_design``): (1) sample
~5,000 rows per attack class (10-class candidate pool — every canonical
class with "enough usable samples" once its ``CLASS_DATASETS`` restriction
is applied; the three smallest, Bot/BruteForce/Infiltration at 116K-143K,
are included for benchmark richness despite being an order of magnitude
below the other seven), (2) compute each class's mean feature vector over
the 37 flow-statistic features, then standardize those per-class means
(z-score across the 10 class means, not the underlying samples), (3) cosine
similarity between the standardized means.

Task assignment is **isolate-and-bundle** with a **size-aware tie-break**:
classes that are all pairwise above a similarity threshold (a "conflict
clique") cannot avoid a conflicting co-location no matter how they're
split, so every member of the largest such clique gets its own singleton
task; the remaining classes are paired off via minimum-weight matching,
still respecting the threshold, but among pairings that tie on the
threshold constraint the one minimizing the *largest resulting task's
size* wins (total similarity is only the secondary tie-break) --
see ``trench_ids.task_design.min_weight_grouping``'s ``sizes`` parameter.
For this 10-class pool, threshold 0.35 (stable across 0.21-0.55) isolates
{Scanning} and {Reconnaissance}; among the 78 threshold-valid pairings of
the remaining 8 classes, {DDoS, Infiltration}, {DoS, Injection}, {Password,
Bot}, {XSS, BruteForce} minimizes the largest task's size (3.79M flows,
bounded by Scanning itself -- no valid pairing can do better) instead of
total pairwise similarity, which happened to select the single
worst-balanced valid pairing ({XSS, DDoS} at 5.88M flows, the design's
original choice). See
docs/superpowers/specs/2026-07-14-benchmark-finalization-design.md §1 for
the full enumeration.

If the candidate pool or task assignment changes, change it here too --
nothing else should hardcode label strings, dataset restrictions, or task
numbers.
"""

from __future__ import annotations

BENIGN = "Benign"

# Raw ``Attack`` string (exactly as it appears in each CSV) -> canonical class.
# Every distinct raw string measured across ToN-IoT, CSE-CIC-IDS2018, and
# BoT-IoT is listed here; an unmapped string is treated as an error (see
# canonical_label).
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
    # --- NF-BoT-IoT-v2 (PascalCase) ---
    "DDoS": "DDoS",
    "DoS": "DoS",
    "Reconnaissance": "Reconnaissance",
    "Theft": "Theft",
}

# Canonical class -> the dataset code(s) it is allowed to draw rows from.
# Required because a canonical label can appear in more than one dataset's
# raw Attack column without being a sanctioned source for that class here
# (BoT-IoT's DDoS/DoS rows are the motivating case — see module docstring).
# Dataset codes match trench_ids.preprocess / configs/*.yaml's
# ``datasets: <dir_name>: <code>`` mapping (ToN, CSE, BoT).
CLASS_DATASETS: dict[str, set[str]] = {
    "Scanning": {"ToN"},
    "XSS": {"ToN"},
    "Password": {"ToN"},
    "DDoS": {"ToN", "CSE"},
    "DoS": {"ToN", "CSE"},
    "Injection": {"ToN", "CSE"},
    "Reconnaissance": {"BoT"},
    "Bot": {"CSE"},
    "BruteForce": {"CSE"},
    "Infiltration": {"CSE"},
}

# Canonical attack classes dropped entirely before task assignment. Their raw
# strings still map in RAW_TO_CANONICAL (so mapping never fails), but rows
# are filtered out during preprocessing and the class is assigned no task.
#   Backdoor, MITM, Ransomware: ToN-IoT classes below the candidate-pool
#   threshold.
#   Web Attacks: CSE-CIC class below threshold.
#   Theft: BoT-IoT class, not in the 10-class candidate pool (2,431 rows —
#   far below even Bot/BruteForce/Infiltration's 116K-143K floor).
# Bot, BruteForce, and Infiltration are deliberately NOT here any more (they
# were excluded in the prior 4-task design; this design includes them for
# benchmark richness — see module docstring).
EXCLUDED_CLASSES: frozenset[str] = frozenset(
    {"Backdoor", "MITM", "Ransomware", "Web Attacks", "Theft"}
)

# Canonical ATTACK class -> continual-learning task id (T1..T6). Size-aware
# isolate-and-bundle task assignment (module docstring,
# docs/superpowers/specs/2026-07-14-benchmark-finalization-design.md §1):
# {Scanning} and {Reconnaissance} are singleton (isolated) tasks; the
# remaining eight classes pair off minimizing the largest resulting task's
# size (3.79M flows), all respecting the 0.35 similarity threshold.
# Benign is deliberately NOT assigned a task: it is the shared negative/
# background class injected fresh into EVERY task, not a task of its own.
CANONICAL_TO_TASK: dict[str, int] = {
    "Scanning": 1,
    "Reconnaissance": 2,
    "DDoS": 3,
    "Infiltration": 3,
    "DoS": 4,
    "Injection": 4,
    "Password": 5,
    "Bot": 5,
    "XSS": 6,
    "BruteForce": 6,
}

# Human-readable task descriptions — the class(es) itself, since these tasks
# are similarity-driven groupings rather than attack-family themes.
TASK_THEMES: dict[int, str] = {
    1: "Scanning",
    2: "Reconnaissance",
    3: "DDoS + Infiltration",
    4: "DoS + Injection",
    5: "Password + Bot",
    6: "XSS + BruteForce",
}

# Datasets that contribute each task's *attack* classes, and therefore the
# datasets a task draws its fresh benign subset from. Derived from
# CLASS_DATASETS. T1 (ToN-only) and T2 (BoT-IoT-only) are single-dataset;
# every other task now spans both ToN and CSE (T3-T6 each pair a
# ToN-restricted or multi-dataset class with a CSE-restricted one).
TASK_DATASETS: dict[int, tuple[str, ...]] = {
    1: ("ToN",),
    2: ("BoT",),
    3: ("ToN", "CSE"),
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
    """Return the task id (1..6) for a canonical class."""
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
