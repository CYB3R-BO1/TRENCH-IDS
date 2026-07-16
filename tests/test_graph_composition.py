from __future__ import annotations

import pytest
import torch
from torch_geometric.data import HeteroData

from trench_ids.graph_composition import (
    build_matrix,
    combination_label,
    dominant_class,
    graph_class_counts,
    graph_class_proportions,
    graph_class_set,
    load_task_graphs,
    task_average_class_proportions,
    task_combination_counts,
    task_dominant_class_counts,
)
from trench_ids.labels import canonical_classes


def _graph(labels: list[str]) -> HeteroData:
    """One flow per entry in `labels`; label_names/y built the same way
    trench_ids.graphs.build_task_graph does (sorted uniques + index map)."""
    label_names = sorted(set(labels))
    lookup = {name: i for i, name in enumerate(label_names)}
    g = HeteroData()
    g["flow"].x = torch.zeros((len(labels), 1))
    g["flow"].label_names = label_names
    g["flow"].y = torch.tensor([lookup[label] for label in labels], dtype=torch.int64)
    return g


def test_graph_class_set_reads_label_names() -> None:
    g = _graph(["Benign", "DDoS"])
    assert graph_class_set(g) == frozenset({"Benign", "DDoS"})


def test_graph_class_counts_tallies_flows_per_class() -> None:
    g = _graph(["DoS", "DoS", "DoS", "Injection", "Benign"])
    counts = graph_class_counts(g)
    assert counts == {"DoS": 3, "Injection": 1, "Benign": 1}


def test_dominant_class_picks_the_majority() -> None:
    from collections import Counter

    assert dominant_class(Counter({"DoS": 3, "Injection": 1})) == "DoS"


def test_dominant_class_reports_equal_on_tie() -> None:
    from collections import Counter

    assert dominant_class(Counter({"DoS": 2, "Injection": 2})) == "Equal"


def test_dominant_class_empty_counts_is_equal() -> None:
    from collections import Counter

    assert dominant_class(Counter()) == "Equal"


def test_graph_class_proportions_computes_share_of_flows() -> None:
    g = _graph(["DoS", "DoS", "DoS", "Injection"])
    props = graph_class_proportions(g)
    assert props == {"DoS": 0.75, "Injection": 0.25}


def test_graph_class_proportions_empty_graph_is_empty_dict() -> None:
    g = _graph([])
    assert graph_class_proportions(g) == {}


def test_task_average_class_proportions_weights_graphs_equally() -> None:
    # Graph 1: 90/10 DoS/Injection. Graph 2: 50/50. A flow-weighted average
    # would differ from this if the graphs had different flow counts --
    # here they're equal size so it also sanity-checks the simple case.
    graphs = [
        _graph(["DoS"] * 9 + ["Injection"] * 1),
        _graph(["DoS"] * 5 + ["Injection"] * 5),
    ]

    avg = task_average_class_proportions(graphs)

    assert avg["DoS"] == pytest.approx((0.9 + 0.5) / 2)
    assert avg["Injection"] == pytest.approx((0.1 + 0.5) / 2)


def test_task_average_class_proportions_weights_small_and_large_graphs_equally() -> None:
    # A 10-flow graph that's 100% DoS and a 100-flow graph that's 100%
    # Injection should average to 50/50 -- NOT 10/100 flow-weighted.
    graphs = [_graph(["DoS"] * 10), _graph(["Injection"] * 100)]

    avg = task_average_class_proportions(graphs)

    assert avg == {"DoS": pytest.approx(0.5), "Injection": pytest.approx(0.5)}


def test_task_average_class_proportions_empty_list_is_empty_dict() -> None:
    assert task_average_class_proportions([]) == {}


def test_task_dominant_class_counts_tallies_across_graphs() -> None:
    graphs = [
        _graph(["DoS", "DoS", "Injection"]),  # DoS dominant
        _graph(["DoS", "DoS", "Injection"]),  # DoS dominant
        _graph(["Injection", "Injection", "DoS"]),  # Injection dominant
        _graph(["DoS", "Injection"]),  # tie -> Equal
    ]

    counts = task_dominant_class_counts(graphs)

    assert counts == {"DoS": 2, "Injection": 1, "Equal": 1}


def test_task_combination_counts_separates_pure_and_mixed() -> None:
    graphs = [
        _graph(["Benign"]),
        _graph(["Benign"]),
        _graph(["DoS"]),
        _graph(["DoS", "Injection"]),
        _graph(["DoS", "Injection", "Benign"]),
    ]

    counts = task_combination_counts(graphs)

    assert counts[frozenset({"Benign"})] == 2
    assert counts[frozenset({"DoS"})] == 1
    assert counts[frozenset({"DoS", "Injection"})] == 1
    assert counts[frozenset({"DoS", "Injection", "Benign"})] == 1
    assert sum(counts.values()) == 5


def test_combination_label_formats_pure_and_mixed() -> None:
    assert combination_label(frozenset({"Benign"})) == "pure Benign"
    assert combination_label(frozenset({"DoS", "Injection"})) == "mix of DoS + Injection"
    assert (
        combination_label(frozenset({"Injection", "Benign", "DoS"}))
        == "mix of Benign + DoS + Injection"
    )


def test_build_matrix_diagonal_is_pure_counts_and_off_diagonal_is_symmetric() -> None:
    task_combos = {
        4: task_combination_counts(
            [
                _graph(["Benign"]),
                _graph(["Benign"]),
                _graph(["DoS"]),
                _graph(["DoS", "Injection"]),
                _graph(["DoS", "Injection", "Benign"]),
            ]
        )
    }
    classes = canonical_classes()

    matrix = build_matrix(task_combos, classes)

    assert matrix.shape == (len(classes), len(classes))
    # Diagonal: pure counts only (the 3-way graph does NOT count toward
    # "pure DoS" even though DoS is present in it).
    assert matrix.loc["Benign", "Benign"] == 2
    assert matrix.loc["DoS", "DoS"] == 1
    assert matrix.loc["Injection", "Injection"] == 0
    # Off-diagonal: co-occurrence inclusive of the 3-way graph -- DoS+Injection
    # appears together in both the pure pair graph and the 3-way graph.
    assert matrix.loc["DoS", "Injection"] == 2
    assert matrix.loc["Injection", "DoS"] == 2
    assert matrix.loc["DoS", "Benign"] == 1
    assert matrix.loc["Injection", "Benign"] == 1
    # A class from an unrelated task never co-occurs.
    assert matrix.loc["Scanning", "DoS"] == 0
    # Symmetric.
    assert (matrix.to_numpy() == matrix.to_numpy().T).all()


def test_build_matrix_sums_across_multiple_tasks() -> None:
    task_combos = {
        1: task_combination_counts([_graph(["Scanning"]), _graph(["Benign"])]),
        2: task_combination_counts([_graph(["Reconnaissance", "Benign"])]),
    }
    classes = canonical_classes()

    matrix = build_matrix(task_combos, classes)

    assert matrix.loc["Scanning", "Scanning"] == 1
    assert matrix.loc["Benign", "Benign"] == 1  # only task 1's pure-Benign graph
    assert matrix.loc["Reconnaissance", "Benign"] == 1
    # Classes from different tasks never co-occur, even though both involve Benign.
    assert matrix.loc["Scanning", "Reconnaissance"] == 0


def test_load_task_graphs_combines_all_splits(tmp_path) -> None:
    out_dir = tmp_path
    torch.save([_graph(["Benign"])], out_dir / "task_1_train.pt")
    torch.save([_graph(["Scanning"]), _graph(["Benign"])], out_dir / "task_1_val.pt")
    torch.save([_graph(["Scanning"])], out_dir / "task_1_test.pt")

    graphs = load_task_graphs(out_dir, 1)

    assert len(graphs) == 4
