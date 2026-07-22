"""Gene subsets are metric views, not model-vocabulary mutations."""

import numpy as np

from src.evaluation.audit_evaluation import _resolve_gene_panels
from src.training.train import _resolved_evaluation_gene_panels


def test_gene_panels_preserve_requested_order_and_report_missing_names():
    indices, metadata = _resolve_gene_panels(
        ["A", "B", "C", "D"],
        {"comparison_50": ["C", "missing", "A", "C"]},
    )

    np.testing.assert_array_equal(indices["comparison_50"], np.asarray([2, 0]))
    assert metadata["comparison_50"] == {
        "requested_count": 3,
        "evaluated_count": 2,
        "genes": ["C", "A"],
        "missing_genes": ["missing"],
    }


def test_gene_panel_name_must_be_safe_for_metric_keys():
    try:
        _resolve_gene_panels(["A"], {"bad panel": ["A"]})
    except ValueError as exc:
        assert "letters, digits and underscores" in str(exc)
    else:
        raise AssertionError("unsafe panel name should fail closed")


def test_train_variance_panels_use_only_supplied_training_expression():
    cfg = {"evaluation": {"train_variance_gene_panel_sizes": [1, 2]}}
    train_samples = [
        (None, np.asarray([[0.0, 1.0, 0.0], [0.0, 3.0, 1.0]])),
        (None, np.asarray([[0.0, 5.0, 2.0]])),
    ]

    panels = _resolved_evaluation_gene_panels(cfg, train_samples, ["A", "B", "C"])

    assert panels["train_variance_top1"] == ["B"]
    assert panels["train_variance_top2"] == ["B", "C"]
