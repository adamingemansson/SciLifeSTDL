"""
Smoke test for the independent cell-type-plausibility classifier
(src/evaluation/cell_type_classifier.py). Synthetic data with a clear
class structure — no real dataset needed for this check, just confirms
the classifier can actually learn and score plausibility above chance.

Run with: python -m tests.test_cell_type_classifier
"""
import numpy as np

from src.evaluation.cell_type_classifier import CellTypePlausibilityClassifier


def test_fit_predict_above_chance():
    rng = np.random.default_rng(0)
    n_per_class, n_genes, n_classes = 40, 20, 3

    # synthetic classes: each has a distinct mean expression offset, so a
    # classifier should trivially learn to separate them
    expression, labels = [], []
    for c in range(n_classes):
        offset = np.zeros(n_genes)
        offset[c * (n_genes // n_classes):(c + 1) * (n_genes // n_classes)] = 3.0
        expression.append(rng.normal(loc=offset, scale=0.5, size=(n_per_class, n_genes)))
        labels += [f"type_{c}"] * n_per_class
    expression = np.concatenate(expression, axis=0)
    labels = np.array(labels)

    clf = CellTypePlausibilityClassifier(n_estimators=50, seed=0)
    clf.fit(expression, labels)

    # score plausibility on the same clearly-separated synthetic classes —
    # should be well above chance (1/n_classes)
    acc = clf.plausibility_accuracy(expression, labels)
    chance = 1.0 / n_classes
    assert acc > chance, f"accuracy {acc:.3f} not above chance {chance:.3f}"
    print(f"[plausibility_accuracy] OK — {acc:.3f} (chance = {chance:.3f})")


if __name__ == "__main__":
    test_fit_predict_above_chance()
    print("\nAll cell-type classifier smoke tests passed.")
