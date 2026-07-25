import pytest

from gen2_architectures.training import data_prep


def test_skips_a_sample_missing_genes_from_the_reference_panel(monkeypatch):
    """Real bug found 2026-07-25 on the actual training server: a
    held-out sample can pass resolve_compatible_sample_ids' coarser
    compatibility check yet still be missing a few genes from the exact
    train-derived reference panel -- load_multi_sample correctly raises
    rather than silently zero-filling, but that used to crash the WHOLE
    training run over one held-out sample. load_held_out_samples_with_images
    must skip just that sample instead."""
    def fake_load(cfg, sample_ids, reference_genes=None):
        assert len(sample_ids) == 1
        sid = sample_ids[0]
        if sid == "BAD1":
            raise ValueError(
                f"held-out sample {sid!r} is missing 21 genes from the fit-derived "
                "reference panel (examples: ['A', 'B']). Refusing to intersect with "
                "test data because that would make the test set participate in "
                "model-vocabulary selection."
            )
        return [f"adata-{sid}"], [f"images-{sid}"]

    monkeypatch.setattr(data_prep, "load_multi_sample_with_images", fake_load)

    kept_ids, adatas, images_list = data_prep.load_held_out_samples_with_images(
        cfg=object(), sample_ids=["GOOD1", "BAD1", "GOOD2"], reference_genes=["A", "B", "C"],
    )
    assert kept_ids == ["GOOD1", "GOOD2"]
    assert adatas == ["adata-GOOD1", "adata-GOOD2"]
    assert images_list == ["images-GOOD1", "images-GOOD2"]


def test_unrelated_value_errors_are_not_swallowed(monkeypatch):
    def fake_load(cfg, sample_ids, reference_genes=None):
        raise ValueError("some genuinely unrelated real bug")

    monkeypatch.setattr(data_prep, "load_multi_sample_with_images", fake_load)

    with pytest.raises(ValueError, match="genuinely unrelated real bug"):
        data_prep.load_held_out_samples_with_images(object(), ["S1"], ["A"])


def test_all_compatible_keeps_every_sample(monkeypatch):
    def fake_load(cfg, sample_ids, reference_genes=None):
        sid = sample_ids[0]
        return [f"adata-{sid}"], [f"images-{sid}"]

    monkeypatch.setattr(data_prep, "load_multi_sample_with_images", fake_load)

    kept_ids, adatas, images_list = data_prep.load_held_out_samples_with_images(
        object(), ["S1", "S2"], ["A"],
    )
    assert kept_ids == ["S1", "S2"]
    assert len(adatas) == 2
    assert len(images_list) == 2
