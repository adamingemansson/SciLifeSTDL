import pytest
import torch

from gen3_multiscale.gen6.contract import GEN6_ARM_SPECS, gen6_cache_requirements, get_gen6_arm_spec
from gen3_multiscale.gen6.fusion import BidirectionalCrossAttentionFusion, MoMEFusion, SimpleFusion
from gen3_multiscale.gen6.preflight import audit_gen6_manifest_cache_coverage
from gen3_multiscale.models.losses import combined_reconstruction_loss


def test_gen6_contract_has_exact_twelve_unique_arms():
    assert list(GEN6_ARM_SPECS) == [f"gen6{letter}" for letter in "abcdefghijkl"]
    assert len({spec.description for spec in GEN6_ARM_SPECS.values()}) == 12


@pytest.mark.parametrize("arm", ["gen6b", "gen6c", "gen6f", "gen6g", "gen6h", "gen6i"])
def test_uni2_spot_arms_do_not_require_dense_uni2(arm):
    requirements = gen6_cache_requirements({"model": {"arm": arm}})
    assert requirements["uses_uni2_primary"]
    assert not requirements["uses_uni2_dense"]


def test_full_spatial_field_alone_requires_dense_uni2():
    requirements = gen6_cache_requirements({"model": {"arm": "gen6j"}})
    assert requirements["uses_uni2_primary"]
    assert requirements["uses_uni2_dense"]


def test_factorial_encoder_cache_requirements_are_exact():
    b = get_gen6_arm_spec("gen6b").cache_requirements()
    c = get_gen6_arm_spec("gen6c").cache_requirements()
    d = get_gen6_arm_spec("gen6d").cache_requirements()
    e = get_gen6_arm_spec("gen6e").cache_requirements()
    assert b["uses_uni2_primary"] and not b["uses_scfoundation"]
    assert c["uses_uni2_primary"] and c["uses_scfoundation"]
    assert d["uses_gigapath_dense"] and not d["uses_scfoundation"]
    assert e["uses_gigapath_dense"] and e["uses_scfoundation"]


@pytest.mark.parametrize("fusion_cls", [SimpleFusion, MoMEFusion, BidirectionalCrossAttentionFusion])
def test_fusion_shapes_and_gradients(fusion_cls):
    torch.manual_seed(3)
    fusion = fusion_cls(7, 5, 12)
    image = torch.randn(9, 7, requires_grad=True)
    gene = torch.randn(9, 5, requires_grad=True)
    available = torch.tensor([True, False, True, True, False, True, True, True, False])
    if fusion_cls is SimpleFusion:
        output = fusion(image, gene)
    else:
        output = fusion(image, gene, available)
    assert output.shape == (9, 12)
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    assert image.grad is not None and torch.isfinite(image.grad).all()
    assert gene.grad is not None and torch.isfinite(gene.grad).all()


def test_mome_missing_image_rows_are_invariant_to_image_values():
    torch.manual_seed(9)
    fusion = MoMEFusion(7, 5, 12)
    image = torch.randn(4, 7)
    gene = torch.randn(4, 5)
    available = torch.tensor([True, False, True, False])
    first = fusion(image, gene, available)
    mutated = image.clone()
    mutated[~available] = torch.randn_like(mutated[~available]) * 1e6
    second = fusion(mutated, gene, available)
    torch.testing.assert_close(first[~available], second[~available])


def test_gen6_rmse_pcc_loss_is_finite_with_constant_genes_and_has_gradients():
    prediction = torch.randn(8, 5, requires_grad=True)
    target = torch.randn(8, 5)
    target[:, 0] = 1.0
    result = combined_reconstruction_loss(
        prediction, target, torch.randn(8, 2), primary_mode="rmse_pcc", pcc_weight=0.1,
    )
    assert set(("total", "primary", "gradient", "rmse_loss", "pcc_loss")) <= set(result)
    assert all(torch.isfinite(value) for value in result.values())
    result["total"].backward()
    assert prediction.grad is not None and torch.isfinite(prediction.grad).all()


@pytest.mark.parametrize(
    ("arm", "missing_name"),
    (("gen6a", "stpath_checkpoint"), ("gen6d", "gigapath_checkpoint")),
)
def test_real_cache_preflight_fails_before_loading_samples_when_model_artifact_is_missing(
    tmp_path, arm, missing_name,
):
    config = {
        "model": {"arm": arm, "kind": "conditioner", "params": {}},
        "required_fingerprints": {
            "stpath_checkpoint": str(tmp_path / "missing_stpath.pt"),
            "stpath_gene_vocab": str(tmp_path / "missing_vocab.json"),
            "gigapath_checkpoint": str(tmp_path / "missing_gigapath.pt"),
        },
    }
    with pytest.raises(FileNotFoundError, match=missing_name):
        audit_gen6_manifest_cache_coverage(tmp_path, ["never_loaded"], config)
