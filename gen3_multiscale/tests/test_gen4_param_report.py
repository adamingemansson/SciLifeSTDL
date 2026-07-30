from __future__ import annotations

from gen3_multiscale.gen4.conditioner import Gen4Conditioner
from gen3_multiscale.gen4.param_report import format_report_table, report_parameters
from gen3_multiscale.tests._gen4_fixtures import GEN4_MODEL_KWARGS


def test_frozen_context_arm_reports_gene_encoder_as_frozen():
    model = Gen4Conditioner(
        n_genes=6, gex_feature_dim=4, image_feature_dim=8, gex_feature_source="frozen_context",
        gex_context_embedding_dim=5, use_regional_he=False, global_context_source="none", **GEN4_MODEL_KWARGS,
    )
    report = report_parameters(model)
    assert report["by_submodule"]["gene_encoder"]["status"] == "frozen"
    assert report["by_submodule"]["gex_context_proj"]["status"] == "trainable"
    assert report["total_params"] > 0
    assert isinstance(format_report_table(report), str)


def test_weighted_linear_arm_reports_gene_encoder_as_trainable():
    model = Gen4Conditioner(
        n_genes=6, gex_feature_dim=4, image_feature_dim=8, use_regional_he=False,
        global_context_source="none", **GEN4_MODEL_KWARGS,
    )
    report = report_parameters(model)
    assert report["by_submodule"]["gene_encoder"]["status"] == "trainable"
