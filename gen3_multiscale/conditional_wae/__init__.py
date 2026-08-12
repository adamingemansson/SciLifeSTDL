"""Conditional H&E-to-GEX Wasserstein autoencoders."""

from gen3_multiscale.conditional_wae.inputs import FullImageExpressionInputs
from gen3_multiscale.conditional_wae.data import (
    ConditionalWAEMaskedGEXDataset,
    build_conditional_wae_example,
    conditional_wae_identity_collate,
)
from gen3_multiscale.conditional_wae.model import (
    Architecture1ImageConditioner,
    ConditionalGaussianPrior,
    ConditionalWAE,
    DeterministicSpatialPredictor,
    FiLMConditionedExpressionEncoder,
    FrozenGeneEmbeddingExpressionEncoder,
    LocalImageConditioner,
    conditional_imq_mmd,
    imq_mmd,
)
from gen3_multiscale.conditional_wae.structured_field import (
    CenteredGeneStructureArtifact,
    CenteredGeneStructureRefinement,
    fit_centered_organ_balanced_gene_structure,
    load_centered_gene_structure_artifact,
    save_centered_gene_structure_artifact,
)

__all__ = [
    "Architecture1ImageConditioner",
    "ConditionalGaussianPrior",
    "ConditionalWAE",
    "CenteredGeneStructureArtifact",
    "CenteredGeneStructureRefinement",
    "ConditionalWAEMaskedGEXDataset",
    "DeterministicSpatialPredictor",
    "FiLMConditionedExpressionEncoder",
    "FrozenGeneEmbeddingExpressionEncoder",
    "FullImageExpressionInputs",
    "LocalImageConditioner",
    "build_conditional_wae_example",
    "conditional_imq_mmd",
    "conditional_wae_identity_collate",
    "imq_mmd",
    "fit_centered_organ_balanced_gene_structure",
    "load_centered_gene_structure_artifact",
    "save_centered_gene_structure_artifact",
]
