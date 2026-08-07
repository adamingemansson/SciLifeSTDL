"""Static scientific contract for the supervisor conditional-WAE suite."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConditionalWAEArmSpec:
    task: str
    regularizer: str
    include_observed_gex: bool


ARM_SPECS = {
    "wae_he_mmd": ConditionalWAEArmSpec("he_to_st", "mmd", False),
    "wae_he_gan": ConditionalWAEArmSpec("he_to_st", "gan", False),
    "wae_he_st_mmd": ConditionalWAEArmSpec("he_plus_st_to_st", "mmd", True),
    "wae_he_st_gan": ConditionalWAEArmSpec("he_plus_st_to_st", "gan", True),
    # Training-hyperparameter ablation arms (accumulation / lr / compact
    # dims). Same immutable task/regularizer/include_observed_gex contract
    # as "wae_he_gan" -- only training.* and model.params.* differ between
    # them, which this static contract does not govern.
    "wae_he_gan_control": ConditionalWAEArmSpec("he_to_st", "gan", False),
    "wae_he_gan_accum8": ConditionalWAEArmSpec("he_to_st", "gan", False),
    "wae_he_gan_lr3e5": ConditionalWAEArmSpec("he_to_st", "gan", False),
    "wae_he_gan_small": ConditionalWAEArmSpec("he_to_st", "gan", False),
    # FiLM-conditioned-encoder architecture ablation arms. Same immutable
    # task/regularizer/include_observed_gex contract as "wae_he_gan" --
    # only model.params.encoder_conditioning/film_layers/film_shared_generator
    # differ, which this static contract does not govern beyond shape checks.
    "wae_he_gan_film_control": ConditionalWAEArmSpec("he_to_st", "gan", False),
    "wae_he_gan_film_first_only": ConditionalWAEArmSpec("he_to_st", "gan", False),
    "wae_he_gan_film_last_only": ConditionalWAEArmSpec("he_to_st", "gan", False),
    "wae_he_gan_film_shared": ConditionalWAEArmSpec("he_to_st", "gan", False),
    # Image-encoder backbone ablation: same immutable task/regularizer/
    # include_observed_gex contract as "wae_he_gan" -- only
    # data.image_encoder (and its matching pinned-revision field) differs,
    # which this static contract does not govern beyond the checks below.
    "wae_he_gan_uni2_control": ConditionalWAEArmSpec("he_to_st", "gan", False),
    "wae_he_gan_uni2": ConditionalWAEArmSpec("he_to_st", "gan", False),
    # Gene-coexpression decoder-side refinement ablation: same immutable
    # task/regularizer/include_observed_gex contract as "wae_he_gan" --
    # only model.params.use_gene_coexpression_refinement and
    # data.gene_coexpression_basis_path differ. Two treatment arms share
    # one control: "wae_he_gan_coexpression" (basis fit from-scratch on
    # this project's own training expression) and
    # "wae_he_gan_coexpression_scfoundation" (basis derived from
    # scFoundation's pretrained gene embeddings instead) -- the basis
    # SOURCE is opaque to this contract; both feed the identical
    # GeneResidualBasis/GeneCoexpressionRefinement machinery.
    "wae_he_gan_coexpression_control": ConditionalWAEArmSpec("he_to_st", "gan", False),
    "wae_he_gan_coexpression": ConditionalWAEArmSpec("he_to_st", "gan", False),
    "wae_he_gan_coexpression_scfoundation": ConditionalWAEArmSpec("he_to_st", "gan", False),
    # Histology-structure (deterministic multiscale morphology) context
    # injection ablation: same immutable task/regularizer/
    # include_observed_gex contract as "wae_he_gan" -- only
    # model.params.use_histology_context and data.use_histology_features
    # differ.
    "wae_he_gan_histology_control": ConditionalWAEArmSpec("he_to_st", "gan", False),
    "wae_he_gan_histology": ConditionalWAEArmSpec("he_to_st", "gan", False),
    # Gene-encoder x FiLM factorial: scFoundation's frozen per-gene
    # embedding table (real expression @ frozen_table.T -> trainable
    # projection, sourced from the already-fit scFoundation-derived
    # gene-coexpression basis artifact reused as the table here) versus
    # the plain from-scratch Linear(n_genes, hidden_dim) MLP encoder every
    # OTHER WAE arm already uses ("mlp" -- gene_encoder_source="linear",
    # no table/basis file needed at all). Crossed with encoder_conditioning
    # (FiLM on/off) to isolate whether FiLM is doing anything independent
    # of the encoder swap. "wae_he_mmd_geneencoder_mlp_nofilm" is therefore
    # a genuine no-intervention control (plain encoder, no FiLM, just
    # regularizer=mmd/latent_dim=64) -- the other three each add exactly
    # one real change on top of it. Regularizer is "mmd" (WAE-Wasserstein)
    # for all four -- no GAN arm in this ablation.
    "wae_he_mmd_geneencoder_scfoundation_film": ConditionalWAEArmSpec("he_to_st", "mmd", False),
    "wae_he_mmd_geneencoder_scfoundation_nofilm": ConditionalWAEArmSpec("he_to_st", "mmd", False),
    "wae_he_mmd_geneencoder_mlp_film": ConditionalWAEArmSpec("he_to_st", "mmd", False),
    "wae_he_mmd_geneencoder_mlp_nofilm": ConditionalWAEArmSpec("he_to_st", "mmd", False),
    # Same gene-encoder x FiLM factorial as above, but with data.image_encoder
    # fixed to "omiclip" instead of "uni2" -- OmiCLIP's own pretraining was
    # supervised by real, paired H&E+expression contrastive alignment
    # (unlike GigaPath/UNI2's image-only pretraining), so this suite isolates
    # whether an image tower that already "knows about" expression changes
    # which gene-encoder/FiLM choice matters most. Still generative
    # (ConditionalWAE, regularizer="mmd") -- only the frozen image encoder
    # backbone differs from the four arms immediately above.
    "wae_he_mmd_geneencoder_omiclip_scfoundation_film": ConditionalWAEArmSpec("he_to_st", "mmd", False),
    "wae_he_mmd_geneencoder_omiclip_scfoundation_nofilm": ConditionalWAEArmSpec("he_to_st", "mmd", False),
    "wae_he_mmd_geneencoder_omiclip_mlp_film": ConditionalWAEArmSpec("he_to_st", "mmd", False),
    "wae_he_mmd_geneencoder_omiclip_mlp_nofilm": ConditionalWAEArmSpec("he_to_st", "mmd", False),
}

_VALID_FILM_LAYERS = frozenset({"first", "second"})
_VALID_IMAGE_ENCODERS = frozenset({"gigapath", "uni2", "omiclip"})


def static_audit_conditional_wae_config(config: dict) -> dict:
    model = config.get("model") or {}
    arm = str(model.get("arm", ""))
    if arm not in ARM_SPECS:
        raise ValueError(f"unknown conditional-WAE arm {arm!r}")
    spec = ARM_SPECS[arm]
    if model.get("kind") != "conditional_wae":
        raise ValueError("model.kind must be 'conditional_wae'")
    if model.get("task") != spec.task or model.get("regularizer") != spec.regularizer:
        raise ValueError(f"{arm}: task/regularizer do not match the immutable arm contract")
    if bool(model.get("include_observed_gex")) != spec.include_observed_gex:
        raise ValueError(f"{arm}: include_observed_gex does not match the task contract")
    if model.get("image_mode") != "full_visible":
        raise ValueError("model.image_mode must be 'full_visible'; query H&E may not be masked")
    params = model.get("params") or {}
    if int(params.get("image_feature_dim", 0)) < 1:
        raise ValueError("model.params.image_feature_dim must be positive")
    for field in ("latent_dim", "hidden_dim", "gex_feature_dim", "autoencoder_hidden_dim"):
        if int(params.get(field, 0)) < 1:
            raise ValueError(f"model.params.{field} must be positive")
    encoder_conditioning = params.get("encoder_conditioning", "none")
    if encoder_conditioning not in {"none", "film"}:
        raise ValueError("model.params.encoder_conditioning must be 'none' or 'film'")
    if encoder_conditioning == "film":
        film_layers = frozenset(params.get("film_layers", ("first", "second")))
        if not film_layers or not film_layers.issubset(_VALID_FILM_LAYERS):
            raise ValueError(f"model.params.film_layers must be a non-empty subset of {_VALID_FILM_LAYERS}")
        if bool(params.get("film_shared_generator", False)) and film_layers != _VALID_FILM_LAYERS:
            raise ValueError("model.params.film_shared_generator requires film_layers to include both layers")
    use_gene_coexpression_refinement = bool(params.get("use_gene_coexpression_refinement", False))
    data = config.get("data") or {}
    if use_gene_coexpression_refinement and not data.get("gene_coexpression_basis_path"):
        raise ValueError(
            "model.params.use_gene_coexpression_refinement requires data.gene_coexpression_basis_path"
        )
    gene_encoder_source = str(params.get("gene_encoder_source", "linear"))
    if gene_encoder_source not in {"linear", "frozen_table"}:
        raise ValueError("model.params.gene_encoder_source must be 'linear' or 'frozen_table'")
    if gene_encoder_source == "frozen_table" and not data.get("gene_encoder_table_path"):
        raise ValueError(
            "model.params.gene_encoder_source='frozen_table' requires data.gene_encoder_table_path"
        )
    use_histology_context = bool(params.get("use_histology_context", False))
    if use_histology_context and not bool(data.get("use_histology_features", False)):
        raise ValueError(
            "model.params.use_histology_context requires data.use_histology_features to also be true, "
            "or the dataset would never load the features this arm's conditioner expects"
        )
    if not data.get("gen3_manifest_path"):
        raise ValueError("data.gen3_manifest_path must point to an immutable manifest")
    image_encoder = str(data.get("image_encoder", "gigapath"))
    if image_encoder not in _VALID_IMAGE_ENCODERS:
        raise ValueError(f"data.image_encoder must be one of {sorted(_VALID_IMAGE_ENCODERS)}")
    if image_encoder == "gigapath":
        if not data.get("tile_encoder_revision"):
            raise ValueError("data.tile_encoder_revision must be pinned")
    elif image_encoder == "uni2":
        if not data.get("uni2_pinned_revision"):
            raise ValueError("data.uni2_pinned_revision must be pinned")
    else:
        if not data.get("omiclip_pinned_revision"):
            raise ValueError("data.omiclip_pinned_revision must be pinned")
    training = config.get("training") or {}
    if not training.get("checkpoint_dir"):
        raise ValueError("training.checkpoint_dir is required")
    loss = config.get("loss") or {}
    for field in ("pcc_weight", "regularizer_weight", "conditional_mean_weight"):
        if float(loss.get(field, -1)) < 0:
            raise ValueError(f"loss.{field} must be non-negative")
    return {
        "passed": True,
        "arm": arm,
        "task": spec.task,
        "regularizer": spec.regularizer,
        "query_he_visible": True,
        "query_gex_visible": False,
        "surrounding_gex_visible": spec.include_observed_gex,
        "image_encoder": image_encoder,
        "use_gene_coexpression_refinement": use_gene_coexpression_refinement,
        "use_histology_context": use_histology_context,
        "gene_encoder_source": gene_encoder_source,
    }
