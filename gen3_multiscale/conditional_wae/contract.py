"""Static scientific contract for the supervisor conditional-WAE suite."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConditionalWAEArmSpec:
    task: str
    regularizer: str
    include_observed_gex: bool


ARM_SPECS = {
    # Four-arm MK architecture comparison (Aug 2026).  These names encode
    # the actual scientific intervention; the checks below additionally make
    # local/spatial and standard/conditional-prior claims fail closed.
    "mk_local_wae_mmd": ConditionalWAEArmSpec("he_to_st", "mmd", False),
    "mk_spatial_deterministic": ConditionalWAEArmSpec("he_to_st", "none", False),
    "mk_local_conditional_wae_mmd": ConditionalWAEArmSpec("he_to_st", "mmd", False),
    "mk_spatial_conditional_wae_mmd": ConditionalWAEArmSpec("he_to_st", "mmd", False),
    "mk_field_within": ConditionalWAEArmSpec("he_to_st", "none", False),
    "mk_field_between": ConditionalWAEArmSpec("he_to_st", "none", False),
    "mk_field_gradient": ConditionalWAEArmSpec("he_to_st", "none", False),
    "mk_field_combined": ConditionalWAEArmSpec("he_to_st", "none", False),
    "mk_wb_serial": ConditionalWAEArmSpec("he_to_st", "none", False),
    "mk_bw_serial": ConditionalWAEArmSpec("he_to_st", "none", False),
    "mk_wbw_sandwich": ConditionalWAEArmSpec("he_to_st", "none", False),
    "mk_wb_parallel_gated": ConditionalWAEArmSpec("he_to_st", "none", False),
    # Residual WAE-MMD factorial on the frozen, best deterministic
    # within/between parallel-gated H&E predictor.  The two factors are the
    # prior (global versus H&E-conditional) and posterior FiLM conditioning.
    "mk_rwae_standard_nofilm": ConditionalWAEArmSpec("he_to_st", "mmd", False),
    "mk_rwae_standard_film": ConditionalWAEArmSpec("he_to_st", "mmd", False),
    "mk_rwae_conditional_nofilm": ConditionalWAEArmSpec("he_to_st", "mmd", False),
    "mk_rwae_conditional_film": ConditionalWAEArmSpec("he_to_st", "mmd", False),
    "wae_within": ConditionalWAEArmSpec("he_to_st", "mmd", False),
    "wae_between": ConditionalWAEArmSpec("he_to_st", "mmd", False),
    "wae_gradient": ConditionalWAEArmSpec("he_to_st", "mmd", False),
    "wae_combined": ConditionalWAEArmSpec("he_to_st", "mmd", False),
    "cwae_within": ConditionalWAEArmSpec("he_to_st", "mmd", False),
    "cwae_between": ConditionalWAEArmSpec("he_to_st", "mmd", False),
    "cwae_gradient": ConditionalWAEArmSpec("he_to_st", "mmd", False),
    "cwae_combined": ConditionalWAEArmSpec("he_to_st", "mmd", False),
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
_MK_ARCHITECTURE_DESIGNS = {
    "mk_local_wae_mmd": ("local", "standard", False),
    "mk_spatial_deterministic": ("spatial", "none", True),
    "mk_local_conditional_wae_mmd": ("local", "conditional", False),
    "mk_spatial_conditional_wae_mmd": ("spatial", "conditional", False),
}
_MK_STRUCTURED_FIELD_DESIGNS = {
    # conditioner, centered gene programs, graph-refinement steps,
    # local-gradient enabled, wide-gradient enabled
    "mk_field_within": ("local", True, 0, False, False),
    "mk_field_between": ("spatial", False, 1, False, False),
    "mk_field_gradient": ("local", False, 0, True, True),
    "mk_field_combined": ("spatial", True, 1, True, True),
    "wae_within": ("local", True, 0, False, False),
    "wae_between": ("spatial", False, 1, False, False),
    "wae_gradient": ("local", False, 0, True, True),
    "wae_combined": ("spatial", True, 1, True, True),
    "cwae_within": ("local", True, 0, False, False),
    "cwae_between": ("spatial", False, 1, False, False),
    "cwae_gradient": ("local", False, 0, True, True),
    "cwae_combined": ("spatial", True, 1, True, True),
    # Clean deterministic composition screen: both structured mechanisms are
    # present, while gradient supervision stays off in every arm.
    "mk_wb_serial": ("spatial", True, 1, False, False),
    "mk_bw_serial": ("spatial", True, 1, False, False),
    "mk_wbw_sandwich": ("spatial", True, 1, False, False),
    "mk_wb_parallel_gated": ("spatial", True, 1, False, False),
}
_MK_STRUCTURED_FIELD_FAMILIES = {
    "mk_field_within": ("none", True),
    "mk_field_between": ("none", True),
    "mk_field_gradient": ("none", True),
    "mk_field_combined": ("none", True),
    "wae_within": ("standard", False),
    "wae_between": ("standard", False),
    "wae_gradient": ("standard", False),
    "wae_combined": ("standard", False),
    "cwae_within": ("conditional", False),
    "cwae_between": ("conditional", False),
    "cwae_gradient": ("conditional", False),
    "cwae_combined": ("conditional", False),
    "mk_wb_serial": ("none", True),
    "mk_bw_serial": ("none", True),
    "mk_wbw_sandwich": ("none", True),
    "mk_wb_parallel_gated": ("none", True),
}
_MK_STRUCTURED_COMPOSITIONS = {
    "mk_wb_serial": "within_then_between",
    "mk_bw_serial": "between_then_within",
    "mk_wbw_sandwich": "within_between_within",
    "mk_wb_parallel_gated": "parallel_gated",
}
_MK_RESIDUAL_WAE_DESIGNS = {
    # prior_mode, posterior encoder conditioning
    "mk_rwae_standard_nofilm": ("standard", "none"),
    "mk_rwae_standard_film": ("standard", "film"),
    "mk_rwae_conditional_nofilm": ("conditional", "none"),
    "mk_rwae_conditional_film": ("conditional", "film"),
}
_VALID_STRUCTURED_COMPOSITIONS = frozenset({
    "within_then_between", "between_then_within",
    "within_between_within", "parallel_gated",
})


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
    deterministic_only = bool(params.get("deterministic_only", False))
    required_positive = ["hidden_dim", "gex_feature_dim", "autoencoder_hidden_dim"]
    if not deterministic_only:
        required_positive.append("latent_dim")
    for field in required_positive:
        if int(params.get(field, 0)) < 1:
            raise ValueError(f"model.params.{field} must be positive")
    if float(params.get("z_noise_std", 0.0)) < 0:
        raise ValueError("model.params.z_noise_std must be non-negative")
    n_refinement_steps = int(params.get("n_refinement_steps", 0))
    if n_refinement_steps < 0:
        raise ValueError("model.params.n_refinement_steps must be non-negative")
    if n_refinement_steps > 0:
        for field in ("refinement_k_neighbors", "refinement_hidden_dim",
                      "refinement_gex_feature_dim"):
            if int(params.get(field, 1)) < 1:
                raise ValueError(f"model.params.{field} must be positive")
    structured_composition = str(
        params.get("structured_composition", "within_then_between")
    )
    if structured_composition not in _VALID_STRUCTURED_COMPOSITIONS:
        raise ValueError(
            "model.params.structured_composition must be one of "
            f"{sorted(_VALID_STRUCTURED_COMPOSITIONS)}"
        )
    likelihood = str(params.get("likelihood", "gaussian_mse"))
    if likelihood not in {"gaussian_mse", "zero_inflated_gaussian"}:
        raise ValueError(
            "model.params.likelihood must be 'gaussian_mse' or 'zero_inflated_gaussian'"
        )
    if float(params.get("distributional_weight", 1.0)) < 0:
        raise ValueError("model.params.distributional_weight must be non-negative")
    encoder_conditioning = params.get("encoder_conditioning", "none")
    if encoder_conditioning not in {"none", "film"}:
        raise ValueError("model.params.encoder_conditioning must be 'none' or 'film'")
    if encoder_conditioning == "film":
        film_layers = frozenset(params.get("film_layers", ("first", "second")))
        if not film_layers or not film_layers.issubset(_VALID_FILM_LAYERS):
            raise ValueError(f"model.params.film_layers must be a non-empty subset of {_VALID_FILM_LAYERS}")
        if bool(params.get("film_shared_generator", False)) and film_layers != _VALID_FILM_LAYERS:
            raise ValueError("model.params.film_shared_generator requires film_layers to include both layers")
    conditioner_mode = str(params.get("conditioner_mode", "spatial"))
    if conditioner_mode not in {"local", "spatial"}:
        raise ValueError("model.params.conditioner_mode must be 'local' or 'spatial'")
    prior_mode = str(params.get("prior_mode", "standard"))
    if prior_mode not in {"standard", "conditional", "none"}:
        raise ValueError("model.params.prior_mode must be 'standard', 'conditional', or 'none'")
    is_new_mk_arm = arm.startswith("mk_") or arm in _MK_STRUCTURED_FIELD_DESIGNS
    is_structured_field_arm = arm in _MK_STRUCTURED_FIELD_DESIGNS
    is_residual_wae_arm = arm in _MK_RESIDUAL_WAE_DESIGNS
    if arm in _MK_ARCHITECTURE_DESIGNS:
        actual_design = (conditioner_mode, prior_mode, deterministic_only)
        if actual_design != _MK_ARCHITECTURE_DESIGNS[arm]:
            raise ValueError(
                f"{arm}: conditioner/prior/deterministic design {actual_design} does not "
                f"match immutable contract {_MK_ARCHITECTURE_DESIGNS[arm]}"
            )
    if deterministic_only:
        if model.get("regularizer") != "none" or prior_mode != "none":
            raise ValueError("deterministic_only requires regularizer='none' and prior_mode='none'")
        if encoder_conditioning != "none":
            raise ValueError("deterministic_only may not configure a posterior encoder/FiLM")
        if arm == "mk_spatial_deterministic":
            if conditioner_mode != "spatial":
                raise ValueError("the deterministic MK arm must use the spatial conditioner")
            if n_refinement_steps != 0:
                raise ValueError("deterministic MK arm tests the transformer only; spatial refinement must be off")
    elif is_residual_wae_arm:
        expected_prior, expected_conditioning = _MK_RESIDUAL_WAE_DESIGNS[arm]
        if model.get("regularizer") != "mmd":
            raise ValueError(f"{arm}: residual generative screen requires WAE-MMD")
        if (prior_mode, encoder_conditioning) != (
            expected_prior, expected_conditioning
        ):
            raise ValueError(
                f"{arm}: prior/conditioning {(prior_mode, encoder_conditioning)} "
                "does not match its immutable factorial cell "
                f"{(expected_prior, expected_conditioning)}"
            )
        if conditioner_mode != "spatial":
            raise ValueError(f"{arm}: frozen point predictor must be spatial")
        if structured_composition != "parallel_gated":
            raise ValueError(f"{arm}: frozen point predictor must be parallel_gated")
        if n_refinement_steps != 1 or not bool(
            params.get("use_centered_gene_structure", False)
        ):
            raise ValueError(
                f"{arm}: frozen point predictor requires within-gene and "
                "between-spot refinement"
            )
        if str(params.get("latent_residual_mode", "free")) != "antithetic_zero_mean":
            raise ValueError(f"{arm}: latent residual must be antithetic_zero_mean")
        if not bool(params.get("freeze_deterministic_backbone", False)):
            raise ValueError(f"{arm}: deterministic backbone must be frozen")
        if not params.get("deterministic_backbone_checkpoint"):
            raise ValueError(f"{arm}: deterministic backbone checkpoint is required")
        if not params.get("deterministic_backbone_weights_sha256"):
            raise ValueError(f"{arm}: deterministic backbone weights hash is required")
    elif is_new_mk_arm:
        if structured_composition != "within_then_between":
            raise ValueError(
                "non-default structured composition is currently defined only for "
                "deterministic predictors"
            )
        if model.get("regularizer") != "mmd" and prior_mode == "conditional":
            raise ValueError("conditional prior is currently supported only with WAE-MMD")
        if encoder_conditioning != "film":
            raise ValueError("the new MK WAE arms require the matched MLP+FiLM posterior encoder")
    if conditioner_mode == "local" and n_refinement_steps != 0:
        raise ValueError("local conditioner arms may not enable spatial refinement")
    if float(params.get("conditional_prior_context_weight", 1.0)) <= 0:
        raise ValueError("conditional_prior_context_weight must be positive")
    if float(params.get("conditional_prior_anchor_weight", 0.1)) < 0:
        raise ValueError("conditional_prior_anchor_weight must be non-negative")
    use_gene_coexpression_refinement = bool(params.get("use_gene_coexpression_refinement", False))
    data = config.get("data") or {}
    loss = config.get("loss") or {}
    local_gradient_weight = float(loss.get("local_gradient_weight", 0.0))
    wide_gradient_weight = float(loss.get("wide_gradient_weight", 0.0))
    if local_gradient_weight < 0 or wide_gradient_weight < 0:
        raise ValueError("structured-field gradient weights must be non-negative")
    if is_residual_wae_arm and (
        local_gradient_weight != 0 or wide_gradient_weight != 0
    ):
        raise ValueError(f"{arm}: gradient losses are outside this controlled screen")
    if is_structured_field_arm:
        expected_prior, expected_deterministic = _MK_STRUCTURED_FIELD_FAMILIES[arm]
        if (prior_mode, deterministic_only) != (expected_prior, expected_deterministic):
            raise ValueError(
                f"{arm}: structured-field family must use prior={expected_prior!r}, "
                f"deterministic_only={expected_deterministic}"
            )
        if not data.get("centered_gene_structure_path"):
            raise ValueError(
                f"{arm}: data.centered_gene_structure_path is required for shared, training-only scales"
            )
        if not data.get("centered_gene_structure_basis_sha256"):
            raise ValueError(
                f"{arm}: data.centered_gene_structure_basis_sha256 is required"
            )
        actual = (
            conditioner_mode,
            bool(params.get("use_centered_gene_structure", False)),
            n_refinement_steps,
            local_gradient_weight > 0,
            wide_gradient_weight > 0,
        )
        expected = _MK_STRUCTURED_FIELD_DESIGNS[arm]
        if actual != expected:
            raise ValueError(
                f"{arm}: structured-field design {actual} does not match immutable contract {expected}"
            )
        expected_composition = _MK_STRUCTURED_COMPOSITIONS.get(
            arm, "within_then_between"
        )
        if structured_composition != expected_composition:
            raise ValueError(
                f"{arm}: structured composition {structured_composition!r} does not "
                f"match immutable contract {expected_composition!r}"
            )
        if int(params.get("local_gradient_k", 0)) < 1 or int(params.get("wide_gradient_k", 0)) < 1:
            raise ValueError("structured-field gradient neighbourhood sizes must be positive")
    spatial_prior_path = data.get("spatial_prior_path")
    if spatial_prior_path and n_refinement_steps < 1:
        raise ValueError(
            "data.spatial_prior_path is set but model.params.n_refinement_steps is 0, so there is "
            "no spatial refiner to load the pretrained prior into -- the weights would be silently "
            "discarded"
        )
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
        "z_noise_std": float(params.get("z_noise_std", 0.0)),
        "n_refinement_steps": n_refinement_steps,
        "spatial_prior_path": str(spatial_prior_path) if spatial_prior_path else None,
        "likelihood": likelihood,
        "conditioner_mode": conditioner_mode,
        "prior_mode": prior_mode,
        "deterministic_only": deterministic_only,
        "use_centered_gene_structure": bool(params.get("use_centered_gene_structure", False)),
        "local_gradient_weight": local_gradient_weight,
        "wide_gradient_weight": wide_gradient_weight,
        "structured_composition": structured_composition,
    }
