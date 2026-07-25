import json

from omegaconf import OmegaConf

from gen2_architectures.training.train_local_neighborhood import _model_config_dict


def test_model_config_dict_with_list_valued_params_is_json_serializable():
    """Real bug hit on the actual training server at the first periodic
    checkpoint save: dict(cfg.model.params) is only a SHALLOW conversion
    -- nested list-valued params (organ_vocab, tech_vocab, hidden_dims,
    ...) stayed as OmegaConf ListConfig objects, which json.dump cannot
    serialize ("Object of type ListConfig is not JSON serializable").
    Every config with any list-valued model param (all 7 shipped configs,
    via organ_vocab/tech_vocab auto-injection or a literal YAML list like
    hidden_dims) would have hit this on its first checkpoint save."""
    cfg = OmegaConf.create({
        "model": {
            "architecture": "1",
            "params": {
                "feat_dim": 256,
                "organ_vocab": ["Kidney", "Lung"],
                "tech_vocab": ["Visium"],
                "nested": {"hidden_dims": [4096, 1024]},
            },
        },
    })
    payload = _model_config_dict(cfg, scfoundation_dim=None)
    serialized = json.dumps(payload, indent=2)  # must not raise
    reloaded = json.loads(serialized)
    assert reloaded["params"]["organ_vocab"] == ["Kidney", "Lung"]
    assert reloaded["params"]["nested"]["hidden_dims"] == [4096, 1024]
    assert isinstance(payload["params"], dict)
    assert type(payload["params"]["organ_vocab"]) is list
