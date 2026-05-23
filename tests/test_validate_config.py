from bench import validate_config, VALID_AGENTS, VALID_BACKENDS

VALID_AGENT = "pi" if "pi" in VALID_AGENTS else sorted(VALID_AGENTS)[0]
VALID_BACKEND = "ollama" if "ollama" in VALID_BACKENDS else sorted(VALID_BACKENDS)[0]


def _valid_cfg(**overrides):
    cfg = {
        "models": [{"id": "qwen3-1.7b", "ollama": "qwen3:1.7b"}],
        "agents": [VALID_AGENT],
        "backends": [VALID_BACKEND],
        "iterations": 1,
        "warmup": 0,
        "prompt": "Hello world",
    }
    cfg.update(overrides)
    return cfg


class TestValidateConfigValid:
    def test_valid_config_returns_empty(self):
        assert validate_config(_valid_cfg()) == []

    def test_all_backends_listed(self):
        for be in sorted(VALID_BACKENDS):
            assert validate_config(_valid_cfg(backends=[be])) == []

    def test_multiple_models(self):
        assert validate_config(_valid_cfg(
            models=[
                {"id": "m1", "ollama": "m1"},
                {"id": "m2", "lmstudio": "m2"},
            ]
        )) == []

    def test_direct_with_ollama(self):
        assert validate_config(_valid_cfg(direct=True, backends=["ollama"])) == []


class TestValidateConfigModels:
    def test_missing_models(self):
        errors = validate_config({**_valid_cfg(), "models": []})
        assert any("non-empty" in e for e in errors)

    def test_models_not_list(self):
        errors = validate_config(_valid_cfg(models="bad"))
        assert any("must be a list" in e for e in errors)

    def test_model_no_id(self):
        errors = validate_config(_valid_cfg(models=[{"ollama": "x"}]))
        assert any("missing required field: id" in e for e in errors)

    def test_model_empty_id(self):
        errors = validate_config(_valid_cfg(models=[{"id": "", "ollama": "x"}]))
        assert any("must be non-empty" in e for e in errors)

    def test_duplicate_model_ids(self):
        errors = validate_config(_valid_cfg(models=[
            {"id": "dup", "ollama": "dup"},
            {"id": "dup", "lmstudio": "dup"},
        ]))
        assert any("must be unique" in e for e in errors)

    def test_model_no_backend_alias(self):
        errors = validate_config(_valid_cfg(models=[{"id": "x"}]))
        assert any("backend alias" in e for e in errors)

    def test_model_id_not_string(self):
        errors = validate_config(_valid_cfg(models=[{"id": 123}]))
        assert any("must be a string" in e for e in errors)


class TestValidateConfigAgents:
    def test_missing_agents(self):
        errors = validate_config({**_valid_cfg(), "agents": []})
        assert any("non-empty" in e for e in errors)

    def test_unknown_agent(self):
        errors = validate_config(_valid_cfg(agents=["nonexistent"]))
        assert any("unknown agent" in e for e in errors)


class TestValidateConfigBackends:
    def test_missing_backends(self):
        errors = validate_config({**_valid_cfg(), "backends": []})
        assert any("non-empty" in e for e in errors)

    def test_unknown_backend(self):
        errors = validate_config(_valid_cfg(backends=["fake"]))
        assert any("unknown backend" in e for e in errors)


class TestValidateConfigIterations:
    def test_zero_iterations(self):
        errors = validate_config(_valid_cfg(iterations=0))
        assert any("greater than 0" in e for e in errors)

    def test_negative_iterations(self):
        errors = validate_config(_valid_cfg(iterations=-1))
        assert any("greater than 0" in e for e in errors)

    def test_float_iterations(self):
        errors = validate_config(_valid_cfg(iterations=1.5))
        assert any("must be an integer" in e for e in errors)


class TestValidateConfigWarmup:
    def test_negative_warmup(self):
        errors = validate_config(_valid_cfg(warmup=-1))
        assert any("greater than or equal to 0" in e for e in errors)

    def test_float_warmup(self):
        errors = validate_config(_valid_cfg(warmup=1.5))
        assert any("must be an integer" in e for e in errors)


class TestValidateConfigPrompt:
    def test_missing_prompt(self):
        errors = validate_config({**_valid_cfg(), "prompt": ""})
        assert any("non-empty" in e for e in errors)

    def test_prompt_not_string(self):
        errors = validate_config(_valid_cfg(prompt=123))
        assert any("must be a string" in e for e in errors)


class TestValidateConfigDirect:
    def test_direct_requires_ollama(self):
        errors = validate_config(_valid_cfg(direct=True, backends=["lmstudio"]))
        assert any("direct mode requires ollama" in e for e in errors)

    def test_direct_not_boolean(self):
        errors = validate_config(_valid_cfg(direct="yes"))
        assert any("must be a boolean" in e for e in errors)


class TestValidateConfigMultipleErrors:
    def test_multiple_errors_reported_together(self):
        errors = validate_config({
            "models": [],
            "agents": [],
            "backends": [],
        })
        assert len(errors) >= 3


class TestValidateConfigTimeout:
    def test_valid_timeout_accepted(self):
        assert validate_config(_valid_cfg(timeout_s=300)) == []

    def test_valid_no_output_timeout_accepted(self):
        assert validate_config(_valid_cfg(no_output_timeout_s=120)) == []

    def test_zero_timeout_rejected(self):
        errors = validate_config(_valid_cfg(timeout_s=0))
        assert any("must be greater than 0" in e for e in errors)

    def test_zero_no_output_timeout_rejected(self):
        errors = validate_config(_valid_cfg(no_output_timeout_s=0))
        assert any("no_output_timeout_s must be greater than 0" in e for e in errors)

    def test_negative_timeout_rejected(self):
        errors = validate_config(_valid_cfg(timeout_s=-1))
        assert any("must be greater than 0" in e for e in errors)

    def test_float_timeout_rejected(self):
        errors = validate_config(_valid_cfg(timeout_s=1.5))
        assert any("must be an integer" in e for e in errors)

    def test_float_no_output_timeout_rejected(self):
        errors = validate_config(_valid_cfg(no_output_timeout_s=1.5))
        assert any("no_output_timeout_s must be an integer" in e for e in errors)


class TestValidateConfigValidators:
    def test_valid_validators_accepted(self):
        assert validate_config(_valid_cfg(validators={"html": True, "button": False})) == []

    def test_validators_not_dict(self):
        errors = validate_config(_valid_cfg(validators="yes"))
        assert any("must be a dict" in e for e in errors)

    def test_validator_value_not_boolean(self):
        errors = validate_config(_valid_cfg(validators={"html": 1}))
        assert any("must be a boolean" in e for e in errors)
