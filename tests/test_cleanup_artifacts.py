from bench import _get_artifacts_to_cleanup


def _state(**overrides):
    state = {
        "installed_by_us": {},
        "model_pulled_by_us": {},
    }
    state.update(overrides)
    return state


class TestGetArtifactsEmpty:
    def test_no_artifacts_when_empty_state(self):
        artifacts = _get_artifacts_to_cleanup(_state(), {})
        assert artifacts["tools_to_uninstall"] == []
        assert artifacts["directories_to_remove"] == []
        assert artifacts["models_to_remove"] == []

    def test_returns_expected_keys(self):
        artifacts = _get_artifacts_to_cleanup(_state(), {})
        assert "tools_to_uninstall" in artifacts
        assert "files_to_remove" in artifacts
        assert "directories_to_remove" in artifacts
        assert "models_to_remove" in artifacts


class TestGetArtifactsTools:
    def test_fzf_installed(self):
        artifacts = _get_artifacts_to_cleanup(
            _state(installed_by_us={"fzf": True}), {}
        )
        names = [t[0] for t in artifacts["tools_to_uninstall"]]
        assert "fzf" in names

    def test_pi_installed(self):
        artifacts = _get_artifacts_to_cleanup(
            _state(installed_by_us={"pi": True}), {}
        )
        names = [t[0] for t in artifacts["tools_to_uninstall"]]
        assert "pi" in names

    def test_opencode_installed(self):
        artifacts = _get_artifacts_to_cleanup(
            _state(installed_by_us={"opencode": True}), {}
        )
        names = [t[0] for t in artifacts["tools_to_uninstall"]]
        assert "opencode" in names


class TestGetArtifactsDirectories:
    def test_pi_install_does_not_imply_home_directory_removal(self):
        artifacts = _get_artifacts_to_cleanup(
            _state(installed_by_us={"pi": True}), {}
        )
        assert artifacts["directories_to_remove"] == []

    def test_opencode_install_does_not_imply_home_directory_removal(self):
        artifacts = _get_artifacts_to_cleanup(
            _state(installed_by_us={"opencode": True}), {}
        )
        assert artifacts["directories_to_remove"] == []

    def test_recorded_created_directory_is_listed(self):
        artifacts = _get_artifacts_to_cleanup(
            _state(created_dirs=["/tmp/ai-bench-created"]), {}
        )
        paths = [d["path"] for d in artifacts["directories_to_remove"]]
        assert "/tmp/ai-bench-created" in paths


class TestGetArtifactsModels:
    def test_ollama_models_tracked(self):
        artifacts = _get_artifacts_to_cleanup(
            _state(model_pulled_by_us={"ollama": ["qwen3:1.7b"]}), {}
        )
        models = [m["name"] for m in artifacts["models_to_remove"]]
        assert "qwen3:1.7b" in models

    def test_lmstudio_models_tracked(self):
        artifacts = _get_artifacts_to_cleanup(
            _state(model_pulled_by_us={"lmstudio": ["publisher/model"]}), {}
        )
        models = [m["name"] for m in artifacts["models_to_remove"]]
        assert "publisher/model" in models

    def test_omlx_models_tracked(self):
        artifacts = _get_artifacts_to_cleanup(
            _state(model_pulled_by_us={"omlx": ["my-model"]}), {}
        )
        models = [m["name"] for m in artifacts["models_to_remove"]]
        assert "my-model" in models

    def test_multiple_backends(self):
        artifacts = _get_artifacts_to_cleanup(
            _state(model_pulled_by_us={
                "ollama": ["phi3:3.8b"],
                "omlx": ["llama-3.1"],
            }), {}
        )
        models = [m["name"] for m in artifacts["models_to_remove"]]
        assert "phi3:3.8b" in models
        assert "llama-3.1" in models

    def test_empty_model_list(self):
        artifacts = _get_artifacts_to_cleanup(
            _state(model_pulled_by_us={"ollama": []}), {}
        )
        assert artifacts["models_to_remove"] == []
