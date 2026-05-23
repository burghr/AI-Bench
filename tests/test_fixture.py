import json
from pathlib import Path

FIXTURE = Path(__file__).parent / "results.schema1.json"


class TestSchema1Fixture:
    def _load(self):
        return json.loads(FIXTURE.read_text())

    def test_fixture_loads(self):
        data = self._load()
        assert data is not None

    def test_schema_version_present(self):
        assert self._load()["schema_version"] == 1

    def test_top_level_keys(self):
        data = self._load()
        for key in ("timestamp", "platform", "config", "combos"):
            assert key in data, f"missing key: {key}"

    def test_config_has_models(self):
        cfg = self._load()["config"]
        assert isinstance(cfg["models"], list)
        assert len(cfg["models"]) > 0

    def test_config_has_agents_and_backends(self):
        cfg = self._load()["config"]
        assert isinstance(cfg["agents"], list)
        assert isinstance(cfg["backends"], list)

    def test_combos_have_structured_fields(self):
        for combo in self._load()["combos"]:
            for key in ("label", "agent", "backend", "model_id", "summary"):
                assert key in combo, f"combo missing key: {key}"

    def test_summary_has_metrics(self):
        s = self._load()["combos"][0]["summary"]
        for key in ("wall_s_mean", "wall_s_median", "ttft_s_mean",
                     "throughput_tok_per_s_mean_est"):
            assert key in s, f"summary missing key: {key}"

    def test_iterations_have_required_fields(self):
        iters = self._load()["combos"][0]["iterations"]
        assert len(iters) == 3
        for it in iters:
            for key in ("iter", "wall_s", "ttft_s", "streamed",
                         "looks_like_html", "has_button"):
                assert key in it, f"iteration missing key: {key}"
