from bench import estimate_tokens, _normalize_label


class TestEstimateTokens:
    def test_empty_string_returns_zero(self):
        assert estimate_tokens("") == 0

    def test_none_returns_zero(self):
        assert estimate_tokens(None) == 0

    def test_short_text_rounds_up_to_min_one(self):
        assert estimate_tokens("a") == 1

    def test_moderate_text(self):
        assert estimate_tokens("Hello world") == 3

    def test_long_text(self):
        assert estimate_tokens("a" * 4000) == 1000


class TestNormalizeLabel:
    def test_basic_normalization(self):
        assert _normalize_label("qwen3-1.7b") == "qwen3-1.7b"

    def test_slash_path(self):
        assert _normalize_label("publisher/model-name") == "model-name"

    def test_colon_separator(self):
        assert _normalize_label("qwen3:1.7b-mlx") == "qwen3-1.7b"

    def test_underscore_separator(self):
        assert _normalize_label("my_model_4bit") == "my-model"

    def test_space_separator(self):
        assert _normalize_label("My Model gguf") == "my-model"

    def test_strips_mlx_suffix(self):
        assert _normalize_label("llama-mlx-bf16") == "llama"

    def test_strips_4bit_suffix(self):
        assert _normalize_label("llama-4bit") == "llama"

    def test_underscore_converts_to_hyphen(self):
        # underscores are converted to hyphens before suffix stripping,
        # so -q4_k_m never matches as a suffix after normalization
        assert _normalize_label("llama-q4_k_m") == "llama-q4-k-m"

    def test_lowercases(self):
        assert _normalize_label("MyModel-GGUF") == "mymodel"

    def test_strips_fp16_suffix(self):
        assert _normalize_label("mistral-fp16") == "mistral"
