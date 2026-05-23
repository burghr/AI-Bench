from bench import summarize


def _iter_row(**overrides):
    row = {
        "wall_s": 1.0,
        "ttft_s": 0.2,
        "throughput_tok_per_s_est": 10.0,
        "streamed": True,
        "looks_like_html": True,
        "has_button": True,
    }
    row.update(overrides)
    return row


class TestSummarizeEmpty:
    def test_empty_iteration_list(self):
        result = summarize("test", [])
        assert result["label"] == "test"
        assert result["iterations"] == []
        assert result["summary"] is None


class TestSummarizeSingle:
    def test_single_iteration(self):
        result = summarize("test", [_iter_row()])
        assert result["label"] == "test"
        assert result["summary"]["wall_s_mean"] == 1.0
        assert result["summary"]["wall_s_median"] == 1.0
        assert result["summary"]["wall_s_stdev"] == 0.0
        assert result["summary"]["ttft_s_mean"] == 0.2
        assert result["summary"]["throughput_tok_per_s_mean_est"] == 10.0
        assert result["summary"]["streamed_all"] is True
        assert result["summary"]["all_runs_html"] is True
        assert result["summary"]["all_runs_have_buttons"] is True


class TestSummarizeMultiple:
    def test_mean_calculated(self):
        iters = [_iter_row(wall_s=1.0), _iter_row(wall_s=3.0)]
        result = summarize("two", iters)
        assert result["summary"]["wall_s_mean"] == 2.0

    def test_median_calculated(self):
        iters = [_iter_row(wall_s=1.0), _iter_row(wall_s=2.0), _iter_row(wall_s=3.0)]
        result = summarize("three", iters)
        assert result["summary"]["wall_s_median"] == 2.0

    def test_stdev_nonzero(self):
        iters = [_iter_row(wall_s=1.0), _iter_row(wall_s=5.0)]
        result = summarize("spread", iters)
        assert result["summary"]["wall_s_stdev"] > 0

    def test_all_flags_false_when_one_fails(self):
        iters = [_iter_row(), _iter_row(streamed=False)]
        result = summarize("mixed", iters)
        assert result["summary"]["streamed_all"] is False

    def test_html_flag(self):
        iters = [_iter_row(), _iter_row(looks_like_html=False)]
        result = summarize("nohtml", iters)
        assert result["summary"]["all_runs_html"] is False

    def test_button_flag(self):
        iters = [_iter_row(), _iter_row(has_button=False)]
        result = summarize("nobtn", iters)
        assert result["summary"]["all_runs_have_buttons"] is False


class TestSummarizeTtftNone:
    def test_missing_ttft_excluded_from_mean(self):
        iters = [_iter_row(ttft_s=None), _iter_row(ttft_s=0.5)]
        result = summarize("notuft", iters)
        assert result["summary"]["ttft_s_mean"] == 0.5

    def test_all_ttft_none(self):
        iters = [_iter_row(ttft_s=None), _iter_row(ttft_s=None)]
        result = summarize("noft", iters)
        assert result["summary"]["ttft_s_mean"] is None


class TestSummarizeIterationsIncluded:
    def test_iterations_list_preserved(self):
        iters = [_iter_row(wall_s=1.0), _iter_row(wall_s=2.0)]
        result = summarize("preserved", iters)
        assert len(result["iterations"]) == 2
