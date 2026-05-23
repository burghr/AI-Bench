import os
import sys
from bench import bench_one, run_agent_streamed


class TestRunAgentStreamed:
    def test_stdout_and_exit_zero(self):
        result = run_agent_streamed(
            [sys.executable, "-c", "print('hello')"],
            env=os.environ.copy(),
            total_timeout=10,
        )
        assert result["rc"] == 0
        assert "hello" in result["stdout"]
        assert result["wall_s"] >= 0
        assert result["end_reason"] in ("exit", "killed")

    def test_stderr_and_nonzero_exit(self):
        result = run_agent_streamed(
            [sys.executable, "-c", "import sys; sys.stderr.write('err\\n'); sys.exit(42)"],
            env=os.environ.copy(),
            total_timeout=10,
        )
        assert result["rc"] == 42
        assert "err" in result["stderr"]

    def test_timeout_kills_process(self):
        result = run_agent_streamed(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            env=os.environ.copy(),
            total_timeout=2,
        )
        assert result["wall_s"] < 5

    def test_ttft_recorded(self):
        result = run_agent_streamed(
            [sys.executable, "-c", "import sys; sys.stdout.write('A')"],
            env=os.environ.copy(),
            total_timeout=10,
        )
        assert result["ttft_s"] is not None
        assert result["ttft_s"] >= 0

    def test_env_variables_passed(self):
        env = os.environ.copy()
        env["BENCH_TEST_VAR"] = "works"
        result = run_agent_streamed(
            [sys.executable, "-c", "import os; print(os.environ['BENCH_TEST_VAR'])"],
            env=env,
            total_timeout=10,
        )
        assert result["rc"] == 0
        assert "works" in result["stdout"]

    def test_no_output_timeout_kills_silent_process(self):
        result = run_agent_streamed(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            env=os.environ.copy(),
            total_timeout=10,
            no_output_timeout=1,
        )
        assert result["wall_s"] < 5
        assert result["end_reason"] == "no_output_timeout"

    def test_no_output_timeout_allows_late_process_after_output(self):
        result = run_agent_streamed(
            [sys.executable, "-c", "import sys, time; sys.stdout.write('A'); sys.stdout.flush(); time.sleep(1.5); print('B')"],
            env=os.environ.copy(),
            total_timeout=10,
            no_output_timeout=1,
        )
        assert result["rc"] == 0
        assert result["end_reason"] == "exit"
        assert "AB" in result["stdout"].replace("\n", "")


class TestBenchOne:
    def test_records_stderr_file_in_iteration_row(self, tmp_path):
        result = bench_one(
            "agent+backend+model",
            [sys.executable, "-c", "import sys; sys.stderr.write('err\\n'); print('<html><button></button><script></script>')"],
            os.environ.copy(),
            tmp_path,
            n_iter=1,
            warmup=0,
            total_timeout=10,
        )
        row = result["iterations"][0]
        assert row["stderr_file"] == "agent+backend+model__iter-0.stderr.log"
        assert (tmp_path / row["stderr_file"]).read_text() == "err\n"
