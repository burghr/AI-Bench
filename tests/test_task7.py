import os
import sys
from bench import run_agent_streamed


class TestEndReasonTimeout:
    def test_timeout_reports_timeout_not_exit(self):
        result = run_agent_streamed(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            env=os.environ.copy(),
            total_timeout=1,
        )
        assert result["end_reason"] == "timeout"

    def test_normal_exit_reports_exit(self):
        result = run_agent_streamed(
            [sys.executable, "-c", "print('done')"],
            env=os.environ.copy(),
            total_timeout=10,
        )
        assert result["end_reason"] == "exit"


class TestTimeoutValue:
    def test_timeout_value_affects_behavior(self):
        result = run_agent_streamed(
            [sys.executable, "-c", "import time; time.sleep(5)"],
            env=os.environ.copy(),
            total_timeout=1,
        )
        assert result["wall_s"] < 5
        assert result["end_reason"] == "timeout"
