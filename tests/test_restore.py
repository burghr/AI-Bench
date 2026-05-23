import pytest
from pathlib import Path
from bench import _restore_backed_up


class TestRestoreBackedUp:
    def test_restores_from_backup(self, tmp_path: Path):
        orig = tmp_path / "config.json"
        bak = tmp_path / "config.json.bak"
        orig.write_text("modified")
        bak.write_text("original")

        _restore_backed_up(orig, bak)

        assert bak.exists() is False
        assert orig.read_text() == "original"

    def test_preserves_original_when_no_backup_and_no_created_marker(self, tmp_path: Path):
        orig = tmp_path / "config.json"
        bak = tmp_path / "config.json.bak"
        orig.write_text("orphan")

        _restore_backed_up(orig, bak)

        assert orig.read_text() == "orphan"
        assert bak.exists() is False

    def test_removes_created_original_when_marker_exists(self, tmp_path: Path):
        orig = tmp_path / "config.json"
        bak = tmp_path / "config.json.bak"
        created = tmp_path / "config.json.created"
        orig.write_text("created by bench")
        created.write_text("")

        _restore_backed_up(orig, bak, created)

        assert orig.exists() is False
        assert created.exists() is False

    def test_noop_when_neither_exists(self, tmp_path: Path):
        orig = tmp_path / "config.json"
        bak = tmp_path / "config.json.bak"

        _restore_backed_up(orig, bak)

        assert orig.exists() is False
        assert bak.exists() is False

    def test_second_call_preserves_restored_original(self, tmp_path: Path):
        orig = tmp_path / "config.json"
        bak = tmp_path / "config.json.bak"
        orig.write_text("modified")
        bak.write_text("original")

        _restore_backed_up(orig, bak)
        assert orig.read_text() == "original"

        _restore_backed_up(orig, bak)
        assert orig.read_text() == "original"

    def test_creates_parent_dirs_not_required(self, tmp_path: Path):
        # The helper does not create parent dirs; it works with existing paths.
        orig = tmp_path / "sub" / "config.json"
        bak = tmp_path / "sub" / "config.json.bak"
        (tmp_path / "sub").mkdir()
        orig.write_text("m")
        bak.write_text("o")

        _restore_backed_up(orig, bak)

        assert orig.read_text() == "o"

    def test_preserves_sibling_files(self, tmp_path: Path):
        orig = tmp_path / "config.json"
        bak = tmp_path / "config.json.bak"
        sibling = tmp_path / "other.txt"
        orig.write_text("modified")
        bak.write_text("original")
        sibling.write_text("keep me")

        _restore_backed_up(orig, bak)

        assert sibling.read_text() == "keep me"
