from ai_bench.cli import _launch_viewer


def test_launch_viewer_opens_file_url(tmp_path):
    viewer = tmp_path / "viewer.html"
    viewer.write_text("<!doctype html>")
    opened_urls = []

    def opener(url):
        opened_urls.append(url)
        return True

    assert _launch_viewer(viewer, opener=opener) is True
    assert opened_urls == [viewer.resolve().as_uri()]


def test_launch_viewer_reports_failed_open(tmp_path):
    viewer = tmp_path / "viewer.html"
    viewer.write_text("<!doctype html>")

    assert _launch_viewer(viewer, opener=lambda _url: False) is False
