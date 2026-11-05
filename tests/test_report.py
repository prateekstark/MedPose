from html.parser import HTMLParser

import numpy as np

from medpose.trial import Trial
from medpose.report import render_report, save_report
from medpose.report.html import DISCLAIMER, ERROR_CAVEAT


class OfflineCheck(HTMLParser):
    def handle_starttag(self, tag, attrs):
        assert tag not in {"script", "iframe", "object", "embed", "link", "form"}
        for key, value in attrs:
            assert key not in {"src", "href", "srcset", "action"}


def synthetic(empty=False):
    return Trial([] if empty else [0, 1, 2], ("knee",), np.empty((0, 1)) if empty else [[0], [np.nan], [20]],
                 "1_phone", "<script>synthetic</script>", "1", 1, 1, ("AFO <test>",))


def test_offline_report_and_escaping(tmp_path):
    trial = synthetic()
    output = render_report(trial, planes={"knee":"sagittal"}, task="synthetic")
    OfflineCheck().feed(output)
    assert DISCLAIMER in output and ERROR_CAVEAT in output
    assert '<script>' not in output and '&lt;script&gt;' in output
    assert 'class="uncertain"' in output and "error not yet measured" in output
    assert 'missing frames 1' in output and 'AFO &lt;test&gt;' in output
    # Two isolated finite points are not connected across the missing frame.
    assert output.count('<polyline') == 2
    path = tmp_path / "synthetic.html"
    save_report(trial, path, planes={"knee":"sagittal"}, task="synthetic")
    assert path.read_text() == output


def test_empty_report():
    output = render_report(synthetic(True), planes={"knee":"sagittal"}, task="synthetic")
    assert "No finite observations" in output
    OfflineCheck().feed(output)
