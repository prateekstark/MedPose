"""Static offline-resource checks, plus JavaScript syntax when Node is available.

This is not browser interaction testing or a captured browser network trace.
"""
from html.parser import HTMLParser
from pathlib import Path
import re
import shutil
import subprocess

import pytest

VIEWER = Path(__file__).resolve().parents[1] / "medpose" / "viewer" / "viewer.html"


class Resources(HTMLParser):
    def __init__(self):
        super().__init__()
        self.policy = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        assert tag not in {"iframe", "object", "embed", "link", "form"}
        assert not any(key in attrs for key in ("src", "href", "srcset", "action"))
        if tag == "meta" and attrs.get("http-equiv") == "Content-Security-Policy":
            self.policy = attrs["content"]


def test_no_network_capabilities_or_resources():
    html = VIEWER.read_text()
    parsed = Resources(); parsed.feed(html)
    assert "connect-src 'none'" in parsed.policy
    assert "default-src 'none'" in parsed.policy
    assert "media-src blob:" in parsed.policy
    script = re.search(r"<script>(.*?)</script>", html, re.S).group(1)
    forbidden = r"\b(fetch|XMLHttpRequest|WebSocket|EventSource|sendBeacon|importScripts|Worker|serviceWorker)\b|https?://|import\s*\("
    assert not re.search(forbidden, script)
    assert "URL.createObjectURL(file)" in script
    assert "URL.revokeObjectURL" in script
    assert "not an image of the person's bones" in html
    assert "Anatomical bones — unavailable" in html
    assert "element.innerHTML" not in script and ".innerHTML" not in script


def test_javascript_syntax(tmp_path):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node not installed; JavaScript syntax check not run")
    script = re.search(r"<script>(.*?)</script>", VIEWER.read_text(), re.S).group(1)
    path = tmp_path / "viewer.js"
    path.write_text(script)
    result = subprocess.run([node, "--check", str(path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_trial_validation_and_time_sampling(tmp_path):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node not installed; viewer logic check not run")
    script = re.search(r"<script>(.*?)</script>", VIEWER.read_text(), re.S).group(1)
    # Pure functions above event binding run without a browser or any network API.
    prefix = script.split('for (const [id, slot] of [["fileA", 0], ["fileB", 1]])')[0]
    checks = '''
const assert = require("node:assert/strict");
const trial = {schema_version:1,time:[5,6,8],coordinate_names:["knee"],units:["deg"],
values:[[0],[null],[20]],recording_mode:"1_phone",engine_name:"synthetic",engine_version:"1",
camera_count:1,frame_rate:1,supports:[],confidence:null};
validateTrial(trial);
assert.equal(frameAt(trial,0),0);
assert.equal(frameAt(trial,1.5),1);
assert.equal(frameAt(trial,3),2);
assert.equal(frameAt(trial,3.1),null);
assert.throws(()=>validateTrial({...trial,schema_version:99}));
assert.throws(()=>validateTrial({...trial,time:[5,5,8]}));
assert.throws(()=>validateTrial({...trial,values:[[0],[Infinity],[20]]}));
assert.throws(()=>validateTrial({...trial,camera_count:2}));
assert.throws(()=>validateTrial({...trial,confidence:[[0],[2],[1]]}));
assert.throws(()=>validateTrial({...trial,name:"forbidden"}));
'''
    path = tmp_path / "viewer_checks.cjs"
    path.write_text(prefix + checks)
    result = subprocess.run([node, str(path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
