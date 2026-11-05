"""Synthetic-only UI checks in the project Python environment.

Chrome is used solely as an external test browser, not a runtime dependency.
Browser profiles, generated videos, screenshots and fixtures stay in this repo.
"""
import base64
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

import cv2
import numpy as np
import pytest

from medpose.ui.build import synthetic_trial, joints
from medpose.trial import Trial

ROOT = Path(__file__).resolve().parents[1]
UI = ROOT / "medpose" / "ui"


def payload():
    html = (UI / "demo.html").read_text()
    encoded = re.search(r"const DEMO = (.*);\nconst \$", html).group(1)
    return html, json.loads(encoded)


class OfflineResources(HTMLParser):
    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        assert tag not in {"iframe", "object", "embed", "link", "form"}
        assert not any(key in attrs for key in ("src", "href", "srcset", "action"))


def test_demo_provenance_and_geometry():
    trial = synthetic_trial()
    assert trial.engine_name == "synthetic-demo"
    assert len(trial.time) == 240 and trial.frame_rate == 30
    # t=0: sin0=0; both hip flexions=12deg; left knee remains at baseline12deg.
    np.testing.assert_allclose(trial.values[0, :3], [12, 12, 12])
    assert trial.confidence is None
    body = joints([0, 0, 0, 0])
    # Straight leg: thigh .49 + shank .47 = .96 arbitrary display units.
    assert body["ankle_l"][1] == pytest.approx(-.96)
    assert body["hip_l"][0] == -.14


def test_embedded_video_and_offline_resources(tmp_path):
    html, data = payload()
    OfflineResources().feed(html)
    trial = Trial.from_dict(data["trial"])
    assert len(trial.time) == 240
    assert data["video"].startswith("data:video/mp4;base64,")
    video = tmp_path / "synthetic.mp4"
    video.write_bytes(base64.b64decode(data["video"].split(",", 1)[1]))
    capture = cv2.VideoCapture(str(video))
    assert capture.isOpened()
    assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 240
    assert int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) == 720
    capture.release()
    assert "connect-src 'none'" in html
    script = html.split("<script>", 1)[1].split("</script>", 1)[0]
    # Embedded report contains citation prose, so inspect executable code separately.
    executable = script.split("const $", 1)[1]
    assert not re.search(r"\b(fetch|XMLHttpRequest|WebSocket|EventSource|sendBeacon|importScripts|Worker)\s*\(", executable)
    assert "http://" not in executable and "https://" not in executable
    assert "No movement was estimated from the video" in html


BROWSER_CHECKS = r'''
(async function runChecks() {
    const output=document.createElement("pre");output.id="ui-test-result";output.hidden=true;document.body.append(output);
    const assert=(condition,message)=>{if(!condition)throw Error(message);};
    const wait=ms=>new Promise(resolve=>setTimeout(resolve,ms));
    try {
        for(let i=0;i<60&&!state.videoReady;i++)await wait(100);
        assert(state.videoReady,"Embedded MP4 did not load");
        assert($("video").videoWidth===720,"Video width");
        assert(state.demo&&state.trial.time.length===240,"Demo Trial initialization");
        assert(frameAt(state.trial,1)===30,"Frame sampling at 1 second");
        assert(frameAt(state.trial,99)===null,"Out-of-range frame");
        assert(document.documentElement.scrollWidth<=innerWidth+1,"Horizontal page overflow");
        let rejected=false;try{validateTrial({...DEMO.trial,schema_version:99});}catch(e){rejected=true;}
        assert(rejected,"Unknown schema accepted");
        rejected=false;try{validateTrial({...DEMO.trial,values:[[null]]});}catch(e){rejected=true;}
        assert(rejected,"Malformed dimensions accepted");
        $("scrub").value=2.4;$("scrub").dispatchEvent(new Event("input"));
        assert(state.elapsed===2.4&&$("model-frame").textContent.includes("072"),"Scrub/model synchronization");
        document.querySelector('[data-joint="hip_flexion"]').click();
        assert(state.joint==="hip_flexion","Joint selection");
        document.querySelector('[data-view="side"]').click();assert(state.yaw===Math.PI/2,"Preset view");
        $("play").click();assert(state.playing,"Play control");pause();assert(!state.playing,"Pause control");
        document.querySelector('[data-page="compare"]').click();
        const comparison=JSON.parse(JSON.stringify(DEMO.trial));comparison.supports=["synthetic AFO"];
        const transfer=new DataTransfer();transfer.items.add(new File([JSON.stringify(comparison)],"synthetic-comparison.json",{type:"application/json"}));
        $("comparison-file").files=transfer.files;$("comparison-file").dispatchEvent(new Event("change"));
        for(let i=0;i<20&&!state.comparison;i++)await wait(50);
        assert($("comparison-warning").textContent.includes("Not comparable"),"Support mismatch warning");
        assert(!$("comparison-chart").hidden,"Comparison chart missing");
        $("clear-comparison").click();assert(state.comparison===null,"Clear comparison");
        document.querySelector('[data-page="benchmarks"]').click();
        const row={method:"synthetic",mode:"1_phone",coordinate:"knee_angle_l",plane:"sagittal",task:"walking",statistic:"mae",aggregation:"test",reference_system:"simulated truth",correction:"none",source:"simulated",origin:"generated test fixture",run_status:"reproduced",unit:"deg",value:3};
        loadBenchmarks([row,{...row,correction:"within-trial mean offset",value:1}]);
        assert($("benchmark-warning").textContent.includes("Mixed protocols"),"Benchmark mismatch warning");
        assert($("benchmark-table").textContent.includes("simulated"),"Error provenance missing");
        document.querySelector('[data-page="notes"]').click();$("notes").value="<script>not executable</script>";
        let exported=null;const objectURL=URL.createObjectURL;
        URL.createObjectURL=function(blob){exported=blob;return objectURL(blob);};
        const originalClick=HTMLAnchorElement.prototype.click;HTMLAnchorElement.prototype.click=function(){};
        $("export").click();const report=await exported.text();
        assert(report.includes("&lt;script&gt;not executable"),"Report notes not escaped");
        assert(report.includes("<svg"),"Report curves missing");
        assert(report.includes("Research use only, not clinically validated"),"Report disclaimer missing");
        URL.createObjectURL=objectURL;HTMLAnchorElement.prototype.click=originalClick;
        const local=JSON.parse(JSON.stringify(DEMO.trial));local.engine_name="local-test";
        const input=new DataTransfer();input.items.add(new File([JSON.stringify(local)],"synthetic-local.json",{type:"application/json"}));
        $("trial-file").files=input.files;$("trial-file").dispatchEvent(new Event("change"));
        for(let i=0;i<20&&state.demo;i++)await wait(50);
        assert(!state.demo&&state.trial.engine_name==="local-test","Local Trial import");
        assert(!$("video").getAttribute("src"),"Unrelated demo video retained for local Trial");
        loadTrial({...local,time:[],values:[],confidence:null});assert(duration()===0,"Empty Trial");
        // Generate pixels in the test browser: no real picture is opened.
        loadDemo();
        const fixture=document.createElement("canvas");fixture.width=32;fixture.height=24;
        fixture.getContext("2d").fillRect(0,0,32,24);
        const pictureBlob=await new Promise(resolve=>fixture.toBlob(resolve,"image/png"));
        const pictureInput=new DataTransfer();
        pictureInput.items.add(new File([pictureBlob],"synthetic.png",{type:"image/png"}));
        $("picture-file").files=pictureInput.files;$("picture-file").dispatchEvent(new Event("change"));
        for(let i=0;i<40&&!state.pictureURL;i++)await wait(50);
        assert(state.pictureURL&&!$("picture").hidden,"Picture import failed");
        assert(state.trial===null&&!state.demo,"Picture retained unrelated synthetic Trial");
        assert($("video").hidden&&!$("video").getAttribute("src"),"Video not cleared for picture");
        assert($("play").disabled&&$("speed").disabled,"Picture permits temporal playback");
        assert($("video-resolution").textContent==="32 × 24","Picture dimensions");
        assert($("range-left").textContent==="—","Picture invented an angle range");
        let pictureReport=null;
        URL.createObjectURL=function(blob){pictureReport=blob;return objectURL(blob);};
        HTMLAnchorElement.prototype.click=function(){};
        $("export").click();const pictureText=await pictureReport.text();
        assert(pictureText.includes("No pose, joint angles"),"Picture report lacks limitation");
        assert(!pictureText.includes("<img"),"Picture unexpectedly embedded in report");
        URL.createObjectURL=objectURL;HTMLAnchorElement.prototype.click=originalClick;
        const originalPicture=state.pictureURL;
        rejected=false;try{await attachPicture(new File(["broken"],"broken.png",{type:"image/png"}));}catch(e){rejected=true;}
        assert(rejected&&state.pictureURL===originalPicture,"Malformed picture changed current review");
        rejected=false;try{await attachPicture(new File(["<svg/>"],"test.svg",{type:"image/svg+xml"}));}catch(e){rejected=true;}
        assert(rejected,"Unsupported picture format accepted");
        const matchingInput=new DataTransfer();matchingInput.items.add(new File([JSON.stringify(local)],"matching.json",{type:"application/json"}));
        $("trial-file").files=matchingInput.files;$("trial-file").dispatchEvent(new Event("change"));
        for(let i=0;i<20&&!state.trial;i++)await wait(50);
        assert(state.trial&&state.pictureURL===originalPicture,"Matching Trial import cleared picture");
        seek(1);assert(state.elapsed===1&&$("play").disabled,"Manual Trial-frame selection");
        assert($("demo-notice").textContent.includes("loaded separately"),"Picture association not disclosed");
        assert(document.documentElement.scrollWidth<=innerWidth+1,"Picture mode overflows mobile viewport");
        loadDemo();assert(!state.pictureURL&&!$("play").disabled&&$("picture").hidden,"Reset left picture state behind");
        const resources=performance.getEntriesByType("resource").map(r=>r.name);
        assert(!resources.some(url=>/^https?:/i.test(url)),"Remote resource request detected");
        assert(window.__networkAttempts.length===0,"Network API was invoked");
        loadDemo();switchPage("review");$("toast").classList.remove("show");for(let i=0;i<60&&!state.videoReady;i++)await wait(100);
        output.textContent=JSON.stringify({passed:true,videoDecoded:true,requests:window.__networkAttempts,remoteResources:resources.filter(r=>/^https?:/i.test(r)),width:innerWidth});
    } catch(error) { output.textContent=JSON.stringify({passed:false,error:error.message,stack:error.stack}); }
})();
'''


@pytest.mark.parametrize("width,height", [(1440,1100), (390,844)])
def test_browser_workflows_and_no_network(tmp_path, width, height):
    chrome = shutil.which("google-chrome")
    if not chrome:
        pytest.skip("Chrome unavailable; browser workflow checks not run")
    html, _ = payload()
    guards = '''<script>window.__networkAttempts=[];
    for(const key of ["fetch","XMLHttpRequest","WebSocket","EventSource"]){window[key]=function(){window.__networkAttempts.push(key);throw Error("Network forbidden");};}
    navigator.sendBeacon=function(){window.__networkAttempts.push("sendBeacon");throw Error("Network forbidden");};
    </script>'''
    html = html.replace("<script>", guards + "<script>", 1)
    html = html.replace("</body>", "<script>" + BROWSER_CHECKS.replace("</script>", "<\\/script>") + "</script></body>")
    page = tmp_path / "browser-check.html"
    page.write_text(html)
    env = dict(os.environ, XDG_CONFIG_HOME=str(tmp_path / "config"), XDG_CACHE_HOME=str(tmp_path / "cache"))
    node = shutil.which("node")
    if not node:
        pytest.skip("Node unavailable; Chrome pipe driver not run")
    command = [node, str(ROOT / "tools" / "ui_browser_check.cjs"), chrome, page.as_uri(),
               str(width), str(height), str(tmp_path / "profile"), str(tmp_path / "screenshot.png")]
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stderr[-2000:]
    checks = json.loads(result.stdout)
    assert checks["passed"], checks
    assert checks["width"] == width
    assert checks["requests"] == [] and checks["remoteResources"] == []
