"""Synthetic-only model, transport and browser integration checks."""
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from medpose.trial import Trial
from medpose.ui import models
from medpose.ui.serve import Handler, ROOT, ROUTES


def test_landmark_geometry_and_missing_values(tmp_path):
    script = '''
import {bend,packResult,toTrial,confidence} from MATH;
const assert=(x,m)=>{if(!x)throw Error(m)};
const p=(x,y,z=0)=>({x,y,z,visibility:1});
assert(bend([p(-1,0),p(0,0),p(1,0)],[0,1,2]).value===0,'straight');
assert(Math.abs(bend([p(0,1),p(0,0),p(1,0)],[0,1,2]).value-90)<1e-10,'right angle');
assert(bend([p(0,0),p(0,0),p(1,0)],[0,1,2]).value===null,'degenerate');
assert(bend(null,[0,1,2]).value===null,'missing');
assert(confidence({...p(0,0),visibility:NaN})===0,'nan');
assert(confidence({...p(0,0),presence:.2})===.2,'presence');
assert(bend([{...p(0,1),visibility:.499},p(0,0),p(1,0)],[0,1,2]).value===null,'below gate');
assert(bend([{...p(0,1),visibility:.5},p(0,0),p(1,0)],[0,1,2]).value===90,'gate boundary');
const missing=packResult({landmarks:[],worldLandmarks:[]},0);
assert(missing.world===null&&missing.status==='no_pose','empty result');
const multi=packResult({landmarks:[[],[]],worldLandmarks:[[],[]]},.1);
assert(multi.world===null&&multi.status==='multiple_people','multiple people');
let rejected=false;try{toTrial([missing,missing],10,'video')}catch(e){rejected=true}assert(rejected,'duplicate timestamp');
console.log(JSON.stringify(toTrial([missing,multi],10,'video')));
'''.replace('MATH', json.dumps((ROOT / 'medpose/ui/pose_math.js').as_uri()))
    result = subprocess.run(['node', '--input-type=module', '-e', script], capture_output=True, text=True, check=True)
    trial = Trial.from_dict(json.loads(result.stdout))
    assert trial.values.shape == (2, 4)


def test_installer_rejects_bad_checksum_and_keeps_existing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(models, 'ASSETS', tmp_path)
    monkeypatch.setattr(models, 'VERIFIED', {})
    target = models.model_path('lite')
    target.write_bytes(b'old invalid model')
    monkeypatch.setattr(models.urllib.request, 'urlopen', lambda *a, **k: io.BytesIO(b'bad model'))
    with pytest.raises(ValueError, match='checksum'):
        models.install('lite')
    assert target.read_bytes() == b'old invalid model'
    assert not target.with_suffix('.part').exists()
    with pytest.raises(ValueError, match='Unknown'):
        models.install('../escape')


def test_installed_models_are_verified():
    for name in models.MODELS:
        assert models.installed(name), 'Run tools/setup_pose.py before model integration tests'


@pytest.fixture
def studio_server():
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield 'http://127.0.0.1:%d' % server.server_port
    server.shutdown()
    server.server_close()
    thread.join()


def test_server_exposes_only_fixed_assets_and_guards_installs(studio_server, monkeypatch):
    import medpose.ui.serve as serve
    calls = []
    monkeypatch.setattr(serve, 'start_install', lambda name: calls.append(name) or True)
    with urllib.request.urlopen(studio_server + '/api/models') as response:
        assert all(m['installed'] for m in json.load(response))
    with urllib.request.urlopen(studio_server + '/') as response:
        assert "connect-src 'self'" in response.headers['Content-Security-Policy']
    for route in ['/../README.md', '/patient_data/x', '/.git/config', '/api/models/unknown']:
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(studio_server + route)
        assert error.value.code == 404
    for headers, body in [({}, None), ({'Host':'evil.example'}, None),
                           ({'Origin':'https://evil.example','X-MedPose-Action':'install'}, None),
                           ({'Origin':studio_server,'X-MedPose-Action':'install'}, b'picture bytes')]:
        request = urllib.request.Request(studio_server + '/api/models/lite', data=body, method='POST', headers=headers)
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(request)
        assert error.value.code == 403
    request = urllib.request.Request(studio_server + '/api/models/lite', method='POST',
                                     headers={'Origin':studio_server,'X-MedPose-Action':'install'})
    with urllib.request.urlopen(request) as response:
        assert response.status == 202
    assert calls == ['lite']


BROWSER_CHECKS = r'''
(async()=>{
 const app=await import('/pose.js'), math=await import('/pose_math.js');
 const $=id=>document.getElementById(id), assert=(v,m)=>{if(!v)throw Error(m)};
 await app.refreshModels();
 assert(app.state.models.length===3&&app.state.models.every(m=>m.installed),'models unavailable');
 const blob=await(await fetch('/samples/synthetic-adult.png')).blob();
 const picture=()=>new File([blob],'synthetic.png',{type:'image/png'});
 await app.loadPictures([picture(),picture()]);
 assert($('media-label').textContent==='PICTURES'&&!$('gallery').hidden,'multiple picture labels');
 const models={};
 for(const model of MODEL_LIST){
  $('model-select').value=model;
  await app.run();
  assert(app.state.frames.length===1,'inference failed: '+$('status').textContent);
  const frame=app.state.frames[0];
  assert(frame.status==='detected'&&frame.world.length===33,'no synthetic pose: '+model);
  assert(frame.measures.every(m=>Number.isFinite(m.value)),'visible joints have missing bends');
  assert(!$('pose-empty').hidden===false,'pose placeholder still visible');
  assert($('playback').hidden,'picture shows video controls');
  models[model]={landmarks:frame.world.length,detected:true};
 }
 const original=app.state.frames;
 await app.selectPicture(1);assert(!app.state.frames.length,'unrelated picture inherited pose');
 await app.selectPicture(0);assert(app.state.frames===original,'picture result lost');
 let rejected=false;try{await app.loadPictures([new File(['bad'],'broken.png',{type:'image/png'})])}catch(e){rejected=true}
 assert(rejected&&app.state.frames===original,'invalid picture destroyed review');
 rejected=false;try{await app.loadPictures([new File(['<svg/>'],'x.svg',{type:'image/svg+xml'})])}catch(e){rejected=true}
 assert(rejected,'svg accepted');
 const blank=document.createElement('canvas');blank.width=100;blank.height=100;
 const blankBlob=await new Promise(resolve=>blank.toBlob(resolve));
 await app.loadPictures([new File([blankBlob],'blank.png',{type:'image/png'})]);
 $('model-select').value='lite';await app.run();
 assert(app.state.frames[0]?.status==='no_pose'&&app.state.frames[0].world===null,'blank image invented pose');
 const clip=await(await fetch('/samples/synthetic-still.mp4')).blob();
 await app.loadVideo(new File([clip],'synthetic.mp4',{type:'video/mp4'}));
 await app.run();
 assert(app.state.frames.length===20,'video sampling failed: '+$('status').textContent);
 assert(app.state.frames.every(f=>f.status==='detected'),'synthetic video lost pose: '+JSON.stringify(app.state.frames.map(f=>[f.time,f.status])));
 assert(!$('playback').hidden&&!$('curves').hidden,'video results hidden');
 const videoFrames=app.state.frames.length;
 const trial=math.toTrial(app.state.frames,10,'video');
 $('scrub').value='1';$('scrub').dispatchEvent(new Event('input'));
 await new Promise(resolve=>setTimeout(resolve,150));
 assert(Math.abs($('video').currentTime-1)<.02,'scrubbing failed');
 const pending=app.run();await new Promise(resolve=>setTimeout(resolve,20));$('cancel').click();await pending;
 assert(!app.state.busy&&!app.state.frames.length&&$('export').disabled,'cancel left partial results');
 await app.loadPictures([picture()]);$('model-select').value='full';await app.run();
 assert(document.documentElement.scrollWidth<=innerWidth+1,'mobile overflow');
 const exported=app.results();
 assert(exported.measurement_error.status==='not_measured'&&exported.model==='full','export provenance');
 assert(!JSON.stringify(exported).includes('data:image'),'export contains image');
 window.scrollTo(0,0);await new Promise(requestAnimationFrame);
 return {passed:true,width:innerWidth,models,videoFrames,trial};
})()
'''


@pytest.mark.parametrize('width,height', [(1440,1100),(390,844)])
def test_real_model_browser_workflow(studio_server, tmp_path, width, height):
    chrome = shutil.which('google-chrome')
    if not chrome or not shutil.which('node'):
        pytest.skip('Chrome / Node required for local inference integration')
    checks = tmp_path / 'checks.js'
    checks.write_text(BROWSER_CHECKS.replace('MODEL_LIST', json.dumps(['lite','full','heavy'] if width==1440 else ['full'])))
    env = dict(os.environ, XDG_CONFIG_HOME=str(tmp_path/'config'), XDG_CACHE_HOME=str(tmp_path/'cache'))
    result = subprocess.run(['node', str(ROOT/'tools/pose_browser_check.cjs'), chrome, studio_server+'/',
                             str(width),str(height),str(tmp_path/'profile'),str(tmp_path/'screenshot.png'),str(checks)],
                            env=env,capture_output=True,text=True,timeout=190)
    assert result.returncode == 0, result.stderr[-4000:]
    data = json.loads(result.stdout)
    assert data['passed'] and data['width']==width
    assert data['videoFrames']==20
    Trial.from_dict(data['trial'])
    requests = data['browserRequests']
    assert any('/assets/wasm/' in r['url'] for r in requests), requests
    assert all(r['url'].startswith((studio_server+'/', 'data:', 'blob:')) for r in requests), requests
    assert all(r['method']=='GET' and not r['hasPostData'] for r in requests), requests
