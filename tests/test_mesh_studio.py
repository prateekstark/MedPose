"""Synthetic transport/geometry checks; never reads personal media."""
import io
import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import numpy as np
import pytest

from medpose.engine.advanced import mesh_frame, read_media
from medpose.ui import mesh_jobs
from medpose.ui.serve import Handler


def test_mesh_finite_geometry_and_provenance():
    result=mesh_frame(np.zeros((6890,3)),np.zeros((24,3)),0,.99)
    assert len(result['vertices'])==6890
    assert result['detector_confidence']==.99
    with pytest.raises(ValueError,match='nonfinite'):
        mesh_frame([[float('nan'),0,0]],[[0,0,0]],0,.99)


def test_bad_media_and_limits(tmp_path):
    file=tmp_path/'synthetic.png';file.write_bytes(b'not an image')
    with pytest.raises(ValueError,match='decoded'):read_media(file,'picture','hmr2')
    for size in (0,mesh_jobs.MAX_BYTES+1):
        with pytest.raises(ValueError):mesh_jobs.start(io.BytesIO(),size,'picture','hmr2')
    with pytest.raises(ValueError):mesh_jobs.start(io.BytesIO(b'x'),1,'picture','unknown')


def test_mesh_api_rejects_cross_origin_and_accepts_explicit_local_request(monkeypatch):
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    base='http://127.0.0.1:%d'%server.server_port
    calls=[]
    def start(source,length,kind,engine):
        calls.append((source.read(length),kind,engine));return 'synthetic-job'
    monkeypatch.setattr(mesh_jobs,'start',start)
    try:
        for origin in ('','https://elsewhere.invalid'):
            req=urllib.request.Request(base+'/api/mesh',data=b'x',headers={'Origin':origin,'X-MedPose-Action':'infer'})
            with pytest.raises(urllib.error.HTTPError) as e:urllib.request.urlopen(req)
            assert e.value.code==403
        req=urllib.request.Request(base+'/api/mesh',data=b'synthetic',headers={'Origin':base,
            'X-MedPose-Action':'infer','X-MedPose-Engine':'hmr2','X-MedPose-Media':'picture'})
        with urllib.request.urlopen(req) as response:
            assert response.status==202
            assert json.load(response)=={'id':'synthetic-job'}
        assert calls==[(b'synthetic','picture','hmr2')]
        with urllib.request.urlopen(base+'/mesh') as response:
            assert "connect-src 'self'" in response.headers['Content-Security-Policy']
        with pytest.raises(urllib.error.HTTPError) as e:urllib.request.urlopen(base+'/api/mesh/unknown')
        assert e.value.code==404
    finally:server.shutdown();server.server_close();thread.join()


def test_cancel_and_busy_guard(monkeypatch):
    class Process:
        stopped=False
        def terminate(self):self.stopped=True
        def wait(self,timeout=None):return 0
    process=Process();job=dict(status='running',process=process,created=0)
    monkeypatch.setattr(mesh_jobs,'JOBS',{'synthetic':job})
    with pytest.raises(RuntimeError,match='Another'):
        mesh_jobs.start(io.BytesIO(b'x'),1,'picture','hmr2')
    assert mesh_jobs.cancel('synthetic') and process.stopped
    assert mesh_jobs.snapshot('synthetic') is None
    assert job['status']=='cancelled'
    assert not mesh_jobs.cancel('missing')


def test_memory_file_is_readable_without_disk_path():
    import os
    fd=mesh_jobs.memory_file()
    try:
        os.write(fd,b'synthetic bytes')
        with open('/proc/self/fd/%d'%fd,'rb') as source:
            assert source.read()==b'synthetic bytes'
        assert 'memfd:medpose-media' in os.readlink('/proc/self/fd/%d'%fd)
    finally:os.close(fd)


def test_mesh_browser_with_installed_models(tmp_path):
    """Opt-in real model check on the committed synthetic adult only."""
    import os
    import shutil
    import subprocess
    from pathlib import Path
    from medpose.paths import ROOT
    if os.environ.get('MEDPOSE_TEST_MESH')!='1':
        pytest.skip('Set MEDPOSE_TEST_MESH=1 for the real model browser check')
    chrome=shutil.which('google-chrome')
    if not chrome:pytest.skip('Chrome unavailable')
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    base='http://127.0.0.1:%d'%server.server_port
    try:
        check=subprocess.run(['node',str(ROOT/'tools/pose_browser_check.cjs'),chrome,base+'/mesh',
            '1280','900',str(tmp_path/'browser'),str(tmp_path/'mesh.png'),
            str(ROOT/'tools/mesh_browser_check.js')],capture_output=True,text=True,timeout=190)
        assert check.returncode==0,check.stderr
        result=json.loads(check.stdout)
        assert 'complete' in result['status']
        assert 'HMR2' in result['metrics']
        requests=result['browserRequests']
        assert all(r['url'].startswith((base+'/', 'blob:', 'data:')) for r in requests)
        assert any(r['method']=='POST' and r['url']==base+'/api/mesh' for r in requests)
    finally:
        server.shutdown();server.server_close();thread.join()
        for key in list(mesh_jobs.JOBS):mesh_jobs.cancel(key)


def test_auxiliary_installer_rejects_unverified_archive(tmp_path):
    from tools.setup_advanced import install_archive
    archive=tmp_path/'download.tar.gz';archive.write_bytes(b'untrusted content')
    target=tmp_path/'assets'
    with pytest.raises(ValueError,match='checksum'):
        install_archive(archive,assets=target,destination=tmp_path/'links')
    assert not target.exists()
