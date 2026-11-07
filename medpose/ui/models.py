"""Pinned public pose models. Downloads occur only on explicit setup/install actions."""
from __future__ import annotations

import hashlib
from pathlib import Path
import threading
import urllib.request

ASSETS = Path(__file__).resolve().parents[2] / '.conda-envs' / 'pose-web'
MODELS = {
    'lite': {'bytes': 5777746, 'sha256': '59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a',
             'description': 'Lower compute requirement; useful on slower devices'},
    'full': {'bytes': 9398198, 'sha256': '5134a3aad27a58b93da0088d431f366da362b44e3ccfbe3462b3827a839011b1',
             'description': 'Balanced local default for pictures and video'},
    'heavy': {'bytes': 30664242, 'sha256': '64437af838a65d18e5ba7a0d39b465540069bc8aae8308de3e318aad31fcbc7b',
              'description': 'Highest-capacity MediaPipe option here; more compute, not validated as best for CP'},
}
LOCK = threading.Lock()
JOBS = {}
VERIFIED = {}


def model_path(name):
    if name not in MODELS:
        raise ValueError('Unknown model')
    return ASSETS / ('pose_landmarker_' + name + '.task')


def installed(name):
    path = model_path(name)
    if not path.is_file():
        return False
    stat = path.stat()
    fingerprint = (stat.st_size, stat.st_mtime_ns)
    if stat.st_size != MODELS[name]['bytes']:
        return False
    if VERIFIED.get(name) != fingerprint:
        if hashlib.sha256(path.read_bytes()).hexdigest() != MODELS[name]['sha256']:
            return False
        VERIFIED[name] = fingerprint
    return True


def install(name):
    path = model_path(name)
    if installed(name):
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix('.part')
    url = ('https://storage.googleapis.com/mediapipe-models/pose_landmarker/'
           'pose_landmarker_%s/float16/1/pose_landmarker_%s.task' % (name, name))
    digest = hashlib.sha256()
    size = 0
    try:
        with urllib.request.urlopen(url, timeout=60) as source, partial.open('wb') as target:
            while True:
                chunk = source.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > MODELS[name]['bytes']:
                    raise ValueError('Unexpected model size')
                digest.update(chunk)
                target.write(chunk)
        if size != MODELS[name]['bytes'] or digest.hexdigest() != MODELS[name]['sha256']:
            raise ValueError('Model checksum did not match; file was not installed')
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)


def start_install(name):
    model_path(name)
    with LOCK:
        if any(job.get('downloading') for job in JOBS.values()):
            return False
        JOBS[name] = {'downloading': True}
    def work():
        error = None
        try:
            install(name)
        except Exception as exc:
            error = 'Download failed: ' + str(exc)
        with LOCK:
            JOBS[name] = {'downloading': False, 'error': error}
    threading.Thread(target=work, daemon=True).start()
    return True


def catalog():
    return [dict(id=name, installed=installed(name), **spec, **JOBS.get(name, {}))
            for name, spec in MODELS.items()]
