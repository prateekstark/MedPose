"""Single local inference worker; media and results remain in RAM."""
import ctypes
import json
import os
import subprocess
import sys
import threading
import time
import uuid
from medpose.paths import ROOT

MAX_BYTES = 64 * 1024 * 1024
LOCK = threading.Lock()
JOBS = {}


def memory_file():
    # Some Conda Python builds omit os.memfd_create despite Linux support.
    if hasattr(os, 'memfd_create'):
        return os.memfd_create('medpose-media', flags=0)
    libc = ctypes.CDLL(None, use_errno=True)
    create = libc.memfd_create
    create.argtypes = [ctypes.c_char_p, ctypes.c_uint]
    create.restype = ctypes.c_int
    fd = create(b'medpose-media', 0)
    if fd < 0:
        raise OSError(ctypes.get_errno(), 'Could not allocate in-memory media')
    return fd


def start(source, length, kind, engine):
    if not 0 < length <= MAX_BYTES:
        raise ValueError('Choose a file between 1 byte and 64 MB')
    if kind not in ('picture', 'video') or engine not in ('hmr2', 'wham'):
        raise ValueError('Unsupported model or media type')
    with LOCK:
        for key in list(JOBS):
            if time.monotonic()-JOBS[key]['created'] > 900 and JOBS[key]['status'] != 'running':
                del JOBS[key]
        if any(j['status'] == 'running' for j in JOBS.values()):
            raise RuntimeError('Another body-mesh analysis is running. Cancel it or wait.')
        # Keep a bounded number of large results in memory.
        while len(JOBS) >= 2:
            del JOBS[next(iter(JOBS))]
        fd = memory_file()
        try:
            remaining = length
            while remaining:
                data = source.read(min(1024*1024, remaining))
                if not data:
                    raise ValueError('Incomplete media upload')
                pending = memoryview(data)
                while pending:
                    pending = pending[os.write(fd, pending):]
                remaining -= len(data)
            os.lseek(fd, 0, os.SEEK_SET)
            process = subprocess.Popen([sys.executable, '-m', 'medpose.engine.advanced',
                '/proc/self/fd/%d' % fd, '--kind', kind, '--engine', engine], cwd=ROOT,
                pass_fds=(fd,), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        finally:
            os.close(fd)
        key = uuid.uuid4().hex
        job = dict(status='running', stage='Starting local model', created=time.monotonic(), process=process)
        JOBS[key] = job
    threading.Thread(target=_collect, args=(job,), daemon=True).start()
    return key


def _collect(job):
    process = job['process']
    errors = []
    def diagnostics():
        for line in process.stderr:
            if line.startswith('MEDPOSE_PROGRESS '):
                try:
                    p = json.loads(line[len('MEDPOSE_PROGRESS '):])
                    job['stage'] = '%s (%s/%s)' % (p['stage'], p['completed'], p['total'])
                except (ValueError, KeyError):
                    pass
            else:
                errors.append(line.strip())
                del errors[:-12]
    reader = threading.Thread(target=diagnostics, daemon=True)
    reader.start()
    timer = threading.Timer(1800, lambda: _stop(job, 'Analysis exceeded 30 minutes. Try a shorter clip.'))
    timer.daemon = True
    timer.start()
    try:
        output = process.stdout.read()
        process.wait()
        reader.join()
        with LOCK:
            if job['status'] != 'running':
                return
            if process.returncode:
                job.update(status='error', error=(errors[-1] if errors else 'Local model failed'))
            else:
                result = json.loads(output)
                job.update(status='complete', result=result)
    except Exception:
        job.update(status='error', error='Could not read the local model result')
    finally:
        timer.cancel()
        process.stdout.close()
        process.stderr.close()
        expiry = threading.Timer(900, lambda: _expire(job))
        expiry.daemon = True
        expiry.start()


def _stop(job, reason='Cancelled'):
    with LOCK:
        if job['status'] != 'running':
            return
        job.update(status='cancelled', error=reason)
        process = job['process']
        process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _expire(job):
    with LOCK:
        for key in list(JOBS):
            if JOBS[key] is job and job['status'] != 'running':
                del JOBS[key]


def cancel(key):
    job = JOBS.get(key)
    if job is None:
        return False
    _stop(job)
    _expire(job)
    return True


def snapshot(key):
    with LOCK:
        job = JOBS.get(key)
        if job is None:
            return None
        return {k:v for k,v in job.items() if k in ('status','stage','error','result')}
