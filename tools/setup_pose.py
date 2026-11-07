#!/usr/bin/env python3
"""One-time, explicit public-model setup; inference itself needs no internet."""
import argparse
import base64
import hashlib
import io
from pathlib import Path
import sys
import subprocess
import tarfile
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from medpose.ui.models import ASSETS, MODELS, install

SDK_URL = 'https://registry.npmjs.org/@mediapipe/tasks-vision/-/tasks-vision-0.10.21.tgz'
SDK_SHA512 = 'TuhKH+credq4zLksGbYrnvJ1aLIWMc5r0UHwzxzql4BHECJwIAoBR61ZrqwGOW6ZmSBIzU1t4VtKj8hbxFaKeA=='


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=list(MODELS) + ['all'], default='all')
    args = parser.parse_args()
    ASSETS.mkdir(parents=True, exist_ok=True)
    archive = ASSETS / 'runtime.tgz'
    data = archive.read_bytes() if archive.exists() else b''
    if base64.b64encode(hashlib.sha512(data).digest()).decode() != SDK_SHA512:
        print('Downloading pinned MediaPipe Tasks Vision 0.10.21 runtime…', flush=True)
        with urllib.request.urlopen(SDK_URL, timeout=60) as source:
            data = source.read(25 * 1024 * 1024)
        if base64.b64encode(hashlib.sha512(data).digest()).decode() != SDK_SHA512:
            raise ValueError('Runtime checksum mismatch')
        archive.write_bytes(data)
    with tarfile.open(fileobj=io.BytesIO(data)) as source:
        for member in source.getmembers():
            # No symlinks, arbitrary paths, install scripts or package manager execution.
            if not member.isfile() or not member.name.startswith('package/') or '..' in Path(member.name).parts:
                continue
            target = ASSETS / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.extractfile(member).read())
    for name in MODELS if args.model == 'all' else [args.model]:
        install(name)
        print(name + ': installed and checksum verified', flush=True)
    subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-loop', '1',
        '-i', str(Path(__file__).resolve().parents[1] / 'medpose/ui/samples/synthetic-adult.png'),
        '-t', '2', '-vf', 'scale=480:720', '-r', '10', '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(ASSETS / 'synthetic-still.mp4')], check=True)
    print('Start: .conda-envs/opencap/bin/python -m medpose.ui.serve')

if __name__ == '__main__':
    main()
