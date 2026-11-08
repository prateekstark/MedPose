"""Install a user's licensed SMPL archive locally and expose it to OpenCap."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import zipfile

from medpose.paths import ROOT, MONO_ROOT

STORE = ROOT / '.conda-envs' / 'smpl'
MODEL_NAMES = ('SMPL_MALE.pkl', 'SMPL_FEMALE.pkl', 'SMPL_NEUTRAL.pkl')
SOURCE_NAMES = {
    'SMPL_MALE.pkl': 'basicmodel_m_lbs_10_207_0_v1.1.0.pkl',
    'SMPL_FEMALE.pkl': 'basicmodel_f_lbs_10_207_0_v1.1.0.pkl',
    'SMPL_NEUTRAL.pkl': 'basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl',
}
AUXILIARY = ('J_regressor_wham.npy', 'J_regressor_h36m.npy', 'J_regressor_feet.npy')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def install(archive, store=STORE, mono=MONO_ROOT):
    """Copy only known model members; never execute archive scripts or unpickle."""
    archive, store, mono = Path(archive), Path(store), Path(mono)
    if not mono.is_dir():
        raise ValueError('OpenCap Monocular checkout is missing')
    destination = mono / 'WHAM/dataset/body_models/smpl'
    # Refuse replacing any unrelated installation, including broken links.
    for name in MODEL_NAMES:
        link = destination / name
        if (link.exists() or link.is_symlink()) and link.resolve() != (store / name).resolve():
            raise ValueError('Existing model at %s; refusing to replace it' % link)
    store.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as source:
        members = {}
        for name, basename in SOURCE_NAMES.items():
            matches = [m for m in source.infolist() if Path(m.filename).name.lower() == basename]
            if len(matches) != 1 or not 0 < matches[0].file_size < 1024**3:
                raise ValueError('Archive must contain exactly one valid ' + basename)
            members[name] = matches[0]
        licenses = [m for m in source.infolist() if m.filename.endswith('/smpl_webuser/LICENSE.txt')]
        if len(licenses) != 1 or licenses[0].file_size > 1024**2:
            raise ValueError('Expected SMPL licence text is missing or invalid')
        with tempfile.TemporaryDirectory(prefix='smpl-install-', dir=store.parent) as temp:
            staging = Path(temp)
            records = {}
            for name, member in members.items():
                with source.open(member) as src, (staging / name).open('wb') as out:
                    shutil.copyfileobj(src, out)
                records[name] = {'bytes': (staging / name).stat().st_size,
                                 'sha256': sha256(staging / name)}
                if (store / name).exists() and sha256(store / name) != records[name]['sha256']:
                    raise ValueError('Different installed SMPL model; refusing to overwrite ' + name)
            (staging / 'LICENSE.txt').write_bytes(source.read(licenses[0]))
            manifest = {'version': '1.1.0', 'shape_components': 300,
                        'archive_sha256': sha256(archive), 'models': records,
                        'licence': 'User-supplied licensed SMPL; no redistribution'}
            (staging / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
            store.mkdir(parents=True, exist_ok=True)
            for file in staging.iterdir():
                file.replace(store / file.name)
    destination.mkdir(parents=True, exist_ok=True)
    for name in MODEL_NAMES:
        link = destination / name
        if not link.is_symlink():
            link.symlink_to((store / name).resolve())
    return manifest


def status(store=STORE, mono=MONO_ROOT):
    store, mono = Path(store), Path(mono)
    folder = mono / 'WHAM/dataset/body_models'
    manifest_path = store / 'manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    models = {name: (store / name).is_file() and (folder / 'smpl' / name).is_file()
              and (folder / 'smpl' / name).resolve() == (store / name).resolve()
              for name in MODEL_NAMES}
    return {'smpl_installed': all(models.values()), 'models': models,
            'version': manifest.get('version'),
            'missing_auxiliary': [name for name in AUXILIARY if not (folder / name).is_file()],
            'pipeline_ready': False,
            'note': 'Body geometry only. Full inference requires additional assets, dependencies and hardware validation.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    command = sub.add_parser('install', help='Install a licensed SMPL Python 1.1.0 zip')
    command.add_argument('archive', type=Path)
    sub.add_parser('status')
    args = parser.parse_args()
    if args.command == 'install':
        install(args.archive)
    print(json.dumps(status(), indent=2))


if __name__ == '__main__':
    main()
