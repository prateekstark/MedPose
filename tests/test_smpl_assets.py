"""Licensed asset integration: synthetic archives and synthetic body poses only."""
import json
from pathlib import Path
import zipfile

import pytest

from medpose.engine.smpl_assets import install, status, MODEL_NAMES, SOURCE_NAMES
from medpose.paths import MONO_ROOT


def archive_at(path, duplicate=False, missing=False):
    with zipfile.ZipFile(path, 'w') as archive:
        for name, source in SOURCE_NAMES.items():
            if missing and name == 'SMPL_NEUTRAL.pkl':
                continue
            archive.writestr('smpl/models/' + source, ('fixture:' + name).encode())
        archive.writestr('smpl/smpl_webuser/LICENSE.txt', 'Synthetic test licence')
        archive.writestr('../../escape.txt', 'must not extract')
        if duplicate:
            archive.writestr('other/' + next(iter(SOURCE_NAMES.values())), 'duplicate')
    return path


def test_install_known_members_and_connect_paths(tmp_path):
    mono, store = tmp_path/'mono', tmp_path/'store'
    mono.mkdir()
    archive = archive_at(tmp_path/'models.zip')
    manifest = install(archive, store, mono)
    assert manifest['version'] == '1.1.0'
    for name in MODEL_NAMES:
        link = mono/'WHAM/dataset/body_models/smpl'/name
        assert link.is_symlink() and link.resolve() == store/name
        assert link.read_bytes() == ('fixture:' + name).encode()
        assert manifest['models'][name]['bytes'] == len(link.read_bytes())
    assert not (tmp_path/'escape.txt').exists()
    assert status(store, mono)['smpl_installed']
    assert not status(store, mono)['pipeline_ready']
    assert len(status(store, mono)['missing_auxiliary']) == 3
    assert install(archive, store, mono) == manifest  # repeat is safe
    assert json.loads((store/'manifest.json').read_text()) == manifest


@pytest.mark.parametrize('duplicate,missing', [(True,False),(False,True)])
def test_refuse_ambiguous_or_incomplete_archive(tmp_path, duplicate, missing):
    mono=tmp_path/'mono';mono.mkdir()
    with pytest.raises(ValueError, match='exactly one'):
        install(archive_at(tmp_path/'models.zip',duplicate,missing),tmp_path/'store',mono)
    assert not (mono/'WHAM').exists()


def test_never_replace_unrelated_model(tmp_path):
    mono=tmp_path/'mono'
    folder=mono/'WHAM/dataset/body_models/smpl';folder.mkdir(parents=True)
    existing=folder/'SMPL_NEUTRAL.pkl';existing.write_bytes(b'keep')
    with pytest.raises(ValueError,match='refusing to replace'):
        install(archive_at(tmp_path/'models.zip'),tmp_path/'store',mono)
    assert existing.read_bytes()==b'keep'


@pytest.mark.parametrize('gender', ['male','female','neutral'])
def test_installed_smpl_responds_to_synthetic_pose(gender):
    if not status()['smpl_installed']:
        pytest.skip('User-licensed SMPL models not installed on this machine')
    import torch
    from smplx import SMPL
    torch.set_num_threads(2)
    model=SMPL(str(MONO_ROOT/'WHAM/dataset/body_models/smpl'),gender=gender,num_betas=10,batch_size=2).eval()
    pose=torch.zeros(2,69)
    pose[1,9]=0.4
    with torch.no_grad():
        result=model(betas=torch.zeros(2,10),body_pose=pose,global_orient=torch.zeros(2,3))
    assert result.vertices.shape==(2,6890,3)
    assert model.faces.shape==(13776,3)
    assert torch.isfinite(result.vertices).all() and torch.isfinite(result.joints).all()
    assert not torch.allclose(result.vertices[0],result.vertices[1])
    assert model.num_betas==10
