#!/usr/bin/env python3
"""Install the small official WHAM auxiliary assets and compatible adapter packages.

Use the existing OpenCap environment. Public checkpoints are installed separately
by download_public_weights.py; licensed SMPL by medpose.engine.smpl_assets.
"""
import argparse
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[1]
ASSETS=ROOT/'.conda-envs/advanced-assets'
SHA256='533a8b6b05ba3bc3d35c1dcaec757a4164a669625116a16d30d7642f3b200c8c'
NAMES={'coco_aug_dict.pth','J_regressor_wham.npy','smpl_mean_params.npz','J_regressor_feet.npy',
       'J_regressor_coco.npy','smplx2smpl.pkl','J_regressor_h36m.npy'}
PACKAGES=['ultralytics==8.0.100','seaborn==0.12.2','py-cpuinfo==9.0.0',
          'thop==0.1.1.post2209072238','sentry-sdk==1.45.0']


def install_archive(archive, assets=ASSETS, destination=None):
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=SHA256:
        raise ValueError('Auxiliary asset archive checksum mismatch')
    assets.mkdir(parents=True,exist_ok=True)
    with tarfile.open(archive) as tar:
        members=[m for m in tar.getmembers() if Path(m.name).name in NAMES]
        if len(members)!=len(NAMES) or {Path(m.name).name for m in members}!=NAMES:
            raise ValueError('Archive has missing or duplicate assets')
        if any(not m.isfile() or m.size>600_000_000 for m in members):
            raise ValueError('Unsafe archive member')
        for member in members:
            target=assets/Path(member.name).name
            with tar.extractfile(member) as source, target.open('wb') as out:
                shutil.copyfileobj(source,out)
    destination=destination or ROOT/'opencap-monocular/WHAM/dataset/body_models'
    destination.mkdir(parents=True,exist_ok=True)
    for name in sorted(NAMES):
        target=destination/name
        if target.is_symlink() and target.resolve()==(assets/name).resolve():continue
        if target.exists() or target.is_symlink():raise ValueError('Refusing to overwrite '+str(target))
        target.symlink_to((assets/name).resolve())


def synthetic_video():
    import cv2
    image=cv2.imread(str(ROOT/'medpose/ui/samples/synthetic-adult.png'))
    height,width=image.shape[:2]
    size=(640,round(height*640/width/2)*2)
    frame=cv2.resize(image,size)
    writer=cv2.VideoWriter(str(ASSETS/'synthetic-mesh.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),10,size)
    if not writer.isOpened():raise RuntimeError('Could not encode the synthetic video')
    try:
        for _ in range(30):writer.write(frame)
    finally:writer.release()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-packages',action='store_true')
    args=parser.parse_args()
    if not args.skip_packages:
        # Do not let pip upgrade the pinned OpenCap scientific stack.
        subprocess.run([sys.executable,'-m','pip','install','--no-deps',*PACKAGES],check=True)
    ASSETS.mkdir(parents=True,exist_ok=True)
    archive=ASSETS/'body_models.tar.gz'
    if not archive.is_file():
        import gdown
        gdown.download(id='1pbmzRbWGgae6noDIyQOnohzaVnX_csUZ',output=str(archive),use_cookies=False)
    install_archive(archive)
    synthetic_video()
    print('Auxiliary assets installed. Run the model checks before claiming inference readiness.')

if __name__=='__main__':main()
