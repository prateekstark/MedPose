"""Offline HMR2 image meshes and fixed-camera WHAM video meshes.

Runs in an isolated child process. Inputs and results are passed in memory by
Pose Studio; no patient files, telemetry or model downloads are written here.
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

from medpose.paths import ROOT, MONO_ROOT
from medpose.engine.smpl_assets import status as smpl_status

ASSETS = ROOT / '.conda-envs/advanced-assets'
CHECKPOINTS = MONO_ROOT / 'WHAM/checkpoints'
MAX_VIDEO_FRAMES = 120


def readiness():
    smpl = smpl_status()
    common = [CHECKPOINTS/'hmr2a.ckpt', CHECKPOINTS/'yolov8x.pt', ASSETS/'smpl_mean_params.npz']
    missing = [p.name for p in common if not p.is_file()]
    dependencies = [n for n in ('torch','smplx','ultralytics','einops','timm','yacs') if importlib.util.find_spec(n) is None]
    hmr = smpl['smpl_installed'] and not missing and not dependencies
    temporal_missing = [n for n in ('J_regressor_wham.npy','J_regressor_h36m.npy','J_regressor_feet.npy') if not (ASSETS/n).is_file()]
    for name in ('vitpose-h-multi-coco.pth','wham_vit_bedlam_w_3dpw.pth.tar'):
        if not (CHECKPOINTS/name).is_file(): temporal_missing.append(name)
    if importlib.util.find_spec('mmpose') is None: temporal_missing.append('mmpose')
    return dict(smpl, hmr2_ready=bool(hmr), wham_ready=bool(hmr and not temporal_missing),
                missing_dependencies=dependencies, missing_image_assets=missing,
                missing_temporal_assets=temporal_missing,
                pipeline_ready=False, note='Readiness is asset availability; OpenCap clinical fitting is separate.')


def offline_setup():
    ASSETS.mkdir(parents=True, exist_ok=True)
    config=ASSETS/'yolo';config.mkdir(exist_ok=True)
    os.environ.update(YOLO_CONFIG_DIR=str(config),YOLO_AUTOINSTALL='false',MPLCONFIGDIR=str(config/'mpl'),
                      XDG_CONFIG_HOME=str(config),XDG_CACHE_HOME=str(config/'cache'))
    (config/'settings.yaml').write_text(json.dumps({
        'datasets_dir':str(config/'datasets'),'weights_dir':str(config/'weights'),
        'runs_dir':str(config/'runs'),'uuid':'local-offline','sync':False,
        'api_key':'','settings_version':'0.0.3'}))
    def audit(event, args):
        if event in ('socket.connect','socket.getaddrinfo'):
            raise OSError('External networking disabled during inference')
    sys.addaudithook(audit)
    os.chdir(MONO_ROOT)
    sys.path.insert(0,str(MONO_ROOT/'WHAM'))
    sys.path.insert(0,str(MONO_ROOT))


def progress(stage, completed=0, total=1):
    print('MEDPOSE_PROGRESS '+json.dumps(dict(stage=stage,completed=completed,total=total)), file=sys.stderr,flush=True)


def load_hmr(device):
    import torch
    from WHAM.lib.models.preproc.backbone.hmr2 import HMR2
    # mmap+assign avoids allocating a second 2.5GB copy of the checkpoint.
    checkpoint=torch.load(str(CHECKPOINTS/'hmr2a.ckpt'),map_location='cpu',mmap=True)['state_dict']
    linspace=torch.linspace
    # Upstream calls .item() on a tiny drop-path schedule during construction;
    # only that schedule must live on CPU while parameters initialize on meta.
    with patch('torch.linspace',lambda *a,**kw:linspace(*a,**dict(kw,device='cpu'))),torch.device('meta'):
        model=HMR2()
    keys=model.state_dict().keys()
    if set(keys)-set(checkpoint):
        raise ValueError('HMR2 checkpoint is missing required model weights')
    model.load_state_dict({k:checkpoint[k] for k in keys},strict=True,assign=True)
    del checkpoint
    return model.eval().to(device)


def release(*objects):
    import torch
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


def read_media(path,kind,engine):
    import cv2
    import numpy as np
    if kind=='picture':
        image=cv2.imdecode(np.frombuffer(Path(path).read_bytes(),dtype=np.uint8),cv2.IMREAD_COLOR)
        if image is None: raise ValueError('Picture could not be decoded')
        if image.shape[0]*image.shape[1]>20_000_000:raise ValueError('Picture exceeds 20 megapixels')
        return [image],[0.0],None
    cap=cv2.VideoCapture(str(path))
    try:
        fps=cap.get(cv2.CAP_PROP_FPS);count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not cap.isOpened() or not np.isfinite(fps) or fps<=0 or count<1:raise ValueError('Video timing could not be decoded')
        if count>MAX_VIDEO_FRAMES:raise ValueError('Body-mesh video is limited to 120 frames. Choose a shorter clip.')
        if engine=='wham' and count<30:raise ValueError('WHAM requires at least 30 continuous frames')
        frames=[]
        for i in range(count):
            ok,image=cap.read()
            if not ok:raise ValueError('Video ended before its declared frame count')
            h,w=image.shape[:2]
            if h*w>20_000_000:raise ValueError('Video frame exceeds 20 megapixels')
            # Decode one clip into bounded working resolution; preserve aspect.
            scale=min(1,960/max(h,w))
            frames.append(cv2.resize(image,(round(w*scale),round(h*scale))))
        return frames,[i/fps for i in range(count)],fps
    finally:cap.release()


def detect_boxes(images):
    from ultralytics import YOLO
    detector=YOLO(str(CHECKPOINTS/'yolov8x.pt'))
    boxes=[];scores=[]
    for i,image in enumerate(images):
        pred=detector.predict(image,device='cpu',classes=[0],conf=.5,save=False,verbose=False)[0].boxes
        boxes.append(pred.xyxy[0].cpu().numpy() if len(pred)==1 else None)
        scores.append(float(pred.conf[0]) if len(pred)==1 else None)
        progress('Detecting a single person',i+1,len(images))
    del detector;release()
    return boxes,scores


def image_tensor(image,box,device):
    import cv2
    import torch
    from WHAM.lib.models.preproc.backbone.utils import process_image
    center=(box[:2]+box[2:])/2
    scale=float(max(box[2:]-box[:2]))*1.05/200
    crop,_=process_image(cv2.cvtColor(image,cv2.COLOR_BGR2RGB),center,scale,256,256)
    return torch.from_numpy(crop).float().unsqueeze(0).to(device)


def mesh_frame(vertices,joints,time,detector_score,pose=None,betas=None):
    import numpy as np
    vertices=np.asarray(vertices);joints=np.asarray(joints)
    if not np.isfinite(vertices).all() or not np.isfinite(joints).all():raise ValueError('Model returned nonfinite geometry')
    return dict(time=float(time),status='detected',vertices=np.round(vertices,5).tolist(),
                joints=np.round(joints,5).tolist(),detector_confidence=detector_score,
                pose=None if pose is None else np.asarray(pose).round(6).tolist(),
                betas=None if betas is None else np.asarray(betas).round(6).tolist())


def hmr_frames(images,times,boxes,scores,device):
    import torch
    from smplx import SMPL
    from WHAM.lib.utils.transforms import matrix_to_axis_angle
    if all(box is None for box in boxes):
        return [dict(time=t,status='no_single_person',vertices=None,joints=None) for t in times], []
    progress('Loading HMR2 body-mesh model')
    network=load_hmr(device)
    smpl=SMPL(str(MONO_ROOT/'WHAM/dataset/body_models/smpl'),gender='neutral',num_betas=10).eval().to(device)
    result=[]
    with torch.inference_mode():
        for i,(image,box) in enumerate(zip(images,boxes)):
            if box is None:
                result.append(dict(time=times[i],status='no_single_person',vertices=None,joints=None));continue
            orient,pose,betas,_=network(image_tensor(image,box,device))
            out=smpl(global_orient=orient,body_pose=pose,betas=betas,pose2rot=False)
            root=out.joints[0,0]
            angles=matrix_to_axis_angle(torch.cat([orient,pose],dim=1))[0].cpu().numpy()
            result.append(mesh_frame((out.vertices[0]-root).cpu().numpy(),(out.joints[0,:24]-root).cpu().numpy(),times[i],scores[i],angles,betas[0].cpu().numpy()))
            progress('Estimating body mesh',i+1,len(images))
    faces=smpl.faces.tolist()
    del network,smpl;release()
    return result,faces


def wham_frames(images,times,boxes,scores,device,fps):
    import numpy as np
    import torch
    from mmpose.apis import init_pose_model,inference_top_down_pose_model
    from WHAM.configs.config import get_cfg_defaults
    from WHAM.lib.data.datasets.dataset_custom import CustomDataset
    from WHAM.lib.models import build_body_model,build_network
    if any(box is None for box in boxes):raise ValueError('WHAM needs one continuously visible person in every frame; ambiguous/missing detections were not filled in')
    progress('Loading ViTPose keypoint model')
    config=MONO_ROOT/'WHAM/third-party/ViTPose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_huge_coco_256x192.py'
    detector=init_pose_model(str(config),str(CHECKPOINTS/'vitpose-h-multi-coco.pth'),device=device)
    keypoints=[]
    for i,(image,box) in enumerate(zip(images,boxes)):
        detections,_=inference_top_down_pose_model(detector,image,person_results=[{'bbox':box}],format='xyxy')
        if len(detections)!=1:raise ValueError('Missing keypoint detection')
        keypoints.append(detections[0]['keypoints'][:17])
        progress('Estimating video keypoints',i+1,len(images))
    del detector;release()
    progress('Loading HMR2 video features')
    encoder=load_hmr(device);features=[];init={}
    with torch.inference_mode():
        for i,(image,box) in enumerate(zip(images,boxes)):
            x=image_tensor(image,box,device)
            features.append(encoder(x,encode=True).cpu())
            if i==0:
                orient,pose,betas,_=encoder(x)
                init=dict(init_global_orient=orient.cpu(),init_body_pose=pose.cpu(),init_betas=betas.cpu())
            progress('Encoding video frames',i+1,len(images))
    del encoder;release()
    cfg=get_cfg_defaults();cfg.merge_from_file(str(MONO_ROOT/'WHAM/configs/yamls/demo.yaml'))
    cfg.DEVICE=device;cfg.FLIP_EVAL=False
    bbox=np.asarray([np.r_[(b[:2]+b[2:])/2,max(b[2:]-b[:2])*1.05/200] for b in boxes])
    tracking={0:dict(keypoints=np.asarray(keypoints),bbox=bbox,features=torch.cat(features),frame_id=np.arange(len(images)),**init)}
    slam=np.zeros((len(images),7));slam[:,6]=1 # Explicit fixed-camera assumption; no SLAM.
    h,w=images[0].shape[:2]
    dataset=CustomDataset(cfg,tracking,slam,w,h,fps)
    progress('Loading WHAM temporal model')
    if not Path(cfg.TRAIN.CHECKPOINT).is_file():raise ValueError('WHAM checkpoint missing; refusing an untrained model')
    body=build_body_model(device,batch_size=len(images),gender='neutral')
    network=build_network(cfg,body).eval()
    _,x,inits,feat,mask,init_root,angvel,frame_ids,kwargs=dataset.load_data(0)
    with torch.inference_mode():
        pred=network(x,inits,feat,mask=mask,init_root=init_root,cam_angvel=angvel,return_y_up=True,**kwargs)
    # Return camera-relative mesh only. Do not export learned contacts/world motion
    # as observed gait or calibrated translation.
    meshes=pred['verts_cam'].reshape(-1,6890,3).cpu().numpy()
    joints=network.output.joints.reshape(len(images),-1,3).cpu().numpy()
    frames=[mesh_frame(v,j,t,s) for v,j,t,s in zip(meshes,joints,times,scores)]
    progress('Video meshes complete',len(frames),len(frames))
    return frames,body.faces.tolist()


def infer(path,kind='picture',engine='hmr2',device='auto'):
    import torch
    path=Path(path).absolute()
    offline_setup();torch.set_num_threads(2)
    if engine not in ('hmr2','wham') or kind not in ('picture','video'):raise ValueError('Unsupported inference mode')
    if engine=='wham' and kind!='video':raise ValueError('WHAM is a video model; choose HMR2 for pictures')
    device=('cuda' if torch.cuda.is_available() else 'cpu') if device=='auto' else device
    if device not in ('cpu','cuda'):raise ValueError('Unsupported device')
    if device=='cuda':torch.cuda.reset_peak_memory_stats()
    start=time.monotonic()
    images,times,fps=read_media(path,kind,engine)
    boxes,scores=detect_boxes(images)
    if engine=='hmr2':frames,faces=hmr_frames(images,times,boxes,scores,device)
    else:frames,faces=wham_frames(images,times,boxes,scores,device,fps)
    return dict(schema_version=1,type='medpose_body_mesh',engine=engine,device=device,
        gpu=torch.cuda.get_device_name(0) if device=='cuda' else None,
        peak_gpu_MiB=round(torch.cuda.max_memory_allocated()/1048576,1) if device=='cuda' else None,
        elapsed_seconds=round(time.monotonic()-start,2),fps=fps,frames=frames,faces=faces,
        coordinate_system='Model-estimated camera-relative geometry; not calibrated world coordinates',
        measurement_error='not_measured',clinical_validated=False,
        limitations=['Research only; not clinically validated','SMPL generic statistical body, not anatomical bones',
                     'No clinical joint angles, ground contacts, forces or diagnosis',
                     'Fixed-camera assumption for WHAM; HMR2 video frames are independent'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input',type=Path)
    parser.add_argument('--kind',choices=['picture','video'],default='picture')
    parser.add_argument('--engine',choices=['hmr2','wham'],default='hmr2')
    parser.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    args=parser.parse_args()
    # Upstream loaders print diagnostics. Keep stdout machine-readable JSON.
    original=sys.__stdout__
    try:
        sys.__stdout__=sys.stderr
        with contextlib.redirect_stdout(sys.stderr):
            result=infer(args.input,args.kind,args.engine,args.device)
    finally:
        sys.__stdout__=original
    print(json.dumps(result,separators=(',',':'),allow_nan=False))

if __name__=='__main__':
    main()
