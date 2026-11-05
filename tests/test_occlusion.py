"""Generated video/points only; no patient or upstream recordings."""
import json

import cv2
import numpy as np
import pytest

from benchmarks.occlusion import OcclusionSpec, raster_mask, occlude_keypoints, occlude_video


def spec(**kwargs):
    args = dict(style="helper", box=(0,0,1,1), start_frame=1, duration_frames=2, visible_body_fraction=.5)
    args.update(kwargs)
    return OcclusionSpec(**args)


def test_exact_keypoint_intervention_and_duration():
    points = np.tile([[1.,1.], [1.,3.], [np.nan,np.nan], [4.,1.]], (4,1,1))
    original = points.copy()
    masked, manifest = occlude_keypoints(points, 4, 4, spec())
    # Bottom 2 rows of 4x4 mask ->8 pixels. Only keypoint1 in frames1,2 is hidden.
    assert manifest["hidden_pixel_runs_row_major"] == [[8,16]]
    assert manifest["hidden_keypoints"] == [[1,1],[2,1]]
    assert manifest["hidden_pixels_per_active_frame"] == 8
    assert np.isnan(masked[1:3,1]).all()
    np.testing.assert_allclose(points, original, equal_nan=True)
    np.testing.assert_allclose(masked[[0,3]], points[[0,3]], equal_nan=True)


def test_empty_zero_duration_walker_and_fraction_boundaries():
    output, manifest = occlude_keypoints(np.empty((0,2,2)), 4, 4, spec())
    assert output.shape == (0,2,2) and manifest["frame_interval"] == [0,0]
    assert not raster_mask(4,4,spec(visible_body_fraction=1)).any()
    assert raster_mask(4,4,spec(visible_body_fraction=0)).all()
    # Walker: two one-pixel vertical bars (8 pixels) plus top middle2 =10.
    assert raster_mask(4,4,spec(style="walker", visible_body_fraction=0)).sum() == 10
    _, manifest = occlude_keypoints(np.zeros((2,1,2)),4,4,spec(duration_frames=0))
    assert manifest["hidden_keypoints"] == []


def test_generated_video(tmp_path):
    original, output, manifest_path = tmp_path/'generated.avi', tmp_path/'masked.avi', tmp_path/'mask.json'
    writer = cv2.VideoWriter(str(original), cv2.VideoWriter_fourcc(*'MJPG'), 10, (32,32))
    assert writer.isOpened()
    for _ in range(4):
        writer.write(np.full((32,32,3), 255, np.uint8))
    writer.release()
    manifest = occlude_video(original, output, manifest_path, spec())
    assert manifest["total_frames"] == 4 and manifest["frame_interval"] == [1,3]
    assert manifest["hidden_pixels_per_active_frame"] == 512
    assert json.loads(manifest_path.read_text())["source"] == "simulated"
    capture = cv2.VideoCapture(str(output))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok: break
        frames.append(frame)
    capture.release()
    assert len(frames) == 4
    assert frames[0][24,16].mean() > 240 and frames[1][24,16].mean() < 15
    assert frames[3][24,16].mean() > 240


def test_malformed_inputs(tmp_path):
    with pytest.raises(ValueError): spec(box=(0,0,2,1))
    with pytest.raises(ValueError): spec(duration_frames=-1)
    with pytest.raises(ValueError): spec(visible_body_fraction=np.nan)
    with pytest.raises(ValueError): occlude_keypoints([[1,2]],4,4,spec())
    invalid = tmp_path/'bad.avi'; invalid.write_bytes(b'not video')
    with pytest.raises(ValueError): occlude_video(invalid,tmp_path/'out.avi',tmp_path/'mask.json',spec())
    with pytest.raises(ValueError): occlude_video(invalid,invalid,tmp_path/'mask.json',spec())
