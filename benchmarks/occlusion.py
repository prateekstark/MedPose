"""Controlled synthetic masking. No pose model or robustness claim.

The visible-body-fraction setting is an ROI-area proxy, not anatomical visibility.
Caller supplies a normalized bounding region; the tool does not locate a person.
"""
from dataclasses import dataclass, asdict
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class OcclusionSpec:
    style: str
    box: tuple  # normalized left, top, right, bottom
    start_frame: int
    duration_frames: int
    visible_body_fraction: float
    bar_fraction: float = .1  # geometric walker-bar width, not a clinical threshold

    def __post_init__(self):
        if self.style not in {"walker", "helper", "region"}:
            raise ValueError("style must be walker, helper or region")
        if not isinstance(self.box, (tuple, list)) or len(self.box) != 4:
            raise ValueError("box must contain left, top, right, bottom")
        if any(type(v) not in (int, float) or not np.isfinite(v) for v in self.box):
            raise ValueError("Box values must be finite numbers")
        x0, y0, x1, y1 = self.box
        if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
            raise ValueError("Box must be ordered within [0,1]")
        for name in ("start_frame", "duration_frames"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError("Frame offsets/durations must be nonnegative integers")
        for name in ("visible_body_fraction", "bar_fraction"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Fractions must lie in [0,1]")
        if self.bar_fraction == 0:
            raise ValueError("bar_fraction must be positive")
        object.__setattr__(self, "box", tuple(self.box))


def raster_mask(width, height, spec):
    """Return exact boolean mask; normalized rectangles use floor/ceil bounds."""
    if type(width) is not int or type(height) is not int or min(width, height) <= 0:
        raise ValueError("Raster dimensions must be positive integers")
    if not isinstance(spec, OcclusionSpec):
        raise ValueError("Provide OcclusionSpec")
    x0, y0 = np.floor(np.array(spec.box[:2]) * [width, height]).astype(int)
    x1, y1 = np.ceil(np.array(spec.box[2:]) * [width, height]).astype(int)
    mask = np.zeros((height, width), dtype=bool)
    # Occluded bottom fraction of the supplied ROI; not an inferred body silhouette.
    hidden_height = int(np.ceil((y1 - y0) * (1 - spec.visible_body_fraction)))
    top = y1 - hidden_height
    if hidden_height:
        if spec.style in {"helper", "region"}:
            mask[top:y1, x0:x1] = True
        else:
            thickness = max(1, int(np.ceil((x1 - x0) * spec.bar_fraction)))
            mask[top:y1, x0:min(x1, x0 + thickness)] = True
            mask[top:y1, max(x0, x1 - thickness):x1] = True
            mask[top:min(y1, top + thickness), x0:x1] = True
    return mask


def _runs(mask):
    """Exact row-major half-open runs [start, stop) of hidden pixels."""
    flat = mask.ravel().astype(np.int8)
    boundaries = np.diff(np.r_[0, flat, 0])
    return np.column_stack([np.flatnonzero(boundaries == 1), np.flatnonzero(boundaries == -1)]).tolist()


def _manifest(mask, spec, frames):
    start = min(spec.start_frame, frames)
    stop = min(spec.start_frame + spec.duration_frames, frames)
    return dict(schema_version=1, source="simulated", spec=asdict(spec),
                frame_interval=[start, stop], interval_convention="half-open",
                width=int(mask.shape[1]), height=int(mask.shape[0]), total_frames=frames,
                hidden_pixel_runs_row_major=_runs(mask), hidden_pixels_per_active_frame=int(mask.sum()),
                fraction_note="visible_body_fraction is an ROI proxy, not measured anatomical visibility",
                robustness="not evaluated")


def occlude_keypoints(points, width, height, spec):
    """Pixel coordinates (frames, keypoints, xy); floor to raster pixels, NaN out hidden.

    Missing and off-image keypoints are recorded separately, never counted as
    newly hidden. Input and ground truth arrays are not modified.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 3 or points.shape[2] != 2 or np.isinf(points).any():
        raise ValueError("Keypoints must be (frames, keypoints, 2), finite or NaN")
    mask = raster_mask(width, height, spec)
    output = points.copy()
    hidden = np.zeros(points.shape[:2], dtype=bool)
    present = np.isfinite(points).all(axis=2)
    inside = present & (points[:, :, 0] >= 0) & (points[:, :, 0] < width) & (points[:, :, 1] >= 0) & (points[:, :, 1] < height)
    for frame in range(spec.start_frame, min(len(points), spec.start_frame + spec.duration_frames)):
        indices = np.flatnonzero(inside[frame])
        pixels = np.floor(points[frame, indices]).astype(int)
        hidden[frame, indices] = mask[pixels[:, 1], pixels[:, 0]]
    output[hidden] = np.nan
    manifest = _manifest(mask, spec, len(points))
    manifest.update(hidden_keypoints=np.argwhere(hidden).tolist(),
                    missing_keypoints=np.argwhere(~present).tolist(),
                    off_image_keypoints=np.argwhere(present & ~inside).tolist())
    return output, manifest


def occlude_video(input_path, output_path, manifest_path, spec):
    """Apply masks to a local video and record exact intervention geometry.

    OpenCV already exists in the project lock. AVI/MJPG output is a visualization
    derivative: lossy encoding may change unmasked pixels too. Mask geometry is
    exact, but decoded pixel identity is not promised. No files are uploaded.
    """
    import cv2

    input_path, output_path, manifest_path = map(Path, (input_path, output_path, manifest_path))
    if len({p.resolve() for p in (input_path, output_path, manifest_path)}) != 3:
        raise ValueError("Input, output and manifest must be different files")
    if output_path.exists() or manifest_path.exists():
        raise ValueError("Refusing to overwrite existing output or manifest")
    if output_path.suffix.lower() != ".avi":
        raise ValueError("Output must use .avi for the specified MJPG codec")
    capture = cv2.VideoCapture(str(input_path))
    writer = None
    try:
        if not capture.isOpened():
            raise ValueError("Could not open input video")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("Video has no valid frame rate")
        ok, frame = capture.read()
        if not ok:
            raise ValueError("Video contains no readable frames")
        height, width = frame.shape[:2]
        mask = raster_mask(width, height, spec)
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height))
        if not writer.isOpened():
            raise ValueError("Could not create output video")
        count = 0
        while ok:
            if frame.shape[:2] != (height, width):
                raise ValueError("Video frame dimensions changed")
            if spec.start_frame <= count < spec.start_frame + spec.duration_frames:
                frame[mask] = 0
            writer.write(frame)
            count += 1
            ok, frame = capture.read()
        writer.release(); writer = None
        manifest = _manifest(mask, spec, count)
        manifest.update(fps=fps, requested_duration_seconds=spec.duration_frames / fps,
                        applied_duration_seconds=(manifest["frame_interval"][1] - manifest["frame_interval"][0]) / fps,
                        codec="MJPG", encoding_caveat="Lossy encoding can also change unmasked pixels")
        manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        return manifest
    except Exception:
        if writer is not None:
            writer.release()
            writer = None
        if output_path.exists():
            output_path.unlink()
        raise
    finally:
        capture.release()
        if writer is not None:
            writer.release()
