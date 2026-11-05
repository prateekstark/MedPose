#!/usr/bin/env python3
"""Download upstream public inference checkpoints, excluding license-gated SMPL assets."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib
import shutil
import time
import urllib.request
from pathlib import Path

import gdown


ROOT = Path(__file__).resolve().parent.parent
MONO = ROOT / "opencap-monocular"
CORE = ROOT / "opencap-core"

WHAM_DRIVE = {
    "wham_vit_w_3dpw.pth.tar": "1i7kt9RlCCCNEW2aYaDWVr-G778JkLNcB",
    "wham_vit_bedlam_w_3dpw.pth.tar": "19qkI-a6xuwob9_RFNSPWf1yWErwVVlks",
    "hmr2a.ckpt": "1J6l8teyZrL0zFzHhzkC7efRhU0ZJ5G9Y",
    "yolov8x.pt": "1zJ0KP23tXD42D47cw1Gs7zE2BA_V_ERo",
    "vitpose-h-multi-coco.pth": "1xyF7F3I7lWtdq82xmEPVQ5zl4HaasBso",
}
CORE_HTTP = {
    "faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth": (
        "https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/"
        "faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth"
    ),
    "hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth": (
        "https://download.openmmlab.com/mmpose/top_down/hrnet/"
        "hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth"
    ),
}


def download_drive(file_id: str, output: Path) -> None:
    # Google Drive supports byte ranges for these public checkpoints. Download
    # ranges concurrently so a slow single connection does not dominate setup.
    gdown_module = importlib.import_module("gdown.download")
    session, _ = gdown_module._get_session(
        proxy=None,
        use_cookies=True,
        user_agent="Mozilla/5.0",
        return_cookies_file=True,
    )
    response = session.get(f"https://drive.google.com/uc?id={file_id}", stream=True)
    try:
        if "Content-Disposition" in response.headers:
            url = response.url
        else:
            url = gdown_module.get_url_from_gdrive_confirmation(response.text)
        response.close()

        probe = session.get(url, headers={"Range": "bytes=0-0"}, stream=True, timeout=(30, 120))
        content_range = probe.headers.get("Content-Range", "")
        probe.close()
        if probe.status_code != 206 or not content_range.startswith("bytes 0-0/"):
            raise RuntimeError(f"Google Drive did not provide a byte-range response: {content_range}")
        total_size = int(content_range.rsplit("/", 1)[1])

        partials = sorted(
            output.parent.glob(output.name + "*.part"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if len(partials) > 1:
            raise RuntimeError(f"Multiple interrupted downloads need cleanup: {partials}")
        prefix = partials[0] if partials else None
        prefix_size = prefix.stat().st_size if prefix else 0
        if prefix_size > total_size:
            raise RuntimeError(f"Partial file is larger than the checkpoint: {prefix_size} > {total_size}")

        block_size = 16 * 1024 * 1024
        part_dir = Path("/tmp/medpose-gdrive-ranges") / output.name
        part_dir.mkdir(parents=True, exist_ok=True)

        def fetch(index: int) -> int:
            start = prefix_size + index * block_size
            end = min(total_size - 1, start + block_size - 1)
            if start >= total_size:
                return index
            part = part_dir / f"{index:04}.part"
            expected = end - start + 1
            if part.exists() and part.stat().st_size == expected:
                return index
            for attempt in range(6):
                temporary = part.with_suffix(".tmp")
                try:
                    ranged = session.get(
                        url,
                        headers={"Range": f"bytes={start}-{end}"},
                        stream=True,
                        timeout=(30, 180),
                    )
                    expected_range = f"bytes {start}-{end}/{total_size}"
                    if ranged.status_code != 206 or ranged.headers.get("Content-Range") != expected_range:
                        raise RuntimeError(
                            f"unexpected range response {ranged.status_code}: {ranged.headers.get('Content-Range')}"
                        )
                    received = 0
                    with temporary.open("wb") as target:
                        for data in ranged.iter_content(1024 * 1024):
                            if data:
                                target.write(data)
                                received += len(data)
                    ranged.close()
                    if received != expected:
                        raise RuntimeError(f"expected {expected} bytes, received {received}")
                    temporary.replace(part)
                    print(f"  range {index + 1}: {received // (1024 * 1024)} MiB", flush=True)
                    return index
                except Exception as error:
                    print(f"  range {index + 1} retry {attempt + 1}: {error}", flush=True)
                    time.sleep(min(3 * (attempt + 1), 18))
            raise RuntimeError(f"Google Drive range {index + 1} failed")

        count = (total_size - prefix_size + block_size - 1) // block_size
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(fetch, index) for index in range(count)]
            for future in as_completed(futures):
                future.result()

        assembled = output.with_suffix(output.suffix + ".assembling")
        with assembled.open("wb") as target:
            if prefix is not None:
                with prefix.open("rb") as source:
                    shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
            for index in range(count):
                with (part_dir / f"{index:04}.part").open("rb") as source:
                    shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
        if assembled.stat().st_size != total_size:
            assembled.unlink(missing_ok=True)
            raise RuntimeError(f"assembled checkpoint has wrong size: {assembled.stat().st_size}")
        assembled.replace(output)
        if prefix is not None:
            prefix.unlink(missing_ok=True)
        shutil.rmtree(part_dir, ignore_errors=True)
    finally:
        session.close()


def download_http(url: str, output: Path) -> None:
    probe_request = urllib.request.Request(
        url,
        headers={"User-Agent": "MedPose local setup", "Range": "bytes=0-0", "Accept-Encoding": "identity"},
    )
    with urllib.request.urlopen(probe_request, timeout=60) as probe:
        content_range = probe.headers.get("Content-Range", "")
        if probe.status != 206 or not content_range.startswith("bytes 0-0/"):
            raise RuntimeError(f"model host did not provide a byte-range response: {content_range}")
        total_size = int(content_range.rsplit("/", 1)[1])

    block_size = 16 * 1024 * 1024
    part_dir = Path("/tmp/medpose-http-ranges") / output.name
    part_dir.mkdir(parents=True, exist_ok=True)

    def fetch(index: int) -> int:
        start = index * block_size
        end = min(total_size - 1, start + block_size - 1)
        part = part_dir / f"{index:04}.part"
        expected = end - start + 1
        if part.exists() and part.stat().st_size == expected:
            return index
        for attempt in range(6):
            temporary = part.with_suffix(".tmp")
            try:
                request = urllib.request.Request(
                    url,
                    headers={
                        "User-Agent": "MedPose local setup",
                        "Range": f"bytes={start}-{end}",
                        "Accept-Encoding": "identity",
                    },
                )
                with urllib.request.urlopen(request, timeout=180) as response:
                    expected_range = f"bytes {start}-{end}/{total_size}"
                    if response.status != 206 or response.headers.get("Content-Range") != expected_range:
                        raise RuntimeError(
                            f"unexpected range response {response.status}: {response.headers.get('Content-Range')}"
                        )
                    received = 0
                    with temporary.open("wb") as target:
                        while True:
                            block = response.read(1024 * 1024)
                            if not block:
                                break
                            target.write(block)
                            received += len(block)
                if received != expected:
                    raise RuntimeError(f"expected {expected} bytes, received {received}")
                temporary.replace(part)
                print(f"  range {index + 1}: {received // (1024 * 1024)} MiB", flush=True)
                return index
            except Exception as error:
                print(f"  range {index + 1} retry {attempt + 1}: {error}", flush=True)
                time.sleep(min(3 * (attempt + 1), 18))
        raise RuntimeError(f"HTTP model range {index + 1} failed")

    count = (total_size + block_size - 1) // block_size
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(fetch, index) for index in range(count)]
        for future in as_completed(futures):
            future.result()

    assembled = output.with_suffix(output.suffix + ".assembling")
    with assembled.open("wb") as target:
        for index in range(count):
            with (part_dir / f"{index:04}.part").open("rb") as source:
                shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
    if assembled.stat().st_size != total_size:
        actual_size = assembled.stat().st_size
        assembled.unlink(missing_ok=True)
        raise RuntimeError(f"assembled checkpoint has wrong size: {actual_size} != {total_size}")
    assembled.replace(output)
    shutil.rmtree(part_dir, ignore_errors=True)


def save_if_missing(output: Path, downloader, url_or_id: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and output.stat().st_size:
        print(f"Already present: {output}")
        return
    print(f"Downloading: {output.name}")
    try:
        downloader(url_or_id, output)
        if not output.exists() or output.stat().st_size == 0:
            raise RuntimeError("download completed without producing a file")
    except Exception:
        output.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch public WHAM/ViTPose and OpenCap Core pose checkpoints."
    )
    parser.add_argument("--workflow", choices=("all", "mono", "core"), default="all")
    args = parser.parse_args()

    if args.workflow in ("all", "mono"):
        for filename, file_id in WHAM_DRIVE.items():
            save_if_missing(MONO / "WHAM" / "checkpoints" / filename, download_drive, file_id)
    if args.workflow in ("all", "core"):
        for filename, url in CORE_HTTP.items():
            save_if_missing(CORE / "models" / filename, download_http, url)
    print("Public checkpoints are ready. SMPL body-model files were intentionally excluded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
