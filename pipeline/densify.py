#!/usr/bin/env python3
"""Neural dense reconstruction: monocular depth fused over solved cameras.

COLMAP's sparse stage gives exact camera poses but few points. This stage
runs a monocular depth network (Depth Anything V2) on selected keyframes,
anchors each frame's relative depth to the sparse points visible in it
(per-frame scale/shift fit), back-projects every pixel into world space,
and merges the result into one dense colored cloud.

Usage:
    python3 pipeline/densify.py spaces/<name> [--keyframes 12] [--stride 4]

Needs: torch, transformers (Depth Anything V2 small downloads on first run).
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from pointcloud import PointCloud, remove_outliers, save_ply, trim_far_points


def quat_to_rot(qw, qx, qy, qz):
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
        [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
        [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)],
    ])


def read_cameras_bin(path):
    cameras = {}
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            cam_id, model, width, height = struct.unpack("<iiQQ", f.read(24))
            num_params = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8, 6: 12}.get(model, 4)
            params = struct.unpack(f"<{num_params}d", f.read(8 * num_params))
            cameras[cam_id] = {"model": model, "width": width, "height": height,
                               "params": params}
    return cameras


def read_images_bin(path):
    """Full parse: pose, camera id, name, and 2D->3D observations."""
    images = {}
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            image_id = struct.unpack("<I", f.read(4))[0]
            qw, qx, qy, qz = struct.unpack("<dddd", f.read(32))
            t = np.array(struct.unpack("<ddd", f.read(24)))
            cam_id = struct.unpack("<I", f.read(4))[0]
            name = b""
            while (ch := f.read(1)) != b"\x00":
                name += ch
            npts = struct.unpack("<Q", f.read(8))[0]
            raw = np.frombuffer(f.read(24 * npts), dtype=np.uint8)
            xys = raw.view("<f8").reshape(npts, 3)[:, :2] if npts else np.zeros((0, 2))
            p3d = raw.view("<i8").reshape(npts, 3)[:, 2] if npts else np.zeros(0, np.int64)
            images[image_id] = {
                "R": quat_to_rot(qw, qx, qy, qz), "t": t, "camera_id": cam_id,
                "name": name.decode(), "xys": xys, "point3D_ids": p3d,
            }
    return images


def read_points3d_bin(path):
    points = {}
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            pid = struct.unpack("<Q", f.read(8))[0]
            xyz = np.array(struct.unpack("<ddd", f.read(24)))
            f.read(3 + 8)  # rgb + error
            track_len = struct.unpack("<Q", f.read(8))[0]
            f.read(8 * track_len)
            points[pid] = xyz
    return points


def pick_keyframes(images, count):
    """Evenly spread keyframes, preferring frames with many 3D observations."""
    ordered = sorted(images.items(), key=lambda kv: kv[1]["name"])
    if len(ordered) <= count:
        return [k for k, _ in ordered]
    bins = np.array_split(np.arange(len(ordered)), count)
    chosen = []
    for b in bins:
        best = max(b, key=lambda i: (ordered[i][1]["point3D_ids"] >= 0).sum())
        chosen.append(ordered[best][0])
    return chosen


def fit_scale_shift(pred_inv, sparse_depth):
    """Robust fit: sparse_inverse_depth ~ a * predicted_inverse + b."""
    target = 1.0 / np.maximum(sparse_depth, 1e-6)
    a, b = 1.0, 0.0
    mask = np.ones(len(target), bool)
    for _ in range(4):
        A = np.column_stack([pred_inv[mask], np.ones(mask.sum())])
        (a, b), *_ = np.linalg.lstsq(A, target[mask], rcond=None)
        residual = np.abs(a * pred_inv + b - target)
        cutoff = 2.5 * np.median(residual) + 1e-9
        mask = residual < cutoff
        if mask.sum() < 8:
            break
    return a, b


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("space", help="space folder (e.g. spaces/first-test)")
    parser.add_argument("--model-dir", default=None,
                        help="COLMAP model dir (default: workspace/sparse/<best>)")
    parser.add_argument("--keyframes", type=int, default=12)
    parser.add_argument("--stride", type=int, default=4,
                        help="back-project every Nth pixel (default 4)")
    args = parser.parse_args()

    space = Path(args.space)
    workspace = space / "workspace"
    if args.model_dir:
        model_dir = Path(args.model_dir)
    else:
        candidates = sorted((workspace / "sparse").iterdir())
        model_dir = max(
            candidates,
            key=lambda d: len(read_points3d_bin(d / "points3D.bin")),
        )
    print(f"Using model {model_dir}")

    cameras = read_cameras_bin(model_dir / "cameras.bin")
    images = read_images_bin(model_dir / "images.bin")
    points3d = read_points3d_bin(model_dir / "points3D.bin")
    keys = pick_keyframes(images, args.keyframes)
    print(f"{len(images)} registered frames; densifying {len(keys)} keyframes")

    import torch
    from PIL import Image
    from transformers import pipeline as hf_pipeline

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    depth_model = hf_pipeline(
        "depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf",
        device=device,
    )
    print(f"Depth Anything V2 (small) on {device}")

    all_pts, all_cols = [], []
    for i, image_id in enumerate(keys):
        info = images[image_id]
        cam = cameras[info["camera_id"]]
        fx, fy, cx, cy = cam["params"][:4]
        img = Image.open(workspace / "images" / info["name"])
        W, H = img.size

        pred = depth_model(img)["predicted_depth"].squeeze().float().cpu().numpy()
        pred = np.array(Image.fromarray(pred).resize((W, H), Image.BILINEAR))
        # Depth Anything outputs relative INVERSE depth (bigger = closer)
        pred_inv = np.maximum(pred, 1e-6)

        # anchor to sparse points seen in this frame
        valid = info["point3D_ids"] >= 0
        xy = info["xys"][valid]
        ids = info["point3D_ids"][valid]
        world = np.array([points3d[p] for p in ids if p in points3d])
        xy = xy[[p in points3d for p in ids]]
        cam_space = (info["R"] @ world.T).T + info["t"]
        in_front = cam_space[:, 2] > 0.05
        xy, depth_true = xy[in_front], cam_space[in_front, 2]
        px = np.clip(xy[:, 0].astype(int), 0, W - 1)
        py = np.clip(xy[:, 1].astype(int), 0, H - 1)
        if len(px) < 12:
            print(f"  {info['name']}: too few anchors, skipped")
            continue
        a, b = fit_scale_shift(pred_inv[py, px], depth_true)
        metric_depth = 1.0 / np.maximum(a * pred_inv + b, 1e-6)

        # back-project a pixel grid
        s = args.stride
        us, vs = np.meshgrid(np.arange(0, W, s), np.arange(0, H, s))
        us, vs = us.ravel(), vs.ravel()
        z = metric_depth[vs, us]
        ok = (z > 0.05) & (z < np.percentile(depth_true, 98) * 3)
        us, vs, z = us[ok], vs[ok], z[ok]
        rays = np.column_stack([(us - cx) / fx, (vs - cy) / fy, np.ones(len(us))])
        cam_pts = rays * z[:, None]
        world_pts = (info["R"].T @ (cam_pts - info["t"]).T).T
        rgb = np.asarray(img)[vs, us]

        all_pts.append(world_pts.astype(np.float32))
        all_cols.append(rgb.astype(np.uint8))
        print(f"  [{i+1}/{len(keys)}] {info['name']}: "
              f"{len(world_pts):,} points (scale fit on {len(px)} anchors)")

    cloud = PointCloud(np.vstack(all_pts), np.vstack(all_cols))
    print(f"Fused: {len(cloud):,} points; cleaning up")
    cloud = trim_far_points(cloud, factor=8)
    extent = np.linalg.norm(cloud.points.max(0) - cloud.points.min(0))
    cloud = remove_outliers(cloud, voxel_size=extent / 200, min_neighbors=4)
    out = space / "cloud-dense.ply"
    save_ply(cloud, out)
    print(f"Dense cloud: {len(cloud):,} points -> {out}")


if __name__ == "__main__":
    main()
