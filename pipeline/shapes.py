#!/usr/bin/env python3
"""Shape detection & regeneration: point cloud -> structured room model.

Detects the geometric structure of a reconstructed space and regenerates
it as clean parametric shapes:

  1. Estimate the up direction from the solved camera poses.
  2. RANSAC plane detection -> floor and walls.
  3. Cluster the remaining points -> furniture volumes (oriented boxes).
  4. Emit shapes.json (parameters) — the Blender builder
     (tools/blender_room.py) turns it into an editable .blend scene.

Usage:
    python3 pipeline/shapes.py spaces/<name> [--cloud cloud-dense.ply]
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from pointcloud import load_ply


def read_camera_rotations(path):
    rotations = []
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            struct.unpack("<I", f.read(4))
            qw, qx, qy, qz = struct.unpack("<dddd", f.read(32))
            f.read(24)
            struct.unpack("<I", f.read(4))
            while f.read(1) != b"\x00":
                pass
            npts = struct.unpack("<Q", f.read(8))[0]
            f.read(24 * npts)
            rotations.append(np.array([
                [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
                [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
                [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)],
            ]))
    return rotations


def estimate_up(model_dir: Path) -> np.ndarray:
    """A phone held upright films with its image-Y pointing at the floor,
    so world-up is the average of the cameras' -Y axes."""
    ups = [-R.T @ np.array([0.0, 1.0, 0.0]) for R in
           read_camera_rotations(model_dir / "images.bin")]
    up = np.mean(ups, axis=0)
    return up / np.linalg.norm(up)


def ransac_plane(points, threshold, iterations=400, rng=None):
    rng = rng or np.random.default_rng(0)
    best_inliers, best = 0, None
    for _ in range(iterations):
        sample = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            continue
        normal = normal / norm
        d = -normal @ sample[0]
        distance = np.abs(points @ normal + d)
        inliers = int((distance < threshold).sum())
        if inliers > best_inliers:
            best_inliers, best = inliers, (normal, d)
    normal, d = best
    mask = np.abs(points @ normal + d) < threshold
    # refine with least squares on inliers
    inlier_pts = points[mask]
    centroid = inlier_pts.mean(axis=0)
    _, _, Vt = np.linalg.svd(inlier_pts - centroid, full_matrices=False)
    normal = Vt[2]
    d = -normal @ centroid
    mask = np.abs(points @ normal + d) < threshold
    return normal, d, mask


def oriented_rect(points_2d):
    """Tight rectangle (center, axes 2x2, half-sizes) around 2D points."""
    lo, hi = np.percentile(points_2d, [1, 99], axis=0)
    return (lo + hi) / 2, (hi - lo) / 2


def cluster_grid(points, cell):
    """Connected-component clustering on a voxel grid (26-connectivity)."""
    grid = np.floor(points / cell).astype(np.int64)
    grid -= grid.min(axis=0)
    dims = grid.max(axis=0) + 2
    keys = (grid[:, 0] * dims[1] + grid[:, 1]) * dims[2] + grid[:, 2]
    unique, inverse = np.unique(keys, return_inverse=True)
    index = {k: i for i, k in enumerate(unique.tolist())}
    parent = np.arange(len(unique))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    offsets = [(dx * dims[1] + dy) * dims[2] + dz
               for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
               if (dx, dy, dz) != (0, 0, 0)]
    for i, k in enumerate(unique.tolist()):
        for off in offsets:
            j = index.get(k + off)
            if j is not None:
                ra, rb = find(i), find(j)
                if ra != rb:
                    parent[rb] = ra
    labels = np.array([find(i) for i in range(len(unique))])
    return labels[inverse]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("space")
    parser.add_argument("--cloud", default=None,
                        help="which cloud to analyze (default: densest available)")
    parser.add_argument("--max-walls", type=int, default=4)
    args = parser.parse_args()

    space = Path(args.space)
    cloud_path = (space / args.cloud) if args.cloud else next(
        p for p in [space / "cloud-dense.ply", space / "cloud.ply"] if p.exists())
    cloud = load_ply(cloud_path)
    print(f"{len(cloud):,} points from {cloud_path.name}")

    # cap points for speed
    if len(cloud) > 800_000:
        pick = np.random.default_rng(0).choice(len(cloud), 800_000, replace=False)
        pts, cols = cloud.points[pick].astype(np.float64), cloud.colors[pick]
    else:
        pts, cols = cloud.points.astype(np.float64), cloud.colors

    models = sorted((space / "workspace" / "sparse").iterdir())
    up = estimate_up(models[-1] if len(models) == 1 else
                     max(models, key=lambda d: (d / "points3D.bin").stat().st_size))
    print(f"up vector: {np.round(up, 3)}")

    # world frame: up = +Z
    z = up
    x = np.cross([0.0, 1.0, 0.0], z)
    if np.linalg.norm(x) < 1e-6:
        x = np.cross([1.0, 0.0, 0.0], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    world = np.stack([x, y, z])
    P = pts @ world.T

    extent = np.linalg.norm(np.percentile(P, 98, 0) - np.percentile(P, 2, 0))
    threshold = extent * 0.012
    remaining = np.ones(len(P), bool)
    shapes = {"up": up.tolist(), "world": world.tolist(), "planes": [], "boxes": []}
    rng = np.random.default_rng(7)

    for _ in range(args.max_walls + 2):
        active = np.where(remaining)[0]
        if len(active) < 5000:
            break
        normal, d, mask = ransac_plane(P[active], threshold, rng=rng)
        if mask.sum() < len(P) * 0.02:
            break
        idx = active[mask]
        vertical = abs(normal[2])
        kind = "floor_or_ceiling" if vertical > 0.85 else (
            "wall" if vertical < 0.35 else "slanted")
        # plane frame for the rectangle
        n = normal if normal[2] >= 0 or vertical <= 0.85 else -normal
        a = np.cross(n, [0, 0, 1.0])
        if np.linalg.norm(a) < 1e-6:
            a = np.array([1.0, 0.0, 0.0])
        a /= np.linalg.norm(a)
        b = np.cross(n, a)
        uv = np.column_stack([P[idx] @ a, P[idx] @ b])
        center_uv, half = oriented_rect(uv)
        center = center_uv[0] * a + center_uv[1] * b - d * n
        color = cols[idx].mean(axis=0)
        shapes["planes"].append({
            "kind": kind, "normal": n.tolist(), "center": center.tolist(),
            "axis_a": a.tolist(), "axis_b": b.tolist(),
            "half_a": float(half[0]), "half_b": float(half[1]),
            "points": int(mask.sum()), "color": color.astype(int).tolist(),
        })
        print(f"  plane: {kind}, {mask.sum():,} pts, "
              f"{2*half[0]:.1f} x {2*half[1]:.1f} units")
        remaining[idx] = False

    # furniture: cluster what's left
    leftover = np.where(remaining)[0]
    if len(leftover):
        labels = cluster_grid(P[leftover], cell=extent * 0.02)
        for lab in np.unique(labels):
            members = leftover[labels == lab]
            if len(members) < len(P) * 0.005:
                continue
            lo, hi = np.percentile(P[members], [2, 98], axis=0)
            if np.any(hi - lo < extent * 0.01):
                continue
            shapes["boxes"].append({
                "min": lo.tolist(), "max": hi.tolist(),
                "points": int(len(members)),
                "color": cols[members].mean(axis=0).astype(int).tolist(),
            })
            print(f"  box: {len(members):,} pts, size "
                  f"{np.round(hi - lo, 1).tolist()}")

    out = space / "shapes.json"
    out.write_text(json.dumps(shapes, indent=1))
    print(f"{len(shapes['planes'])} planes + {len(shapes['boxes'])} boxes -> {out}")


if __name__ == "__main__":
    main()
