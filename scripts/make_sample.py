#!/usr/bin/env python3
"""Generate a synthetic room point cloud (spaces/sample-room/cloud.ply).

Lets the editor be exercised end-to-end before any real capture exists.
The room: floor, three walls, a table, two box seats, and some scattered
clutter to practice deleting.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
from pointcloud import PointCloud, save_ply

rng = np.random.default_rng(7)

ROOM_W, ROOM_D, ROOM_H = 5.0, 4.0, 2.7  # meters


def surface(n, u_range, v_range, plane, offset, color, jitter=0.008):
    """n points on an axis-aligned rectangle, slightly noisy like a scan."""
    u = rng.uniform(*u_range, n)
    v = rng.uniform(*v_range, n)
    w = np.full(n, offset) + rng.normal(0, jitter, n)
    axes = {"xy": (u, v, w), "xz": (u, w, v), "yz": (w, u, v)}
    x, y, z = axes[plane]
    points = np.column_stack([x, y, z])
    base = np.array(color, dtype=np.float64)
    shade = rng.normal(1.0, 0.05, (n, 1))
    colors = np.clip(base * shade, 0, 255).astype(np.uint8)
    return points, colors


def box(center, size, n_per_face, color):
    cx, cy, cz = center
    sx, sy, sz = size
    parts = [
        surface(n_per_face, (cx - sx / 2, cx + sx / 2), (cy - sy / 2, cy + sy / 2),
                "xy", cz + sz / 2, color),  # top
        surface(n_per_face, (cx - sx / 2, cx + sx / 2), (cz - sz / 2, cz + sz / 2),
                "xz", cy - sy / 2, color),
        surface(n_per_face, (cx - sx / 2, cx + sx / 2), (cz - sz / 2, cz + sz / 2),
                "xz", cy + sy / 2, color),
        surface(n_per_face, (cy - sy / 2, cy + sy / 2), (cz - sz / 2, cz + sz / 2),
                "yz", cx - sx / 2, color),
        surface(n_per_face, (cy - sy / 2, cy + sy / 2), (cz - sz / 2, cz + sz / 2),
                "yz", cx + sx / 2, color),
    ]
    return parts


parts = []

# Floor (warm wood) and walls (off-white / sage accent)
parts.append(surface(140_000, (0, ROOM_W), (0, ROOM_D), "xy", 0.0, (168, 128, 92)))
parts.append(surface(90_000, (0, ROOM_W), (0, ROOM_H), "xz", 0.0, (226, 222, 214)))
parts.append(surface(90_000, (0, ROOM_W), (0, ROOM_H), "xz", ROOM_D, (226, 222, 214)))
parts.append(surface(80_000, (0, ROOM_D), (0, ROOM_H), "yz", 0.0, (156, 176, 152)))

# Table in the middle
parts.extend(box((2.5, 2.0, 0.72), (1.6, 0.9, 0.05), 9_000, (120, 84, 56)))
for lx, ly in [(1.8, 1.6), (3.2, 1.6), (1.8, 2.4), (3.2, 2.4)]:
    parts.extend(box((lx, ly, 0.36), (0.07, 0.07, 0.72), 800, (90, 62, 40)))

# Two box seats
parts.extend(box((1.1, 3.3, 0.22), (0.7, 0.7, 0.44), 5_000, (70, 96, 138)))
parts.extend(box((3.9, 3.3, 0.22), (0.7, 0.7, 0.44), 5_000, (150, 84, 78)))

# Scan-noise clutter floating in the air: practice targets for deletion
clutter = rng.uniform(
    [0.3, 0.3, 1.6], [ROOM_W - 0.3, ROOM_D - 0.3, ROOM_H - 0.2], (4_000, 3)
)
clutter_colors = rng.integers(60, 200, (4_000, 3), dtype=np.uint8)
parts.append((clutter, clutter_colors))

points = np.vstack([p for p, _ in parts]).astype(np.float32)
colors = np.vstack([c for _, c in parts])

out = Path(__file__).resolve().parent.parent / "spaces" / "sample-room" / "cloud.ply"
save_ply(PointCloud(points, colors), out)
print(f"Wrote {len(points):,} points -> {out}")
