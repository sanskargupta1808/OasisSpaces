#!/usr/bin/env python3
"""Classify detected shapes in a space's shapes.json.

Usage:
    python3 tools/classify_shapes.py spaces/<name>

Reads spaces/<name>/shapes.json, assigns a "label" to every plane and box
(rewriting the file in place), and prints a table of the decisions.

Plane labels:
    floor    large horizontal plane at the room's floor level
    ceiling  horizontal plane clearly above the floor by ~room height
    wall     vertical plane
    surface  other horizontal plane (e.g. a table top)
    slanted  anything else

Box labels (all thresholds are RELATIVE to room size derived from the floor
plane and wall heights, so the classifier works regardless of scan units):
    seat, table, bed, wardrobe, clutter, block (fallback)

Coordinates in shapes.json are already in a Z-up frame (up = +Z).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median

# --------------------------------------------------------------------------
# Thresholds. Angles are absolute (orientation is unit-free); everything else
# is a fraction of room height (H) or floor area (A).
# --------------------------------------------------------------------------
HORIZONTAL_MAX_TILT_DEG = 25.0   # normal within this angle of +Z  -> horizontal
VERTICAL_MIN_TILT_DEG = 65.0     # normal at least this far from +Z -> vertical

FLOOR_AREA_MIN_FRAC = 0.40       # floor candidate: area >= this frac of largest horizontal plane
FLOOR_LEVEL_TOL_FRAC = 0.25      # floor candidate: |center_z - wall-bottom estimate| <= this * H
CEILING_MIN_HEIGHT_FRAC = 0.65   # ceiling: at least this * H above floor level

SEAT_MAX_FOOTPRINT_FRAC = 0.06   # seat: footprint <= 6% of floor area
SEAT_MAX_HEIGHT_FRAC = 0.25      # seat/table height band: < ~25% of room height
TABLE_MAX_HEIGHT_FRAC = 0.35     # table/bed may be slightly taller
TABLE_MIN_ELONGATION = 1.60      # elongated footprint => table
TABLE_MAX_FOOTPRINT_FRAC = 0.15  # a table still has a bounded footprint
BED_MIN_FOOTPRINT_FRAC = 0.15    # bed: footprint > ~15% of floor area
WARDROBE_MIN_HEIGHT_FRAC = 0.55  # wardrobe: taller than ~55% of room height
WARDROBE_MAX_FOOTPRINT_FRAC = 0.20
WARDROBE_MIN_FOOTPRINT_FRAC = 0.002
WARDROBE_WALL_CLEARANCE_FRAC = 0.10  # of sqrt(floor area)

NEAR_FLOOR_MAX_GAP_FRAC = 0.20   # bottom within this * H above floor counts as "on the floor"
NEAR_FLOOR_MIN_GAP_FRAC = -0.25  # tolerate the box sinking slightly below the floor fit
FLOATING_MIN_GAP_FRAC = 0.35     # bottom higher than this * H above floor -> floating clutter
LOW_DENSITY_MAX_REL = 0.10       # box surface density below 10% of scene density -> clutter
OVERSIZE_HEIGHT_FRAC = 1.10      # taller than the room itself -> not furniture
OVERSIZE_FOOTPRINT_FRAC = 0.70   # covers most of the floor -> not furniture


# --------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------
def plane_tilt_deg(plane: dict) -> float:
    """Angle between the plane normal and the up axis (+Z), folded to [0, 90]."""
    nz = abs(plane["normal"][2])
    return math.degrees(math.acos(max(-1.0, min(1.0, nz))))


def plane_area(plane: dict) -> float:
    return 4.0 * plane["half_a"] * plane["half_b"]


def plane_z_extent(plane: dict) -> float:
    """Half-extent of the plane patch along Z."""
    return (abs(plane["axis_a"][2]) * plane["half_a"]
            + abs(plane["axis_b"][2]) * plane["half_b"])


def box_size(box: dict) -> tuple[float, float, float]:
    return tuple(box["max"][i] - box["min"][i] for i in range(3))


def box_surface_area(box: dict) -> float:
    sx, sy, sz = box_size(box)
    return 2.0 * (sx * sy + sx * sz + sy * sz)


def box_center(box: dict) -> tuple[float, float, float]:
    return tuple((box["min"][i] + box["max"][i]) / 2.0 for i in range(3))


# --------------------------------------------------------------------------
# Room context
# --------------------------------------------------------------------------
@dataclass
class RoomContext:
    floor_z: float
    room_height: float
    floor_area: float
    ref_density: float | None       # scene points-per-area reference (median over planes)
    floor_source: str
    height_source: str
    walls: list[dict] = field(default_factory=list)
    surfaces: list[dict] = field(default_factory=list)


def orientation_class(plane: dict) -> str:
    tilt = plane_tilt_deg(plane)
    if tilt <= HORIZONTAL_MAX_TILT_DEG:
        return "horizontal"
    if tilt >= VERTICAL_MIN_TILT_DEG:
        return "vertical"
    return "slanted"


def classify_planes(planes: list[dict]) -> tuple[list[str], list[str], RoomContext]:
    """Label every plane and derive the room context used for boxes."""
    orient = [orientation_class(p) for p in planes]
    walls = [p for p, o in zip(planes, orient) if o == "vertical"]
    horiz = [(i, p) for i, (p, o) in enumerate(zip(planes, orient)) if o == "horizontal"]

    # Room height and floor level, estimated from the vertical span of walls.
    # Walls are the most reliable cue: horizontal planes may be table tops or
    # duplicate ceiling detections, so the lowest one is NOT necessarily the floor.
    if walls:
        room_height = median(2.0 * plane_z_extent(w) for w in walls)
        floor_est = median(w["center"][2] - plane_z_extent(w) for w in walls)
        height_source = f"median of {len(walls)} wall heights"
    elif horiz:
        zs = [p["center"][2] for _, p in horiz]
        floor_est = min(zs)
        spread = max(zs) - min(zs)
        if spread > 1e-6:
            room_height = spread
            height_source = "horizontal-plane z spread (no walls found)"
        else:
            # All horizontal planes at one level: no vertical cue at all.
            # A room's height is on the order of its floor width, so use that
            # (scale-aware) rather than a degenerate near-zero spread.
            largest = max(plane_area(p) for _, p in horiz)
            room_height = max(math.sqrt(largest), 1.0)
            height_source = "sqrt of largest horizontal plane area (no vertical cue)"
    else:
        floor_est, room_height, height_source = 0.0, 1.0, "default (no planes)"

    # Floor: large horizontal planes sitting near the wall-bottom estimate.
    floor_idx: set[int] = set()
    if horiz:
        max_area = max(plane_area(p) for _, p in horiz)
        candidates = [
            (i, p) for i, p in horiz
            if plane_area(p) >= FLOOR_AREA_MIN_FRAC * max_area
            and (not walls
                 or abs(p["center"][2] - floor_est) <= FLOOR_LEVEL_TOL_FRAC * room_height)
        ]
        if candidates:
            lowest_z = min(p["center"][2] for _, p in candidates)
            floor_idx = {i for i, p in candidates
                         if p["center"][2] <= lowest_z + 0.05 * room_height}

    if floor_idx:
        floor_z = min(planes[i]["center"][2] for i in floor_idx)
        floor_area = max(plane_area(planes[i]) for i in floor_idx)
        floor_source = "floor plane"
    else:
        floor_z = floor_est
        floor_area = max((plane_area(p) for _, p in horiz), default=0.0)
        floor_source = "wall-bottom estimate (no floor plane detected)"
    if floor_area <= 0.0:  # last-resort so ratios stay finite
        floor_area = room_height * room_height
        floor_source += "; area from room height"

    ref_density = None
    densities = [p["points"] / plane_area(p) for p in planes if plane_area(p) > 0]
    if densities:
        ref_density = median(densities)

    labels, reasons = [], []
    for i, (plane, o) in enumerate(zip(planes, orient)):
        tilt = plane_tilt_deg(plane)
        z = plane["center"][2]
        if o == "vertical":
            labels.append("wall")
            reasons.append(f"vertical (tilt {tilt:.1f} deg)")
        elif o == "slanted":
            labels.append("slanted")
            reasons.append(f"tilted {tilt:.1f} deg from up")
        elif i in floor_idx:
            labels.append("floor")
            reasons.append(f"large horizontal at floor level z={z:.2f}")
        elif z >= floor_z + CEILING_MIN_HEIGHT_FRAC * room_height:
            labels.append("ceiling")
            reasons.append(f"horizontal, {(z - floor_z) / room_height:.0%} of room height above floor")
        else:
            labels.append("surface")
            reasons.append(f"horizontal, not floor/ceiling (z={z:.2f})")

    ctx = RoomContext(
        floor_z=floor_z,
        room_height=max(room_height, 1e-9),
        floor_area=floor_area,
        ref_density=ref_density,
        floor_source=floor_source,
        height_source=height_source,
        walls=walls,
        surfaces=[p for p, lab in zip(planes, labels) if lab == "surface"],
    )
    return labels, reasons, ctx


# --------------------------------------------------------------------------
# Box classification
# --------------------------------------------------------------------------
def _wall_clearance(box: dict, wall: dict) -> float:
    """Distance between the box and the wall plane (0 if they touch/intersect)."""
    n = wall["normal"]
    c = box_center(box)
    d = sum((c[k] - wall["center"][k]) * n[k] for k in range(3))
    half_along_n = sum(0.5 * s * abs(n[k]) for k, s in enumerate(box_size(box)))
    return max(0.0, abs(d) - half_along_n)


def _has_surface_above(box: dict, ctx: RoomContext) -> bool:
    """A 'surface' plane hovering just above the box footprint => table pairing."""
    cx, cy, _ = box_center(box)
    sx, sy, _ = box_size(box)
    top = box["max"][2]
    h = ctx.room_height
    for s in ctx.surfaces:
        z = s["center"][2]
        if not (top - 0.05 * h <= z <= top + 0.30 * h):
            continue
        if (abs(s["center"][0] - cx) <= 0.5 * sx + 0.10 * math.sqrt(ctx.floor_area)
                and abs(s["center"][1] - cy) <= 0.5 * sy + 0.10 * math.sqrt(ctx.floor_area)):
            return True
    return False


def classify_box(box: dict, ctx: RoomContext) -> tuple[str, str, dict]:
    sx, sy, sz = box_size(box)
    footprint = sx * sy
    fp_frac = footprint / ctx.floor_area
    h_frac = sz / ctx.room_height
    gap_frac = (box["min"][2] - ctx.floor_z) / ctx.room_height
    elong = max(sx, sy) / max(min(sx, sy), 1e-9)
    rel_density = None
    if ctx.ref_density:
        rel_density = (box["points"] / max(box_surface_area(box), 1e-9)) / ctx.ref_density
    metrics = {"fp_frac": fp_frac, "h_frac": h_frac, "gap_frac": gap_frac,
               "elong": elong, "rel_density": rel_density}

    near_floor = NEAR_FLOOR_MIN_GAP_FRAC <= gap_frac <= NEAR_FLOOR_MAX_GAP_FRAC

    # Sanity gates first: scan ghosts and debris, in relative terms.
    if h_frac > OVERSIZE_HEIGHT_FRAC or fp_frac > OVERSIZE_FOOTPRINT_FRAC:
        return ("clutter",
                f"implausibly large ({h_frac:.0%} of room height, "
                f"{fp_frac:.0%} of floor area)", metrics)
    if rel_density is not None and rel_density < LOW_DENSITY_MAX_REL:
        return ("clutter",
                f"sparse: {rel_density:.1%} of scene surface density", metrics)
    if gap_frac > FLOATING_MIN_GAP_FRAC:
        return ("clutter",
                f"floating {gap_frac:.0%} of room height above floor", metrics)
    if gap_frac < NEAR_FLOOR_MIN_GAP_FRAC:
        return ("clutter",
                f"detached: {-gap_frac:.0%} of room height below floor level", metrics)

    if (h_frac >= WARDROBE_MIN_HEIGHT_FRAC
            and WARDROBE_MIN_FOOTPRINT_FRAC <= fp_frac <= WARDROBE_MAX_FOOTPRINT_FRAC
            and near_floor):
        limit = WARDROBE_WALL_CLEARANCE_FRAC * math.sqrt(ctx.floor_area)
        if any(_wall_clearance(box, w) <= limit for w in ctx.walls):
            return ("wardrobe",
                    f"tall ({h_frac:.0%} of room height), near a wall", metrics)

    if fp_frac >= BED_MIN_FOOTPRINT_FRAC and h_frac <= TABLE_MAX_HEIGHT_FRAC and near_floor:
        return ("bed",
                f"broad footprint ({fp_frac:.0%} of floor), low ({h_frac:.0%} of room height)",
                metrics)

    if h_frac <= TABLE_MAX_HEIGHT_FRAC and near_floor and fp_frac <= TABLE_MAX_FOOTPRINT_FRAC:
        if _has_surface_above(box, ctx):
            return ("table", "surface plane detected just above it", metrics)
        if elong >= TABLE_MIN_ELONGATION and h_frac <= TABLE_MAX_HEIGHT_FRAC:
            return ("table", f"elongated footprint ({elong:.1f}:1), low, on floor", metrics)

    if fp_frac <= SEAT_MAX_FOOTPRINT_FRAC and h_frac <= SEAT_MAX_HEIGHT_FRAC and near_floor:
        return ("seat",
                f"small footprint ({fp_frac:.1%} of floor), "
                f"low ({h_frac:.0%} of room height), on floor", metrics)

    return ("block", "no furniture rule matched", metrics)


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
def _print_table(headers: list[str], rows: list[list[str]]) -> None:
    widths = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h)
              for i, h in enumerate(headers)]
    line = "  ".join(h.ljust(w) for h, w in zip(headers, widths))
    print(line)
    print("-" * len(line))
    for r in rows:
        print("  ".join(c.ljust(w) for c, w in zip(r, widths)))


def report(space: str, planes: list[dict], boxes: list[dict],
           plane_labels: list[str], plane_reasons: list[str],
           box_results: list[tuple[str, str, dict]], ctx: RoomContext) -> None:
    print(f"== {space} ==")
    dens = f"{ctx.ref_density:.0f} pts/area" if ctx.ref_density else "n/a"
    print(f"room context: floor_z={ctx.floor_z:.3f} ({ctx.floor_source}), "
          f"room_height={ctx.room_height:.2f} ({ctx.height_source}), "
          f"floor_area={ctx.floor_area:.2f}, scene_density={dens}")
    print()

    print(f"PLANES ({len(planes)})")
    rows = []
    for i, (p, lab, why) in enumerate(zip(planes, plane_labels, plane_reasons)):
        rows.append([
            str(i), p["kind"], lab, f"{plane_tilt_deg(p):5.1f}",
            f"{p['center'][2]:8.3f}",
            f"{2 * p['half_a']:.2f} x {2 * p['half_b']:.2f}",
            str(p["points"]), why,
        ])
    _print_table(["#", "kind", "label", "tilt", "center_z", "size", "points", "reason"], rows)
    print()

    print(f"BOXES ({len(boxes)})")
    rows = []
    for i, (b, (lab, why, m)) in enumerate(zip(boxes, box_results)):
        sx, sy, sz = box_size(b)
        dens = f"{m['rel_density']:.2f}" if m["rel_density"] is not None else "n/a"
        rows.append([
            str(i), lab, f"{sx:.2f} x {sy:.2f} x {sz:.2f}",
            f"{m['fp_frac']:6.1%}", f"{m['h_frac']:5.0%}", f"{m['gap_frac']:+5.0%}",
            dens, str(b["points"]), why,
        ])
    _print_table(["#", "label", "size", "foot/A", "hgt/H", "gap/H", "dens", "points", "reason"],
                 rows)
    print()


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Label planes and boxes in a space's shapes.json (in place).")
    parser.add_argument("space", help="space directory, e.g. spaces/sample-room")
    args = parser.parse_args(argv)

    shapes_path = Path(args.space) / "shapes.json"
    if not shapes_path.is_file():
        print(f"error: {shapes_path} not found", file=sys.stderr)
        return 1

    with open(shapes_path, encoding="utf-8") as f:
        data = json.load(f)

    planes = data.get("planes", [])
    boxes = data.get("boxes", [])

    plane_labels, plane_reasons, ctx = classify_planes(planes)
    box_results = [classify_box(b, ctx) for b in boxes]

    for plane, label in zip(planes, plane_labels):
        plane["label"] = label
    for box, (label, _, _) in zip(boxes, box_results):
        box["label"] = label

    with open(shapes_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1)
        f.write("\n")

    report(args.space, planes, boxes, plane_labels, plane_reasons, box_results, ctx)
    print(f"wrote labels to {shapes_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
