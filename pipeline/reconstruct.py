#!/usr/bin/env python3
"""OasisSpaces reconstruction pipeline.

Turns a folder of photos (or a walkthrough video) of a space into a point
cloud ready for the editor:

    images/video -> COLMAP SfM (features, matching, mapping) -> sparse cloud
                 -> optional CUDA dense reconstruction -> cleanup -> cloud.ply

Usage:
    python3 pipeline/reconstruct.py <images_dir | video_file> --name kitchen
    python3 pipeline/reconstruct.py walkthrough.mp4 --name loft --fps 2
    python3 pipeline/reconstruct.py photos/ --name office --dense

Output lands in spaces/<name>/:
    workspace/   COLMAP database, extracted frames, sparse model
    cloud.ply    cleaned point cloud (open it in the editor)
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from pointcloud import (
    load_ply, remove_outliers, save_ply, trim_far_points, voxel_downsample,
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic", ".tif", ".tiff", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def run(command: list[str], log_file: Path) -> None:
    print(f"  $ {' '.join(command[:4])} ...")
    with open(log_file, "a") as log:
        log.write(f"\n=== {' '.join(command)} ===\n")
        log.flush()
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        sys.exit(
            f"Step failed ({command[0]} {command[1] if len(command) > 1 else ''}), "
            f"see log: {log_file}"
        )


def require_binary(name: str, install_hint: str) -> None:
    if shutil.which(name) is None:
        sys.exit(f"{name} not found. Install it with: {install_hint}")


def extract_frames(video: Path, images_dir: Path, fps: float, log: Path) -> None:
    print(f"Extracting frames from {video.name} at {fps} fps")
    images_dir.mkdir(parents=True, exist_ok=True)
    run(
        ["ffmpeg", "-y", "-i", str(video), "-vf", f"fps={fps}",
         "-qscale:v", "2", str(images_dir / "frame_%05d.jpg")],
        log,
    )


def collect_images(source: Path, images_dir: Path) -> int:
    images_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for path in sorted(source.iterdir()):
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            target = images_dir / path.name
            if not target.exists():
                shutil.copy2(path, target)
            count += 1
    return count


def cuda_available() -> bool:
    try:
        output = subprocess.run(
            ["colmap", "-h"], capture_output=True, text=True
        ).stdout
    except OSError:
        return False
    return "with CUDA" in output


def gpu_flags(subcommand: str, new_prefix: str, old_prefix: str) -> list[str]:
    """Force CPU SIFT on headless non-CUDA Linux (e.g. a bare cloud VM).

    COLMAP's GPU SIFT needs CUDA or a display there, while macOS builds
    handle themselves — so only intervene where use_gpu=1 would crash.
    The option prefix also changed between COLMAP 3.x and 4.x, so detect
    which spelling this build understands instead of assuming a version.
    """
    if sys.platform == "darwin" or cuda_available():
        return []
    result = subprocess.run(
        ["colmap", subcommand, "--help"], capture_output=True, text=True
    )
    help_text = result.stdout + result.stderr
    for prefix in (new_prefix, old_prefix):
        if f"--{prefix}.use_gpu" in help_text:
            return [f"--{prefix}.use_gpu", "0"]
    return []


def sparse_reconstruction(
    workspace: Path, images_dir: Path, sequential: bool, log: Path
) -> Path:
    database = workspace / "database.db"
    sparse_dir = workspace / "sparse"
    sparse_dir.mkdir(parents=True, exist_ok=True)

    print("COLMAP: extracting features")
    run(
        ["colmap", "feature_extractor",
         "--database_path", str(database),
         "--image_path", str(images_dir),
         "--ImageReader.camera_model", "OPENCV",
         "--ImageReader.single_camera", "1",
         *gpu_flags("feature_extractor", "FeatureExtraction", "SiftExtraction")],
        log,
    )

    matcher = "sequential_matcher" if sequential else "exhaustive_matcher"
    print(f"COLMAP: matching features ({matcher})")
    run(
        ["colmap", matcher,
         "--database_path", str(database),
         *gpu_flags(matcher, "FeatureMatching", "SiftMatching")],
        log,
    )

    print("COLMAP: mapping (sparse reconstruction) — this is the slow part")
    run(
        ["colmap", "mapper",
         "--database_path", str(database),
         "--image_path", str(images_dir),
         "--output_path", str(sparse_dir)],
        log,
    )

    models = sorted(d for d in sparse_dir.iterdir() if d.is_dir())
    if not models:
        sys.exit(
            "COLMAP could not register the images into a model. "
            "Capture more overlapping photos (60-80% overlap between shots) "
            f"and retry. Log: {log}"
        )

    # The mapper may fragment a difficult capture into several models;
    # export each and keep the one with the most points.
    best_ply, best_model, best_points = None, None, -1
    for model in models:
        model_ply = workspace / f"model_{model.name}.ply"
        run(
            ["colmap", "model_converter",
             "--input_path", str(model),
             "--output_path", str(model_ply),
             "--output_type", "PLY"],
            log,
        )
        n = len(load_ply(model_ply))
        print(f"  model {model.name}: {n:,} points")
        if n > best_points:
            best_ply, best_model, best_points = model_ply, model, n
    if len(models) > 1:
        print(
            f"Note: reconstruction fragmented into {len(models)} pieces "
            f"(weak matching); using the largest. More overlap or more "
            f"texture in the capture will help."
        )
    return best_ply, best_model


def dense_reconstruction(
    workspace: Path, images_dir: Path, model_dir: Path, log: Path
) -> Path:
    dense_dir = workspace / "dense"
    dense_dir.mkdir(parents=True, exist_ok=True)
    print("COLMAP: undistorting images")
    run(
        ["colmap", "image_undistorter",
         "--image_path", str(images_dir),
         "--input_path", str(model_dir),
         "--output_path", str(dense_dir)],
        log,
    )
    print("COLMAP: dense stereo (CUDA)")
    run(
        ["colmap", "patch_match_stereo", "--workspace_path", str(dense_dir)],
        log,
    )
    print("COLMAP: fusing depth maps")
    fused = dense_dir / "fused.ply"
    run(
        ["colmap", "stereo_fusion",
         "--workspace_path", str(dense_dir),
         "--output_path", str(fused)],
        log,
    )
    return fused


def cleanup(raw_ply: Path, output: Path, voxel_size: float | None) -> None:
    cloud = load_ply(raw_ply)
    print(f"Cleanup: {len(cloud):,} raw points")
    if len(cloud) == 0:
        sys.exit("Reconstruction produced an empty cloud — nothing to save.")
    if voxel_size is not None and voxel_size <= 0:
        sys.exit("--voxel must be positive.")
    # Always drop far-field triangulation junk: a handful of wildly distant
    # points otherwise wreck camera fitting in every viewer.
    cloud = trim_far_points(cloud)
    if len(cloud) < 20_000 and voxel_size is None:
        # A small sparse cloud has no density to spare — keep every point.
        save_ply(cloud, output)
        print(f"Cloud is sparse; keeping all {len(cloud):,} points -> {output}")
        return
    if voxel_size is None:
        # COLMAP's scene scale is arbitrary; size voxels off the extent.
        extent = cloud.points.max(axis=0) - cloud.points.min(axis=0)
        voxel_size = float(np.linalg.norm(extent)) / 800
    if voxel_size > 0:
        cloud = remove_outliers(cloud, voxel_size=voxel_size * 4, min_neighbors=3)
        cloud = voxel_downsample(cloud, voxel_size=voxel_size)
    save_ply(cloud, output)
    print(f"Cleanup: {len(cloud):,} points kept -> {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="folder of photos, or a video file")
    parser.add_argument("--name", required=True, help="space name (output folder)")
    parser.add_argument("--fps", type=float, default=2.0,
                        help="frames per second to extract from video (default 2)")
    parser.add_argument("--dense", action="store_true",
                        help="attempt CUDA dense reconstruction after SfM")
    parser.add_argument("--voxel", type=float, default=None,
                        help="cleanup voxel size in scene units "
                             "(default: scene extent / 800)")
    args = parser.parse_args()

    require_binary("colmap", "brew install colmap")
    source = Path(args.source).expanduser()
    if not source.exists():
        sys.exit(f"Source not found: {source}")

    root = Path(__file__).resolve().parent.parent
    space_dir = root / "spaces" / args.name
    workspace = space_dir / "workspace"
    images_dir = workspace / "images"
    workspace.mkdir(parents=True, exist_ok=True)
    log = workspace / "colmap.log"

    # A rerun must not mix with stale state: features/models/frames from a
    # previous attempt (or a different --fps / source) would corrupt the
    # solve. Wipe everything derived, including old frames.
    for stale in [workspace / "database.db", workspace / "sparse",
                  workspace / "dense", images_dir,
                  *workspace.glob("model_*.ply"), workspace / "sparse.ply"]:
        if stale.is_dir():
            shutil.rmtree(stale)
        elif stale.exists():
            stale.unlink()

    is_video = source.is_file() and source.suffix.lower() in VIDEO_EXTENSIONS
    if is_video:
        require_binary("ffmpeg", "brew install ffmpeg")
        extract_frames(source, images_dir, args.fps, log)
    elif source.is_dir():
        count = collect_images(source, images_dir)
        if count < 10:
            sys.exit(
                f"Only {count} images found in {source} — a space needs at "
                "least a few dozen overlapping photos from every angle."
            )
        print(f"Using {count} images from {source}")
    else:
        sys.exit(f"{source} is neither an image folder nor a video file")

    sparse_ply, best_model = sparse_reconstruction(
        workspace, images_dir, sequential=is_video, log=log
    )

    result_ply = sparse_ply
    if args.dense:
        if cuda_available():
            result_ply = dense_reconstruction(
                workspace, images_dir, best_model, log
            )
        else:
            print(
                "Skipping dense reconstruction: this COLMAP build has no CUDA "
                "(normal on macOS). The sparse cloud is still produced; for a "
                "dense cloud run the dense step on a CUDA machine."
            )

    cleanup(result_ply, space_dir / "cloud.ply", args.voxel)
    print(
        f"\nDone. View it: serve the project root (python3 -m http.server 8734)"
        f"\nthen open http://localhost:8734/editor/?load=/spaces/{args.name}/cloud.ply"
    )


if __name__ == "__main__":
    main()
