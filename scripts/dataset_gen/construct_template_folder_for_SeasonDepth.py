import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from shutil import copy2

import cv2
import numpy as np

from etri_depth.utils.utils import write_intrinsic_json

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


def get_calib_infos(file_path):

    with open(file_path, "r", encoding="utf-8") as f:
        lines = [x.rstrip() for x in f if x.startswith("OPENCV", 3)]

    calib_infos = {}
    for line in lines:
        cam_name = line.split()[0]
        _, _, fx, fy, cx, cy, _, _, _, _ = (float(x) for x in line.split()[2:])

        K = np.array(
            [
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0],
            ]
        )

        calib_infos[cam_name] = {
            "K": K,
        }

    return calib_infos


def worker_only_copy(arguments):
    # unpack arguments
    dst_seq_dir: Path
    cam_name: str
    frame_idx: int
    src_img_path: Path
    src_depth_gt_path: Path
    dst_seq_dir, cam_name, frame_idx, src_img_path, src_depth_gt_path = arguments

    # save img
    dst_img_path = dst_seq_dir / "image" / f"cam_{cam_name}" / f"{frame_idx:010d}.jpg"
    dst_img_path.parent.mkdir(parents=True, exist_ok=True)
    copy2(src_img_path, dst_img_path)

    # copy
    dst_depth_gt_path = dst_seq_dir / "gt_depth" / f"cam_{cam_name}" / f"{frame_idx:010d}.png"
    dst_depth_gt_path.parent.mkdir(parents=True, exist_ok=True)
    copy2(src_depth_gt_path, dst_depth_gt_path)


def gen_train_set(src_root: Path, dst_root: Path):

    calib_infos = get_calib_infos(src_root / "intrinsics.txt")

    # Prepare arguments
    seq_roots = sorted(src_root.glob("*/*"))
    for x in seq_roots:
        assert x.is_dir() and x.name.startswith("env"), x

    # Exclude slice4 (As, it is not undistorted.)
    seq_roots = [x for x in seq_roots if x.parent.name != "slice4"]

    arguments = []
    for seq_root in seq_roots:
        cam_names = sorted(x.name for x in seq_root.iterdir())

        # Collect arguments
        for cam_name in cam_names:
            seq_name = f"{seq_root.parent.name}_{seq_root.name}_{cam_name}"
            img_paths = sorted((seq_root / cam_name / "images").glob("*.jpg"))
            depth_gt_paths = sorted((seq_root / cam_name / "depth_maps").glob("*.png"))
            assert len(img_paths) > 10
            for frame_idx, (img_path, depth_gt_path) in enumerate(
                zip(img_paths, depth_gt_paths, strict=True)
            ):
                assert img_path.stem == depth_gt_path.stem
                arguments.append(
                    (
                        dst_root / seq_name,
                        cam_name,
                        frame_idx,
                        img_path,
                        depth_gt_path,
                    )
                )

            # Save intrinsics
            (dst_root / seq_name).mkdir(parents=True, exist_ok=True)
            write_intrinsic_json(
                dst_root / seq_name / f"cam_{cam_name}.json",
                calib_infos[cam_name]["K"],
            )

    # Run parallel
    with ProcessPoolExecutor() as executor:
        for it, _ in enumerate(executor.map(worker_only_copy, arguments)):
            sys.stdout.write(f"\r{it+1}/{len(arguments)}")
            sys.stdout.flush()
    print("")


def gen_test_set(src_root: Path, dst_root: Path):

    calib_infos = get_calib_infos(src_root / "intrinsics.txt")

    # Prepare arguments
    seq_roots = sorted((src_root / "images").iterdir())
    for x in seq_roots:
        assert x.is_dir() and x.name.startswith("env"), x

    arguments = []
    for seq_root in seq_roots:
        cam_names = sorted(x.name for x in seq_root.iterdir())

        # Collect arguments
        for cam_name in cam_names:
            seq_name = f"{seq_root.name}_{cam_name}"
            img_paths = sorted((seq_root / cam_name).glob("*.jpg"))
            depth_gt_paths = sorted((src_root / "depth" / seq_root.name / cam_name).glob("*.png"))
            assert len(img_paths) > 10
            for frame_idx, (img_path, depth_gt_path) in enumerate(
                zip(img_paths, depth_gt_paths, strict=True)
            ):
                assert img_path.stem == depth_gt_path.stem
                arguments.append(
                    (
                        dst_root / seq_name,
                        cam_name,
                        frame_idx,
                        img_path,
                        depth_gt_path,
                    )
                )

            # Save intrinsics
            (dst_root / seq_name).mkdir(parents=True, exist_ok=True)
            write_intrinsic_json(
                dst_root / seq_name / f"cam_{cam_name}.json",
                calib_infos[cam_name]["K"],
            )

    # Run parallel
    with ProcessPoolExecutor() as executor:
        for it, _ in enumerate(executor.map(worker_only_copy, arguments)):
            sys.stdout.write(f"\r{it+1}/{len(arguments)}")
            sys.stdout.flush()
    print("")


if __name__ == "__main__":
    gen_train_set(
        Path("/mnt/U2_01/Sync/OpenData/SeasonDepth/SeasonDepth_trainingset_v1.1"),
        Path("Data/SeasonDepth_train"),
    )
    gen_test_set(
        Path("/mnt/U2_01/Sync/OpenData/SeasonDepth/SeasonDepth_testset"),
        Path("Data/SeasonDepth_val"),
    )
