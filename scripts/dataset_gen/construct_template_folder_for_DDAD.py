import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from etri_depth.utils.utils import write_intrinsic_json

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


def worker(arguments):
    # unpack arguments
    src_seq_dir: Path
    dst_seq_dir: Path
    img_basename: str
    index: int
    src_seq_dir, dst_seq_dir, img_basename, index = arguments

    # PNG -> JPG
    for cam_no in CAMS:
        src_path = src_seq_dir / "rgb" / f"CAMERA_{cam_no}" / f"{img_basename}.png"
        dst_path = dst_seq_dir / "image" / f"cam{cam_no}" / f"{index:010d}.jpg"
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dst_path), cv2.imread(str(src_path)))


def gen_train_set():

    # Prepare arguments
    seq_names = sorted([x.name for x in DDAD_ROOT.iterdir() if x.is_dir()])
    arguments = []
    for seq_name in seq_names:

        seq_dir = DDAD_ROOT / seq_name

        cam_basenames = sorted(
            [x.stem for x in (seq_dir / "rgb" / f"CAMERA_{CAMS[0]}").glob("*.png")]
        )

        # Validation check
        assert len(cam_basenames) > 10
        for c in CAMS[1:]:
            assert cam_basenames == sorted(
                [x.stem for x in (seq_dir / "rgb" / f"CAMERA_{c}").glob("*.png")]
            )

        # Append arguments
        for index, cam_basename in enumerate(cam_basenames):
            arguments.append([seq_dir, DST_ROOT / seq_name, cam_basename, index])

    # Run parallel
    with ProcessPoolExecutor() as executor:
        for it, _ in enumerate(executor.map(worker, arguments)):
            print(f"\r{it+1}/{len(arguments)}", end="", flush=True)
    print("")


def gen_calibration():
    seq_names = sorted([x.name for x in DDAD_ROOT.iterdir() if x.is_dir()])
    for seq_name in seq_names:
        seq_dir = DDAD_ROOT / seq_name
        calib_file = sorted((seq_dir / "calibration").glob("*.json"))
        assert len(calib_file) == 1
        calib_file = calib_file[0]

        with open(calib_file, encoding="utf-8") as fp:
            calib_data = json.load(fp)

        for cam_no in CAMS:
            name = f"CAMERA_{cam_no}"

            intrinsics = calib_data["intrinsics"][calib_data["names"].index(name)]

            K_cam = np.array(
                [
                    [intrinsics["fx"], 0.0, intrinsics["cx"]],
                    [0.0, intrinsics["fy"], intrinsics["cy"]],
                    [0.0, 0.0, 1.0],
                ]
            )

            # Save intrinsics
            json_path = DST_ROOT / seq_name / f"cam{cam_no}.json"
            json_path.parent.mkdir(parents=True, exist_ok=True)
            write_intrinsic_json(json_path, K_cam)


if __name__ == "__main__":
    DDAD_ROOT = Path("/mnt/U2_01/Sync/OpenData/packnet_data/ddad_train_val")
    DST_ROOT = Path("Data/DDAD_DEPTH")
    CAMS = ("01",)  # , "05", "06")
    gen_train_set()
    gen_calibration()
