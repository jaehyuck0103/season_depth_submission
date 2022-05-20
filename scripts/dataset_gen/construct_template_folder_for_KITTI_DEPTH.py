import sys
from pathlib import Path

import cv2
import numpy as np

from etri_depth.utils.kitti_tools import read_calib_file
from etri_depth.utils.utils import write_intrinsic_json

EXPLICIT_STATIC_SCENES = [
    "2011_09_28_drive_0016_sync",
    "2011_09_28_drive_0021_sync",
    "2011_09_28_drive_0034_sync",
    "2011_09_28_drive_0035_sync",
    "2011_09_28_drive_0039_sync",
    "2011_09_28_drive_0043_sync",
    "2011_09_28_drive_0053_sync",
    "2011_09_28_drive_0054_sync",
    "2011_09_28_drive_0057_sync",
    "2011_09_28_drive_0065_sync",
    "2011_09_28_drive_0066_sync",
    "2011_09_28_drive_0068_sync",
    "2011_09_28_drive_0070_sync",
    "2011_09_28_drive_0071_sync",
    "2011_09_28_drive_0075_sync",
    "2011_09_28_drive_0077_sync",
    "2011_09_28_drive_0078_sync",
    "2011_09_28_drive_0080_sync",
    "2011_09_28_drive_0082_sync",
    "2011_09_28_drive_0086_sync",
    "2011_09_28_drive_0087_sync",
    "2011_09_28_drive_0089_sync",
    "2011_09_28_drive_0090_sync",
    "2011_09_28_drive_0094_sync",
    "2011_09_28_drive_0095_sync",
    "2011_09_28_drive_0096_sync",
    "2011_09_28_drive_0098_sync",
    "2011_09_28_drive_0100_sync",
    "2011_09_28_drive_0102_sync",
    "2011_09_28_drive_0103_sync",
    "2011_09_28_drive_0104_sync",
    "2011_09_28_drive_0106_sync",
    "2011_09_28_drive_0108_sync",
    "2011_09_28_drive_0110_sync",
    "2011_09_28_drive_0113_sync",
    "2011_09_28_drive_0117_sync",
    "2011_09_28_drive_0119_sync",
    "2011_09_28_drive_0121_sync",
    "2011_09_28_drive_0122_sync",
    "2011_09_28_drive_0125_sync",
    "2011_09_28_drive_0126_sync",
    "2011_09_28_drive_0128_sync",
    "2011_09_28_drive_0132_sync",
    "2011_09_28_drive_0134_sync",
    "2011_09_28_drive_0135_sync",
    "2011_09_28_drive_0136_sync",
    "2011_09_28_drive_0138_sync",
    "2011_09_28_drive_0141_sync",
    "2011_09_28_drive_0143_sync",
    "2011_09_28_drive_0145_sync",
    "2011_09_28_drive_0146_sync",
    "2011_09_28_drive_0149_sync",
    "2011_09_28_drive_0153_sync",
    "2011_09_28_drive_0154_sync",
    "2011_09_28_drive_0155_sync",
    "2011_09_28_drive_0156_sync",
    "2011_09_28_drive_0160_sync",
    "2011_09_28_drive_0161_sync",
    "2011_09_28_drive_0162_sync",
    "2011_09_28_drive_0165_sync",
    "2011_09_28_drive_0166_sync",
    "2011_09_28_drive_0167_sync",
    "2011_09_28_drive_0168_sync",
    "2011_09_28_drive_0171_sync",
    "2011_09_28_drive_0174_sync",
    "2011_09_28_drive_0177_sync",
    "2011_09_28_drive_0179_sync",
    "2011_09_28_drive_0183_sync",
    "2011_09_28_drive_0184_sync",
    "2011_09_28_drive_0185_sync",
    "2011_09_28_drive_0186_sync",
    "2011_09_28_drive_0187_sync",
    "2011_09_28_drive_0191_sync",
    "2011_09_28_drive_0192_sync",
    "2011_09_28_drive_0195_sync",
    "2011_09_28_drive_0198_sync",
    "2011_09_28_drive_0199_sync",
    "2011_09_28_drive_0201_sync",
    "2011_09_28_drive_0204_sync",
    "2011_09_28_drive_0205_sync",
    "2011_09_28_drive_0208_sync",
    "2011_09_28_drive_0209_sync",
    "2011_09_28_drive_0214_sync",
    "2011_09_28_drive_0216_sync",
    "2011_09_28_drive_0220_sync",
    "2011_09_28_drive_0222_sync",
]


def _filter_static_seqs(seq_names):
    print("All Seqs", len(seq_names))
    filtered_seq_names = []
    for seq_name in seq_names:
        is_static = False
        for static_name in EXPLICIT_STATIC_SCENES:
            if static_name in seq_name:
                is_static = True
                break
        if is_static is False:
            filtered_seq_names.append(seq_name)
    print("Filtered Seqs", len(filtered_seq_names))
    return filtered_seq_names


def gen_train_set():
    SRC_TRAIN_ROOT = SRC_DEPTH_ROOT / "train"

    seq_names = sorted([x.name for x in SRC_TRAIN_ROOT.iterdir() if x.is_dir()])
    seq_names = _filter_static_seqs(seq_names)

    for idx, seq_name in enumerate(seq_names):
        sys.stdout.write(f"\rProcessing {idx+1} / {len(seq_names)} ")
        sys.stdout.flush()

        seq_dir = SRC_TRAIN_ROOT / seq_name
        seq_date = seq_name[:10]

        # Validation check
        cam02_filenames = sorted([x.stem for x in (seq_dir / "image_02" / "data").glob("*.png")])
        cam03_filenames = sorted([x.stem for x in (seq_dir / "image_03" / "data").glob("*.png")])
        assert cam02_filenames == cam03_filenames
        assert len(cam02_filenames) > 10

        # Convert image (png -> jpg)
        for fname in cam02_filenames:
            for cam_no in ("02", "03"):
                src_path = seq_dir / f"image_{cam_no}" / "data" / f"{fname}.png"
                dst_path = DST_ROOT / seq_name / "image" / f"cam{cam_no}" / f"{fname}.jpg"
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(dst_path), cv2.imread(str(src_path)))

        # Copy
        for cam_no in ["02", "03"]:
            # save intrinsics
            calib_path = SRC_RAW_ROOT / seq_date / "calib_cam_to_cam.txt"
            P_rect = read_calib_file(calib_path)[f"P_rect_{cam_no}"]

            json_path = DST_ROOT / seq_name / f"cam{cam_no}.json"
            json_path.parent.mkdir(exist_ok=True)
            write_intrinsic_json(json_path, np.array(P_rect).reshape(3, 4)[:, :3])
    print("")


if __name__ == "__main__":
    SRC_ROOT = Path("/mnt/U2_01/Sync/OpenData/KITTI")
    SRC_DEPTH_ROOT = SRC_ROOT / "Depth"
    SRC_RAW_ROOT = SRC_ROOT / "Raw"

    DST_ROOT = Path("Data/KITTI_DEPTH_train")
    assert not DST_ROOT.exists()
    gen_train_set()
