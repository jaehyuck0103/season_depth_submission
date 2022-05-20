import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from etri_depth.utils.vision import colorize_depth_map, imread_depth_png

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

TEST_SET_ROOT = Path("/mnt/U2_01/Sync/OpenData/SeasonDepth/ICRA2022_SeasonDepth_Test_RGB")


@torch.no_grad()
def evaluate(output_dir: Path):

    img_filepaths = sorted(TEST_SET_ROOT.rglob("*.jpg"))
    depth_filepaths = sorted(output_dir.rglob("*.png"))

    for img_path, depth_path in zip(img_filepaths, depth_filepaths, strict=True):
        assert img_path.stem == depth_path.stem

        img = cv2.imread(str(img_path))
        depth = imread_depth_png(depth_path)
        colorized_depth = colorize_depth_map(depth, "numpy", 0.0, 255.0)

        assert img.shape == colorized_depth.shape

        concat = np.concatenate([img, colorized_depth], axis=0)
        dst_path = (
            Path("Logs/submissions_vis")
            / depth_path.parts[-5]
            / depth_path.parts[-4]
            / f"{depth_path.parts[-3]}_{depth_path.parts[-2]}_{depth_path.parts[-1]}"
        )
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dst_path), concat)


def main():
    # ------------
    # Argparse
    # ------------
    parser = argparse.ArgumentParser()

    parser.add_argument("output_root", type=Path)

    args = parser.parse_args()

    evaluate(args.output_root)


if __name__ == "__main__":
    main()
