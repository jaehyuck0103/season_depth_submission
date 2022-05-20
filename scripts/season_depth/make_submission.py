import argparse
import pprint
from pathlib import Path

import cv2
import torch
from box import Box
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import to_tensor

from etri_depth.agents import get_agent
from etri_depth.utils.vision import imread_uint8, imwrite_depth_png

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

TEST_SET_ROOT = Path("/mnt/U2_01/Sync/OpenData/SeasonDepth/ICRA2022_SeasonDepth_Test_RGB")


class SimpleDataset(Dataset):
    def __init__(self):
        super().__init__()

        self.filepaths = sorted(TEST_SET_ROOT.rglob("*.jpg"))

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, index):

        img_path = self.filepaths[index]
        img = imread_uint8(img_path)
        origH, origW, _ = img.shape

        img = cv2.resize(img, (512, 384), cv2.INTER_AREA)

        dst_sub_path = img_path.relative_to(TEST_SET_ROOT)
        dst_sub_path = dst_sub_path.with_suffix(".png")

        return {
            ("color_aug", 0): to_tensor(img),
            "dst_sub_path": str(dst_sub_path),
            "origH": origH,
            "origW": origW,
        }


@torch.no_grad()
def evaluate(cfg, output_dir: Path):

    agent = get_agent(cfg.agent.name, cfg, predict_only=True)

    dataset = SimpleDataset()
    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=8,
        drop_last=False,
    )

    for batch_idx, inputs in enumerate(dataloader):
        print(f"\r{batch_idx+1}/{len(dataloader)}", end="", flush=True)

        dst_sub_paths = inputs.pop("dst_sub_path")
        origHs = inputs.pop("origH")
        origWs = inputs.pop("origW")
        batch_depth_pred = agent.test_step(inputs, lambda x: x)

        for depth_pred, dst_sub_path, origH, origW in zip(
            batch_depth_pred.squeeze(1).numpy(), dst_sub_paths, origHs, origWs, strict=True
        ):
            assert 0 <= depth_pred.min() and depth_pred.max() <= 1.0
            depth_pred = depth_pred * 255
            depth_pred = depth_pred.clip(0.0, 255.0)
            depth_pred = cv2.resize(depth_pred, (origW.item(), origH.item()))

            dst_path = output_dir / dst_sub_path
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            imwrite_depth_png(dst_path, depth_pred)
    print()


def main():
    # ------------
    # Argparse
    # ------------
    parser = argparse.ArgumentParser()

    parser.add_argument("config_path", type=Path)
    parser.add_argument("load_weights_folder", type=Path)
    parser.add_argument("--output_root", type=Path, default="Logs/submissions")

    args = parser.parse_args()

    # --------------------------
    # Load and update settings
    # --------------------------
    cfg = Box.from_toml(filename=args.config_path)
    cfg.agent.log_dir = None
    cfg.agent.load_weights_folder = args.load_weights_folder

    # output directory setting
    output_dir = (
        args.output_root
        / args.config_path.stem
        / f"{args.load_weights_folder.parts[-3]}_{args.load_weights_folder.parts[-1]}"
    )

    # summarize settings
    pprint.pprint(cfg.to_dict(), width=88, sort_dicts=False)
    evaluate(cfg, output_dir)


if __name__ == "__main__":
    main()
