import json
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
import typer
from torchvision.models.optical_flow import raft_small
from torchvision.transforms.functional import normalize, to_tensor

from etri_depth.utils.vision import imread_uint8

app = typer.Typer()

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


class Workers:
    def __init__(self, workers_per_device: int, target_WH: tuple[int, int]):
        self.models = []
        self.devices = []
        self.target_WH = target_WH

        for i_device in range(torch.cuda.device_count()):
            for _ in range(workers_per_device):
                self.devices.append(torch.device(f"cuda:{i_device}"))
                self.models.append(raft_small(pretrained=True).to(self.devices[-1]).eval())

    @torch.no_grad()
    def __call__(self, pair):
        # Unpack arguments
        idx: int
        img_dir: Path
        flow_dir: Path
        basename_prev: str
        basename_next: str
        idx, img_dir, flow_dir, basename_prev, basename_next = pair

        model_idx = idx % len(self.models)
        model = self.models[model_idx]
        device = self.devices[model_idx]

        # Read input image
        img_path_prev = img_dir / f"{basename_prev}.jpg"
        img_path_next = img_dir / f"{basename_next}.jpg"
        img_prev = imread_uint8(img_path_prev)
        img_next = imread_uint8(img_path_next)
        img_prev = cv2.resize(img_prev, self.target_WH, interpolation=cv2.INTER_AREA)
        img_next = cv2.resize(img_next, self.target_WH, interpolation=cv2.INTER_AREA)

        # to_tensor
        img_prev = to_tensor(img_prev).unsqueeze(0).to(device)
        img_next = to_tensor(img_next).unsqueeze(0).to(device)

        img_prev = normalize(img_prev, mean=0.5, std=0.5)
        img_next = normalize(img_next, mean=0.5, std=0.5)

        # infer
        flow = model(img_prev, img_next)[-1]

        # Save as file
        save_path = flow_dir / f"{basename_prev}.npy"
        np.save(save_path, flow.squeeze().cpu().numpy())

        return save_path

    def __len__(self):
        return len(self.models)


@app.command()
def main_flow(data_root: Path, workers_per_device: int = 1):
    """
    Step1
    Extract flow, Save as npy.
    """
    if "KITTI" in str(data_root):
        target_WH = (608, 176)
    elif "NUSCENES" in str(data_root):
        target_WH = (640, 360)
    elif "DDAD" in str(data_root):
        target_WH = (640, 400)
    elif "WAYMO" in str(data_root):
        target_WH = (640, 424)
    elif "SeasonDepth" in str(data_root):
        target_WH = (640, 472)
    else:
        raise ValueError(data_root)

    workers = Workers(workers_per_device, target_WH)

    seq_roots = sorted(x for x in data_root.iterdir() if x.is_dir())

    pairs = []
    idx = 0
    for seq_root in seq_roots:

        selected_cam = sorted(x.name for x in (seq_root / "image").iterdir() if x.is_dir())[0]

        img_dir = seq_root / "image" / selected_cam
        img_paths = sorted(img_dir.glob("*.jpg"))
        basenames = [x.stem for x in img_paths]

        flow_dir = seq_root / "flow"
        flow_dir.mkdir(parents=True, exist_ok=True)  # exist not ok. prevent overwriting

        for basename_prev, basename_next in zip(basenames[:-1], basenames[1:]):
            pairs.append((idx, img_dir, flow_dir, basename_prev, basename_next))
            idx += 1

    with ThreadPoolExecutor(max_workers=len(workers)) as executor:
        for it, save_path in enumerate(executor.map(workers, pairs)):
            print(f"\r{it+1}/{len(pairs)}: {save_path}", end="", flush=True)
    print("")


def worker_extract_median(seq_root: Path):
    flow_dir = seq_root / "flow"
    flow_paths = sorted(flow_dir.glob("*.npy"))

    flow_medians = {}
    for flow_path in flow_paths:
        flow = np.load(flow_path)
        flow_medians[flow_path.stem] = np.median(np.abs(flow)).item()

    json_path = seq_root / "flow_median.json"
    with open(json_path, "w", encoding="utf-8") as fp:
        json.dump(flow_medians, fp, indent=4, sort_keys=True)

    return json_path


@app.command()
def main_extract_median(data_root: Path):
    """
    Step2
    Extract flow median. Save as json.
    """
    seq_roots = sorted(x for x in data_root.iterdir() if x.is_dir())
    with ProcessPoolExecutor() as executor:
        for it, json_path in enumerate(executor.map(worker_extract_median, seq_roots)):
            print(f"\r{it+1}/{len(seq_roots)}: {json_path}", end="", flush=True)

    print("")


if __name__ == "__main__":
    app()
