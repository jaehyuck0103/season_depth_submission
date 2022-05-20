import json
import random
from functools import partial

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ColorJitter, RandomApply
from torchvision.transforms.functional import to_tensor

from etri_depth.utils.vision import imread_depth_png, imread_uint8

from .transforms import (
    RandomCrop,
    equalize_focal_length_with_minimum_size,
    pad_img,
    resize,
)

# jitter = RandomApply([ColorJitter(0.4, 0.4, 0.4, 0.2)], p=0.8)
jitter = RandomApply([ColorJitter(0.6, 0.6, 0.6, 0.3)], p=0.8)


class DepthDataset(Dataset):
    def __init__(self, dataset_cfg, is_train: bool):
        super().__init__()

        self.cfg = dataset_cfg

        self.filepaths = []
        for json_path in self.cfg.splits:
            with open(json_path, encoding="utf-8") as fp:
                self.filepaths += json.load(fp)

        self.is_train = is_train

        # augments
        if self.cfg.scaling_method == "rand_crop":
            self.scaling = RandomCrop(self.cfg.cropW, self.cfg.cropH)
        elif self.cfg.scaling_method == "resize":
            self.scaling = partial(resize, outW=self.cfg.cropW, outH=self.cfg.cropH)
        else:
            raise ValueError(self.cfg.scaling_method)

    def augment_on_gpu(self, inputs):

        # Batchwise Augment color
        if self.cfg.color_aug:
            inputs["color_aug", 0] = torch.stack([jitter(xx) for xx in inputs["color", 0]])
        else:
            inputs["color_aug", 0] = inputs["color", 0].clone()

        return inputs

    def get_color(self, line, offset: int, do_flip: bool):
        if offset == 0:
            path = line["image"]
        elif offset == -1:
            path = random.choice(line["prev_images"])
        elif offset == 1:
            path = random.choice(line["next_images"])
        else:
            raise ValueError(offset)

        color = imread_uint8(path)

        if do_flip:
            color = cv2.flip(color, 1)  # LR flip

        return color

    def get_intrinsic(self, json_path: str, do_flip: bool, imgW: int):
        with open(json_path, encoding="utf-8") as fp:
            meta = json.load(fp)

        K = np.array(meta["K"], dtype=np.float32)

        if do_flip:
            K[0, 2] = imgW - K[0, 2]

        return K

    @staticmethod
    def get_depth(path: str, do_flip: bool):
        depth = imread_depth_png(path)

        if do_flip:
            depth = np.fliplr(depth)

        return depth

    def __len__(self):
        return len(self.filepaths) // self.cfg.epoch_reduce_rate

    def __getitem__(self, index):
        # Select index, Select wheter do_flip
        index += len(self) * random.randrange(self.cfg.epoch_reduce_rate)
        line = self.filepaths[index]
        do_flip = self.is_train and random.random() > 0.5

        inputs = {}

        # Read RGB
        for frame_offset in self.cfg.frame_offsets:
            inputs["color", frame_offset] = self.get_color(line, frame_offset, do_flip)
        origH, origW, _ = inputs["color", 0].shape

        # Read intrinsic
        if self.is_train:
            K = self.get_intrinsic(line["meta"], do_flip, origW)
        else:
            K = np.zeros([3, 3], dtype=np.float32)

        # Read gt
        if "gt_depth" in line:
            inputs[("gt_depth",)] = self.get_depth(line["gt_depth"], do_flip)
        else:
            inputs[("gt_depth",)] = np.zeros((origH, origW), dtype=np.float32)

        # Equalize Focal Length
        if self.is_train:
            inputs, K = equalize_focal_length_with_minimum_size(
                inputs, K, self.cfg.target_focal_length, self.cfg.cropW, self.cfg.cropH
            )

        # Canvas
        if self.is_train:
            rgb_canvas = {}
            for frame_offset in self.cfg.frame_offsets:
                rgb_canvas[frame_offset], K_canvas = pad_img(
                    inputs["color", frame_offset], K, (self.cfg.canvasW, self.cfg.canvasH)
                )

        # Scaling inputs
        inputs, K = self.scaling(inputs, K)

        # Copy rgb_canvas to inputs
        if self.is_train:
            for frame_offset in self.cfg.frame_offsets:
                inputs["color_canvas", frame_offset] = rgb_canvas[frame_offset]  # type: ignore

        # np -> tensor
        for k, v in inputs.items():
            if "color" in k or "color_canvas" in k:
                inputs[k] = to_tensor(v)

        target_keys = ["gt_depth"]
        for key in inputs.keys():
            if key[0] in target_keys:
                inputs[key] = np.expand_dims(inputs[key], 0).astype(np.float32)
                inputs[key] = torch.from_numpy(inputs[key])

        if self.is_train:
            inputs[("inv_K_crop",)] = torch.from_numpy(np.linalg.inv(K))
            inputs[("K_canvas",)] = torch.from_numpy(K_canvas)  # type: ignore

        return inputs
