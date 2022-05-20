import datetime
import json
import os
import time

import cv2
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from etri_depth.datasets import get_dataset
from etri_depth.models import get_model
from etri_depth.modules.layers import depth_to_logdepth, normalize_image
from etri_depth.scores import depth_scores
from etri_depth.utils import ddp_utils
from etri_depth.utils.vision import colorize_depth_map

torch.backends.cudnn.benchmark = True
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
np.set_printoptions(linewidth=100)


class BaseAgent:
    def __init__(self, options, predict_only):
        self.opt = options
        self.device = torch.device("cuda")

        ddp_utils.init_distributed_mode()

        # network setting
        net, loss_module = get_model(self.opt.net.model_name, self.opt, predict_only)
        self.net = net.to(self.device)
        self.loss_module = loss_module.to(self.device)

        """
        if ddp_utils.is_dist_avail_and_initialized():
            self.net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.net)
        """

        # DDP
        if ddp_utils.is_dist_avail_and_initialized():
            self.net = torch.nn.parallel.DistributedDataParallel(
                self.net,
                device_ids=[ddp_utils.get_local_rank()],
                # find_unused_parameters=True,
            )
            self.net_without_ddp = self.net.module
        else:
            self.net_without_ddp = self.net

        # Load Pretrained
        if self.opt.agent.load_weights_folder:
            self.load_snapshot()

        if predict_only:
            return

        # Dataset Setting
        local_batch_size = ddp_utils.get_local_batch_size(self.opt.optim.batch_size)
        self.train_samplers = []
        self.train_loaders = []
        for opt_dataset in self.opt.train_datasets:
            dataset = get_dataset(opt_dataset, is_train=True)
            if ddp_utils.is_dist_avail_and_initialized():
                sampler = torch.utils.data.distributed.DistributedSampler(dataset, drop_last=True)
            else:
                sampler = torch.utils.data.RandomSampler(dataset)

            self.train_samplers.append(sampler)

            self.train_loaders.append(
                DataLoader(
                    dataset=dataset,
                    batch_size=local_batch_size,
                    sampler=sampler,
                    num_workers=self.opt.agent.num_workers,
                    drop_last=True,
                    pin_memory=True,
                )
            )
        self.val_loaders = []
        for opt_dataset in self.opt.val_datasets:
            dataset = get_dataset(opt_dataset, is_train=False)
            if ddp_utils.is_dist_avail_and_initialized():
                sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False)
            else:
                sampler = torch.utils.data.SequentialSampler(dataset)

            self.val_loaders.append(
                DataLoader(
                    dataset=dataset,
                    batch_size=1,
                    num_workers=self.opt.agent.num_workers,
                    drop_last=False,
                )
            )

        self.epoch = 0

        # Optimizer
        self.optimizer = optim.AdamW(self.net.parameters(), lr=self.opt.optim.learning_rate)
        self.lr_scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, self.opt.optim.lr_decay_step, self.opt.optim.lr_decay
        )

        self.log_path = os.path.join(self.opt.agent.log_dir, self.opt.agent.model_name)
        print("Training model named:\n  ", self.opt.agent.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.agent.log_dir)

        # Writers
        self.train_writers = [
            SummaryWriter(os.path.join(self.log_path, f"train{x}"))
            for x in range(len(self.opt.train_datasets))
        ]
        self.val_writers = [
            SummaryWriter(os.path.join(self.log_path, f"val{x}"))
            for x in range(len(self.opt.val_datasets))
        ]

        # Save opts
        self.save_opts()

    def train(self):
        start_epoch = self.epoch + 1
        for self.epoch in range(start_epoch, self.opt.agent.num_epochs + 1):
            if ddp_utils.is_dist_avail_and_initialized():
                for x in self.train_samplers:
                    x.set_epoch(self.epoch)

            print(f"\nEpoch {self.epoch} - LR {self.optimizer.param_groups[0]['lr']}")
            epoch_start = time.time()
            for data_loader, tb_writer in zip(self.train_loaders, self.train_writers):
                self._train_epoch(data_loader, tb_writer)
            epoch_duration = time.time() - epoch_start
            print("Train Epoch Duration: ", str(datetime.timedelta(seconds=epoch_duration)))

            self.lr_scheduler.step()
            self.save_snapshot()

            for data_loader, tb_writer in zip(self.val_loaders, self.val_writers):
                self._validate_epoch(data_loader, tb_writer)

    def _train_epoch(self, data_loader, tb_writer, early_return_step=None):
        raise NotImplementedError

    @torch.no_grad()
    def _validate_epoch(self, data_loader, tb_writer):
        raise NotImplementedError

    @torch.no_grad()
    def compute_scores(self, inputs, outputs):

        used_metrics = [
            depth_scores.MAE(),
            depth_scores.RMSE(),
            depth_scores.InvMAE(),
            depth_scores.InvRMSE(),
            depth_scores.LogMAE(),
            depth_scores.LogRMSE(),
            depth_scores.ScaleInvariantLog(),
            depth_scores.AbsRel(),
            depth_scores.SquareRel(),
            depth_scores.A1(),
            depth_scores.A2(),
            depth_scores.A3(),
        ]

        depth_gt = inputs[("gt_depth",)]
        depth_pred = outputs["depth", 0, 0]

        mask = depth_gt > 0
        if mask.sum().item() > 0:
            # median scaling
            depth_pred = torch.stack(
                [
                    each_pred * each_gt[each_mask].median() / each_pred[each_mask].median()
                    if each_mask.sum() > 0
                    else each_pred
                    for each_pred, each_gt, each_mask in zip(
                        depth_pred, depth_gt, mask, strict=True
                    )
                ]
            )

        depth_errors = {}
        for m in used_metrics:
            depth_errors[m.name] = m(depth_pred, depth_gt)

        return depth_errors

    @ddp_utils.on_rank_0
    def save_opts(self):
        with open(os.path.join(self.log_path, "opt.json"), "w", encoding="utf-8") as f:
            json.dump(self.opt, f, indent=2)

    @ddp_utils.on_rank_0
    def save_snapshot(self):
        save_folder = os.path.join(self.log_path, "models", f"weights_{self.epoch}")
        os.makedirs(save_folder, exist_ok=True)

        for name, elem in self.net_without_ddp.snapshot_elements().items():
            save_path = os.path.join(save_folder, f"{name}.pt")
            torch.save(elem.state_dict(), save_path)
            print(f"Save at {save_path}")

    def load_snapshot(self):

        print(f"Loading model from folder {self.opt.agent.load_weights_folder}")

        for name, elem in self.net_without_ddp.snapshot_elements().items():
            path = os.path.join(self.opt.agent.load_weights_folder, f"{name}.pt")
            state_dict = torch.load(path, map_location="cpu")
            elem.load_state_dict(state_dict)
            print(f"Load from {path}")

    @ddp_utils.on_rank_0
    def log_tensorboard_images(self, writer, inputs, outputs, step):

        pred_depth = outputs["depth", 0, 0][0]
        pred_depth = torch.cat(
            [
                inputs["color", 0][0].detach().cpu(),
                colorize_depth_map(
                    normalize_image(pred_depth), return_type="torch", min_d=0.0, max_d=1.0
                ),
            ],
            dim=1,
        )
        writer.add_image(f"pred_depth/{step}", pred_depth, self.epoch)

        for key in outputs.keys():
            if "color_pred" in key:
                writer.add_image(
                    f"color_pred_{key[1]}_{key[2]}/{step}",
                    outputs[key][0],
                    self.epoch,
                )

            if key[0] in ("color_posenet_input",):
                writer.add_image(
                    f"{key[0]}_{key[1]}/{step}",
                    outputs[key][0],
                    self.epoch,
                )

            if "automask" in key:
                writer.add_image(
                    f"{key}/{step}",
                    outputs[key][0][None, ...],
                    self.epoch,
                )

            if key[0] == "depth":
                writer.add_image(
                    f"{key}/{step}",
                    colorize_depth_map(
                        normalize_image(outputs[key][0]), return_type="torch", min_d=0.0, max_d=1.0
                    ),
                    self.epoch,
                )

        for key in inputs.keys():
            if key[0] in ("color", "color_aug", "color_canvas", "color_posenet_input"):
                writer.add_image(
                    f"{key[0]}_{key[1]}/{step}",
                    inputs[key][0],
                    self.epoch,
                )
            if key[0] in ("gt_depth",):
                writer.add_image(
                    f"{key[0]}/{step}",
                    (1 - depth_to_logdepth(inputs[key][0])).clamp(0, 1),
                    self.epoch,
                )
