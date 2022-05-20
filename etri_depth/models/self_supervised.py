import torch
from torch import nn

from etri_depth.geometry.camera import (
    GridSampleWithoutExtrinsic,
    grid_sample_with_extrinsic,
)
from etri_depth.geometry.transform import transformation_from_parameters
from etri_depth.modules import get_module
from etri_depth.modules.losses import SSIM, get_smooth_loss
from etri_depth.modules.pose.pose_encoder import PoseEncoder


class NetBuilder(nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.depth_estimator = get_module(opt.net.depth_module_name, opt)
        self.pose_encoder = PoseEncoder(
            opt.net.posenet.transl_scale,
            opt.net.posenet.encoder_type,
            opt.net.posenet.use_additional_layers,
        )
        self.grid_sample_pose_input = GridSampleWithoutExtrinsic(
            opt.net.posenet_inH, opt.net.posenet_inW, opt.net.posenet_inK
        )

        self.opt = opt

    def snapshot_elements(self):
        return {
            "depth_estimator": self.depth_estimator,
            "pose_encoder": self.pose_encoder,
        }

    def forward(self, inputs):
        outputs = self.depth_estimator(inputs)

        if self.training:
            outputs.update(self.predict_poses(inputs))
            self.generate_images_pred(inputs, outputs)

        return outputs

    def predict_poses(self, inputs):

        outputs = {}
        for f_i in inputs["dataset_cfg"].frame_offsets:
            outputs["color_posenet_input", f_i] = self.grid_sample_pose_input(
                inputs["color_canvas", f_i], inputs[("K_canvas",)]
            )

        for f_i in inputs["dataset_cfg"].frame_offsets[1:]:
            angle_params, transl_params = self.pose_encoder(
                outputs["color_posenet_input", f_i], outputs["color_posenet_input", 0]
            )

            outputs[("angle_params", f_i)] = angle_params
            outputs[("transl_params", f_i)] = transl_params
            outputs[("cam_T_cam", f_i)] = transformation_from_parameters(
                angle_params, transl_params
            )  # (B, 4, 4)

        return outputs

    def generate_images_pred(self, inputs, outputs):

        for scale in range(self.depth_estimator.num_output_scales):

            depth = outputs[("depth", 0, scale)]

            for frame_id in inputs["dataset_cfg"].frame_offsets[1:]:

                outputs[
                    ("color_pred", f"{frame_id}to{0}", f"s{scale}")
                ] = grid_sample_with_extrinsic(
                    tgt_depth=depth,
                    tgt_invK=inputs[("inv_K_crop",)],
                    src_img=inputs["color_canvas", frame_id],
                    srcK=inputs[("K_canvas",)],
                    T=outputs[("cam_T_cam", frame_id)],
                )


class LossModule(nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.ssim = SSIM(opt.loss.photo.ssim.window_size)

        self.opt = opt

        assert len(opt.loss.scale_weights.schedule) + 1 == len(opt.loss.scale_weights.list)
        assert sorted(opt.loss.scale_weights.schedule) == opt.loss.scale_weights.schedule

        self.scale_weights = []
        self.use_automask = True

    def update_epoch_setting(self, epoch):

        self.scale_weights = self.opt.loss.scale_weights.list[-1]
        for i, sche in enumerate(self.opt.loss.scale_weights.schedule):
            if epoch < sche:
                self.scale_weights = self.opt.loss.scale_weights.list[i]
                break

        self.use_automask = epoch >= self.opt.loss.automask_begin_epoch

        print("-- Loss Settings --")
        print(f"w: {self.scale_weights}")
        print(f"use_automask: {self.use_automask}")
        print("")

    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images"""
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)
        ssim_loss = self.ssim(pred, target).mean(1, True)  # (4, 1, H, W)
        reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    def forward(self, inputs, outputs):

        losses = {}
        total_loss = 0

        target_color = inputs["color", 0]
        for scale, scale_weight in enumerate(self.scale_weights):

            pred_depth = outputs["depth", 0, scale]

            # -----------------
            # Photometric Loss
            # -----------------
            reprojection_losses = []
            for frame_id in inputs["dataset_cfg"].frame_offsets[1:]:
                pred = outputs[("color_pred", f"{frame_id}to{0}", f"s{scale}")]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target_color))
            reprojection_losses = torch.cat(reprojection_losses, 1)

            if self.use_automask:
                identity_reprojection_losses = []
                for frame_id in inputs["dataset_cfg"].frame_offsets[1:]:
                    pred = inputs["color", frame_id]
                    identity_reprojection_losses.append(
                        self.compute_reprojection_loss(pred, target_color)
                    )
                identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

                combined = torch.cat((identity_reprojection_losses, reprojection_losses), dim=1)

                to_optimise, idxs = torch.min(combined, dim=1)
                outputs[f"automask_{scale}"] = (
                    idxs > identity_reprojection_losses.shape[1] - 1
                ).float()

                photo_loss = self.opt.loss.w.photo * to_optimise.mean()
            else:
                photo_loss = self.opt.loss.w.photo * reprojection_losses.mean()

            # ----------------
            # Smooth Loss
            # ----------------
            mean_depth = pred_depth.mean([2, 3], True).detach()
            norm_depth = pred_depth / (mean_depth + 1e-7)
            norm_depth = torch.log(norm_depth)

            smooth_loss = get_smooth_loss(norm_depth, target_color)
            smooth_loss = self.opt.loss.w.smooth * smooth_loss

            losses[f"loss/{scale}/photo"] = scale_weight * photo_loss
            losses[f"loss/{scale}/smooth"] = scale_weight * smooth_loss
            losses[f"loss/{scale}"] = (
                losses[f"loss/{scale}/photo"] + losses[f"loss/{scale}/smooth"]
            )
            total_loss += losses[f"loss/{scale}"]
        losses["total_loss"] = total_loss
        return losses, outputs
