import torch

from etri_depth.utils import ddp_utils

from .base import BaseAgent


class DepthAgent(BaseAgent):
    def _train_epoch(self, data_loader, tb_writer, early_return_step=None):
        self.net.train()
        print("")

        self.loss_module.update_epoch_setting(self.epoch)

        metric_logger = ddp_utils.MetricLogger()

        header = f"Epoch {self.epoch} | Training "
        for step, inputs in enumerate(metric_logger.log_every(data_loader, 10, header)):

            # Load data
            for key, ipt in inputs.items():
                inputs[key] = ipt.to(self.device, non_blocking=True)
            inputs["dataset_cfg"] = data_loader.dataset.cfg

            inputs = data_loader.dataset.augment_on_gpu(inputs)

            # Forward Pass
            outputs = self.net(inputs)
            losses, outputs = self.loss_module(inputs, outputs)

            # Backpropagation
            self.optimizer.zero_grad()
            losses["total_loss"].backward()
            self.optimizer.step()

            # Print Summary
            for key, value in losses.items():
                metric_logger.meters[key].update(value.item(), n=1)

            # Logging
            if step > len(data_loader) - 8 or early_return_step:
                self.log_tensorboard_images(tb_writer, inputs, outputs, step)

            if early_return_step:
                if step >= early_return_step:
                    break

    @torch.no_grad()
    def _validate_epoch(self, data_loader, tb_writer):
        self.net.eval()

        metric_logger = ddp_utils.MetricLogger()

        print("")

        header = f"Epoch {self.epoch} | Validation "
        for step, inputs in enumerate(metric_logger.log_every(data_loader, 10, header)):
            # Load Data
            for key, ipt in inputs.items():
                inputs[key] = ipt.to(self.device)

            inputs = data_loader.dataset.augment_on_gpu(inputs)

            torch.cuda.synchronize()

            outputs = self.net(inputs)

            torch.cuda.synchronize()

            # Logging
            batch_size = len(inputs["color", 0])
            errors = self.compute_scores(inputs, outputs)
            for key, val in errors.items():
                metric_logger.meters[key].update(val, n=batch_size)

            if (step + 1) % 100 == 0:
                self.log_tensorboard_images(tb_writer, inputs, outputs, step)

            if step >= 2000:
                break

        metric_logger.synchronize_between_processes()

        print(f"Epoch {self.epoch} | Validation" f" | epoch errors: {metric_logger}")

        # Log errors to the tensorboard
        if ddp_utils.is_main_process():
            for l, v in metric_logger.meters.items():
                tb_writer.add_scalar(f"{l}", v.global_avg, self.epoch)
            tb_writer.flush()

    @torch.no_grad()
    def test_step(self, inputs_, augment_on_gpu_fn):
        self.net.eval()

        # Prepare data
        inputs = {}
        for key, ipt in inputs_.items():
            inputs[key] = ipt.detach().to(self.device)

        inputs = augment_on_gpu_fn(inputs)

        # Forward
        outputs = self.net(inputs)
        return outputs["depth", 0, 0].detach().cpu()
