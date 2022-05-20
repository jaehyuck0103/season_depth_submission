import torch
from torch import nn


class MAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "err/mae"

    def forward(self, pred, target):
        val_pixels = target > 0
        num_pixels = val_pixels.sum().item()
        if num_pixels < 10:
            return -1

        abs_err = (pred[val_pixels] - target[val_pixels]).abs()
        return abs_err.mean().item()


class RMSE(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "err/rmse"

    def forward(self, pred, target):
        val_pixels = target > 0
        num_pixels = val_pixels.sum().item()
        if num_pixels < 10:
            return -1

        sq_err = (pred[val_pixels] - target[val_pixels]) ** 2
        return sq_err.mean().sqrt().item()


class InvMAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "err/inv_mae"

    def forward(self, pred, target):
        val_pixels = target > 0
        num_pixels = val_pixels.sum().item()
        if num_pixels < 10:
            return -1

        inv_err = (1.0 / pred[val_pixels] - 1.0 / target[val_pixels]).abs()
        return inv_err.mean().item()


class InvRMSE(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "err/inv_rmse"

    def forward(self, pred, target):
        val_pixels = target > 0
        num_pixels = val_pixels.sum().item()
        if num_pixels < 10:
            return -1

        inv_squared_err = (1.0 / pred[val_pixels] - 1.0 / target[val_pixels]) ** 2
        return inv_squared_err.mean().sqrt().item()


class LogMAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "err/log_mae"

    def forward(self, pred, target):
        val_pixels = target > 0
        num_pixels = val_pixels.sum().item()
        if num_pixels < 10:
            return -1

        log_err = (pred[val_pixels].log() - target[val_pixels].log()).abs()
        return log_err.mean().item()


class LogRMSE(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "err/log_rmse"

    def forward(self, pred, target):
        val_pixels = target > 0
        num_pixels = val_pixels.sum().item()
        if num_pixels < 10:
            return -1

        inv_squared_err = (pred[val_pixels].log() - target[val_pixels].log()) ** 2
        return inv_squared_err.mean().sqrt().item()


class ScaleInvariantLog(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "err/SIlog"

    def forward(self, pred, target):
        val_pixels = target > 0
        num_pixels = val_pixels.sum()
        if num_pixels < 10:
            return -1

        pred_valid = pred[val_pixels]
        target_valid = target[val_pixels]

        d_err_log_squared = (target_valid.log() - pred_valid.log()) ** 2
        log_sum = (target_valid.log() - pred_valid.log()).sum()

        return (d_err_log_squared.mean() - (log_sum / num_pixels) ** 2).sqrt().item()


class AbsRel(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "err/abs_rel"

    def forward(self, pred, target):
        val_pixels = target > 0
        num_pixels = val_pixels.sum().item()
        if num_pixels < 10:
            return -1

        abs_rel_err = torch.abs(pred[val_pixels] - target[val_pixels]) / target[val_pixels]
        return abs_rel_err.mean().item()


class SquareRel(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "err/sq_rel"

    def forward(self, pred, target):
        val_pixels = target > 0
        num_pixels = val_pixels.sum().item()
        if num_pixels < 10:
            return -1
        sq_rel_err = (pred[val_pixels] - target[val_pixels]) ** 2 / target[val_pixels] ** 2
        return sq_rel_err.mean().item()


class A1(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "err/a1"

    def forward(self, pred, target):
        val_pixels = target > 0
        num_pixels = val_pixels.sum().item()
        if num_pixels < 10:
            return -1

        thresh = torch.maximum(
            pred[val_pixels] / target[val_pixels], target[val_pixels] / pred[val_pixels]
        )
        return (thresh < 1.25).float().mean().item()


class A2(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "err/a2"

    def forward(self, pred, target):
        val_pixels = target > 0
        num_pixels = val_pixels.sum().item()
        if num_pixels < 10:
            return -1

        thresh = torch.maximum(
            pred[val_pixels] / target[val_pixels], target[val_pixels] / pred[val_pixels]
        )
        return (thresh < 1.25**2).float().mean().item()


class A3(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "err/a3"

    def forward(self, pred, target):
        val_pixels = target > 0
        num_pixels = val_pixels.sum().item()
        if num_pixels < 10:
            return -1

        thresh = torch.maximum(
            pred[val_pixels] / target[val_pixels], target[val_pixels] / pred[val_pixels]
        )
        return (thresh < 1.25**3).float().mean().item()
