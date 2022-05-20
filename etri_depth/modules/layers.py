import numpy as np
import torch


def depth_to_logdepth(depth):
    return torch.log(depth + 1) / np.log(100)


def depth_to_logdepth_np(depth):
    return np.log(depth + 1) / np.log(100)


def logdepth_to_depth(logdepth):
    return torch.exp(logdepth * np.log(100)) - 1


def logdepth_to_depth_np(logdepth):
    return np.exp(logdepth * np.log(100)) - 1


def normalize_image_torch(x):
    """Rescale image pixels to span range [0, 1]"""
    # ma = float(x.max())
    # mi = float(x.min())
    ma = torch.kthvalue(x.view(-1), int(x.numel() * 0.9)).values.item()
    mi = torch.kthvalue(x.view(-1), int(x.numel() * 0.1)).values.item()
    x = torch.clamp(x, mi, ma)
    d = ma - mi if ma != mi else 1e5
    return (x - mi) / d


def normalize_image(x: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    is_numpy = isinstance(x, np.ndarray)
    if is_numpy:
        x = torch.from_numpy(x)

    x = normalize_image_torch(x)

    if is_numpy:
        x = x.numpy()

    return x
