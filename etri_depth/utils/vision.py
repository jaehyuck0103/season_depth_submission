from pathlib import Path
from typing import Union, cast

import cv2
import numpy as np
from torch import Tensor
from torchvision.transforms.functional import to_tensor


def imread_float(path: Union[str, Path]) -> np.ndarray:
    path = str(path)
    img = cv2.imread(path).astype(np.float32) / 255
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def imread_gray_float(path: Union[str, Path]) -> np.ndarray:
    path = str(path)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
    return img


def imread_uint8(path: Union[str, Path]) -> np.ndarray:
    path = str(path)
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def imread_depth_png(path: Union[str, Path]) -> np.ndarray:
    path = str(path)
    return cv2.imread(path, cv2.IMREAD_ANYDEPTH).astype(np.float32) / 256.0


def imwrite_depth_png(path: Union[str, Path], depth_map_as_meters: np.ndarray):
    path = str(path)
    cv2.imwrite(path, (depth_map_as_meters * 256).astype(np.uint16))


def colorize_depth_map(
    depth_map: Union[Tensor, np.ndarray],
    return_type: str,
    min_d: float = 5,
    max_d: float = 120,
) -> Union[Tensor, np.ndarray]:
    """
    depth_map (meters): Tensor ([1,H,W] or [H,W]), np.ndarray ([H, W, 1] or [H,W])
    return_type: "torch" or "numpy"
    min_d: min distance
    max_d: max distance

    return: Tensor ([3, H, W], RGB float), nd.ndarray ([H, W, 3], BRG uint8)
    """

    # Tensor -> np.ndarray
    if isinstance(depth_map, Tensor):
        depth_map = depth_map.detach().cpu().numpy()
        depth_map = cast(np.ndarray, depth_map)

    # color mapping (https://ai.googleblog.com/2019/08/turbo-improved-rainbow-colormap-for.html)
    depth_map = depth_map.squeeze().clip(min_d, max_d)
    depth_map = ((depth_map - min_d) / (max_d - min_d) * 255).astype(np.uint8)
    color_map = cv2.applyColorMap(255 - depth_map, cv2.COLORMAP_TURBO)  # cv2.COLORMAP_JET

    if return_type == "numpy":
        return color_map
    elif return_type == "torch":
        color_map = cv2.cvtColor(color_map, cv2.COLOR_BGR2RGB)
        return to_tensor(color_map)
    else:
        raise ValueError(return_type)
