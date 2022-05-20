import random

import cv2
import numpy as np


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def _is_numpy_intrinsic(K):
    return isinstance(K, np.ndarray) and (K.shape == (3, 3))


def pad_img(img: np.ndarray, inK: np.ndarray, outWH: tuple[int, int]):
    assert _is_numpy_image(img), type(img)
    assert _is_numpy_intrinsic(inK), type(inK)

    inH = img.shape[0]
    inW = img.shape[1]

    outH = outWH[1]
    outW = outWH[0]

    assert inH <= outH and inW <= outW, f"{inH} {outH} {inW} {outW}"
    if inH == outH and inW == outW:
        return img, inK

    padU = (outH - inH) // 2
    padD = outH - inH - padU
    padL = (outW - inW) // 2
    padR = outW - inW - padL

    img = np.pad(img, ((padU, padD), (padL, padR), (0, 0)), mode="edge")

    outK = inK.copy()
    outK[0, 2] += padL
    outK[1, 2] += padU

    assert img.shape[0] == outH and img.shape[1] == outW

    return img, outK


class RandomCrop:
    def __init__(self, cropW, cropH, bottom_crop=False):
        self.cropW = cropW
        self.cropH = cropH
        self.bottom_crop = bottom_crop

    def __call__(self, inputs, base_intrinsics):
        output_intrinsics = np.copy(base_intrinsics)

        inH, inW, _ = inputs["color", 0].shape
        for _, val in inputs.items():
            assert inH == val.shape[0], f"{inH}, {val.shape[0]}"
            assert inW == val.shape[1], f"{inW}, {val.shape[1]}"
        assert inH >= self.cropH, f"{inH}, {self.cropH}"
        assert inW >= self.cropW, f"{inW}, {self.cropW}"

        if inH == self.cropH and inW == self.cropW:
            return inputs, output_intrinsics

        # Crop
        if self.bottom_crop:
            offsetY = inH - self.cropH
        else:
            offsetY = random.randint(0, inH - self.cropH)
        offsetX = random.randint(0, inW - self.cropW)

        for key in list(inputs.keys()):
            inputs[key] = inputs[key][
                offsetY : offsetY + self.cropH, offsetX : offsetX + self.cropW
            ]

        output_intrinsics[0, 2] -= offsetX
        output_intrinsics[1, 2] -= offsetY
        return inputs, output_intrinsics


def resize(inputs, base_intrinsics, outW: int, outH: int):
    output_intrinsics = np.copy(base_intrinsics)

    inH, inW, _ = inputs["color", 0].shape
    for _, val in inputs.items():
        assert inH == val.shape[0], f"{inH}, {val.shape[0]}"
        assert inW == val.shape[1], f"{inW}, {val.shape[1]}"

    if inH == outH and inW == outW:
        return inputs, output_intrinsics

    # Scaling
    for key in list(inputs.keys()):
        if "color" in key[0]:
            if outW * outH < inW * inH:
                intp = cv2.INTER_AREA
            else:
                intp = cv2.INTER_LINEAR
        else:
            intp = cv2.INTER_NEAREST
        inputs[key] = cv2.resize(inputs[key], (outW, outH), interpolation=intp)

    output_intrinsics[0] *= outW / inW
    output_intrinsics[1] *= outH / inH
    return inputs, output_intrinsics


def equalize_focal_length(inputs, K: np.ndarray, target_f: float):
    if abs(K[0, 0] - target_f) < 100:
        return inputs, K

    inH, inW, _ = inputs["color", 0].shape

    outW = int(round((target_f / K[0, 0]) * inW))
    outH = int(round((target_f / K[1, 1]) * inH))
    return resize(inputs, K, outW, outH)


def equalize_focal_length_with_minimum_size(
    inputs, K: np.ndarray, target_f: float, minW: int, minH: int
):

    """
    if abs(K[0, 0] - target_f) < 100:
        return inputs, K
    """

    inH, inW, _ = inputs["color", 0].shape

    outW = round((target_f / K[0, 0]) * inW)
    outH = round((target_f / K[1, 1]) * inH)

    if outW < minW or outH < minH:
        target_f = max(K[0, 0] / inW * minW, K[1, 1] / inH * minH)
        outW = round((target_f / K[0, 0]) * inW)
        outH = round((target_f / K[1, 1]) * inH)

    return resize(inputs, K, outW, outH)
