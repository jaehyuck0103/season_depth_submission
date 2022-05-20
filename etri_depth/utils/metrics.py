from typing import Dict, List

import numpy as np


class AverageMeter:
    def __init__(self):
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    @property
    def val(self):
        return self.avg


class AverageMeterVec:
    def __init__(self, size):
        self.size = size
        self.avg = np.zeros(size)
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    @property
    def val(self):
        return self.avg


class AverageMeterDic:
    def __init__(self, keys: List[str]):
        self.sum = {key: 0.0 for key in keys}
        self.count = 0

    def update(self, val: Dict[str, float], n: int):
        for key in self.sum.keys():
            self.sum[key] += val[key] * n
        self.count += n

    @property
    def val(self):
        return {k: v / self.count for k, v in self.sum.items()}
