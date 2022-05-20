import json
from pathlib import Path

import numpy as np


def readlines(filename):
    """Read all the lines in a text file and return as a list"""
    with open(filename, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    return lines


class StaticPrinter:
    def __init__(self):
        self.num_lines = 0

    def print(self, *line):
        print(*line)
        self.num_lines += 1

    def reset(self):
        for _ in range(self.num_lines):
            print("\033[F", end="")  # Cursor up one line
            print("\033[K", end="")  # Clear to the end of line
        self.num_lines = 0


def write_intrinsic_json(json_path: Path, K: np.ndarray):

    assert K.shape == (3, 3)

    with open(json_path, "w", encoding="utf-8") as fp:
        json.dump({"K": K.tolist()}, fp, indent=4, sort_keys=True)
