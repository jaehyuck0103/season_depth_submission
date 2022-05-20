import json
from pathlib import Path


def expand(json_path: Path):

    with open(json_path, encoding="utf-8") as fp:
        json_data = json.load(fp)

    repeated_data = []
    for _ in range(10):
        repeated_data += json_data

    dst_path = json_path.parent / f"{json_path.stem}_repeat.json"
    with open(dst_path, "w", encoding="utf-8") as fp:
        json.dump(repeated_data, fp, indent=4, sort_keys=True)


if __name__ == "__main__":
    expand(Path("Data/splits/season_depth_train_dynamic.json"))
    expand(Path("Data/splits/season_depth_val_dynamic.json"))
