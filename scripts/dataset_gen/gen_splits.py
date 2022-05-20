import json
import sys
from pathlib import Path

import typer

app = typer.Typer()


def _write_files(
    seq_root: Path, min_flow: float, max_flow: float, selected_keys: list[str], cam_name: str
):

    if min_flow > 0:
        with open(seq_root / "flow_median.json", encoding="utf-8") as fp:
            flow_medians = json.load(fp)
    else:
        flow_medians = {}

    image_dir = seq_root / "image" / cam_name
    basenames = sorted([x.stem for x in image_dir.glob("*.jpg")])

    files_list = []
    for bname in basenames:
        files = {}
        for key in [
            "gt_depth",
        ]:
            if key in selected_keys:
                files[key] = seq_root / key / cam_name / f"{bname}.png"

        for key in [
            "image",
        ]:
            if key in selected_keys:
                files[key] = seq_root / key / cam_name / f"{bname}.jpg"

        if "meta" in selected_keys:
            files["meta"] = seq_root / f"{cam_name}.json"

        for key, val in files.items():
            assert val.is_file(), val
            files[key] = str(val)

        if min_flow > 0:

            # Scan prev 3 frames
            accum = 0
            prev_images = []
            for offset in range(-1, -4, -1):
                flow = flow_medians.get(f"{int(bname)+offset:010d}")
                if flow is None:
                    break

                accum += flow
                if accum > min_flow:
                    prev_images.append(
                        str(seq_root / "image" / cam_name / f"{int(bname)+offset:010d}.jpg")
                    )
                if accum > max_flow:
                    break
            if len(prev_images) == 0:
                continue

            # Scan next 3 frames
            accum = 0
            next_images = []
            for offset in range(1, 4):
                flow = flow_medians.get(f"{int(bname)+offset-1:010d}")
                if flow is None:
                    break

                accum += flow
                if accum > min_flow:
                    next_images.append(
                        str(seq_root / "image" / cam_name / f"{int(bname)+offset:010d}.jpg")
                    )
                if accum > max_flow:
                    break
            if len(next_images) == 0:
                continue

            # Add adjacent frames to files
            files["prev_images"] = prev_images
            files["next_images"] = next_images

            files_list.append(files)
        else:
            files_list.append(files)

    return files_list


def write_files(
    data_root: Path,
    min_flow: float,
    max_flow: float,
    selected_keys: list[str],
    out_json_fname: str,
):
    seq_roots = sorted(x for x in data_root.iterdir() if x.is_dir())

    files_list = []
    for it, seq_root in enumerate(seq_roots):

        sys.stdout.write(f"\r{it+1}/{len(seq_roots)}: {seq_root.name}")
        sys.stdout.flush()

        cam_list = [x.name for x in (seq_root / "image").iterdir() if x.is_dir()]
        for cam_name in cam_list:
            files_list += _write_files(seq_root, min_flow, max_flow, selected_keys, cam_name)
    print("")

    json_path = SPLITS_ROOT / out_json_fname
    with open(json_path, "w", encoding="utf-8") as fp:
        json.dump(files_list, fp, indent=4, sort_keys=True)
    print("Writing Done", json_path)


@app.command()
def write_kitti_depth_train_dynamic():
    data_root = Path("Data") / "KITTI_DEPTH_train"
    write_files(
        data_root,
        min_flow=2.5,
        max_flow=2.5,
        selected_keys=["image", "meta"],
        out_json_fname="kitti_depth_train_dynamic.json",
    )


@app.command()
def write_nuscenes_dynamic():
    data_root = Path("Data") / "NUSCENES"
    write_files(
        data_root,
        min_flow=1.5,
        max_flow=1.5,
        selected_keys=["image", "meta"],
        out_json_fname="nuscenes_dynamic.json",
    )


@app.command()
def write_ddad_dynamic():
    data_root = Path("Data") / "DDAD_DEPTH"
    write_files(
        data_root,
        min_flow=1.5,
        max_flow=1.5,
        selected_keys=["image", "meta"],
        out_json_fname="ddad_depth_dynamic.json",
    )


@app.command()
def write_waymo_train_dynamic():
    data_root = Path("Data") / "WAYMO_train"
    write_files(
        data_root,
        min_flow=1.5,
        max_flow=1.5,
        selected_keys=["image", "meta"],
        out_json_fname="waymo_train_dynamic.json",
    )


@app.command()
def write_season_depth_train_dynamic():
    data_root = Path("Data") / "SeasonDepth_train"
    write_files(
        data_root,
        min_flow=1.5,
        max_flow=1.5,
        selected_keys=["image", "meta"],
        out_json_fname="season_depth_train_dynamic.json",
    )


@app.command()
def write_season_depth_val_dynamic():
    data_root = Path("Data") / "SeasonDepth_val"
    write_files(
        data_root,
        min_flow=1.5,
        max_flow=1.5,
        selected_keys=["image", "meta"],
        out_json_fname="season_depth_val_dynamic.json",
    )


@app.command()
def write_season_depth_val():
    data_root = Path("Data") / "SeasonDepth_val"
    write_files(
        data_root,
        min_flow=0.0,
        max_flow=0.0,
        selected_keys=["image", "gt_depth", "meta"],
        out_json_fname="season_depth_val.json",
    )


if __name__ == "__main__":
    SPLITS_ROOT = Path("Data") / "splits"
    SPLITS_ROOT.mkdir(exist_ok=True)

    app()
