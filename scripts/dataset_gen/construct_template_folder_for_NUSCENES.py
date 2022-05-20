import shutil
from pathlib import Path

import numpy as np
from nuscenes.nuscenes import NuScenes

from etri_depth.utils.utils import write_intrinsic_json


def gen_set(nusc, dst_root: Path):
    for scene_idx, scene in enumerate(nusc.scene):

        print(f"\r{scene_idx+1}/{len(nusc.scene)}: {scene['name']}", end="", flush=True)

        first_sample = nusc.get("sample", scene["first_sample_token"])
        cam_names = [x for x in first_sample["data"].keys() if x.startswith("CAM")]

        for cam_name in cam_names:

            dst_seq_root = dst_root / f"{scene['name']}_{cam_name}"
            dst_seq_root.mkdir(parents=True, exist_ok=True)
            cam_data = nusc.get("sample_data", first_sample["data"][cam_name])

            # Save intrinsics
            cam_calib = nusc.get("calibrated_sensor", cam_data["calibrated_sensor_token"])
            write_intrinsic_json(
                dst_seq_root / f"{cam_name}.json",
                np.asarray(
                    cam_calib["camera_intrinsic"],
                ),
            )

            # Copy jpg
            cam_dst_dir = dst_seq_root / "image" / cam_name
            cam_dst_dir.mkdir(parents=True, exist_ok=True)

            f_no = 0
            while True:
                cam_src_path = nusc.get_sample_data_path(cam_data["token"])
                shutil.copy2(cam_src_path, cam_dst_dir / f"{f_no:010d}.jpg")

                if cam_data["next"] == "":
                    break
                cam_data = nusc.get("sample_data", cam_data["next"])
                f_no += 1
    print()


def main():
    nuscenes_root = Path("/mnt/U2_01/UnSync/OpenData/nuscenes")
    dst_root = Path("Data/NUSCENES")

    # nusc = NuScenes(version="v1.0-mini", dataroot=nuscenes_root / "v1.0-mini", verbose=True)
    nusc = NuScenes(
        version="v1.0-trainval", dataroot=nuscenes_root / "v1.0-trainval", verbose=True
    )

    # assert not DST_ROOT.exists()
    gen_set(nusc, dst_root)


if __name__ == "__main__":
    main()
