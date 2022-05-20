import base64
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from google.protobuf.json_format import MessageToDict
from waymo_open_dataset import dataset_pb2

from etri_depth.utils.utils import write_intrinsic_json

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


def get_calib_info(frame_dict):
    calib_info = {}
    for calib in frame_dict["context"]["camera_calibrations"]:
        cam_name = calib["name"]
        imgW = calib["width"]
        imgH = calib["height"]

        # Intrinsics
        intrinsic = calib["intrinsic"]
        origK = np.array(
            [
                [intrinsic[0], 0.0, intrinsic[2]],
                [0.0, intrinsic[1], intrinsic[3]],
                [0.0, 0.0, 1.0],
            ]
        )
        distort_param = np.array(intrinsic[4:])

        newK, roi = cv2.getOptimalNewCameraMatrix(origK, distort_param, (imgW, imgH), 0)

        mapx, mapy = cv2.initUndistortRectifyMap(
            origK, distort_param, None, newK, (imgW, imgH), cv2.CV_32FC1
        )

        calib_info[cam_name] = {
            "imgW": imgW,
            "imgH": imgH,
            "newK": newK,
            "roi": roi,
            "mapx": mapx,
            "mapy": mapy,
        }

    return calib_info


def worker(segment_name):
    tfrecord_file = INPUT_DIR / f"{segment_name}.tfrecord"
    out_segment_dir: Path = OUTPUT_DIR / segment_name
    out_segment_dir.mkdir(parents=True, exist_ok=True)

    dataset = tf.data.TFRecordDataset(tfrecord_file, compression_type="")
    for f_idx, data in enumerate(dataset):
        frame = dataset_pb2.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        frame_dict = MessageToDict(frame, preserving_proto_field_name=True)

        # Get calib
        calib_info = get_calib_info(frame_dict)
        if f_idx == 0:
            for img_dict in frame_dict["images"]:
                cam_name = img_dict["name"]
                if cam_name in CAM_EXCLUDE:
                    continue
                newK = calib_info[cam_name]["newK"]

                write_intrinsic_json(out_segment_dir / f"cam_{cam_name}.json", newK)

        # Undistort and Save images
        for img_dict in frame_dict["images"]:
            cam_name = img_dict["name"]
            if cam_name in CAM_EXCLUDE:
                continue
            roi = calib_info[cam_name]["roi"]
            mapx = calib_info[cam_name]["mapx"]
            mapy = calib_info[cam_name]["mapy"]

            buf = np.frombuffer(
                base64.b64decode(img_dict["image"].encode("utf-8")), dtype=np.uint8
            )
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            assert img.shape[0] == calib_info[cam_name]["imgH"]
            assert img.shape[1] == calib_info[cam_name]["imgW"]

            img = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
            img = img[roi[1] : roi[1] + roi[3], roi[0] : roi[0] + roi[2]]

            jpeg_path = out_segment_dir / "image" / f"cam_{cam_name}" / f"{f_idx:010d}.jpg"
            jpeg_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(jpeg_path), img)

    return segment_name


def main():

    segment_names = sorted(x.stem for x in INPUT_DIR.glob("*.tfrecord"))

    with ThreadPoolExecutor() as executor:
        for it, _ in enumerate(executor.map(worker, segment_names)):
            print(f"\r{it+1}/{len(segment_names)}", end="", flush=True)


if __name__ == "__main__":

    INPUT_DIR = Path(
        "/mnt/U2_01/UnSync/OpenData/waymo_open_dataset_v_1_3_1_individual_files/training"
    )
    OUTPUT_DIR = Path("Data/WAYMO_train")
    CAM_EXCLUDE = ("SIDE_LEFT", "SIDE_RIGHT")
    tf.config.set_visible_devices([], device_type="GPU")  # Force to use cpu
    main()
