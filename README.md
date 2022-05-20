# Training Monodepth2 with Many Dataset

This code trains Monodepth2(https://github.com/nianticlabs/monodepth2) with 5 open datasets.
- SeasonDepth (https://seasondepth.github.io)
- KITTI (http://www.cvlibs.net/datasets/kitti)
- NuScenes (https://www.nuscenes.org)
- Waymo Open (https://waymo.com/open)
- DDAD (https://github.com/TRI-ML/DDAD)

Each dataset has each characteristic and the size of the training dataset becomes very large.
We can expect better performance and generalization.

**This approach got the 3rd place prize at the self-supervised learning track of the SeasonDepth Predcition Challenge (ICRA2022).** (http://seasondepth-challenge.org/index/)

## Environment

Tested on
```
python==3.10.4
torch==1.11.0
torchvision==0.12.0
typer==0.4.0
python-box==5.4.1
opencv-python==4.5.5
tensorboard==2.8.0
timm==0.5.4
toml==0.10.2
```
All scripts run on root directory of this repo.
Append below line to the `~/.bashrc`.
```
export PYTHONPATH=./:${PYTHONPATH}
```

## Prepare Data
### Equalize Folder Structure
This codes equalize the folder structure of datasets, and save them to `./Data`.

``` bash
# Before running the code, change the path for SeasonDepth_trainingset_v1.1 and SeasonDepth_testset in the code.
python scripts/dataset_gen/construct_template_folder_for_SeasonDepth.py
```
``` bash
# Before running the code, change SRC_ROOT in the code.
# The SRC_ROOT should contain both KITTI_RAW and KITTI_DEPTH_BENCHMARK datasets.
python scripts/dataset_gen/construct_template_folder_for_KITTI_DEPTH.py
```
``` bash
# Before running the code, change the nuscens_root in the code.
pip install nuscenes-devkit  # This package is used only for this script.
python scripts/dataset_gen/construct_template_folder_for_NUSCENES.py
```
``` bash
# Before running the code, change the INPUT_DIR in the code.
# The pypi package for waymo-open-dataset does not support python3.10 yet.
# So, I use python3.9 (with venv) only for this script.
pip install waymo-open-dataset-tf-2-6-0
pip install opencv-contrib-python
CUDA_VISIBLE_DEVICES=-1 python scripts/dataset_gen/construct_template_folder_for_waymo.py
```
``` bash
# Before running the code, change the DDAD_ROOT in the code.
python scripts/dataset_gen/construct_template_folder_for_DDAD.py
```

### Extract Dynamic Frames

Self-supervised learning for depth estimation assumes that there is ego motion between input frames.
Simply, I use a median value of optical flow to determine the dynamic.
The result will be saved on `./Data/splits`.

``` bash
python scripts/dataset_gen/gen_flows.py main-flow ./Data/DDAD_DEPTH
python scripts/dataset_gen/gen_flows.py main-flow ./Data/KITTI_DEPTH_train
python scripts/dataset_gen/gen_flows.py main-flow ./Data/NUSCENES
python scripts/dataset_gen/gen_flows.py main-flow ./Data/SeasonDepth_train
python scripts/dataset_gen/gen_flows.py main-flow ./Data/SeasonDepth_val
python scripts/dataset_gen/gen_flows.py main-flow ./Data/WAYMO_train

python scripts/dataset_gen/gen_flows.py main-extract-median ./Data/DDAD_DEPTH
python scripts/dataset_gen/gen_flows.py main-extract-median ./Data/KITTI_DEPTH_train
python scripts/dataset_gen/gen_flows.py main-extract-median ./Data/NUSCENES
python scripts/dataset_gen/gen_flows.py main-extract-median ./Data/SeasonDepth_train
python scripts/dataset_gen/gen_flows.py main-extract-median ./Data/SeasonDepth_val
python scripts/dataset_gen/gen_flows.py main-extract-median ./Data/WAYMO_train

python scripts/dataset_gen/gen_splits.py write-kitti-depth-train-dynamic
python scripts/dataset_gen/gen_splits.py write-nuscenes-dynamic
python scripts/dataset_gen/gen_splits.py write-ddad-dynamic
python scripts/dataset_gen/gen_splits.py write-waymo-train-dynamic
python scripts/dataset_gen/gen_splits.py write-season-depth-train-dynamic
python scripts/dataset_gen/gen_splits.py write-season-depth-val-dynamic
python scripts/dataset_gen/gen_splits.py write-season-depth-val

python scripts/dataset_gen/expand_SeasonDepth.py
```


## Training

I used four RTX3090s for training.
The result will be saved in `./Logs`.

``` bash
torchrun --nproc_per_node 4 scripts/train.py config/ss_season.toml
torchrun --nproc_per_node 4 scripts/train.py config/ss_season_finetune.toml
```
