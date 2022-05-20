from .depth_dataset import DepthDataset


def get_dataset(dataset_cfg, is_train: bool):
    if dataset_cfg.name == "DEPTH":
        dataset = DepthDataset(dataset_cfg=dataset_cfg, is_train=is_train)
        print(f"Using {'train' if is_train else 'val'}_split: ", dataset_cfg.splits)
        print(f"{len(dataset)} items\n")
    else:
        raise ValueError(f"Unexpected Dataset: name={dataset_cfg.name}")

    return dataset
