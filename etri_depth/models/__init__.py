import torch

from . import self_supervised


def get_model(name, opt, predict_only=False):
    if name == "self_supervised":
        net = self_supervised.NetBuilder(opt)
        loss_module = (
            self_supervised.LossModule(opt) if predict_only is False else torch.nn.Identity()
        )
    else:
        raise ValueError(f"Unknown Agent: {name}")
    return net, loss_module
