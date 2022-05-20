from torch.utils import model_zoo

from .resnet import resnet_encoders
from .timm_regnet import timm_regnet_encoders

encoders = {}
encoders.update(resnet_encoders)
encoders.update(timm_regnet_encoders)


def get_encoder(name, in_channels=3, depth=5, weights=None):

    Encoder = encoders[name]["encoder"]

    params = encoders[name]["params"]
    params.update(depth=depth)
    encoder = Encoder(**params)

    if weights is not None:
        settings = encoders[name]["pretrained_settings"][weights]
        encoder.load_state_dict(model_zoo.load_url(settings["url"]))

    encoder.set_in_channels(in_channels)

    return encoder
