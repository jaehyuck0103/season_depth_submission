import numpy as np
import torch
from torch import nn

from etri_depth.modules.segmentation_models_pytorch.decoder import UnetDecoder
from etri_depth.modules.segmentation_models_pytorch.encoders import get_encoder
from etri_depth.modules.segmentation_models_pytorch.heads import SegmentationHead


class DepthPredictionUNet(nn.Module):
    def __init__(
        self,
        encoder_name: str,  # "timm-regnetx_002", resnet18
        encoder_depth=5,
        decoder_channels=(256, 128, 64, 32, 16),
        use_decoder_bn=True,
    ):
        super().__init__()

        if encoder_name.startswith("timm-regnet"):
            weights_name = "imagenet"
        elif encoder_name.startswith("resnet"):
            weights_name = "swsl"
        else:
            raise ValueError(f"Unknwon encoder_name {encoder_name}")

        self.rgb_encoder = get_encoder(
            encoder_name, in_channels=3, depth=encoder_depth, weights=weights_name
        )
        encoder_channels = np.array(self.rgb_encoder.out_channels)

        # decoder
        self.rgb_decoder = UnetDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels[-encoder_depth:],
            n_blocks=encoder_depth,
            use_batchnorm=use_decoder_bn,
            center=False,
        )

        self.head4 = SegmentationHead(decoder_channels[-5], 1, upsampling=16)
        self.head3 = SegmentationHead(decoder_channels[-4], 1, upsampling=8)
        self.head2 = SegmentationHead(decoder_channels[-3], 1, upsampling=4)
        self.head1 = SegmentationHead(decoder_channels[-2], 1, upsampling=2)
        self.head0 = SegmentationHead(decoder_channels[-1], 1)

        self.num_output_scales = 5

    def snapshot_elements(self):
        return {
            "depth_estimator": self,
        }

    def forward(self, inputs):
        input_rgb = inputs["color_aug", 0]
        input_rgb = (input_rgb - 0.45) / 0.225

        encoded = self.rgb_encoder(input_rgb)

        decoder_out = self.rgb_decoder(*encoded)

        output_d4 = torch.sigmoid(self.head4(decoder_out[-5]))
        output_d3 = torch.sigmoid(self.head3(decoder_out[-4]))
        output_d2 = torch.sigmoid(self.head2(decoder_out[-3]))
        output_d1 = torch.sigmoid(self.head1(decoder_out[-2]))
        output_d0 = torch.sigmoid(self.head0(decoder_out[-1]))

        outputs = {}
        outputs["depth", 0, 0] = output_d0
        outputs["depth", 0, 1] = output_d1
        outputs["depth", 0, 2] = output_d2
        outputs["depth", 0, 3] = output_d3
        outputs["depth", 0, 4] = output_d4

        return outputs
