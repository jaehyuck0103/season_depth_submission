import timm
import torch
import torchvision
from torch import nn


class DualImgResnet18(nn.Sequential):
    def __init__(self):

        # Enlarge input ch
        res18 = torchvision.models.resnet18(pretrained=True)
        new_conv = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            new_conv.weight[:] = 0.5 * res18.conv1.weight.repeat([1, 2, 1, 1])
        res18.conv1 = new_conv

        # Remove original fc layer
        modules = list(res18.children())[:-2]

        # Init Sequential
        super().__init__(*modules)

        self.out_ch = 512


class DualImgRegnetX002(nn.Sequential):
    def __init__(self):

        # Enlarge input ch
        reg002 = timm.create_model("regnetx_002", pretrained=True)
        new_conv = nn.Conv2d(6, 32, kernel_size=3, stride=2, padding=1, bias=False)
        with torch.no_grad():
            new_conv.weight[:] = 0.5 * reg002.stem.conv.weight.repeat([1, 2, 1, 1])
        reg002.stem.conv = new_conv

        # Remove original fc layer
        modules = list(reg002.children())[:-1]

        # Init Sequential
        super().__init__(*modules)

        self.out_ch = 368


class PoseEncoder(nn.Module):
    def __init__(self, transl_scale: float, encoder_type: str, use_additional_layers: bool):
        super().__init__()

        self.transl_scale = transl_scale

        # Select encoder
        if encoder_type == "res18":
            self.encoder = DualImgResnet18()
        elif encoder_type == "reg002":
            self.encoder = DualImgRegnetX002()
        else:
            raise ValueError(f"Unknwon encoder type {encoder_type}")

        # Select conv_squeeze
        if use_additional_layers:
            self.conv_squeeze = nn.Sequential(
                nn.Conv2d(self.encoder.out_ch, 256, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 6, 1),
            )
        else:
            self.conv_squeeze = nn.Conv2d(self.encoder.out_ch, 6, 1, 1, padding=0, bias=False)

    def snapshot_elements(self):
        return {
            "pose_encoder": self,
        }

    def forward(self, img1, img2):

        # Concat and Normalization
        x = torch.cat([img1, img2], 1)
        x = (x - 0.45) / 0.225

        # Foward
        x = self.encoder(x)
        x = self.conv_squeeze(x)
        x = x.mean(dim=(2, 3))
        angle_params = 0.01 * x[..., :3]
        transl_params = self.transl_scale * x[..., 3:]

        return angle_params, transl_params
