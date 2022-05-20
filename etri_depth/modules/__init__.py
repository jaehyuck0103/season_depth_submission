from .depth_prediction import dp_unet


def get_module(name, opt):
    if name == "dp_unet":
        net = dp_unet.DepthPredictionUNet(
            encoder_name=opt.net.dp_unet.encoder_name,
            encoder_depth=opt.net.dp_unet.encoder_depth,
            decoder_channels=opt.net.dp_unet.decoder_channels,
            use_decoder_bn=opt.net.dp_unet.use_decoder_bn,
        )
    else:
        raise ValueError(f"Unknown Module: {name}")
    return net
