"""Model definitions.

* ``UNet``            - the Week 1 U-Net trained from scratch (any number of input channels).
* ``build_smp_model`` - Week 2 pretrained encoders (segmentation_models_pytorch) with a corrected
                        way of feeding them 12 / 17 channels (see ``adapt_first_conv``).
"""
import torch
import torch.nn as nn

from .features import RGB_BANDS


# --------------------------------------------------------------------------- U-Net from scratch
class DoubleConv(nn.Module):
    def __init__(self, cin, cout, dropout=0.0):
        super().__init__()
        layers = [nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                  nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_channels, base=32, dropout=0.0):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8, base * 16]
        self.enc = nn.ModuleList([DoubleConv(in_channels, c[0]), DoubleConv(c[0], c[1]),
                                  DoubleConv(c[1], c[2]), DoubleConv(c[2], c[3], dropout)])
        self.bottleneck = DoubleConv(c[3], c[4], dropout)
        self.pool = nn.MaxPool2d(2)
        self.up = nn.ModuleList([nn.ConvTranspose2d(c[4], c[3], 2, stride=2), nn.ConvTranspose2d(c[3], c[2], 2, stride=2),
                                 nn.ConvTranspose2d(c[2], c[1], 2, stride=2), nn.ConvTranspose2d(c[1], c[0], 2, stride=2)])
        self.dec = nn.ModuleList([DoubleConv(c[3] * 2, c[3], dropout), DoubleConv(c[2] * 2, c[2]),
                                  DoubleConv(c[1] * 2, c[1]), DoubleConv(c[0] * 2, c[0])])
        self.head = nn.Conv2d(c[0], 1, 1)

    def forward(self, x):
        skips = []
        for i, blk in enumerate(self.enc):
            x = blk(x) if i == 0 else blk(self.pool(x))
            skips.append(x)
        x = self.bottleneck(self.pool(x))
        for up, dec, skip in zip(self.up, self.dec, reversed(skips)):
            x = dec(torch.cat([up(x), skip], dim=1))
        return self.head(x)


# --------------------------------------------------------------------------- pretrained encoders
def _first_conv(encoder: nn.Module) -> nn.Conv2d:
    for m in encoder.modules():
        if isinstance(m, nn.Conv2d):
            return m
    raise RuntimeError("No Conv2d found in encoder")


INIT_STRATEGIES = ("rgb_random", "random", "adapter", "rgb_zero", "mean")


@torch.no_grad()
def adapt_first_conv(encoder: nn.Module, in_channels: int, strategy: str = "rgb_random",
                     rgb_bands=RGB_BANDS) -> nn.Module:
    """Widen the encoder's first conv from 3 to ``in_channels`` input channels.

    Strategies
    ----------
    rgb_random (default)  pretrained ImageNet filters go to the Red/Green/Blue band positions; every other
                          band gets *independent random* filters (N(0, std of the pretrained filters)).
                          Each extra band starts with its own distinct filter and can learn its own pattern.
    random                all channels random (Kaiming normal) - no pretrained information in conv1.
    rgb_zero              pretrained filters on RGB bands, zeros elsewhere (extra bands start switched off).
    mean                  DEPRECATED. Averages the RGB filters and repeats them for every band. All bands get
                          identical filters, so the bands are indistinguishable at initialisation (symmetry)
                          and the pretrained colour filters are destroyed. This was the Week 2 review issue.
    (``adapter`` is handled in ``build_model``: it leaves conv1 untouched.)
    """
    conv = _first_conv(encoder)
    w = conv.weight.data
    assert w.shape[1] == 3, "encoder must be built with 3 input channels first"
    out, _, kh, kw = w.shape
    if strategy == "mean":
        new = w.mean(dim=1, keepdim=True).repeat(1, in_channels, 1, 1) * (3.0 / in_channels)
    elif strategy == "random":
        new = torch.empty(out, in_channels, kh, kw, dtype=w.dtype, device=w.device)
        nn.init.kaiming_normal_(new, mode="fan_out", nonlinearity="relu")
    elif strategy == "rgb_random":
        new = torch.randn(out, in_channels, kh, kw, dtype=w.dtype, device=w.device) * w.std()
        for src, dst in enumerate(rgb_bands):
            new[:, dst] = w[:, src]
    elif strategy == "rgb_zero":
        new = torch.zeros(out, in_channels, kh, kw, dtype=w.dtype, device=w.device)
        for src, dst in enumerate(rgb_bands):
            new[:, dst] = w[:, src]
    else:
        raise ValueError(f"unknown strategy {strategy!r}; choose from {INIT_STRATEGIES}")
    conv.weight = nn.Parameter(new)
    conv.in_channels = in_channels
    if hasattr(encoder, "_in_channels"):
        encoder._in_channels = in_channels
    return encoder


class ChannelAdapter(nn.Module):
    """Learnable 1x1 conv that maps N input channels to the 3 channels a pretrained encoder expects.

    Initialised so the R/G/B bands pass straight through (identity) and every other band contributes a small
    random amount, so training starts from the pretrained behaviour and learns how to mix the other bands in.
    """

    def __init__(self, in_channels, rgb_bands=RGB_BANDS, noise_std=0.05):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, 3, kernel_size=1, bias=True)
        with torch.no_grad():
            self.proj.weight.normal_(0.0, noise_std)
            self.proj.bias.zero_()
            for c, band in enumerate(rgb_bands):
                self.proj.weight[c, band] += 1.0

    def forward(self, x):
        return self.proj(x)


class AdaptedSegModel(nn.Module):
    """ChannelAdapter (N -> 3) in front of an unmodified 3-channel smp model."""

    def __init__(self, net: nn.Module, in_channels: int):
        super().__init__()
        self.adapter = ChannelAdapter(in_channels)
        self.net = net

    @property
    def encoder(self):  # lets ``param_groups`` treat adapter + decoder as the "fast" group
        return self.net.encoder

    def forward(self, x):
        return self.net(self.adapter(x))


def build_smp_model(arch: str, encoder: str, in_channels: int = 12, pretrained: bool = True,
                    init_strategy: str = "rgb_random") -> nn.Module:
    import segmentation_models_pytorch as smp  # imported lazily: only needed for pretrained models
    archs = {"unet": smp.Unet, "deeplabv3plus": smp.DeepLabV3Plus}
    net = archs[arch](encoder_name=encoder, encoder_weights="imagenet" if pretrained else None,
                      in_channels=3, classes=1)
    if init_strategy == "adapter":
        return AdaptedSegModel(net, in_channels)
    adapt_first_conv(net.encoder, in_channels, init_strategy)
    return net


def param_groups(model: nn.Module, enc_lr: float, dec_lr: float, weight_decay: float = 1e-4):
    enc = [p for p in model.encoder.parameters() if p.requires_grad]
    enc_ids = {id(p) for p in enc}
    rest = [p for p in model.parameters() if p.requires_grad and id(p) not in enc_ids]
    return [dict(params=enc, lr=enc_lr, weight_decay=weight_decay),
            dict(params=rest, lr=dec_lr, weight_decay=weight_decay)]


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# --------------------------------------------------------------------------- checkpoint -> model
def build_from_spec(spec: dict, in_channels: int) -> nn.Module:
    """Rebuild an architecture from the ``model`` entry stored in a checkpoint (weights loaded separately)."""
    if spec["type"] == "unet_scratch":
        return UNet(in_channels, base=spec.get("base", 32), dropout=spec.get("dropout", 0.0))
    if spec["type"] == "smp":
        return build_smp_model(spec["arch"], spec["encoder"], in_channels, pretrained=False,
                               init_strategy=spec.get("init_strategy", "rgb_random"))
    raise ValueError(f"unknown model type {spec['type']!r}")
