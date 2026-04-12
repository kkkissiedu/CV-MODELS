import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    Two consecutive Conv2d -> BatchNorm -> ReLU layers.
    The basic building block of every encoder and decoder level.

    padding=1 keeps spatial dimensions constant through the block,
    so skip-connection shapes align without cropping.
    bias=False because BatchNorm's learnable shift makes it redundant.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels,  out_channels, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    """
    Classic U-Net encoder-bottleneck-decoder architecture.

    Args:
        in_channels    : number of input image channels (3 for RGB, 1 for grayscale)
        num_classes    : number of output segmentation classes
        feature_list   : channel counts at each encoder/decoder level
        bottleneck_size: channel count at the bottleneck

    Architecture:
        Encoder    -> ConvBlock -> MaxPool  (repeated len(feature_list) times)
        Bottleneck -> ConvBlock
        Decoder    -> ConvTranspose2d -> concat(skip) -> ConvBlock  (reversed)
        Head       -> 1x1 Conv -> raw logits
    """

    def __init__(self,
                 in_channels: int = 3,
                 num_classes: int = 1,
                 feature_list: list = None,
                 bottleneck_size: int = 1024):
        super().__init__()

        if feature_list is None:
            feature_list = [32, 64, 128, 256, 512]

        self.num_classes     = num_classes
        self.feature_list    = feature_list
        self.bottleneck_size = bottleneck_size

        self.encoder    = nn.ModuleList()
        self.pool       = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = ConvBlock(feature_list[-1], bottleneck_size)
        self.decoder    = nn.ModuleList()
        self.classifier = nn.Conv2d(feature_list[0], num_classes, kernel_size=1)

        self._build_encoder(in_channels, feature_list)
        self._build_decoder(feature_list)

    def _build_encoder(self, in_channels: int, feature_list: list):
        ch = in_channels
        for f in feature_list:
            self.encoder.append(ConvBlock(ch, f))
            self.encoder.append(self.pool)
            ch = f

    def _build_decoder(self, feature_list: list):
        for f in reversed(feature_list):
            self.decoder.append(
                nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2)
            )
            self.decoder.append(ConvBlock(f * 2, f))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []

        for i in range(0, len(self.encoder), 2):
            x = self.encoder[i](x)
            skips.append(x)
            x = self.encoder[i + 1](x)

        x = self.bottleneck(x)

        skips = skips[::-1]

        for i in range(0, len(self.decoder), 2):
            x    = self.decoder[i](x)
            skip = skips[i // 2]

            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:])

            x = torch.cat((skip, x), dim=1)
            x = self.decoder[i + 1](x)

        return self.classifier(x)
