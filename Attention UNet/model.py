# %% Import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

# %% ConvBlock definition
class ConvBlock(nn.Module):
    def __init__(self, in_channels:int, out_channels:int):
        super(ConvBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True),
            
            nn.Conv2d(in_channels = out_channels, out_channels = out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True)
        )

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.conv(x)
# %% Attention Gate definition
class AttentionGate(nn.Module):
    def __init__(self, x_channels:int, g_channels:int, inter_channels:int):
        """
        Attention gate that computes an attention map alpha to weight the skip connection from the encoder.

        Inputs:
            x : skip connection from encoder     (fine, high-res, spatial detail)
            g : gating signal from decoder       (coarse, semantic, knows what to find)

        Both are projected to the same channel dimension (inter_channels),
        added together, passed through ReLU, then projected to a single
        channel and sigmoid'd to produce the attention map alpha.

        Alpha is upsampled to match x's spatial size and multiplied
        element-wise onto x before the skip connection is used.
        """
        super(AttentionGate, self).__init__()

        self.W_x = nn.Sequential(
            nn.Conv2d(in_channels = x_channels, out_channels = inter_channels, kernel_size = 1, stride = 1, padding = 0, bias = False),
            nn.BatchNorm2d(inter_channels),
        )

        self.W_g = nn.Sequential(
            nn.Conv2d(in_channels = g_channels, out_channels = inter_channels, kernel_size = 1, stride = 1, padding = 0, bias = False),
            nn.BatchNorm2d(inter_channels)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(in_channels = inter_channels, out_channels = 1, kernel_size = 1, stride = 1, padding = 0, bias = False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace = True)

    def forward(self, x:torch.Tensor, g:torch.Tensor) -> torch.Tensor:
        x1 = self.W_x(x)
        g1 = self.W_g(g)

        # Handling spatial mismatch between x and g
        if x1.shape != g1.shape:
            g1 = F.interpolate(g1, size = x1.shape[2:], mode = 'bilinear', align_corners = False)

        combined = self.relu(x1 + g1)
        alpha = self.psi(combined)

        # Handling spatial mismatch between alpha and x
        if alpha.shape != x.shape:
            alpha = F.interpolate(alpha, size = x.shape[2:], mode = 'bilinear', align_corners = False)

        return alpha * x
# %% Attention UNet definition
class AttentionUNet(nn.Module):
    def __init__(self, in_channels:int = 3, num_classes:int = 1, feature_list = [32, 64, 128, 256, 512], bottleneck_size:int = 1024):
        super(AttentionUNet, self).__init__()

        self.feature_list = feature_list
        self.bottleneck_size = bottleneck_size
        self.num_classes = num_classes

        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.bottleneck = ConvBlock(in_channels = feature_list[-1], out_channels = self.bottleneck_size)
        self.classifier = nn.Conv2d(in_channels = feature_list[0], out_channels = num_classes, kernel_size = 1)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.upconvs = nn.ModuleList()
        self.att_gates = nn.ModuleList()

        self._make_encoders(in_channels)
        self._make_decoders(self.bottleneck_size)



    def _make_encoders(self, in_channels:int):
        for feature in self.feature_list:
            self.encoders.append(
                ConvBlock(in_channels = in_channels, out_channels = feature)
                )
            self.encoders.append(self.pool)

            in_channels = feature

    def _make_decoders(self, in_channels: int):
        ch = in_channels  # starts at bottleneck_size, then tracks previous decoder output

        for feature in reversed(self.feature_list):
            self.upconvs.append(
                nn.ConvTranspose2d(in_channels=ch, out_channels=feature,
                                kernel_size=2, stride=2)
            )

            self.att_gates.append(
                AttentionGate(x_channels=feature, g_channels=feature,
                            inter_channels=max(feature // 2, 1))
            )

            self.decoders.append(ConvBlock(in_channels=feature * 2, out_channels=feature))
            ch = feature

    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        skips = []

        # Encoder
        for i in range(0, len(self.encoders), 2):
            x = self.encoders[i](x)
            skips.append(x)

            x = self.encoders[i + 1](x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        skips = skips[::-1]

        for i, (up, att, dec) in enumerate(
                zip(self.upconvs, self.att_gates, self.decoders)):
            
            # Upsample
            x = up(x)

            skip = skips[i]

            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:])

            # Attention Gate
            skip = att(skip, x)

            x = torch.cat((skip, x), dim = 1)

            # Decode
            x = dec(x)

        return self.classifier(x)