# %%
from xml.parsers.expat import model

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


# %%
class ResNetBackbone(nn.Module):
    def __init__(self):
        super(ResNetBackbone, self).__init__()

        self.backbone = models.resnet101(weights='IMAGENET1K_V2')
        
        self.backbone.avgpool = nn.Identity()
        self.backbone.fc = nn.Identity()

        self._modify_layer(self.backbone.layer4, dilation = 2)

    def _modify_layer(self, layer, dilation):
        for module in layer.modules():

            if isinstance(module, nn.Conv2d):
                if module.kernel_size == (3, 3):
                    module.stride = (1, 1)
                    module.dilation = (dilation, dilation)
                    module.padding = (dilation, dilation)

                elif module.kernel_size == (1, 1):
                    module.stride = (1, 1)


    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        l1 = self.backbone.layer1(x)        # Low level features 
        l2 = self.backbone.layer2(l1)
        l3 = self.backbone.layer3(l2)
        l4 = self.backbone.layer4(l3)       # Encoder Output

        return l1, l4
        
        

# %%
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels,rates = [6, 12, 18]):
        super(ASPP, self).__init__()

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 1, padding = 0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.branch2 = self._make_branch(in_channels = in_channels, out_channels = out_channels, rate = rates[0])
        self.branch3 = self._make_branch(in_channels = in_channels, out_channels = out_channels, rate = rates[1])
        self.branch4 = self._make_branch(in_channels = in_channels, out_channels = out_channels, rate = rates[2])
        self.branch5 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.projection = nn.Sequential(
            nn.Conv2d(in_channels = out_channels * 5, out_channels = out_channels, kernel_size = 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5)
        )


    def _make_branch(self, in_channels, out_channels, rate):
        branch = nn.Sequential(
            nn.Conv2d(in_channels = in_channels,
                      out_channels = out_channels,
                      kernel_size = 3,
                      padding = rate,
                      dilation = rate),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        return branch
    

    def forward(self, x):

        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        b5 = self. branch5(x)

        b5 = F.interpolate(b5, size = b1.shape[2:], mode = 'bilinear', align_corners = False)

        x = torch.cat([b1, b2, b3, b4, b5], dim = 1)
        x = self.projection(x)

        return x
# %%
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)
# %%
class Decoder(nn.Module):
    def __init__(self, num_classes):
        super(Decoder, self).__init__()

        self._reduce = nn.Sequential(
            nn.Conv2d(in_channels = 256, out_channels = 48, kernel_size = 1),
            nn.BatchNorm2d(48),
            nn.ReLU()
        )

        self.convblock1 = ConvBlock(in_channels = 304, out_channels = 256) 
        self.convblock2 = ConvBlock(in_channels = 256, out_channels = 256)


        self.classifier = nn.Conv2d(in_channels = 256, out_channels = num_classes, kernel_size = 1)


    def forward(self, low_level_features, aspp_out, input_size = (640, 640)):

        # Reduce Low Level Features
        low_level_features = self._reduce(low_level_features)

        # Upsample ASPP output to match low level features spatial dimensions
        aspp_out = F.interpolate(input = aspp_out, size = low_level_features.shape[2:], mode = 'bilinear', align_corners = False)

        # Concatenate and Refine
        cat = torch.cat([low_level_features, aspp_out], dim = 1)
        cat = self.convblock1(cat)
        cat = self.convblock2(cat)

        # Upsample to original image size (assuming input images are 4 times the size of low level features)
        cat = F.interpolate(input = cat, size = input_size, mode = 'bilinear', align_corners = False)
        
        output = self.classifier(cat)

        return output

# %%
class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes):

        super(DeepLabV3Plus, self).__init__()

        self.backbone = ResNetBackbone()
        self.aspp = ASPP(in_channels = 2048, out_channels = 256)
        self.decoder = Decoder(num_classes = num_classes)

    def forward(self, x):
        
        low_level_features, encoder_output = self.backbone(x)
        aspp_out = self.aspp(encoder_output)

        # Original input size for upsampling in decoder
        input_size = x.shape[2:]

        output = self.decoder(low_level_features, aspp_out, input_size)

        return output

# %%
