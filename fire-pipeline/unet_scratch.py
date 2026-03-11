import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, inChannels: int, outChannels: int):
        super().__init__()
        # Convolution + activation + convolution block
        self.conv1 = nn.Conv2d(inChannels, outChannels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(outChannels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(outChannels, outChannels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(outChannels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply CONV => BN => RELU => CONV => BN => RELU
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


class Encoder(nn.Module):
    def __init__(self, channels=(8, 16, 32, 64)):
        super().__init__()
        # Encoder blocks + maxpool
        self.encBlocks = nn.ModuleList(
            [Block(channels[i], channels[i + 1]) for i in range(len(channels) - 1)]
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        # Store intermediate outputs for skip connections
        feats: list[torch.Tensor] = []
        for block in self.encBlocks:
            x = block(x)
            feats.append(x)
            x = self.pool(x)
        return feats


class Decoder(nn.Module):
    def __init__(self, channels=(64, 32, 16)):
        super().__init__()
        # Upsamplers + decoder blocks (each block takes concat of skip + upsampled)
        self.upconvs = nn.ModuleList(
            [
                nn.ConvTranspose2d(channels[i], channels[i + 1], kernel_size=2, stride=2)
                for i in range(len(channels) - 1)
            ]
        )
        self.dec_blocks = nn.ModuleList(
            [
                Block(channels[i + 1] * 2, channels[i + 1])
                for i in range(len(channels) - 1)
            ]
        )

    @staticmethod
    def _center_crop(enc_feat: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
        # Center-crop enc_feat to (H,W) = target_hw
        _, _, h, w = enc_feat.shape
        th, tw = target_hw
        if h == th and w == tw:
            return enc_feat

        y0 = max((h - th) // 2, 0)
        x0 = max((w - tw) // 2, 0)
        y1 = y0 + th
        x1 = x0 + tw
        return enc_feat[:, :, y0:y1, x0:x1]

    def forward(self, x: torch.Tensor, encFeatures: list[torch.Tensor]) -> torch.Tensor:
        # encFeatures are expected as [f1, f2, ..., fN] (shallow -> deep)
        # We will use them in reverse for skip connections: f(N-1), ..., f1
        skips = encFeatures[:-1][::-1]

        for i in range(len(self.upconvs)):
            x = self.upconvs[i](x)
            skip = skips[i]

            # Crop skip to match x spatial dims
            skip = self._center_crop(skip, (x.shape[2], x.shape[3]))

            # Concat + decode
            x = torch.cat([x, skip], dim=1)
            x = self.dec_blocks[i](x)

        return x


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 8,
        num_classes: int = 2,
        features=(16, 32, 64),
        retainDim: bool = True,
    ):
        """
        Simple UNet from scratch.

        Args:
            in_channels: input channels
            num_classes: output classes (2 for DEL, 5 for GRA)
            features: encoder feature widths
            retainDim: if True, output is resized to match input HxW
        """
        super().__init__()

        encChannels = (in_channels, *features)
        decChannels = (features[-1], *features[::-1][1:])  # e.g. (64, 32, 16)

        self.encoder = Encoder(encChannels)
        self.decoder = Decoder(decChannels)

        self.head = nn.Conv2d(decChannels[-1], num_classes, kernel_size=1)

        self.retainDim = retainDim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_hw = (x.shape[2], x.shape[3])

        encFeatures = self.encoder(x)          # [f1, f2, f3]
        x = encFeatures[-1]                    # deepest feature
        x = self.decoder(x, encFeatures)       # decode with skips
        x = self.head(x)                       # (B, num_classes, H', W')

        if self.retainDim and (x.shape[2], x.shape[3]) != input_hw:
            x = F.interpolate(x, size=input_hw, mode="bilinear", align_corners=False)

        return x
