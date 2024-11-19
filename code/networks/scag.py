import torch
import torch.nn as nn


class SCAG(nn.Module):
    def __init__(self, in_channels, inter_channels):
        super(SCAG, self).__init__()

        # Depthwise convolutions
        self.conv_enc = nn.Conv2d(in_channels, inter_channels, kernel_size=5, padding=2, groups=in_channels)
        self.conv_upsample = nn.Conv2d(in_channels, inter_channels, kernel_size=5, padding=2, groups=in_channels)

        # 1x1 Convolution and Sigmoid activation
        self.conv1x1 = nn.Conv2d(inter_channels * 2, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

        # Resample layer
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, encoder_features, upsample_features):
        enc = self.conv_enc(encoder_features)
        up = self.conv_upsample(upsample_features)

        combined = torch.cat([enc, up], dim=1)

        attention = self.sigmoid(self.conv1x1(combined))

        # Resample the attention map
        attention = self.upsample(attention)

        # Apply attention to encoder features
        gated_features = attention * encoder_features

        return gated_features
