import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from torchvision.models.resnet import BasicBlock


class DropConnect(nn.Module):
    def __init__(self, drop_prob):
        super(DropConnect, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1 - self.drop_prob
        mask = torch.rand_like(x) < keep_prob
        return x * mask / keep_prob


class BasicBlockWithDropConnect(BasicBlock):
    def __init__(self, *args, drop_prob=0.0, **kwargs):
        super(BasicBlockWithDropConnect, self).__init__(*args, **kwargs)
        self.dropconnect = DropConnect(drop_prob)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.dropconnect(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


def resnet18_with_dropconnect(drop_prob=0.2, pretrained=False, **kwargs):
    model = resnet18(pretrained=pretrained, **kwargs)
    model.layer1 = nn.Sequential(
        *[BasicBlockWithDropConnect(inplanes, planes, drop_prob=drop_prob)
          for inplanes, planes in zip([64] * len(model.layer1), [64] * len(model.layer1))]
    )
    model.layer2 = nn.Sequential(
        *[BasicBlockWithDropConnect(inplanes, planes, drop_prob=drop_prob)
          for inplanes, planes in zip([128] * len(model.layer2), [128] * len(model.layer2))]
    )
    model.layer3 = nn.Sequential(
        *[BasicBlockWithDropConnect(inplanes, planes, drop_prob=drop_prob)
          for inplanes, planes in zip([256] * len(model.layer3), [256] * len(model.layer3))]
    )
    model.layer4 = nn.Sequential(
        *[BasicBlockWithDropConnect(inplanes, planes, drop_prob=drop_prob)
          for inplanes, planes in zip([512] * len(model.layer4), [512] * len(model.layer4))]
    )
    return model
