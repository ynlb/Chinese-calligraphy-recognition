import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo

__all__ = ['ResNet', 'resnet101']


class CalliNet(nn.Module):
    def __init__(self, batch_size, num_classes):
        super(CalliNet, self).__init__()
        self.features = nn.Sequential(
                # b, 1, 64, 64
                nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                # b, 64, 32, 32

                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(True),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.ReLU(True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                # b, 128, 16, 16

                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.ReLU(True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                # b, 256, 8, 8

                nn.Conv2d(256, 512, kernel_size=3, padding=1),
                nn.ReLU(True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.ReLU(True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                # b, 512, 4, 4
                )
        self.classifier = nn.Sequential(
                nn.Linear(512*4*4, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(True),
                nn.Dropout(),

                nn.Linear(1024, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(True),
                nn.Dropout(),

                nn.Linear(1024, num_classes),
                nn.BatchNorm1d(num_classes),
                nn.Softmax(-1),
                )

    def forward(self, x):
        x = self.features(x)
        fc_input = x.view(x.size(0), -1)
        fc_out = self.classifier(fc_input)
        return fc_out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)   # 形状缩小为原来的1/2
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.PReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=100):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)    # size:64*64*64
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.PReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(4, stride=2, padding=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        # self.fc = nn.Linear(512*4*16, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(                                 # 下采样
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet101(**kwargs):
    """Constructs a ResNet-101 model."""
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model
