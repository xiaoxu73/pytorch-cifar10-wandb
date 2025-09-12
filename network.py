import torch.nn as nn


class Residuial_block(nn.Module):
    def __init__(self, ic, oc, stride=1):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(ic, oc, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(oc),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(oc, oc, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(oc),
        )

        self.ReLU = nn.ReLU(inplace=True)

        self.downsample = None
        if ic != oc or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(ic, oc, kernel_size=1, stride=stride),
                nn.BatchNorm2d(oc)
            )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample:
            residual = self.downsample(residual)

        out += residual
        return self.ReLU(out)

class Resnet(nn.Module):
    def __init__(self, block=Residuial_block, num_layer=[2, 2, 2, 2], num_classes=10, dropout=0.2):
        super().__init__()

        self.preconv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            # nn.Conv2d(3, 32, kernel_size=7, stride=3, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.layer1 = self.make_layer(block, 32, 64, num_layer[0], stride=2)
        self.layer2 = self.make_layer(block, 64, 128, num_layer[1], stride=2)
        self.layer3 = self.make_layer(block, 128, 256, num_layer[2], stride=2)
        self.layer4 = self.make_layer(block, 256, 512, num_layer[3], stride=2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def make_layer(self, block, ic, oc, num_layer, stride):
        layer = [block(ic, oc, stride)]
        for _ in range(num_layer):
            layer.append(block(oc, oc))

        return nn.Sequential(*layer)

    def forward(self, x):
        x = self.preconv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pool(x)

        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x

class LeNetBase(nn.Module):
    def __init__(self):
        super(LeNetBase, self).__init__()
        self.conv_params = nn.Sequential(
            nn.Conv2d(3, 20, kernel_size=5),
            nn.Dropout2d(p=0.1),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )
        self.in_features = 50 * 5 * 5

    def forward(self, x):
        x = self.conv_params(x)
        x = x.view(x.size(0), -1)
        return x