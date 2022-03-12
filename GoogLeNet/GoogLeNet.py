import torch
import torch.nn as nn


def conv(in_channels, out_channels, kernel_size, stride=1, padding=0):
    return nn.Sequential(
        nn.Conv2d(in_channels,
                  out_channels,
                  kernel_size=kernel_size,
                  stride=stride,
                  padding=padding), nn.ReLU(inplace=True))


class Inception(nn.Module):
    def __init__(self, in_channels, ch1, ch3reduce, ch3, ch5reduce, ch5,
                 pool_proj):
        super(Inception, self).__init__()
        self.branch1 = conv(in_channels, ch1, kernel_size=1)
        self.branch2 = nn.Sequential(
            conv(in_channels, ch3reduce, kernel_size=1),
            conv(ch3reduce, ch3, kernel_size=3, stride=1, padding=1))
        self.branch3 = nn.Sequential(
            conv(in_channels, ch5reduce, kernel_size=1),
            conv(ch5reduce, ch5, kernel_size=5, stride=1, padding=2))

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            conv(in_channels, pool_proj, kernel_size=1))

    def forward(self, x):
        block1 = self.branch1(x)
        block2 = self.branch2(x)
        block3 = self.branch3(x)
        block4 = self.branch4(x)

        block = (block1, block2, block3, block4)

        return torch.cat(block, dim=1)


class GoogLeNet(nn.Module):
    """GoogLeNet"""
    def __init__(self, class_nums):
        super(GoogLeNet, self).__init__()
        self.block1 = nn.Sequential(
            conv(3, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True))

        self.block2 = nn.Sequential(
            conv(64, 64, kernel_size=1),
            conv(64, 192, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True))

        self.block3 = nn.Sequential(
            Inception(192, 64, 96, 128, 16, 32, 32),
            Inception(256, 128, 128, 192, 32, 96, 64),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True))

        self.block4 = nn.Sequential(
            Inception(480, 192, 96, 208, 16, 48, 64),
            Inception(512, 160, 112, 224, 24, 64, 64),
            Inception(512, 128, 128, 256, 24, 64, 64),
            Inception(512, 112, 144, 288, 32, 64, 64),
            Inception(528, 256, 160, 320, 32, 128, 128),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True))

        self.block5 = nn.Sequential(
            Inception(832, 256, 160, 320, 32, 128, 128),
            Inception(832, 384, 192, 384, 48, 128, 128),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Dropout(0.4))

        self.classifier = nn.Linear(1024, class_nums)

        #  self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)

        return x


#  print(LeNet())
