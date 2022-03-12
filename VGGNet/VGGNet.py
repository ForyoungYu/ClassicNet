import torch
import torch.nn as nn
from torch.nn.modules import padding


class VGGNet(nn.Module):
    """VGGNet"""
    def __init__(self):
        super(VGGNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2))

        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2))
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2))
        self.layer5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2))
        self.conv = nn.Sequential(self.layer1, self.layer2, self.layer3,
                                  self.layer4, self.layer5)
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(512, 512),
                                nn.ReLU(inplace=True), nn.Dropout(),
                                nn.Linear(512, 256), nn.ReLU(inplace=True),
                                nn.Dropout(), nn.Linear(256, 10))

        #  self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x


#  print(LeNet())
