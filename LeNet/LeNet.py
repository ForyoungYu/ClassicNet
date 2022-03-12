import torch
import torch.nn as nn


class LeNet(nn.Module):
    """LeNet"""
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2,
                         padding=0,
                         dilation=1,
                         ceil_mode=False))
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1)), nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2,
                         padding=0,
                         dilation=1,
                         ceil_mode=False))
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(16 * 5 * 5, 120),
                                nn.Sigmoid(), nn.Linear(120, 84), nn.Sigmoid(),
                                nn.Linear(84, 10))

        #  self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.fc(x)
        return x


#  print(LeNet())
