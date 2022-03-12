import torch.nn as nn


class AlexNet(nn.Module):
    """AlexNet"""
    def __init__(self, num_classes):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, 96, 3, 1),
                                   nn.BatchNorm2d(96),
                                   nn.ReLU(inplace=True),
                                   nn.MaxPool2d(3, 2))
        self.conv2 = nn.Sequential(nn.Conv2d(96, 256, 3, 1),
                                   nn.BatchNorm2d(256),
                                   nn.ReLU(inplace=True),
                                   nn.MaxPool2d(3, 2))
        self.conv3 = nn.Sequential(nn.Conv2d(256, 384, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(384, 384, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(384, 256, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   nn.MaxPool2d(3, 2))
        self.fc = nn.Sequential(nn.Flatten(),
                                nn.Linear(256 * 2 * 2, 2048),
                                nn.ReLU(inplace=True),
                                nn.Dropout(0.5),
                                nn.Linear(2048, 2048),
                                nn.ReLU(inplace=True),
                                nn.Dropout(0.5),
                                nn.Linear(2048, num_classes))

        #  self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.kaiming_normal_(m.weight,
                                    a=0,
                                    mode='fan_out',
                                    nonlinearity='relu')

    def forward(self, img):
        img = self.conv1(img)
        img = self.conv2(img)
        img = self.conv3(img)
        img = self.fc(img)
        return img
