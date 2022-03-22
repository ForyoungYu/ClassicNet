import torch
import torch.nn as nn
import torch.nn.functional as F


class SE(nn.Module):
    def __init__(self, in_chnls, ratio: int = 16) -> None:
        super(SE, self).__init__()
        self.sqeeze = nn.AdaptiveAvgPool2d((1, 1))
        self.compress = nn.Conv2d(in_channels=in_chnls,
                                  out_channels=in_chnls // ratio,
                                  kernel_size=1,
                                  stride=1,
                                  padding=0)
        self.excitation = nn.Conv2d(in_channels=in_chnls // ratio,
                                    out_channels=in_chnls,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0)

    def forward(self, x):
        out = self.sqeeze(x)
        out = self.compress(out)
        out = F.relu(out, inplace=True)
        out = self.excitation(out)
        out = torch.sigmoid(out)

        return out
