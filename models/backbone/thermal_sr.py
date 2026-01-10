import torch
import torch.nn as nn

class ThermalSR(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 1, 3, padding=1)
        )

    def forward(self, thermal_lr, rgb_edge):
        x = torch.cat([thermal_lr, rgb_edge], dim=1)
        return self.net(x)
