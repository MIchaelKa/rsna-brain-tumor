import torch
import torch.nn as nn

class Simple3DNet(nn.Module):
    def __init__(self):
        super().__init__()

        channels = [64, 128, 256, 512]
        conv_bias = True

        # nn.BatchNorm3d(channels[0]),
        
        self.backbone = nn.Sequential(
            # 1
            nn.Conv3d(1, channels[0], kernel_size=3, stride=2, padding=1, bias=conv_bias),
            nn.ReLU(),
            # nn.Conv3d(channels[0], channels[0], kernel_size=3, stride=1, padding=1, bias=conv_bias),
            # nn.ReLU(),
            # nn.Conv3d(channels[0], channels[0], kernel_size=3, stride=1, padding=1, bias=conv_bias),
            # nn.ReLU(),
            nn.MaxPool3d(2),
            
            # 2
            nn.Conv3d(channels[0], channels[1], kernel_size=3, stride=1, padding=1, bias=conv_bias),
            nn.ReLU(),
            # nn.Conv3d(channels[1], channels[1], kernel_size=3, stride=1, padding=1, bias=conv_bias),
            # nn.ReLU(),
            nn.MaxPool3d(2),
            
            # 3
            nn.Conv3d(channels[1], channels[2], kernel_size=3, stride=1, padding=1, bias=conv_bias),
            nn.ReLU(),
            nn.MaxPool3d(2),

            # 4
            # nn.Conv3d(channels[2], channels[3], kernel_size=3, stride=1, padding=1, bias=conv_bias),
            # nn.ReLU(),
            # nn.MaxPool3d(2),
        )
        
        # self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        # self.fc = nn.Linear(512, 1)
        
        # 256*4*16*16 = 262144
        self.fc = nn.Linear(256*4*16*16, 1)

        # 256*2*8*8 = 32768
        # self.fc = nn.Linear(512*2*8*8, 1)

        
        
        
    def forward(self, x):  
        x = self.backbone(x)
        # x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x