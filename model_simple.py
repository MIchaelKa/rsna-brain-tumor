import torch
import torch.nn as nn

class Simple3DNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.backbone = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(2),
            # 13x16x16
            
            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.MaxPool3d(2),
            # 6x8x8
            
            nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.MaxPool3d(2),
            # 3x4x4
        )
        
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        self.fc = nn.Linear(256, 1)
        
#         self.fc = nn.Linear(256*3*4*4, 1)
        
        
    def forward(self, x):  
        x = self.backbone(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x