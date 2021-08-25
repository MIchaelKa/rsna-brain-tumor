import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
        )

        if stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=2, padding=0, bias=False),
                # nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm3d(out_channels),
            )
        else:
            self.downsample = None

        self.relu = nn.ReLU(inplace=True)

        
    def forward(self, x):
        identity = x
        x = self.block(x)
        if self.downsample is not None:
            identity = self.downsample(identity)
        x += identity
        x = self.relu(x)
        return x

class ResNet3D(nn.Module):
    def __init__(self):
        super().__init__()

        layers = [2, 2, 2]

        channels = [64, 128, 256]
        # channels = [64, 256, 512]


        # TODO: less aggressive stride along D direction?

        self.stem = nn.Sequential(
            nn.Conv3d(1, channels[0], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(channels[0]),
            nn.ReLU(inplace=True),

            nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
            # nn.MaxPool3d(2)
        )

        self.layer1 = self._make_layer(layers[0], channels[0], channels[0], 1)
        self.layer2 = self._make_layer(layers[1], channels[0], channels[1], 2)
        self.layer3 = self._make_layer(layers[2], channels[1], channels[2], 2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(256, 1)

        # 256*4*16*16 = 262144
        # self.fc = nn.Linear(256*4*16*16, 1)

    
    def _make_layer(self, blocks, in_channels, out_channels, stride=1):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
     
        
        
    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)  
        return x