import torch
import torch.nn as nn

class VGG16EncoderWithSkipConnections(nn.Module):
    def __init__(self, input_channels=3):
        super(VGG16EncoderWithSkipConnections, self).__init__()

        # Block 1: Two Conv layers + MaxPool
        self.block1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Downsample to 128x128
        )

        # Block 2: Two Conv layers + MaxPool
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Downsample to 64x64
        )

        # Block 3: Three Conv layers + MaxPool
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Downsample to 32x32
        )

        # Block 4: Three Conv layers + MaxPool
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Downsample to 16x16
        )

        # Block 5: Three Conv layers + MaxPool
        self.block5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Downsample to 8x8
        )

        self.block6 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Downsample to 4x4
        )

    def forward(self, x):
        skip_connections = []
        x = self.block1(x)
        skip_connections.append(x)
        x = self.block2(x)
        skip_connections.append(x)
        x = self.block3(x)
        skip_connections.append(x)
        x = self.block4(x)
        skip_connections.append(x)
        x = self.block5(x)
        skip_connections.append(x)
        x = self.block6(x)
        return x, skip_connections


class VGG16Decoder(nn.Module):
    def __init__(self):
        super(VGG16Decoder, self).__init__()

        # Block 6
        self.block6 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),


            # nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)  # Upsample to 8x8
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),  # Upsample to 8x8
            nn.ReLU(inplace=True)
        )

        # Block 5
        self.block5 = nn.Sequential(
            nn.Conv2d(512 + 512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),



            #nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)  # Upsample to 16x16

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        # Block 4
        self.block4 = nn.Sequential(
            nn.Conv2d(512 + 512, 512, kernel_size=3, stride=1, padding=1),  # Concatenate channels
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            #nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)  # Upsample to 32x32

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        # Block 3
        self.block3 = nn.Sequential(
            nn.Conv2d(256 + 256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            #nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)  # Upsample to 64x64

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        # Block 2
        self.block2 = nn.Sequential(
            nn.Conv2d(128 + 128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            # nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)  # Upsample to 128x128

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),  # Upsample to 8x8
            nn.ReLU(inplace=True)
        )

        # Block 1
        self.block1 = nn.Sequential(
            nn.Conv2d(64 + 64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # nn.ConvTranspose2d(64, 3, kernel_size=2, stride=2)  # Upsample to 256x256

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),  # Upsample to 8x8
            nn.Sigmoid()
            #nn.ReLU(inplace=True)
        )

    def forward(self, x, skip_connections):
        # Reverse the skip connections
        skip_connections = skip_connections[::-1]
        x = self.block6(x)
        x = torch.cat([x, skip_connections[0]], dim=1)
        x = self.block5(x)
        x = torch.cat([x, skip_connections[1]], dim=1)
        x = self.block4(x)
        x = torch.cat([x, skip_connections[2]], dim=1)
        x = self.block3(x)
        x = torch.cat([x, skip_connections[3]], dim=1)
        x = self.block2(x)
        x = torch.cat([x, skip_connections[4]], dim=1)
        x = self.block1(x)
        return x


class VGG16Autoencoder(nn.Module):
    def __init__(self, input_channels=3):
        super(VGG16Autoencoder, self).__init__()
        self.encoder = VGG16EncoderWithSkipConnections(input_channels=input_channels)
        self.decoder = VGG16Decoder()

    def forward(self, x):
        encoded_output, skip_connections = self.encoder(x)
        reconstructed = self.decoder(encoded_output, skip_connections)
        return reconstructed
