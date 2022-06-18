import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import numpy as np


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, 1, 1), 
                nn.ReLU(inplace=True),
            )

    def forward(self, x):
        return self.conv(x)


class Standardize(nn.Module):
    def __init__(self, method, mean=0, std=1, device="cuda"):
        super(Standardize, self).__init__()

        if method == 'PC':
            self.process = self.standardize_per_channel
            print('use per-channel preprocessing')
        elif method == 'PS':
            self.process = self.standardize_per_sample
            print('use per-sample preprocessing')
        elif method == 'AS':
            self.process = self.standardize_across_sample
            print('use across-sample preprocessing')
            self.mean = torch.tensor(np.reshape(mean, (1,len(mean),1,1))).float().to(device).detach()
            self.std = torch.tensor(np.reshape(std, (1, len(std),1,1))).float().to(device).detach()
        else:
            raise NotImplementedError("Can't find the method")
        

    def forward(self, x):
        return self.process(x)

    def standardize_per_channel(self, x):
        N = x.shape[0]
        C = x.shape[1]
        x_view = x.reshape(N,C,-1)
        x_mean = torch.mean(x_view, dim=2).view(N,C,1,1)
        x_std = 1e-5 + torch.std(x_view, dim=2).view(N,C,1,1)
        
        return (x - x_mean) / x_std

    def standardize_per_sample(self, x):
        N = x.shape[0]      
        x_view = x.reshape(N, -1)
        x_mean = torch.mean(x_view, dim=1).view(N,1,1,1) 
        x_std = 1e-5 + torch.std(x_view, dim=1).view(N,1,1,1)
        return (x - x_mean) / x_std

    def standardize_across_sample(self, x):
        return (x - self.mean)/self.std


class NoNameUNET(nn.Module):
    def __init__(self, in_channels=12, out_channels=1, 
        features=[64, 128, 256, 512], 
        preprocess=nn.Identity()):

        super(NoNameUNET, self).__init__()

        self.preprocess = preprocess

        self.out_channels = out_channels

        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)     

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part 
        for feature in reversed(features):
            self.ups.append(
                    nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2)
                )
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)


    def bands_noise(self, x):
        N = x.shape[0]
        C = x.shape[1]
        noise = 0.01 * torch.randn([N, C, 1, 1]).to("cuda") # TODO: clean this later
        return x + noise


    def forward(self, x):

        # preprocess - standardize input
        x = self.preprocess(x)
        

        # add noise to the bands
        if self.training:
            x = self.bands_noise(x)

        # unet model
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x) # the ConvTransposed
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)  #[b,c,h,w]
            x = self.ups[idx+1](concat_skip) # the DoubleConv

        return self.final_conv(x) 
