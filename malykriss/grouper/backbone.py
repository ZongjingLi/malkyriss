'''
 # @ Author: Zongjing Li
 # @ Create Time: 2023-12-22 11:01:46
 # @ Modified by: Zongjing Li
 # @ Modified time: 2023-12-22 11:06:21
 # @ Description: This file is distributed under the MIT license.
'''
import torch
import torch.nn as nn

class ResidualDenseConv(nn.Module):
    def __init__(self, channel_in, growRate, kernel_size=3):
        super(ResidualDenseConv, self).__init__()
        G  = growRate
        self.conv = nn.Sequential(*[
            nn.Conv2d(channel_in, G, kernel_size, padding=(kernel_size-1)//2, stride=1),
            nn.ReLU()
        ])

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)

class ResidualDenseBlock(nn.Module):
    def __init__(self, grow0, grow_rate, num_conv_layers):
        super(ResidualDenseBlock, self).__init__()
        convs = []
        for c in range(num_conv_layers):
            convs.append(ResidualDenseConv(grow0 + c*grow_rate, grow_rate))
        self.convs = nn.Sequential(*convs)

        # Local Feature Fusion
        self.local_feature_fusion = nn.Conv2d(grow0 + num_conv_layers*grow_rate, grow0, 1, padding=0, stride=1)

    def forward(self, x):
        return self.local_feature_fusion(self.convs(x)) + x

class ResidualDenseNetwork(nn.Module):
    """Residual Dense Network as the convolutional visual backbone
    """
    def __init__(self, grow0, n_colors = 4, kernel_size = 3, scale = [2],rdn_config = [4,3,16], no_upsample = True):
        super(ResidualDenseNetwork, self).__init__()
        self.no_upsample = no_upsample

        r = scale[0]

        # number of RDB blocks, conv layers, out channels
        self.block_num, conv_layer_num, out_channel_num = rdn_config
        """
        {
            'A': (20, 6, 32),
            'B': (16, 8, 64),
        }[args.RDNconfig]
        """

        # Shallow feature extraction net
        self.shallow_feature_net1 = nn.Conv2d(n_colors, grow0, kernel_size, padding=(kernel_size-1)//2, stride=1)
        self.shallow_feature_net2 = nn.Conv2d(grow0, grow0, kernel_size, padding=(kernel_size-1)//2, stride=1)

        # Redidual dense blocks and dense feature fusion
        self.residual_dense_blocks = nn.ModuleList()
        for i in range(self.block_num):
            self.residual_dense_blocks.append(
                ResidualDenseBlock(grow0 = grow0, grow_rate = out_channel_num, num_conv_layers = conv_layer_num)
            )

        # Global Feature Fusion
        self.global_feature_fusion = nn.Sequential(*[
            nn.Conv2d(self.block_num * grow0, grow0, 1, padding=0, stride=1),
            nn.Conv2d(grow0, grow0, kernel_size, padding=(kernel_size-1)//2, stride=1)
        ])

        if no_upsample:
            self.out_dim = grow0
        else:
            self.out_dim = n_colors
            # Up-sampling net
            if r == 2 or r == 3:
                self.upsample_net = nn.Sequential(*[
                    nn.Conv2d(grow0, out_channel_num * r * r, kernel_size, padding=(kernel_size-1)//2, stride=1),
                    nn.PixelShuffle(r),
                    nn.Conv2d(out_channel_num, n_colors, kernel_size, padding=(kernel_size-1)//2, stride=1)
                ])
            elif r == 4:
                self.upsample_net = nn.Sequential(*[
                    nn.Conv2d(grow0, out_channel_num * 4, kernel_size, padding=(kernel_size-1)//2, stride=1),
                    nn.PixelShuffle(2),
                    nn.Conv2d(out_channel_num, out_channel_num * 4, kernel_size, padding=(kernel_size-1)//2, stride=1),
                    nn.PixelShuffle(2),
                    nn.Conv2d(out_channel_num, n_colors, kernel_size, padding=(kernel_size-1)//2, stride=1)
                ])
            else:
                raise ValueError("scale must be 2 or 3 or 4.")

    def forward(self, x):
        f1 = self.shallow_feature_net1(x)
        x = self.shallow_feature_net2(f1)

        residual_out = []
        for i in range(self.block_num):
            x = self.residual_dense_blocks[i](x)
            residual_out.append(x)

        x = self.global_feature_fusion(torch.cat(residual_out,1))
        x += f1

        if self.no_upsample:
            return x
        else:
            return self.upsample_net(x)

class FeatureMapEncoder(nn.Module):
    def __init__(self,input_nc=3,z_dim=64,bottom=False):
        super().__init__()
        self.bottom = bottom

        if self.bottom:
            self.enc_down_0 = nn.Sequential([
                nn.Conv2d(input_nc + 4,z_dim,3,stride=1,padding=1),
                nn.ReLU(True)])
        self.enc_down_1 = nn.Sequential(nn.Conv2d(z_dim if bottom else input_nc+4, z_dim, 3, stride=2 if bottom else 1,  padding=1),
                                        nn.ReLU(True))
        self.enc_down_2 = nn.Sequential(nn.Conv2d(z_dim, z_dim, 3, stride=2, padding=1),
                                        nn.ReLU(True))
        self.enc_down_3 = nn.Sequential(nn.Conv2d(z_dim, z_dim, 3, stride=2, padding=1),
                                        nn.ReLU(True))
        self.enc_up_3 = nn.Sequential(nn.Conv2d(z_dim, z_dim, 3, stride=1, padding=1),
                                      nn.ReLU(True),
                                      nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        self.enc_up_2 = nn.Sequential(nn.Conv2d(z_dim*2, z_dim, 3, stride=1, padding=1),
                                      nn.ReLU(True),
                                      nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        self.enc_up_1 = nn.Sequential(nn.Conv2d(z_dim * 2, z_dim, 3, stride=1, padding=1),
                                      nn.ReLU(True))

    def forward(self,x):
        """
        input:
            x: input image, [B,3,H,W]
        output:
            feature_map: [B,C,H,W]
        """
        W,H = x.shape[3], x.shape[2]
        X = torch.linspace(-1,1,W)
        Y = torch.linspace(-1,1,H)
        y1_m,x1_m = torch.meshgrid([Y,X])
        x2_m,y2_m = 2 - x1_m,2 - y1_m # Normalized distance in the four direction
        pixel_emb = torch.stack([x1_m,x2_m,y1_m,y2_m]).to(x.device).unsqueeze(0) # [1,4,H,W]
        pixel_emb = pixel_emb.repeat([x.size(0),1,1,1])
        inputs = torch.cat([x,pixel_emb],dim=1)

        if self.bottom:
            x_down_0 = self.enc_down_0(inputs)
            x_down_1 = self.enc_down_1(x_down_0)
        else:
            x_down_1 = self.enc_down_1(inputs)
        x_down_2 = self.enc_down_2(x_down_1)
        x_down_3 = self.enc_down_3(x_down_2)
        x_up_3 = self.enc_up_3(x_down_3)
        x_up_2 = self.enc_up_2(torch.cat([x_up_3, x_down_2], dim=1))
        feature_map = self.enc_up_1(torch.cat([x_up_2, x_down_1], dim=1))  # BxCxHxW
        return feature_map