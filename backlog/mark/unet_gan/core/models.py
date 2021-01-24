""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F
import torch as th

from .parts import *

# from compas_struct_ml.ann.dense.parts import DenseLayer


class DenseLayer(nn.Module):
    def __init__(self, dim_in, dim_out, act_func=F.selu, batch_norm=True, **kwargs):
        super(DenseLayer, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out)
        self.batch_norm = batch_norm
        if self.batch_norm:
            self.batchnorm1d = nn.BatchNorm1d(dim_out)
        self.act_func = act_func
        
    def forward(self, X):
        X1 = self.linear(X)
        if self.act_func is not None:
            X1 = self.act_func(X1)
        if self.batch_norm:
            X1 = self.batchnorm1d(X1)
        return X1
        


class UNet(nn.Module):
    # def __init__(self, n_channels, n_classes):
    #     super(UNet, self).__init__()
    #     self.n_channels = n_channels
    #     self.n_classes = n_classes
    #     self.bilinear = bilinear

    #     self.inc = DoubleConv(n_channels, 64)
    #     self.down1 = Down(64, 128)
    #     self.down2 = Down(128, 256)
    #     self.down3 = Down(256, 512)
    #     # factor = 2 if bilinear else 1
    #     self.down4 = Down(512, 1024 // factor)
    #     self.up1 = Up(1024, 512 // factor, bilinear)
    #     self.up2 = Up(512, 256 // factor, bilinear)
    #     self.up3 = Up(256, 128 // factor, bilinear)
    #     self.up4 = Up(128, 64, bilinear)

    #     self.outc = OutConv(64, n_classes)

    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = Conv(n_channels, 64) # in [3, 256, 256] out [64, 256, 256]
        self.down1 = Down(64, 128)  # in [64, 256, 256] out [128, 128, 128]
        self.down2 = Down(128, 256)  # in [128, 128, 128] out [256, 64, 64]
        self.down3 = Down(256, 512)  # in [256, 64, 64] out [512, 32, 32]
        self.down4 = Down(512, 512)  # in [512, 32, 32] out [512, 16, 16]
        self.down5 = Down(512, 512)  # in [512, 16, 16] out [512, 8, 8]
        self.down6 = Down(512, 512)  # in [512, 8, 8] out [512, 4, 4]
        self.up1 = Up(512 * 2, 512)  # in [512 + 512, 8, 8] out [512, 16, 16]
        self.up2 = Up(512 * 2, 512)  # in [512 + 512, 16, 16] out [512, 32, 32]
        self.up3 = Up(512 * 2, 256)  # in [512 + 512, 32, 32] out [256, 64, 64]
        self.up4 = Up(256 * 2, 128)  # in [256 + 256, 64, 64] out [128, 128, 128]
        self.up5 = Up(128 * 2, 64)  # in [128 + 128, 128, 128] out [64, 256, 256]
        self.up6 = Up(64 * 2, 64)
        # self.lastc = Conv(64 * 2, 64)  # in [64 + 64, 256, 256] out [64, 256, 256]
        self.outc = OutConv(64, n_classes)  # in [64, 256, 256] out [256, 256]

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x7 = self.down6(x6)
        x_1 = self.up1(x7, x6)
        x_2 = self.up2(x_1, x5)
        x_3 = self.up3(x_2, x4)
        x_4 = self.up4(x_3, x3)
        x_5 = self.up5(x_4, x2)
        x_6 = self.up6(x_5, x1)
        x_l = self.outc(x_6)
        return th.sigmoid(x_l)

class ImageToBinary(nn.Module):
    def __init__(self, n_channels_in, n_channels_out=1):
        super(ImageToBinary, self).__init__()
        self.n_channels_in = n_channels_in

        self.inc = Conv(n_channels_in, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.outc = OutConv(1024, n_channels_out)
        self.outl = DenseLayer(256, 1, batch_norm=False)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.outc(x5)
        x6 = x6.reshape(x5.shape[0], x5.shape[2] * x5.shape[3])
        return th.sigmoid(self.outl(x6))

