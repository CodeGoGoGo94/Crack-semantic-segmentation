# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ptflops import get_model_complexity_info
import numpy as np
from torchvision.models import vgg16


"""
Conv + BN + RL (*2) + Drop
"""

class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(),
            nn.Conv2d(ch_out, ch_out, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(),
            nn.Dropout2d(p = 0.2)
            )
        
    def forward(self, x):
        x = self.conv(x)
        return x
     
        
"""
Upsample + Conv + BN + RL
"""       

class up_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(ch_in, ch_out, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU()
            )
    def forward(self,x):
        x = self.conv(x)
        return x
    
    
"""
U-net 5 (the depth of U-net)
"""

class Unet_5(nn.Module):
    def __init__(self, img_ch = 3, out_ch = 1):
        super(Unet_5, self).__init__()
        self.down0 = conv_block(3, 64)
        self.down1 = conv_block(64, 128)
        self.down2 = conv_block(128, 256)
        self.down3 = conv_block(256, 512)
        self.down4 = conv_block(512, 1024)
        
        self.down_sample = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        self.up4 = up_block(1024, 512)
        self.up_con4 = conv_block(512+512, 512)
        
        self.up3 = up_block(512, 256)
        self.up_con3 = conv_block(256+256, 256)
        
        self.up2 = up_block(256, 128)
        self.up_con2 = conv_block(128+128, 128)
        
        self.up1 = up_block(128, 64)
        self.up_con1 = conv_block(64+64, 64)   
        
        self.up0 = nn.Conv2d(64,1,kernel_size=1,stride=1,padding=0)  
       
        self.conv64 = nn.Conv2d(64,1,kernel_size=1,stride=1,padding=0)
        self.conv128 = nn.Conv2d(128,1,kernel_size=1,stride=1,padding=0)
        self.conv256 = nn.Conv2d(256,1,kernel_size=1,stride=1,padding=0)
        self.conv512 = nn.Conv2d(512,1,kernel_size=1,stride=1,padding=0)
        self.conv1024 = nn.Conv2d(1024,1,kernel_size=1,stride=1,padding=0)
        
    def forward(self,x):
        d0 = self.down0(x) # 3 -> 64
        
        d1 = self.down_sample(d0) # 64 -> 64
        d1 = self.down1(d1) # 64 -> 128
        
        d2 = self.down_sample(d1) # 128 -> 128
        d2 = self.down2(d2) # 128 -> 256
        
        d3 = self.down_sample(d2) # 256 -> 256
        d3 = self.down3(d3) # 256 -> 512
        
        d4 = self.down_sample(d3) # 512 -> 512
        d4 = self.down4(d4) # 512 -> 1024     
        
        
        u4 = self.up4(d4) # 1024 -> 512 
        c4 = torch.cat((d3,u4), dim = 1) # 512 + 512
        c4 = self.up_con4(c4) # 1024 -> 512 
        
        u3 = self.up3(c4) # 512 -> 256 
        c3 = torch.cat((d2,u3), dim = 1) # 256 + 256
        c3 = self.up_con3(c3) # 512 -> 256 
            
        u2 = self.up2(c3) # 256 -> 128
        c2 = torch.cat((d1,u2), dim = 1) # 128 + 128
        c2 = self.up_con2(c2) # 256 -> 128
        
        u1 = self.up1(c2) # 128 -> 64
        c1 = torch.cat((d0,u1), dim = 1) # 64 + 64
        c1 = self.up_con1(c1) # 128 -> 64
        
        o = self.up0(c1) # 64 -> 1
        return o
        #return self.conv64(d0),self.conv128(d1),self.conv256(d2),self.conv512(d3),self.conv1024(d4),self.conv512(u4),self.conv256(u3),self.conv128(u2),self.conv64(u1),self.conv512(c4),self.conv256(c3),self.conv128(c2),self.conv64(c1), o

"""
FCN - 8s block
"""

class FCN_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(FCN_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU()
            )
        
    def forward(self, x):
        x = self.conv(x)
        return x    


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        #logits = F.sigmoid(logits)
        #logits = F.softmax(logits)
        return logits
 
"""
Dilated block (1-3-9)
"""
class Dilated_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Dilated_block, self).__init__()
        self.dia1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
            )
        self.dia2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=3, dilation=3, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
            )
        self.dia3 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=9, dilation=9, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
            )
        self.fuse = nn.Sequential(
            nn.Conv2d(in_ch+3*out_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
            )
        
    def forward(self,x):
        x1 = self.dia1(x)
        x2 = self.dia2(x1)
        x3 = self.dia3(x2)
        out = torch.cat((x,x1,x2,x3), dim = 1)
        out = self.fuse(out)
        return out    


"""
Dilated block (1-3-9) used for output images of each layer
"""
class Dilated_3_img(nn.Module):
    def __init__(self):
        super(Dilated_3_img, self).__init__()
        self.d11 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
            )
        self.d12 = nn.Sequential(
            nn.Conv2d(35, 32, kernel_size=3, padding=3, dilation=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
            )
        self.d13 = nn.Sequential(
            nn.Conv2d(67, 32, kernel_size=3, padding=9, dilation=9, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
            )
        self.f1 = nn.Sequential(
            nn.Conv2d(3+3*32, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
            )
        
        self.d21 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
            )
        self.d22 = nn.Sequential(
            nn.Conv2d(96, 64, kernel_size=3, padding=3, dilation=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
            )
        self.d23 = nn.Sequential(
            nn.Conv2d(160, 64, kernel_size=3, padding=9, dilation=9, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
            )
        self.f2 = nn.Sequential(
            nn.Conv2d(32+3*64, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
            )
        
        self.d31 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
            )
        self.d32 = nn.Sequential(
            nn.Conv2d(192, 128, kernel_size=3, padding=3, dilation=3, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
            )
        self.d33 = nn.Sequential(
            nn.Conv2d(320, 128, kernel_size=3, padding=9, dilation=9, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
            )
        self.f3 = nn.Sequential(
            nn.Conv2d(64+3*128, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
            )
       
        self.conv64 = nn.Conv2d(64,1,kernel_size=1,stride=1,padding=0)
        self.conv128 = nn.Conv2d(128,1,kernel_size=1,stride=1,padding=0)
        self.conv32 = nn.Conv2d(32,1,kernel_size=1,stride=1,padding=0)
        self.conv448 = nn.Conv2d(64+3*128,1,kernel_size=1,stride=1,padding=0)
        self.conv224 = nn.Conv2d(32+64+128,1,kernel_size=1,stride=1,padding=0)
        self.conv256 = nn.Conv2d(256,1,kernel_size=1,stride=1,padding=0)
        self.conv320 = nn.Conv2d(320,1,kernel_size=1,stride=1,padding=0)
        self.conv352 = nn.Conv2d(352,1,kernel_size=1,stride=1,padding=0)
        self.conv512 = nn.Conv2d(512,1,kernel_size=1,stride=1,padding=0)
        self.convf = nn.Conv2d(128,1,kernel_size=1,stride=1,padding=0)
        self.sp = sp_block()
        self.nonl = nn.Sequential(
            nn.MaxPool2d(kernel_size = 4, stride = 4, padding = 0),
            #NonLocalBlock2(128*2,mask1,mask2),
            NonLocalBlock(128),
            conv_block(128,128),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 128, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(p = 0.2)
            #nn.AdaptiveAvgPool2d((224, 224)) 
            )
        self.se = SE(128, 2)
        self.f3fu = nn.Conv2d(128*3,128,kernel_size=1,stride=1,padding=0)
        
        self.conv_non_conv = nn.Sequential( 
            conv_block(3, 64),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            conv_block(64, 128),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            #NonLocalBlock2(64,mask1,mask2),
            conv_block(128,128),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 128, kernel_size = 3, stride = 1, padding = 1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.BatchNorm2d(128),
            nn.ReLU()
            )
    def forward(self,x):
        d11 = self.d11(x)
        d12 = self.d12(torch.cat((x,d11), dim = 1))
        d13 = self.d13(torch.cat((x,d11,d12), dim = 1))
        f1 = torch.cat((x,d11,d12,d13), dim = 1)
        f1 = self.f1(f1)
        
        d21 = self.d21(f1)
        d22 = self.d22(torch.cat((f1,d21), dim = 1))
        d23 = self.d23(torch.cat((f1,d21,d22), dim = 1))
        f2 = torch.cat((f1,d21,d22,d23), dim = 1)
        f2 = self.f2(f2)
        
        d31 = self.d31(f2)
        d32 = self.d32(torch.cat((f2,d31), dim = 1))
        d33 = self.d33(torch.cat((f2,d31,d32), dim = 1))
        f34 = torch.cat((f2,d31,d32,d33), dim = 1)
        f3 = self.f3(f34)
        
        add3 = self.conv_non_conv(x)
        f3 = torch.add(f3,add3)
        add_f3 = torch.cat((add3,f3), dim = 1)
        
        f3_sp = self.sp(f3)
        f3_se = self.se(f3)
        f3_nonl = self.nonl(f3)
        f3_f = self.f3fu(torch.cat((f3_sp,f3_se,f3_nonl), dim = 1))
        
        ##f4 = f3_f
        out = self.convf(f3_f)
        ##return out, self.conv64(d13), self.conv64(f1), self.conv128(d23), self.conv128(f2), self.conv128(d33), self.conv512(f34),self.conv128(f3), self.conv128(add3), self.conv256(add_f3), self.conv256(f3_se),self.conv256(f3_sp), self.conv256(f3_nonl), self.conv128(f3_f)
        return out

"""
Dilated block (1-5, k=5)
"""
class Dilated_block3(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Dilated_block3, self).__init__()
        self.dia1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=5, padding=2, dilation=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
            )
        self.dia2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=5, padding=10, dilation=5, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
            )
        self.fuse = nn.Sequential(
            nn.Conv2d(in_ch+2*out_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
            )
       
    def forward(self,x):
        x1 = self.dia1(x)
        x2 = self.dia2(x1)
        out = torch.cat((x,x1,x2), dim = 1)
        out = self.fuse(out)
        return out    

"""
Dilated block (1-7, k=7)
"""
class Dilated_block4(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Dilated_block4, self).__init__()
        self.dia1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=7, padding=3, dilation=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
            )
        self.dia2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=7, padding=21, dilation=7, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
            )
        self.fuse = nn.Sequential(
            nn.Conv2d(in_ch+2*out_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
            )
    def forward(self,x):
        x1 = self.dia1(x)
        x2 = self.dia2(x1)
        out = torch.cat((x,x1,x2), dim = 1)
        out = self.fuse(out)
        return out  
    
"""
Dilated block 2 (1-3-9-27)
"""
class Dilated_block2(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Dilated_block2, self).__init__()
        self.dia1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
            )
        self.dia2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=3, dilation=3, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
            )
        self.dia3 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=9, dilation=9, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
            )
        self.dia4 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=27, dilation=27, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
            )
        self.dim = nn.Sequential(
            nn.Conv2d(4*out_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
            )
        #self.out = nn.Conv2d(out_ch, 1, kernel_size=1)
    
    def forward(self,x):
        x1 = self.dia1(x)
        x2 = self.dia2(x1)
        x3 = self.dia3(x2)
        x4 = self.dia3(x3)
        out = torch.cat((x1,x2,x3,x4), dim = 1)
        out = self.dim(out)
        return out    

        
"""
Dilated conv - doubel
"""
class Dilated_2(nn.Module):
    def __init__(self):
        super(Dilated_2, self).__init__()
        self.dia_2 = nn.Sequential(
            Dilated_block(3,64),
            Dilated_block(64,128),
            nn.Conv2d(128, 1, kernel_size=1)
            )
    def forward(self,x):
        x1 = self.dia_2(x)
        return x1

"""
Dilated conv - triple
"""
class Dilated_3(nn.Module):
    def __init__(self):
        super(Dilated_3, self).__init__()
        self.dia_3 = nn.Sequential(
            Dilated_block(3,64),
            Dilated_block(64,128),
            Dilated_block(128,64),
            nn.Conv2d(64, 1, kernel_size=1)
            )
    def forward(self,x):
        x1 = self.dia_3(x)
        return x1

"""
Dilated conv - triple - fusion 3 blocks
"""
class Dilated_3_f(nn.Module):
    def __init__(self):
        super(Dilated_3_f, self).__init__()
        self.dia1 = Dilated_block(3,64)
        self.dia2 = Dilated_block(64,128)
        self.dia3 = Dilated_block(128,64)
        self.conv1 = nn.Conv2d(64, 32, kernel_size=1)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=1)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=1)
        self.conv4 = nn.Conv2d(64*3, 1, kernel_size=1)  
        self.drop = nn.Dropout2d(p=0.2) 
    def forward(self,x):
        x1 = self.dia1(x)
        x2 = self.dia2(x1)
        x3 = self.dia3(x2)
        x1 = (self.conv1(x1))
        x2 = (self.conv2(x2))
        x3 = (self.conv3(x3))
        x4 = torch.cat((x1,x2,x3), dim = 1)
        x4 = (self.conv4(x4))
        return x4

"""
SE Block block

"""
class SE(nn.Module):

    def __init__(self, in_chnls, ratio):
        super(SE, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
        self.compress = nn.Conv2d(in_chnls, in_chnls // ratio, 1, 1, 0)
        self.excitation = nn.Conv2d(in_chnls // ratio, in_chnls, 1, 1, 0)

    def forward(self, x):
        out = self.squeeze(x)
        out = self.compress(out)
        out = F.relu(out)
        out = self.excitation(out)
        return x*F.sigmoid(out)
    
class SegNet(nn.Module):
    def __init__(self, classes=1):
        super(SegNet, self).__init__()

        self.conv11 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn11 = nn.BatchNorm2d(64)
        self.conv12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn12 = nn.BatchNorm2d(64)

        self.conv21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn21 = nn.BatchNorm2d(128)
        self.conv22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn22 = nn.BatchNorm2d(128)

        self.conv31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn31 = nn.BatchNorm2d(256)
        self.conv32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn32 = nn.BatchNorm2d(256)
        self.conv33 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn33 = nn.BatchNorm2d(256)

        self.conv41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn41 = nn.BatchNorm2d(512)
        self.conv42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn42 = nn.BatchNorm2d(512)
        self.conv43 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn43 = nn.BatchNorm2d(512)

        self.conv51 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn51 = nn.BatchNorm2d(512)
        self.conv52 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn52 = nn.BatchNorm2d(512)
        self.conv53 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn53 = nn.BatchNorm2d(512)

        self.conv53d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn53d = nn.BatchNorm2d(512)
        self.conv52d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn52d = nn.BatchNorm2d(512)
        self.conv51d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn51d = nn.BatchNorm2d(512)

        self.conv43d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn43d = nn.BatchNorm2d(512)
        self.conv42d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn42d = nn.BatchNorm2d(512)
        self.conv41d = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.bn41d = nn.BatchNorm2d(256)

        self.conv33d = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn33d = nn.BatchNorm2d(256)
        self.conv32d = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn32d = nn.BatchNorm2d(256)
        self.conv31d = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn31d = nn.BatchNorm2d(128)

        self.conv22d = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn22d = nn.BatchNorm2d(128)
        self.conv21d = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn21d = nn.BatchNorm2d(64,)

        self.conv12d = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn12d = nn.BatchNorm2d(64)
        self.conv11d = nn.Conv2d(64, classes, kernel_size=3, padding=1)

    def forward(self, x):
        # Stage 1
        x11 = F.relu(self.bn11(self.conv11(x)))
        x12 = F.relu(self.bn12(self.conv12(x11)))
        x1_size = x12.size()
        x1p, id1 = F.max_pool2d(x12, kernel_size=2, stride=2, return_indices=True)

        # Stage 2
        x21 = F.relu(self.bn21(self.conv21(x1p)))
        x22 = F.relu(self.bn22(self.conv22(x21)))
        x2_size = x22.size()
        x2p, id2 = F.max_pool2d(x22, kernel_size=2, stride=2, return_indices=True)

        # Stage 3
        x31 = F.relu(self.bn31(self.conv31(x2p)))
        x32 = F.relu(self.bn32(self.conv32(x31)))
        x33 = F.relu(self.bn33(self.conv33(x32)))
        x3_size = x33.size()
        x3p, id3 = F.max_pool2d(x33, kernel_size=2, stride=2, return_indices=True)

        # Stage 4
        x41 = F.relu(self.bn41(self.conv41(x3p)))
        x42 = F.relu(self.bn42(self.conv42(x41)))
        x43 = F.relu(self.bn43(self.conv43(x42)))
        x4_size = x43.size()
        x4p, id4 = F.max_pool2d(x43, kernel_size=2, stride=2, return_indices=True)

        # Stage 5
        x51 = F.relu(self.bn51(self.conv51(x4p)))
        x52 = F.relu(self.bn52(self.conv52(x51)))
        x53 = F.relu(self.bn53(self.conv53(x52)))
        x5_size = x53.size()
        x5p, id5 = F.max_pool2d(x53, kernel_size=2, stride=2, return_indices=True)

        # Stage 5d
        x5d = F.max_unpool2d(x5p, id5, kernel_size=2, stride=2, output_size=x5_size)
        x53d = F.relu(self.bn53d(self.conv53d(x5d)))
        x52d = F.relu(self.bn52d(self.conv52d(x53d)))
        x51d = F.relu(self.bn51d(self.conv51d(x52d)))

        # Stage 4d
        x4d = F.max_unpool2d(x51d, id4, kernel_size=2, stride=2, output_size=x4_size)
        x43d = F.relu(self.bn43d(self.conv43d(x4d)))
        x42d = F.relu(self.bn42d(self.conv42d(x43d)))
        x41d = F.relu(self.bn41d(self.conv41d(x42d)))

        # Stage 3d
        x3d = F.max_unpool2d(x41d, id3, kernel_size=2, stride=2, output_size=x3_size)
        x33d = F.relu(self.bn33d(self.conv33d(x3d)))
        x32d = F.relu(self.bn32d(self.conv32d(x33d)))
        x31d = F.relu(self.bn31d(self.conv31d(x32d)))

        # Stage 2d
        x2d = F.max_unpool2d(x31d, id2, kernel_size=2, stride=2, output_size=x2_size)
        x22d = F.relu(self.bn22d(self.conv22d(x2d)))
        x21d = F.relu(self.bn21d(self.conv21d(x22d)))

        # Stage 1d
        x1d = F.max_unpool2d(x21d, id1, kernel_size=2, stride=2, output_size=x1_size)
        x12d = F.relu(self.bn12d(self.conv12d(x1d)))
        x11d = self.conv11d(x12d)

        return x11d
"""
https://github.com/sniklaus/pytorch-hed/blob/master/run.py
HED
"""
class HED(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.netVggOne = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.netVggTwo = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.netVggThr = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.netVggFou = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.netVggFiv = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.netScoreOne = torch.nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.netScoreTwo = torch.nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.netScoreThr = torch.nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.netScoreFou = torch.nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.netScoreFiv = torch.nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, stride=1, padding=0)

        self.netCombine = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=5, out_channels=1, kernel_size=1, stride=1, padding=0)
        )

        #self.load_state_dict({ strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in torch.hub.load_state_dict_from_url(url='http://content.sniklaus.com/github/pytorch-hed/network-' + arguments_strModel + '.pytorch', file_name='hed-' + arguments_strModel).items() })
    # end

    def forward(self, tenInput):
        #tenInput = tenInput * 255.0
        #tenInput = tenInput - torch.tensor(data=[104.00698793, 116.66876762, 122.67891434], dtype=tenInput.dtype, device=tenInput.device).view(1, 3, 1, 1)

        tenVggOne = self.netVggOne(tenInput)
        tenVggTwo = self.netVggTwo(tenVggOne)
        tenVggThr = self.netVggThr(tenVggTwo)
        tenVggFou = self.netVggFou(tenVggThr)
        tenVggFiv = self.netVggFiv(tenVggFou)

        tenScoreOne = self.netScoreOne(tenVggOne)
        tenScoreTwo = self.netScoreTwo(tenVggTwo)
        tenScoreThr = self.netScoreThr(tenVggThr)
        tenScoreFou = self.netScoreFou(tenVggFou)
        tenScoreFiv = self.netScoreFiv(tenVggFiv)

        tenScoreOne = torch.nn.functional.interpolate(input=tenScoreOne, size=(tenInput.shape[2], tenInput.shape[3]), mode='bilinear', align_corners=False)
        tenScoreTwo = torch.nn.functional.interpolate(input=tenScoreTwo, size=(tenInput.shape[2], tenInput.shape[3]), mode='bilinear', align_corners=False)
        tenScoreThr = torch.nn.functional.interpolate(input=tenScoreThr, size=(tenInput.shape[2], tenInput.shape[3]), mode='bilinear', align_corners=False)
        tenScoreFou = torch.nn.functional.interpolate(input=tenScoreFou, size=(tenInput.shape[2], tenInput.shape[3]), mode='bilinear', align_corners=False)
        tenScoreFiv = torch.nn.functional.interpolate(input=tenScoreFiv, size=(tenInput.shape[2], tenInput.shape[3]), mode='bilinear', align_corners=False)

        return self.netCombine(torch.cat([ tenScoreOne, tenScoreTwo, tenScoreThr, tenScoreFou, tenScoreFiv ], 1))
    # end
# end
def vgg_block(num_convs, in_channels, out_channels):
    blk = []
    for i in range(num_convs):
        if i == 0:
            blk.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        else:
            blk.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        blk.append(nn.ReLU(inplace=True))
    blk.append(nn.MaxPool2d(kernel_size=2, stride=2))  # 宽高减半
    return blk

class VGG16(nn.Module):
    def __init__(self, pretrained=True):
        super(VGG16, self).__init__()
        features = []
        features.extend(vgg_block(2, 3, 64))
        features.extend(vgg_block(2, 64, 128))
        features.extend(vgg_block(3, 128, 256))
        self.index_pool3 = len(features)
        features.extend(vgg_block(3, 256, 512))
        self.index_pool4 = len(features)
        features.extend(vgg_block(3, 512, 512))
        self.features = nn.Sequential(*features)

        self.conv6 = nn.Conv2d(512, 4096, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv7 = nn.Conv2d(4096, 4096, kernel_size=1)

        # load pretrained params from torchvision.models.vgg16(pretrained=True)
        if pretrained:
            pretrained_model = vgg16(pretrained=pretrained)
            pretrained_params = pretrained_model.state_dict()
            keys = list(pretrained_params.keys())
            new_dict = {}
            for index, key in enumerate(self.features.state_dict().keys()):
                new_dict[key] = pretrained_params[keys[index]]
            self.features.load_state_dict(new_dict)

    def forward(self, x):
        pool3 = self.features[:self.index_pool3](x)      # 1/8
        pool4 = self.features[self.index_pool3:self.index_pool4](pool3)  # 1/16
        pool5 = self.features[self.index_pool4:](pool4)  # 1/32

        conv6 = self.relu(self.conv6(pool5))  # 1/32
        conv7 = self.relu(self.conv7(conv6))  # 1/32

        return pool3, pool4, conv7

class FCN(nn.Module):
    def __init__(self, num_classes=1, backbone='vgg'):
        super(FCN, self).__init__()
        if backbone == 'vgg':
            self.features = VGG16()

        self.scores1 = nn.Conv2d(4096, num_classes, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.scores2 = nn.Conv2d(512, num_classes, kernel_size=1)
        self.scores3 = nn.Conv2d(256, num_classes, kernel_size=1)

        self.upsample_8x = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=8, stride=8)
        self.upsample_4x = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=4)
        self.upsample_2x = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=2, stride=2)
        self.resize1 = nn.AdaptiveAvgPool2d((45,80))
        self.resize2 = nn.AdaptiveAvgPool2d((80,45))
        
    def forward(self, x):
        pool3, pool4, conv7 = self.features(x)

        conv7 = self.relu(self.scores1(conv7))  # 

        pool4 = self.relu(self.scores2(pool4))  # 

        pool3 = self.relu(self.scores3(pool3))  # 
      
        if pool3.size() == self.upsample_2x(pool4).size() and pool3.size() == self.upsample_4x(conv7).size():
            s = pool3 + self.upsample_2x(pool4) + self.upsample_4x(conv7)  # 
        elif pool3.size()[2] < pool3.size()[3]:
            s = pool3 + self.resize1(self.upsample_2x(pool4)) + self.resize1(self.upsample_4x(conv7))  # 
        elif pool3.size()[2] > pool3.size()[3]:
            s = pool3 + self.resize2(self.upsample_2x(pool4)) + self.resize2(self.upsample_4x(conv7)) 
        
        out_8s = self.upsample_8x(s)  # 

        return out_8s




