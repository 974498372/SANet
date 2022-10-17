# Copyright (c) Jiewen Zhu. and UESTC.
"""
DBANet
"""
import warnings
from functools import reduce

import torch
from torch import nn

from .SAG import SAG_atten

warnings.filterwarnings('ignore', category=UserWarning)
from .BaseConv import BaseConv


class Focus(nn.Module):
    def __init__(self, in_channels, out_channels, ksize=1, stride=1, act="silu"):
        super().__init__()
        self.conv = BaseConv(in_channels * 4, out_channels, ksize, stride, act=act)

    def forward(self, x):
        patch_top_left  = x[...,  ::2,  ::2]
        patch_bot_left  = x[..., 1::2,  ::2]
        patch_top_right = x[...,  ::2, 1::2]
        patch_bot_right = x[..., 1::2, 1::2]
        
        x = torch.cat((patch_top_left, patch_bot_left, patch_top_right, patch_bot_right,), dim=1,)
        return self.conv(x)
    
    
class SPPBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=(5, 9, 13), activation="silu"):

        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1      = BaseConv(in_channels, hidden_channels, 1, stride=1, act=activation)
        self.m          = nn.ModuleList([nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2) for ks in kernel_sizes])
       
        conv2_channels  = hidden_channels * (len(kernel_sizes) + 1)
        self.conv2      = BaseConv(conv2_channels, out_channels, 1, stride=1, act=activation)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        x = self.conv2(x)
        return x


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, shortcut=True, expansion=0.5, act="silu",):

        super().__init__()
        hidden_channels = int(out_channels * expansion)
        Conv = BaseConv

        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = Conv(hidden_channels, out_channels, 3, stride=1, act=act)
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.use_add:
            y = y + x
        return y
    
class CSPLayer(nn.Module):
    def __init__(self, in_channels, out_channels, n=1, shortcut=True, expansion=0.5, act="silu",):

        # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        hidden_channels = int(out_channels * expansion)  

        self.conv1  = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)

        self.conv2  = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)

        self.conv3  = BaseConv(2 * hidden_channels, out_channels, 1, stride=1, act=act)

        module_list = [Bottleneck(hidden_channels, hidden_channels, shortcut, 1.0, act=act) for _ in range(n)]
        self.m      = nn.Sequential(*module_list)

    def forward(self, x):

        x_1 = self.conv1(x)
        x_2 = self.conv2(x)

        x_1 = self.m(x_1)
        x = torch.cat((x_1, x_2), dim=1)

        return self.conv3(x)
    
    
class SKConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1,M=2, r=16, L=32):
        super(SKConv, self).__init__()
        d = max(in_channels//r, L) 
        self.M = M
        self.out_channels = out_channels
        self.conv = nn.ModuleList() 
        for i in range(M):
            self.conv.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, stride, padding=1+i, dilation=1+i, groups=32, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))
        self.global_pool = nn.AdaptiveAvgPool2d(output_size=1) 
        self.fc1 = nn.Sequential(
            nn.Conv2d(out_channels, d, 1, bias=False),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
        )   
        self.fc2 = nn.Conv2d(d, out_channels*M, 1, 1, bias=False)  
        self.softmax = nn.Softmax(dim=1) 

    def forward(self, input):
        batch_size = input.size(0)
        output = []

        for i,conv in enumerate(self.conv):
            output.append(conv(input))   

        U = reduce(lambda x,y:x+y, output) 
        s = self.global_pool(U)  
        z = self.fc1(s) 
        a_b = self.fc2(z) 
        a_b = a_b.reshape(batch_size, self.M, self.out_channels, -1) 
        a_b = self.softmax(a_b)
        a_b = list(a_b.chunk(self.M, dim=1))  
        a_b = list(map(lambda x:x.reshape(batch_size,self.out_channels,1,1), a_b))

        V = list(map(lambda x,y:x*y, output, a_b))
        V = reduce(lambda x,y:x+y, V)
        return V
    
class SANet(nn.Module):
    def __init__(self, dep_mul, wid_mul, act="silu"):

        super().__init__()
        Conv = BaseConv
        base_channels = int(wid_mul * 64)
        base_depth = max(round(dep_mul * 3), 1)
        # [3, 640, 640] -> [64, 320, 320] -> [128, 160, 160]
        self.stem = nn.Sequential(
            Focus(3, base_channels, ksize=3, act=act),
            Conv(base_channels, 2*base_channels, 3, 2, act=act)
        )
        
        # [128, 160, 160] -> [128, 160, 160]
        # [128, 160, 160] -> [256, 80, 80]
        self.csp1 = nn.Sequential(
            Conv(2*base_channels, 2*base_channels, 3, 1, act=act),
            CSPLayer(2 * base_channels, 2 * base_channels, n=base_depth, act=act)
        )  
        self.csp2 = nn.Sequential(
            Conv(2*base_channels, 4*base_channels, 3, 2, act=act),
            CSPLayer(4 * base_channels, 4 * base_channels, n=3*base_depth, act=act)
        )
        
        # [128, 160, 160] -> [256, 80, 80]
        # [256, 80, 80] -> [256, 80, 80]
        self.SAG_1 = nn.Sequential(
            SAG_atten(2*base_channels),
            Conv(2*base_channels, 4*base_channels, 3, 2, act=act),
            SAG_atten(4*base_channels),
        )
        self.SAG_2 = nn.Sequential(
            Conv(4*base_channels, 4*base_channels, 3, 1, act=act),
            SAG_atten(4*base_channels),
        )
        
        # [512, 80, 80] -> [512, 40, 40]
        # [512, 40, 40] -> [1024, 20, 20]
        self.sk1 = nn.Sequential(
            Conv(8*base_channels, 8*base_channels, 3, 2, act=act),
            SKConv(8*base_channels, 8*base_channels),
            CSPLayer(8 * base_channels, 8 * base_channels, n=3*base_depth, act=act),
            SKConv(8*base_channels, 8*base_channels)
        )
        self.sk2 = nn.Sequential(
            Conv(8*base_channels, 16*base_channels, 3, 2, act=act),
            SKConv(16*base_channels, 16*base_channels),
            CSPLayer(16 * base_channels, 16 * base_channels, n=3*base_depth, act=act),
            SKConv(16*base_channels, 16*base_channels)
        )
        
        # SPP [1024, 20, 20] -> [1024, 20, 20]
        self.spp = SPPBottleneck(16*base_channels, 16*base_channels, activation=act)
        
        # upsample [1024, 20, 20] -> [512, 20, 20] -> [512, 40, 40]
        self.upsample1 = nn.Sequential(
            Conv(16*base_channels, 8*base_channels, 3, 1, act=act),
            nn.Upsample(scale_factor=2, mode='nearest')
        )
        
        # [1024, 40, 40] -> [512, 40, 40]
        self.csp3 = nn.Sequential(
            Conv(16*base_channels, 8*base_channels, 3, 1,act=act),
            CSPLayer(8*base_channels, 8*base_channels, act=act)
        )
        
        # upsample [512, 40, 40] -> [256, 40, 40] -> [256, 80, 80]
        self.upsample2 = nn.Sequential(
            Conv(8*base_channels, 4*base_channels, 3, 1, act=act),
            nn.Upsample(scale_factor=2, mode='nearest')
        )
        # [768, 80, 80] -> [512, 80, 80] -> [256, 80, 80]
        self.sk3 = nn.Sequential(
            Conv(12*base_channels, 8*base_channels, 3, 1, act=act),
            SKConv(8*base_channels, 8*base_channels),
            CSPLayer(8 * base_channels, 4 * base_channels, n=3*base_depth, act=act),
            SKConv(4*base_channels, 4*base_channels),
            
        )
        
    def forward(self, x):
        # [b, 6, 640, 640] -> [b, 128, 160, 160]
        x = self.stem(x)
         
        # [b, 128, 160, 160] -> [b, 128, 160, 160]
        left = self.csp1(x)
        # [b, 128, 160, 160] -> [b, 256, 80, 80]
        left = self.csp2(left)
        
        # [b, 128, 160, 160] -> [b, 256, 80, 80]
        right = self.SAG_1(x)
        
        # [b, 256, 80, 80] -> [b, 256, 80, 80]
        right = self.SAG_2(right)
        
        
        # [b, 256, 80, 80] + [b, 256, 80, 80] -> [b, 512, 80, 80]
        x = torch.cat([left, right], dim=1)
         
        # [b, 512, 80, 80] -> [b, 512, 40, 40]
        x = self.sk1(x)
        residual = x.clone()  
        # [b, 512, 40, 40] -> [b, 1024, 20, 20]
        x = self.sk2(x)
         
        # SPP [b, 1024, 20, 20]
        x = self.spp(x)
        
        # [b, 1024, 20, 20] -> [b, 512, 40, 40]
        x = self.upsample1(x)
        #[b, 512, 40, 40] + [b, 512, 40, 40] -> [b, 1024, 40, 40]
        x = torch.cat([x, residual], dim=1)
        # [b, 1024, 40, 40] -> [b, 512, 40, 40]
        x = self.csp3(x)
        
        # [b, 512, 40, 40] -> [b, 256, 80, 80]
        x = self.upsample2(x)
        # 3*[256, 80, 80] -> [768, 80, 80]
        x = torch.cat([x, left, right], dim=1)
        # [768, 80, 80] -> [256, 80, 80]
        x = self.sk3(x)
        
        return x
    
if __name__ == "__main__":
    # pass
    a = torch.randn((4,3,640,640))
    a = SANet(0.33, 0.5)(a)
    print(a.shape)