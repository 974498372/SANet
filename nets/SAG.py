import torch
import torch.nn as nn

from .BaseConv import BaseConv


class mlp(nn.Module):

    def __init__(self, in_channels, hidden_channels=None, out_channels=None, act_layer=nn.GELU, drop=0.):
        super().__init__()

        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.fc1 = BaseConv(in_channels, hidden_channels, 1, 1)
        self.fc2 = BaseConv(hidden_channels, out_channels, 1, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class SAG_atten(nn.Module):
    def __init__(self, dim, bias=False, proj_drop=0.):
        super().__init__()

        self.fc1 = BaseConv(dim, dim, 1, 1, bias=bias)  
        
        self.fc2 = BaseConv(dim, dim, 3, 1, bias=bias)  
        
        self.mix = BaseConv(2*dim, dim, 3, 1, bias=bias) 
        self.reweight = mlp(dim, dim, dim)
        

    def forward(self, x):
        B, C, H, W = x.shape
        residual = x
        x_1 = x.clone()
        
        
        # 负数为零
        x_1[x_1<0] = 0
        for i in range(B):
            for j in range(C):
                mean = x_1[i,j,:,:].mean() 
                x_1[i,j,:,:] = x[i,j,:,:]/(mean + 1e-4)   

        x_1 = self.fc1(x_1)
        
        x_2 = self.fc2(x)
        
        x_1 = self.mix(torch.cat([x_1, x_2], dim=1))
        x_1 = self.reweight(x_1)
        
        x = residual * x_1
        return x