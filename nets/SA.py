# Copyright (c) Jiewen Zhu. and UESTC.
import torch
import torch.nn as nn

from .BaseConv import BaseConv
from .SANet import SANet


class SAHead(nn.Module):
    def __init__(self, num_calsses, width=1.0, in_channels = 256, act='silu'):
        super().__init__()
        
        self.cls_convs = nn.Sequential(
            BaseConv(int(in_channels*width), int(in_channels*width), 3, 1, act=act),
            BaseConv(int(in_channels*width), int(in_channels*width), 3, 1, act=act)
        )
        # class
        self.cls_preds = nn.Conv2d(in_channels=int(in_channels*width), out_channels=num_calsses, kernel_size=1, stride=1, padding=0)
        
        self.reg_convs = nn.Sequential(
            BaseConv(int(in_channels*width), int(in_channels*width), 3, 1, act=act),
            BaseConv(int(in_channels*width), int(in_channels*width), 3, 1, act=act)
        )
        # regression
        self.reg_preds = nn.Conv2d(in_channels=int(in_channels*width), out_channels=4, kernel_size=1, stride=1, padding=0)
        
        # object
        self.obj_preds = nn.Conv2d(in_channels=int(in_channels*width), out_channels=1, kernel_size=1, stride=1, padding=0)
        
    def forward(self, inputs):
        # inputs [b, 256, 80, 80]
        outputs = []
        # [b, 256, 80, 80] -> [b, 256, 80, 80]
        cls_feat = self.cls_convs(inputs)
        # [b, 256, 80, 80] -> [b, num_classes, 80, 80]
        cls_output = self.cls_preds(cls_feat)
        
        # [b, 256, 80, 80] -> [b, 256, 80, 80]
        reg_feat = self.reg_convs(inputs)
        # [b, 256, 80, 80] -> [b, 4, 80, 80]
        reg_output = self.reg_preds(reg_feat)
        
        # [b, 256, 80, 80] -> [b, 1, 80, 80]
        obj_output = self.obj_preds(reg_feat)
        
        # [b, 4, 80, 80] + [b, 1, 80, 80] + [b, num_classes, 80, 80] -> [b, 5+num_classes, 80, 80]
        output = torch.cat([reg_output, obj_output, cls_output], dim=1)
        outputs.append(output)
        
        return outputs
    
class SABody(nn.Module):
    def __init__(self, num_classes):
       
        super().__init__()
        # depth {0.33, 0.67, 1.0}
        # width {0.5, 0.75, 1.0}
        depth, width = 0.33, 0.50
        self.backbone = SANet(depth, width)
        self.head = SAHead(num_classes, width)
        
    def forward(self, x):
        x = self.backbone(x)
        outputs = self.head(x)
        return outputs
    
if __name__ == "__main__":
    a = torch.randn((4,3,640,640))
    a = SABody(20, 's')(a)
    print(a[0].shape)
        