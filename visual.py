from argparse import ArgumentParser
from typing import Callable, Dict, Iterable

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from nets.SA import SABody
from utils.utils import cvtColor, preprocess_input, resize_image


def parse_args():
    # Setting parameters
    parser = ArgumentParser()
    
    parser.add_argument('model_path', type=str, help='The path of weight')
    parser.add_argument('image_path', type=str, help='The path of picture')
    parser.add_argument('-g','--Cuda', action="store_true", help='if use cuda')
    parser.add_argument('--input-shape', type=int, default=640, help='size of image')
    

    args = parser.parse_args()
    return args

class FeatureExtractor(nn.Module):
    def __init__(self, model: nn.Module, layers: Iterable[str]):
        super().__init__()
        self.model = model
        self.layers = layers
        self._features = {layer: torch.empty(0) for layer in layers}

        for layer_id in layers:
            layer = dict([*self.model.named_modules()])[layer_id]
            layer.register_forward_hook(self.save_outputs_hook(layer_id))

    def save_outputs_hook(self, layer_id: str) -> Callable:
        def fn(_, __, output):
            self._features[layer_id] = output
        return fn

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        _ = self.model(x)
        return self._features

def preprocess_image(image, input_shape):
    # image_shape = np.array(np.shape(image)[0:2])
    image       = cvtColor(image)
    image_data  = resize_image(image, (input_shape[1],input_shape[0]),letterbox_image=False)
    image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

    return image_data


if __name__ == "__main__":
    args = parse_args()
    model_path = args.model_path  # "logs\weight.pth"
    cuda = args.Cuda
    input_shape = [args.input_shape, args.input_shape]  # [640, 640]
    image_path = args.image_path           # "VOCdevkit/VOC2007/JPEGImages/Misc_5.png"

    # model
    net    = SABody(num_classes = 1)
    if cuda:
        device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device      = torch.device('cpu')
    net.load_state_dict(torch.load(model_path, map_location=device))

    net    = net.eval()
    print('{} model, and classes loaded.'.format(model_path))

    if cuda:
        net = torch.nn.DataParallel(net)
        net = net.cuda()

    # image
    image = Image.open(image_path)
    image = preprocess_image(image, input_shape)
    image = torch.from_numpy(image)
    resnet_features = FeatureExtractor(net, layers=["backbone.SAG_2"])
    features = resnet_features(image)
    features = features['backbone.SAG_2'] # 尺度大小，如：torch.Size([1,80,45,45])
    # 1.2 每个通道对应元素求和
    heatmap = torch.sum(features, dim=1)  # 尺度大小， 如torch.Size([1,45,45])
    max_value = torch.max(heatmap)
    min_value = torch.min(heatmap)
    heatmap = (heatmap-min_value)/(max_value-min_value)*255
    heatmap = heatmap.cpu().detach().numpy().astype(np.uint8).transpose(1,2,0)  # 尺寸大小，如：(45, 45, 1)
    src_size = (125,125)  # 原图尺寸大小
    heatmap = cv2.resize(heatmap, src_size,interpolation=cv2.INTER_LINEAR)  # 重整图片到原尺寸
    heatmap=cv2.applyColorMap(heatmap,cv2.COLORMAP_JET)
    # 保存热力图
    cv2.imshow('heatmap',heatmap)
    cv2.imwrite('Misc_317.jpg', heatmap)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    

