# import tensorflow as tf
# from tensorflow.keras import Model
# from tensorflow.keras.layers import Dense, GlobalAvgPool2D

import torch
import torchvision 
from torchvision.models import mobilenet_v2

import torch.nn.functional as F 
import torch.nn as nn


class LDRNet(nn.Module):
    # input_shape = [channels, x, y]
    def __init__(self, input_shapes=[3, 224, 224], points_size=100, classification_list=[1], alpha=0.35):
        super(LDRNet, self).__init__()
        self.classification_list = classification_list
        self.points_size = points_size
        self.input_shapes = input_shapes

        self.base_model = nn.Sequential(*list(mobilenet_v2(width_mult=alpha).children())[:-1])
        if len(classification_list) > 0:
            class_size = sum(self.classification_list)
        else:
            class_size = 0

        self.corner = OutputBranch(1280, 8, "output_corner")
        self.border = OutputBranch(1280, (points_size - 4) * 2, "output_border")
        self.cls = OutputBranch(1280, class_size, "output_class")

    def forward(self, inputs):
        x = self.base_model(inputs)
        x = F.avg_pool2d(x, kernel_size=(x.shape[2], x.shape[3]))   
        corner_output = self.corner(x)
        border_output = self.border(x)
        cls_output = self.cls(x)
        return corner_output, border_output, cls_output


class OutputBranch(nn.Module):
    def __init__(self, in_size, out_size, name=None):
        super(OutputBranch, self).__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_size, out_size)

    def forward(self, inputs):
        inputs = self.flatten(inputs)
        output = self.fc(inputs)
        return output


if __name__ == "__main__":
    import numpy as np

    xx = torch.zeros((1, 3, 224, 224), dtype=torch.float32)
    model = LDRNet()
    y = model(xx)
    print(y)