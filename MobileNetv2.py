from turtle import width
import torch.nn as nn
import math
import torch
import numpy as np

def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def make_divisible(x, divisible_by=8):
    import numpy as np
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class regression_loss(nn.Module):
    def __init__(self):
        super().__init__()
        
class LineLoss(nn.Module):
    def __init__(self,Beta,Gamma):
        super().__init__()
        self.beta = Beta
        self.gamma = Gamma

    def __call__(self, line):
        line_x = line[:, 0::2]              
        # line_x = {{all x co-ordinates of left edge},{all x co-ordinates of right edge},...}
        line_y = line[:, 1::2]             
        # line_y = {{all y co-ordinates of left edge},{all y co-ordinates of right edge},...}
        x_diff = line_x[:, 1:] - line_x[:, 0:-1]         
        # forming x-component of n-4 vectors formed from n points
        y_diff = line_y[:, 1:] - line_y[:, 0:-1]         
        # forming y-component of n-4 vectors formed from n points
        x_diff_start = x_diff[:, 1:]
        x_diff_end = x_diff[:, 0:-1]
        y_diff_start = y_diff[:, 1:]
        y_diff_end = y_diff[:, 0:-1]
        similarity = (x_diff_start * x_diff_end + y_diff_start * y_diff_end) / (
                    torch.sqrt(torch.square(x_diff_start) + torch.square(y_diff_start)+ 0.0000000000001) *torch.sqrt(torch.square(x_diff_end) + torch.square(y_diff_end) + 0.0000000000001))
        # 0.0000000000001 is for ensuring that the denominator does not become zero
        slop_loss = torch.mean(1 - similarity, axis=1)
        x_diff_loss = torch.mean(torch.abs(torch.abs(x_diff[:, 1:]) - torch.abs(x_diff[:, 0:-1])), 1)
        y_diff_loss = torch.mean(torch.abs(torch.abs(y_diff[:, 1:]) - torch.abs(y_diff[:, 0:-1])), 1)
        sim_loss = torch.sum(slop_loss)
        distance_loss = torch.sum(x_diff_loss + y_diff_loss)
        line_loss = self.beta*sim_loss + self.gamma*distance_loss
        return line_loss
    
    
class MobileNetV2(nn.Module):
    def __init__(self, n_class=1000, input_size=224, width_mult=1.):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        assert input_size % 32 == 0, "Fucks"
        # input_channel = make_divisible(input_channel * width_mult)  # first channel is always 32!
        self.last_channel = make_divisible(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = make_divisible(c * width_mult) if t > 1 else c
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))

        # make it nn.Sequential
        self.out1 = nn.Sequential(*self.features[:7])
        self.out2 = nn.Sequential(*self.features[7:10])
        self.out3 = nn.Sequential(*self.features[10:13])
        self.out4 = nn.Sequential(*self.features[13:14])
        self.out5 = nn.Sequential(*self.features[14:])
        self.AvgPool28_14 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.AvgPool14_7 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self._initialize_weights()

    def forward(self, x):
        #full = self.features(x)
        out1 = self.out1(x)
        out2 = self.out2(out1)
        out3 = self.out3(out2)
        out4 = self.out4(out3)
        out5 = self.out5(out4)
        print(self.AvgPool14_7(self.AvgPool28_14(out1)).shape)
        print(self.AvgPool14_7(out2).shape)
        print(self.AvgPool14_7(out3).shape)
        print(self.AvgPool14_7(out4).shape)
        print(out5.shape)
        return torch.mean(self.AvgPool14_7(self.AvgPool28_14(out1)), dim=1) +  torch.mean(self.AvgPool14_7(out2), dim=1) +  torch.mean(self.AvgPool14_7(out3), dim=1)+torch.mean(self.AvgPool14_7(out4), dim=1) +  torch.mean(out5, dim=1)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

if __name__ == "__main__":
    tmp = np.random.rand(1, 3, 224 ,224).astype(np.float32)
    image = torch.from_numpy(tmp)
    model = MobileNetV2()
    model.forward(image)