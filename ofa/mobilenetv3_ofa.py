"""
Uses https://github.com/d-li14/mobilenetv3.pytorch as starting point

Also uses https://github.com/leaderj1001/MobileNetV3-Pytorch for reference

Also https://github.com/zeiss-microscopy/BSConv/blob/master
"""

import torch.nn as nn
import torch.nn.functional as F
import math


__all__ = ['mobilenetv3_large', 'mobilenetv3_small']

import torch.nn.init
from collections import OrderedDict


def same_padding(kernel_size):
    return (kernel_size - 1) / 2


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class DynamicSELayer(nn.Module):
    def __init__(self, channels, reduction=4):
        super(DynamicSELayer, self).__init__()
        self.reduction = reduction
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        hidden_channels = _make_divisible(channels // self.reduction, 8)
        self.squeeze = nn.Linear(channels, hidden_channels)
        self.relu = nn.ReLU(inplace=True)
        self.excite = nn.Linear(hidden_channels, channels)
        self.hsigmoid = nn.Hardsigmoid(inplace=True)
    
    def forward(self, x):
        batch, in_channels, _, _ = x.size()
        hidden_channels = _make_divisible(in_channels // self.reduction, 8)
        y = self.avg_pool(x).view(batch, in_channels)
        y = F.linear(y, self.squeeze.weight[:hidden_channels, :in_channels], self.squeeze.bias[:hidden_channels])
        y = self.relu(y)
        y = F.linear(y, self.excite.weight[:in_channels, :hidden_channels], self.excite.bias[:in_channels])
        y = self.hsigmoid(y).view(batch, in_channels, 1, 1)
        return x * y


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.Hardswish(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.Hardswish(inplace=True)
    )


# class SampleLayerConfiguration:
#     def __init__(self, input_channels, kernel_size=MAX_KERNEL_SIZE, width=MAX_WIDTH):
#         self.kernel_size = kernel_size
#         self.width = width
#         self.input_channels = input_channels

class DynamicBatchNorm(nn.Module):
    def __init__(self, max_dim):
        super(DynamicBatchNorm, self).__init__()
        self.base_batch_norm = nn.BatchNorm2d(max_dim)
        
    def forward(self, x):
        # Adapted from pytorch source code:
        # https://github.com/pytorch/pytorch/blob/10c4b98ade8349d841518d22f19a653a939e260c/torch/nn/modules/batchnorm.py#L58-L81
        dim = x.size(1)

        if self.base_batch_norm.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.base_batch_norm.momentum

        if self.base_batch_norm.training and self.base_batch_norm.track_running_stats:
            if self.base_batch_norm.num_batches_tracked is not None:
                self.base_batch_norm.num_batches_tracked += 1
                if self.base_batch_norm.momentum is None:
                    exponential_average_factor = 1.0 / float(self.base_batch_norm.num_batches_tracked)
                else:
                    exponential_average_factor = self.base_batch_norm.momentum
        return F.batch_norm(
            x,
            self.base_batch_norm.running_mean[:dim],
            self.base_batch_norm.running_var[:dim],
            self.base_batch_norm.weight[:dim],
            self.base_batch_norm.bias[:dim],
            self.base_batch_norm.training or not self.base_batch_norm.track_running_stats,
            exponential_average_factor,
            self.base_batch_norm.eps,
        )


class DynamicTransformableConv(nn.Module):
    def __init__(self, channels, max_kernel_size, stride=1):
        super(DynamicTransformableConv, self).__init__()
        self.base_conv = nn.Conv2d(channels, kernel_size=max_kernel_size,
                                   padding=same_padding(max_kernel_size), stride=stride,
                                   groups=channels, bias=False)
        self.five_by_five_transformation = nn.Parameter(torch.zeros((25, 25)))
        self.three_by_three_transformation = nn.Parameter(torch.zeros((9, 9)))
    
    def forward(self, x: torch.Tensor, kernel_size, channels):
        # TODO - pick best channels
        #  choose channels before or after shrinking kernel?
        #  Do channels just get re-sorted at each stage of progressive shrinking? Or
        #  do they get chosen during the forward pass of each minibatch?
        #  I think they get chosen at each stage of progressive shrinking, but I'm not sure
        weights = self.base_conv.weight[:channels, :channels, :, :]
        # if out_w < MAX_CHANNELS:
        #     channel_importances = torch.norm(self.base_conv.weight, p=1, dim=0)
        
        (n, c, _, _) = weights.shape
        if kernel_size == 7:
            weights = weights
        # lay out the kernels in 1D vectors
        # perform matrix multiplication of these laid out kernels by the appropriate transformation
        # matrices, then reshape the product
        # for example, for a 5x5 kernel, lay it out to be 1x25
        # then multiply by the 25x25 transformation matrix to get a transformed 1x25 matrix
        # then view that as a 5x5 matrix, which is your kernel
        elif kernel_size == 5:
            weights = weights[:, :, 1:6, 1:6].view(n, c, 1, 25)
            weights = torch.matmul(weights, self.five_by_five_transformation).view(n, c, 5, 5)
        elif kernel_size == 3:
            weights = weights[:, :, 2:5, 2:5].view(n, c, 1, 9)
            weights = torch.matmul(weights, self.three_by_three_transformation).view(n, c, 3, 3)
        else:
            raise ValueError("Invalid kernel size supplied to DynamicConvLayer")
        
        # TODO - weight standardization?
        return F.conv2d(x, weights, None,
                        self.base_conv.stride, padding=same_padding(kernel_size))


class DynamicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(DynamicConv, self).__init__()
        self.base_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                   kernel_size=(kernel_size, kernel_size),
                                   padding=same_padding(kernel_size), stride=(stride, stride),
                                   bias=False)
        
    def forward(self, x, in_width, out_width):
        # TODO - pick best channels
        weights = self.base_conv.weight[:out_width, :in_width, :, :]
        return F.conv2d(x, weights, bias=None,
                        stride=self.base_conv.stride, padding=self.base_conv.padding)
    
        
class DynamicBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DynamicBlock, self).__init__()
        self.blocks = [DynamicInvertedResidual() for _ in range(5)]
    
        
class DynamicInvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, max_expansion_ratio=6,
                 max_kernel_size=7, stride=1, use_se=True, use_hs=True):
        super(DynamicInvertedResidual, self).__init__()
        assert stride in [1, 2]
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.identity = stride == 1 and in_channels == out_channels
        max_hidden_channels = _make_divisible(in_channels * max_expansion_ratio, 8)
        
        self.activation = nn.Hardswish(inplace=True) if use_hs else nn.ReLU(inplace=True)
        self.bottleneck = DynamicConv(in_channels, max_hidden_channels, 1, 1)
        self.bottleneck_norm = DynamicBatchNorm(max_hidden_channels)
        self.depthwise_convolution = DynamicTransformableConv(max_hidden_channels, max_kernel_size, stride)
        self.depthwise_norm = DynamicBatchNorm(max_hidden_channels)
        self.se_layer = DynamicSELayer(max_hidden_channels) if use_se else nn.Identity()
        self.pointwise_conv = DynamicConv(max_hidden_channels, out_channels, 1, 1)
        self.pointwise_norm = DynamicBatchNorm(out_channels)
    
    def forward(self, x, kernel_size, width_expansion_ratio):
        hidden_channels = _make_divisible(self.in_channels * width_expansion_ratio, 8)
        y = self.bottleneck.forward(x, self.in_channels, hidden_channels)
        y = self.bottleneck_norm.forward(y)
        y = self.activation(y)
        y = self.depthwise_convolution.forward(y, kernel_size, hidden_channels)
        y = self.depthwise_norm.forward(y)
        y = self.activation(y)
        y = self.se_layer.forward(y)
        y = self.pointwise_conv.forward(y, hidden_channels, self.out_channels)
        y = self.pointwise_norm.forward(y)
        y = self.activation(y)
        
        if self.identity:
            return x + y
        else:
            return y


class MobileNetV3(nn.Module):
    def __init__(self, cfgs, mode, num_classes=1000, width_mult=1.):
        super(MobileNetV3, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = cfgs
        assert mode in ['large', 'small']
        
        # building first layer
        input_channel = _make_divisible(16 * width_mult, 8)
        layers = [conv_3x3_bn(3, input_channel, 1)]
        # building inverted residual blocks
        for k, t, c, use_se, use_hs, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 8)
            exp_size = _make_divisible(input_channel * t, 8)
            layers.append(DynamicInvertedResidual(input_channel, exp_size, output_channel, k, s, use_se, use_hs))
            input_channel = output_channel
        self.features = nn.Sequential(*layers)
        # building last several layers
        self.conv = conv_1x1_bn(input_channel, exp_size)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        output_channel = {'large': 1280, 'small': 1024}
        output_channel = _make_divisible(output_channel[mode] * width_mult, 8) if width_mult > 1.0 else output_channel[mode]
        self.classifier = nn.Sequential(
            nn.Linear(exp_size, output_channel),
            nn.Hardswish(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(output_channel, num_classes)
        )
        
        self._initialize_weights()
    
    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def mobilenetv3_large(**kwargs):
    """
    Constructs a MobileNetV3-Large model
    """
    cfgs = [
        # k, t, c, SE, HS, s
        [3,   1,  16, 0, 0, 1],
        [3,   4,  24, 0, 0, 2],
        [3,   3,  24, 0, 0, 1],
        [5,   3,  40, 1, 0, 2],
        [5,   3,  40, 1, 0, 1],
        [5,   3,  40, 1, 0, 1],
        [3,   6,  80, 0, 1, 2],
        [3, 2.5,  80, 0, 1, 1],
        [3, 2.3,  80, 0, 1, 1],
        [3, 2.3,  80, 0, 1, 1],
        [3,   6, 112, 1, 1, 1],
        [3,   6, 112, 1, 1, 1],
        [5,   6, 160, 1, 1, 2],
        [5,   6, 160, 1, 1, 1],
        [5,   6, 160, 1, 1, 1]
    ]
    return MobileNetV3(cfgs, mode='large', **kwargs)


def mobilenetv3_small(**kwargs):
    """
    Constructs a MobileNetV3-Small model
    """
    cfgs = [
        # k, t, c, SE, HS, s
        [3,    1,  16, 1, 0, 1],
        [3,  4.5,  24, 0, 0, 1],
        [3, 3.67,  24, 0, 0, 1],
        [5,    4,  40, 1, 1, 2],
        [5,    6,  40, 1, 1, 1],
        [5,    3,  48, 1, 1, 1],
        [5,    3,  48, 1, 1, 1],
        [5,    6,  96, 1, 1, 1],
        [5,    6,  96, 1, 1, 1],
    ]
    
    return MobileNetV3(cfgs, mode='small', **kwargs)
