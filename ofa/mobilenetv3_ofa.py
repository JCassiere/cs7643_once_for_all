"""
Uses https://github.com/d-li14/mobilenetv3.pytorch as starting point

Also uses https://github.com/leaderj1001/MobileNetV3-Pytorch for reference

Also https://github.com/zeiss-microscopy/BSConv/blob/master
"""

import torch.nn as nn
import torch.nn.functional as F
import math


# __all__ = ['mobilenetv3_large', 'mobilenetv3_small']

import torch.nn.init
from collections import OrderedDict

def activation_function(use_hard_swish=True):
    if use_hard_swish:
        return nn.Hardswish(inplace=True)
    else:
        return nn.ReLU(inplace=True)

def same_padding(kernel_size):
    return kernel_size // 2


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
    
    def reorganize_channels(self, sorted_indices):
        # reorganize SE excite output channels according to importance of
        # pointwise linear inputs
        self.excite.weight.data = torch.index_select(
            self.excite.weight.data,
            dim=0,
            index=sorted_indices
        )
        self.excite.bias.data = torch.index_select(
            self.excite.bias.data,
            dim=0,
            index=sorted_indices
        )
        # SE squeeze will have the same number of inputs as the excite has outputs,
        # so reorganize the inputs the same way
        self.squeeze.weight.data = torch.index_select(
            self.squeeze.weight.data,
            dim=1,
            index=sorted_indices
        )
        self.squeeze.bias.data = torch.index_select(
            self.squeeze.bias.data,
            dim=1,
            index=sorted_indices
        )
        # Then reorganize the squeeze-excite "hidden" channels according to
        # their importances
        # do this by measuring importance of the input channels to the excitation layer
        importance = torch.sum(torch.abs(self.excite.weight), dim=(0, 2, 3))
        _, internal_sorted_indices = torch.sort(importance, dim=0, descending=True)
        self.excite.weight.data = torch.index_select(
            self.excite.weight.data,
            dim=1,
            index=internal_sorted_indices
        )
        self.excite.bias.data = torch.index_select(
            self.excite.bias.data,
            dim=1,
            index=internal_sorted_indices
        )
        self.squeeze.weight.data = torch.index_select(
            self.squeeze.weight.data,
            dim=0,
            index=internal_sorted_indices
        )
        self.squeeze.bias.data = torch.index_select(
            self.squeeze.bias.data,
            dim=0,
            index=internal_sorted_indices
        )


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
    
    def reorganize_weights(self, sorted_indices):
        self.base_batch_norm.weight.data = torch.index_select(
            self.base_batch_norm.weight.data,
            dim=0,
            index=sorted_indices
        )
        self.base_batch_norm.bias.data = torch.index_select(
            self.base_batch_norm.bias.data,
            dim=0,
            index=sorted_indices
        )


class DynamicDepthwiseConv(nn.Module):
    def __init__(self, channels, max_kernel_size, stride=1):
        super(DynamicDepthwiseConv, self).__init__()
        self.base_conv = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=max_kernel_size,
                                   padding=same_padding(max_kernel_size), stride=stride,
                                   groups=channels, bias=False)
        self.five_by_five_transformation = nn.Parameter(torch.zeros((25, 25)))
        self.three_by_three_transformation = nn.Parameter(torch.zeros((9, 9)))
    
    def forward(self, x: torch.Tensor, kernel_size, channels):
        # TODO - don't copy weights - mask somehow?
        (n, c, _, _) = self.base_conv.weight.shape
        if kernel_size == 7:
            weights = self.base_conv.weight[:channels, :channels, :, :]
        # lay out the kernels in 1D vectors
        # perform matrix multiplication of these laid out kernels by the appropriate transformation
        # matrices, then reshape the product
        # for example, for a 5x5 kernel, lay it out to be 1x25
        # then multiply by the 25x25 transformation matrix to get a transformed 1x25 matrix
        # then view that as a 5x5 matrix, which is your kernel
        elif kernel_size == 5:
            weights = self.base_conv.weight[:channels, :channels, 1:6, 1:6].view(n, c, 1, 25)
            weights = torch.matmul(weights, self.five_by_five_transformation).view(n, c, 5, 5)
        elif kernel_size == 3:
            weights = self.base_conv.weight[:channels, :channels, 2:5, 2:5].view(n, c, 1, 9)
            weights = torch.matmul(weights, self.three_by_three_transformation).view(n, c, 3, 3)
        else:
            raise ValueError("Invalid kernel size supplied to DynamicConvLayer")
        
        # TODO - weight standardization?
        return F.conv2d(x, weights, None,
                        self.base_conv.stride, padding=same_padding(kernel_size),
                        groups=channels)
    
    def reorganize_channels(self, sorted_indices, dim):
        self.base_conv.weight.data = torch.index_select(
            self.base_conv.weight.data,
            dim=dim,
            index=sorted_indices
        )


class DynamicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(DynamicConv, self).__init__()
        self.base_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                   kernel_size=(kernel_size, kernel_size),
                                   padding=same_padding(kernel_size), stride=(stride, stride),
                                   bias=False)
        
    def forward(self, x, in_width, out_width):
        weights = self.base_conv.weight[:out_width, :in_width, :, :]
        return F.conv2d(x, weights, bias=None,
                        stride=self.base_conv.stride, padding=self.base_conv.padding)
    
    def reorganize_channels(self, sorted_indices, dim):
        self.base_conv.weight.data = torch.index_select(
            self.base_conv.weight.data,
            dim=dim,
            index=sorted_indices
        )


class FirstInvertedResidual(nn.Module):
    '''
    The first inverted residual layer is not dynamic. It has an expansion ratio of 1,
    meaning, there is no linear bottleneck at the beginning. It uses a kernel size
    of 3, does not use squeeze-excite, and uses ReLU as its activation function.
    '''
    def __init__(self, in_channels, out_channels, stride=1):
        super(FirstInvertedResidual, self).__init__()
        kernel_size = 3
        self.add_residual = stride == 1 and in_channels == out_channels
        hidden_channels = _make_divisible(in_channels, 8)
        
        self.conv = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size, stride, padding=same_padding(kernel_size)),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, out_channels, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        if self.add_residual:
            return x + self.conv(x)
        else:
            return self.conv(x)


class DynamicInvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, max_kernel_size=7,
                 max_expansion_ratio=6, stride=1, use_se=True, use_hs=True):
        super(DynamicInvertedResidual, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_se = use_se
        
        self.add_residual = stride == 1 and in_channels == out_channels
        max_hidden_channels = _make_divisible(in_channels * max_expansion_ratio, 8)
        
        self.activation = nn.Hardswish(inplace=True) if use_hs else nn.ReLU(inplace=True)
        self.max_expansion_ratio = max_expansion_ratio
        self.bottleneck = DynamicConv(in_channels, max_hidden_channels, 1, 1)
        self.bottleneck_norm = DynamicBatchNorm(max_hidden_channels)
        self.depthwise_convolution = DynamicDepthwiseConv(max_hidden_channels, max_kernel_size, stride)
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
        
        if self.add_residual:
            return x + y
        else:
            return y

    def reorder_channels(self):
        # sort all channels based on the L1 norm of the input channels to the pointwise
        # linear convolution
        importance = torch.sum(torch.abs(self.pointwise_conv.base_conv.weight), dim=(0, 2, 3))
        sorted_values, sorted_indices = torch.sort(importance, dim=0, descending=True)
        self.pointwise_conv.reorganize_channels(sorted_indices, 1)
        self.depthwise_norm.reorganize_weights(sorted_indices)
        self.depthwise_convolution.reorganize_channels(sorted_indices, 0)
        if self.use_se:
            self.se_layer.reorganize_channels(sorted_indices)
            
        self.bottleneck_norm.reorganize_weights(sorted_indices)
        self.bottleneck.reorganize_channels(sorted_indices, 0)


class DynamicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, max_kernel_size=7, max_expansion_ratio=6, stride=1, use_se=True, use_hs=True):
        super(DynamicBlock, self).__init__()
        self.layers = [
            DynamicInvertedResidual(in_channels, out_channels, max_kernel_size, max_expansion_ratio,
                                    stride=stride, use_se=use_se, use_hs=use_hs)
        ] + [
            DynamicInvertedResidual(out_channels, out_channels, max_kernel_size, max_expansion_ratio,
                                    use_se=use_se, use_hs=use_hs) for _ in range(3)
        ]
        self.layers = nn.ModuleList(self.layers)
        
    def forward(self, x, depth, kernel_sizes, expansion_ratios):
        y = self.layers[0].forward(x, kernel_sizes[0], expansion_ratios[0])
        for i in range(1, depth):
            y = self.layers[i].forward(y, kernel_sizes[i], expansion_ratios[i])
        return y
    
    
class MobileNetV3OFA(nn.Module):
    def __init__(self, output_widths=None, use_squeeze_excites=None,
                 use_hard_swishes=None, strides=None, input_data_channels=3, num_classes=1000,
                 width_mult=1., max_kernel_size=7, max_expansion_ratio=6):
        super(MobileNetV3OFA, self).__init__()

        # setting of inverted residual blocks
        if output_widths is None:
            output_widths = [16, 16, 24, 40, 80, 112, 160, 960, 1280]
        if use_squeeze_excites is None:
            use_squeeze_excites = [False, False, True, False, True, True, True, True, True]
        if use_hard_swishes is None:
            use_hard_swishes = [True, False, False, True, True, True, True, True, True]
        if strides is None:
            strides = [1, 1, 2, 2, 1, 2, 1, 1, 1]
        
        output_widths = [_make_divisible(x * width_mult, 8) for x in output_widths]
        self.first_conv = nn.Sequential(
            nn.Conv2d(input_data_channels, output_widths[0],
                      kernel_size=3, stride=strides[0], padding=same_padding(3), bias=False),
            nn.BatchNorm2d(output_widths[0]),
            activation_function(use_hard_swishes[0])
        )
        self.first_inverted_residual = FirstInvertedResidual(output_widths[0], output_widths[1], strides[1])
        # building inverted residual blocks
        num_pre_block_layers = 2
        num_post_block_layers = 2
        self.blocks = []
        for i in range(num_pre_block_layers, len(output_widths) - num_post_block_layers):
            self.blocks.append(DynamicBlock(
                in_channels=output_widths[i - 1],
                out_channels=output_widths[i],
                max_kernel_size=max_kernel_size,
                max_expansion_ratio=max_expansion_ratio,
                stride=strides[i],
                use_se=use_squeeze_excites[i],
                use_hs=use_hard_swishes[i])
            )
        self.blocks = nn.ModuleList(self.blocks)
        self.final_conv = nn.Sequential(
            nn.Conv2d(output_widths[-3], output_widths[-2], 1, strides[-2], 0, bias=False),
            nn.BatchNorm2d(output_widths[-2]),
            activation_function(use_hard_swishes[-2])
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(output_widths[-2], output_widths[-1]),
            activation_function(use_hard_swishes[-1]),
            nn.Dropout(0.2),
            nn.Linear(output_widths[-1], num_classes)
        )
        
        self._initialize_weights()
    
    def forward(self, x, depths, kernel_sizes, expansion_ratios):
        y = self.first_conv(x)
        y = self.first_inverted_residual(y)
        for i in range(len(self.blocks)):
            y = self.blocks[i].forward(y, depths[i], kernel_sizes[i], expansion_ratios[i])
        y = self.final_conv(y)
        y = self.avgpool(y).view(y.size(0), -1)
        y = self.classifier(y)
        
        return y
    
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


# def mobilenetv3_large(**kwargs):
#     """
#     Constructs a MobileNetV3-Large model
#     """
#     cfgs = [
#         # k, t, c, SE, HS, s
#         [3,   1,  16, 0, 0, 1],
#         [3,   4,  24, 0, 0, 2],
#         [3,   3,  24, 0, 0, 1],
#         [5,   3,  40, 1, 0, 2],
#         [5,   3,  40, 1, 0, 1],
#         [5,   3,  40, 1, 0, 1],
#         [3,   6,  80, 0, 1, 2],
#         [3, 2.5,  80, 0, 1, 1],
#         [3, 2.3,  80, 0, 1, 1],
#         [3, 2.3,  80, 0, 1, 1],
#         [3,   6, 112, 1, 1, 1],
#         [3,   6, 112, 1, 1, 1],
#         [5,   6, 160, 1, 1, 2],
#         [5,   6, 160, 1, 1, 1],
#         [5,   6, 160, 1, 1, 1]
#     ]
#     return MobileNetV3(cfgs, mode='large', **kwargs)
#
#
# def mobilenetv3_small(**kwargs):
#     """
#     Constructs a MobileNetV3-Small model
#     """
#     cfgs = [
#         # k, t, c, SE, HS, s
#         [3,    1,  16, 1, 0, 1],
#         [3,  4.5,  24, 0, 0, 1],
#         [3, 3.67,  24, 0, 0, 1],
#         [5,    4,  40, 1, 1, 2],
#         [5,    6,  40, 1, 1, 1],
#         [5,    3,  48, 1, 1, 1],
#         [5,    3,  48, 1, 1, 1],
#         [5,    6,  96, 1, 1, 1],
#         [5,    6,  96, 1, 1, 1],
#     ]
#
#     return MobileNetV3(cfgs, mode='small', **kwargs)

def mobilenetv3_ofa(**kwargs):
    return MobileNetV3OFA(**kwargs)
