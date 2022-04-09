from typing import List

import torch
from torch import nn
import torch.nn.functional as F
MAX_KERNEL_SIZE = 7
MAX_WIDTH = 6
PADDING = 1
FIRST_LAYER_STRIDE = 2
SUBSEQUENT_LAYER_STRIDE = 1
MAX_DEPTH = 4
DEFAULT_KERNEL_SIZES = [MAX_KERNEL_SIZE for _ in range(MAX_DEPTH)]
DEFAULT_WIDTHS = [MAX_WIDTH for _ in range(MAX_DEPTH)]


class SampleLayerConfiguration:
    def __init__(self, input_channels, kernel_size=MAX_KERNEL_SIZE, width=MAX_WIDTH):
        self.kernel_size = kernel_size
        self.width = width
        self.input_channels = input_channels


class SampleUnitConfiguration:
    def __init__(self,
                 unit_input_channels=MAX_WIDTH,
                 kernel_sizes=DEFAULT_KERNEL_SIZES,
                 widths=DEFAULT_WIDTHS,
                 depth=MAX_DEPTH):
        self.unit_input_channels = unit_input_channels
        self.kernel_sizes = kernel_sizes
        self.widths = widths
        self.depth = depth


class DynamicConvLayer(nn.Module):
    def __init__(self, stride=1):
        super().__init__()
        self.base_conv = nn.Conv2d(in_channels=MAX_WIDTH, out_channels=MAX_WIDTH,
                                   kernel_size=(MAX_KERNEL_SIZE, MAX_KERNEL_SIZE),
                                   padding=PADDING, stride=(stride, stride))
        self.five_by_five_transformation = nn.Parameter(torch.zeros((25, 25)))
        self.three_by_three_transformation = nn.Parameter(torch.zeros(9, 9))
        
    def forward(self, x: torch.Tensor, config: SampleLayerConfiguration):
        k = config.kernel_size
        out_w = config.width
        in_w = config.input_channels
        # TODO - pick best channels
        #  choose channels before or after shrinking kernel?
        #  Do channels just get resorted at each stage of progressive shrinking? Or
        #  do they get chosen during the forward pass of each minibatch?
        #  I think they get chosen at each stage of progressive shrinking, but I'm not sure
        weights = self.base_conv.weight[:out_w, :in_w, :, :]
        # if out_w < MAX_CHANNELS:
        #     channel_importances = torch.norm(self.base_conv.weight, p=1, dim=0)

        (n, c, _, _) = weights.shape
        if k == 7:
            self.base_conv.padding = 3
            weights = weights[:, :, :, :]
        # lay out the kernels in 1D vectors
        # perform matrix multiplication of these laid out kernels by the appropriate transformation
        # matrices, then reshape the product
        # for example, for a 5x5 kernel, lay it out to be 1x25
        # then multiply by the 25x25 transformation matrix to get a transformed 1x25 matrix
        # then view that as a 5x5 matrix, which is your kernel
        elif k == 5:
            self.base_conv.padding = 2
            weights = weights[:, :, 1:6, 1:6].view(n, c, 1, 25)
            weights = torch.matmul(weights, self.five_by_five_transformation).view(n, c, 5, 5)
        elif k == 3:
            self.base_conv.padding = 1
            weights = weights[:, :, 2:5, 2:5].view(n, c, 1, 9)
            weights = torch.matmul(weights, self.three_by_three_transformation).view(n, c, 3, 3)
        else:
            raise ValueError("Invalid kernel size supplied to DynamicConvLayer")

        return F.conv2d(x, weights, self.base_conv.bias[:out_w],
                        self.base_conv.stride, self.base_conv.padding)
        
        
class ConvUnit(nn.Module):
    # TODO - add activation functions
    #  and batchnorm?
    def __init__(self, first_layer_stride=2):
        super().__init__()
        # First layer in a unit has stride = 2 in order to decrease feature map size
        self.layers = nn.ModuleList(
            [DynamicConvLayer(stride=first_layer_stride)] + [DynamicConvLayer() for _ in range(MAX_DEPTH - 1)]
        )
        
    def forward(self, x, sample_unit_config: SampleUnitConfiguration):
        layer_configs = []
        first_layer_config = SampleLayerConfiguration(input_channels=sample_unit_config.unit_input_channels,
                                                      kernel_size=sample_unit_config.kernel_sizes[0],
                                                      width=sample_unit_config.widths[0])
        layer_configs.append(first_layer_config)
        for i in range(1, sample_unit_config.depth):
            config = SampleLayerConfiguration(input_channels=sample_unit_config.widths[i - 1],
                                              kernel_size=sample_unit_config.kernel_sizes[i],
                                              width=sample_unit_config.widths[i])
            layer_configs.append(config)
        
        for i in range(sample_unit_config.depth):
            x = self.layers[i].forward(x, layer_configs[i])
            
        return x


class OnceForAll:
    def __init__(self, num_classes, max_image_size=(256, 256), unit_first_layer_strides=[2, 2, 2, 2, 2]):
        self.conv_units = nn.ModuleList([ConvUnit(x) for x in unit_first_layer_strides])
        max_linear_weights = max_image_size[0] * max_image_size[1] * MAX_WIDTH
        self.linear_classifier = nn.Linear(max_linear_weights, num_classes)
        self.flatten = nn.Flatten()

    def forward(self, x, minibatch_config: List[SampleUnitConfiguration]):
        for i in range(len(self.conv_units)):
            x = self.conv_units[i].forward(x, minibatch_config[i])
        
        final_feature_map_size = x.shape[-1] * x.shape[-2]
        num_channels_last_layer = minibatch_config[-1].widths[-1]
        num_conv_output_weights = num_channels_last_layer * final_feature_map_size
        
        x = self.flatten(x)
        linear_weights = self.linear_classifier.weight[:, :num_conv_output_weights]
        x = F.softmax(F.linear(x, linear_weights, self.linear_classifier.bias), dim=1)
        return x
