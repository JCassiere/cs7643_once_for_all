import torch
from torch import nn
import torch.nn.functional as F
MAX_KERNEL_SIZE = 7
MAX_CHANNELS = 6
PADDING = 1
FIRST_LAYER_STRIDE = 2
SUBSEQUENT_LAYER_STRIDE = 1
MAX_DEPTH = 4
DEFAULT_KERNEL_SIZES = [MAX_KERNEL_SIZE for _ in range(MAX_DEPTH)]
DEFAULT_WIDTHS = [MAX_CHANNELS for _ in range(MAX_DEPTH)]

class SampleLayerConfiguration:
    def __init__(self, input_channels, kernel_size=MAX_KERNEL_SIZE, width=MAX_CHANNELS):
        self.kernel_size = kernel_size
        self.width = width
        self.input_channels = input_channels

class SampleUnitConfiguration:
    def __init__(self,
                 kernel_sizes=DEFAULT_KERNEL_SIZES,
                 widths=DEFAULT_WIDTHS,
                 depth=MAX_DEPTH):
        self.kernel_sizes = kernel_sizes
        self.widths = widths
        self.depth = depth


class DynamicConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.base_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                   kernel_size=(MAX_KERNEL_SIZE, MAX_KERNEL_SIZE),
                                   padding=PADDING, stride=(stride, stride))
        self.five_by_five_transformation = nn.Parameter(torch.zeros((25, 25)))
        self.three_by_three_transformation = nn.Parameter(torch.zeros(9, 9))
        
    def forward(self, x: torch.Tensor, config: SampleLayerConfiguration):
        k = config.kernel_size
        out_w = config.width
        in_w = config.input_channels
        weights = self.base_conv.weight[:out_w, :in_w, :, :]
        (n, c, _, _) = weights.shape
        if k == 7:
            weights = weights[:, :, :, :]
        elif k == 5:
            weights = weights[:, :, 1:6, 1:6].view(n, c, 1, 25)
            weights = weights * self.five_by_five_transformation
            weights = weights.view(n, c, 5, 5)
        elif k == 3:
            weights = weights[:, :, 2:5, 2:5].view(n, c, 1, 9)
            weights = weights * self.three_by_three_transformation
            weights = weights.view(n, c, 3, 3)
        else:
            raise ValueError("Invalid kernel size supplied to DynamicConvLayer")

        # choose channels before or after shrinking kernel?
        if out_w < MAX_CHANNELS:
            channel_importances = torch.norm(weights)
        
        # TODO - pick best channels
        # Get appropriate weights and use torch.Functional
        return F.conv2d(x, weights, self.base_conv.bias[:out_w],
                        self.base_conv.stride, self.base_conv.padding)
        
        
class ConvUnit(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Conv2d(in_channels=in_channels, out_channels=MAX_CHANNELS, kernel_size=MAX_KERNEL_SIZE,
                      padding=PADDING, stride=FIRST_LAYER_STRIDE),
            nn.Conv2d(in_channels=MAX_CHANNELS, out_channels=MAX_CHANNELS, kernel_size=MAX_KERNEL_SIZE,
                      padding=PADDING, stride=SUBSEQUENT_LAYER_STRIDE),
            nn.Conv2d(in_channels=MAX_CHANNELS, out_channels=MAX_CHANNELS, kernel_size=MAX_KERNEL_SIZE,
                      padding=PADDING, stride=SUBSEQUENT_LAYER_STRIDE),
            nn.Conv2d(in_channels=MAX_CHANNELS, out_channels=MAX_CHANNELS, kernel_size=MAX_KERNEL_SIZE,
                      padding=PADDING, stride=SUBSEQUENT_LAYER_STRIDE)
        ])
        
    # def get_active_unit(self, sample_unit_configuration):
    
    


class OnceForAll:
    def __init__(self):
        self.units = nn.ModuleList([
            ConvUnit(6),
            ConvUnit(6),
            ConvUnit(6),
            ConvUnit(6),
            ConvUnit(6)
        ])
