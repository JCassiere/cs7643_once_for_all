from ofa.evolutionary_search import EvoSearch
import torch
from ofa.datasets import get_dataloaders
from ofa.utils import get_device
import random
from ofa.progressive_shrinking import get_network_config
from ofa.model_arch import ModelArch

net = torch.load("checkpoint/elastic_width_stage2.pt", map_location=torch.device('cpu'))
search = EvoSearch(2, 2, 2)
# random.seed(100)
num_blocks = 5
kernel_choices = [3, 5, 7]
depth_choices = [2, 3, 4]
expansion_ratio_choices = [3, 4, 6]

config = get_network_config(num_blocks, kernel_choices, depth_choices, expansion_ratio_choices)
model = ModelArch(config,n=num_blocks, d_c=depth_choices, k_c=kernel_choices, e_c=expansion_ratio_choices)
print(model.depth)
print(model.kernel)
print(model.expansion_ratio)
print(model.get_arch_rep().shape)

# blocks * (len(depth_c) + len(kernel_c) * max(depth_c) + len(exp_r) * max(depth_c))
# device = get_device()
# batchsize = 64
# train_data_loader, test_data_loader = get_dataloaders(device)

# result = search.search(net, train_data_loader, batchsize, num_blocks, kernel_choices, depth_choices, expansion_ratio_choices)