from ofa.evolutionary_search import EvoSearch
import torch
from ofa.datasets import get_dataloaders
from ofa.utils import get_device
import random
from ofa.progressive_shrinking import get_network_config
from ofa.model_arch import ModelArch
from ofa.experiment import Experiment
import time

device = get_device()
exp_kwargs = {
        "dataset_name": "cifar10",
        "experiment_name": "big_net_only_cifar10_{}".format(int(time.time()))
    }

exp = Experiment(**exp_kwargs)
exp.net.load_state_dict(torch.load("checkpoint/elastic_width_stage_2.pt", map_location=torch.device(device)))
search = EvoSearch(32, 32, 32)
# random.seed(100)
num_blocks = 5
kernel_choices = [3, 5, 7]
depth_choices = [2, 3, 4]
expansion_ratio_choices = [3, 4, 6]

config = get_network_config(num_blocks, kernel_choices, depth_choices, expansion_ratio_choices)
model = ModelArch(config,n=num_blocks, d_c=depth_choices, k_c=kernel_choices, e_c=expansion_ratio_choices)
# print(model.depth)
# print(model.kernel)
# print(model.expansion_ratio)
# print(model.get_arch_rep().shape)

# blocks * (len(depth_c) + len(kernel_c) * max(depth_c) + len(exp_r) * max(depth_c))

batchsize = 64
train_data_loader, test_data_loader = get_dataloaders(device, dataset_name="cifar10")

result, total_list = search.search(net=exp.net, loader=train_data_loader, batchsize=batchsize, num_blocks=num_blocks, device=device, kernel_choices=kernel_choices, depth_choices=depth_choices, expansion_ratio_choices=expansion_ratio_choices)

print(result)

for i in range(0, 10):
    print(total_list[i])