import torch
import torch.nn as nn
from collections import OrderedDict
import os
import numpy as np
from progressive_shrinking import get_network_config
from mobilenetv3_ofa import MobileNetV3OFA

class AccNet(nn.Module):
    def __init__(self, hidden_size, layers, device, checkpoint=None):
        super(AccNet, self).__init__()

        hidden_layers = [nn.Linear(hidden_size, hidden_size) for i in range(layers-1)]
        hidden_layers.insert(nn.Linear(3, hidden_size), 0)
        relus = [nn.ReLU(inplace=True) for i in range(layers)]
        overall = [arr[i] for arr in (hidden_layers, relus) for i in range(layers)]
        overall.append(nn.Linear(hidden_size, 1, bias=False))
        self.model = nn.Sequential(OrderedDict(overall))
        
        self.base = nn.Parameter(torch.zeros(1, device=device), requires_grad=False)

        if checkpoint is not None and os.path.exists(checkpoint):
            point = torch.load(checkpoint)
            if "state_dict" in point:
                point = point["state_dict"]
            self.load_state_dict(point)
        
        self.model = self.model.to(device)

    def forward(self, d, k, e):
        x = self.model(torch.tensor([d, k, e])).squeeze() + self.base
        return x



class AccNetTrainer():
    def __init__(self, num_samples, net, dataloader, batch_size, num_blocks = 5, kernel_choices = [3, 5, 7], depth_choices = [2, 3, 4], expansion_ratio_choices = [3, 4, 6]):
        self.n_samples = num_samples
        self.net = net
        self.batchsize = batch_size
        self.dataloader = dataloader
        self.model = AccNet()
        self.num_blocks = num_blocks
        self.kernel_choices = kernel_choices
        self.depth_choices = depth_choices
        self.expansion_ratio_choices = expansion_ratio_choices

    def train(self):
        acc_batch = []
        config_batch = []
        opt = torch.optim.Adam(self.model.parameters(), lr=0.05)
        for i in range(self.n_samples):
            config = get_network_config(self.num_blocks, self.kernel_choices, self.depth_choices, self.expansion_ratio_choices)
            depths = config['depths']
            kernels = config['kernel_sizes']
            expansion_ratios = config['expansion_ratios']
            acc_batch.append(self.net(next(iter(self.dataloader)), depths, kernels, expansion_ratios))
            config_batch.append([depths, kernels, expansion_ratios])
            if len(acc_batch) == self.batchsize:
                config_batch = np.array(config_batch)
                res = self.model(torch.tensor(config_batch[:, 0]), torch.tensor(config_batch[:, 1]), torch.tensor(config_batch[:, 2]))
                loss = nn.CrossEntropyLoss()(res, acc_batch)
                loss.backward()
                opt.step()
                acc_batch = []
                config_batch = []

