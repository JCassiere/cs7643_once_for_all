from ofa.model_arch import ModelArch
import torch
import torch.nn as nn
from collections import OrderedDict
import os
import numpy as np
from ofa.progressive_shrinking import get_network_config
from ofa.mobilenetv3_ofa import MobileNetV3OFA
from torch.utils.data import DataLoader
from tqdm import tqdm

class AccNet(nn.Module):
    def __init__(self, device, num_blocks = 5, kernel_choices = [3, 5, 7], depth_choices = [2, 3, 4], expansion_ratio_choices = [3, 4, 6], hidden_size=300, layers=3, checkpoint=None):
        super(AccNet, self).__init__()
        input_dim = num_blocks * (len(depth_choices) + len(kernel_choices) * max(depth_choices) + len(expansion_ratio_choices) * max(depth_choices))
        
        hidden_layers = [nn.Linear(hidden_size, hidden_size) for i in range(layers-1)]
        hidden_layers.insert(0, nn.Linear(input_dim, hidden_size))
        relus = [nn.ReLU(inplace=False) for i in range(layers)]
        overall = [arr[i] for arr in (hidden_layers, relus) for i in range(layers)]
        overall.append(nn.Linear(hidden_size, 1, bias=False))
        self.model = nn.Sequential(*overall)
        self.model = self.model.double()
        self.base = nn.Parameter(torch.zeros(1, device=device, dtype=torch.double), requires_grad=False)

        if checkpoint is not None and os.path.exists(checkpoint):
            point = torch.load(checkpoint)
            if "state_dict" in point:
                point = point["state_dict"]
            self.load_state_dict(point)
        
        self.model = self.model.to(device)

    def forward(self, x):
        x = self.model(x).squeeze() + self.base
        return x



class AccNetTrainer():
    def __init__(self, num_samples, net, dataloader, batch_size, device, num_blocks = 5, kernel_choices = [3, 5, 7], depth_choices = [2, 3, 4], expansion_ratio_choices = [3, 4, 6]):
        self.n_samples = num_samples
        self.net = net
        self.batchsize = batch_size
        self.dataloader = dataloader
        self.model = AccNet(device=device, num_blocks=num_blocks, kernel_choices=kernel_choices, depth_choices=depth_choices, expansion_ratio_choices=expansion_ratio_choices)
        self.num_blocks = num_blocks
        self.kernel_choices = kernel_choices
        self.depth_choices = depth_choices
        self.expansion_ratio_choices = expansion_ratio_choices

    def train(self):
        
        opt = torch.optim.Adam(self.model.parameters(), lr=0.05)
        for img, label in tqdm(self.dataloader):
            config = get_network_config(self.num_blocks, self.kernel_choices, self.depth_choices, self.expansion_ratio_choices)
            arch = ModelArch(config_dict=config, n=self.num_blocks, d_c=self.depth_choices, k_c=self.kernel_choices, e_c=self.expansion_ratio_choices)
            accs = self.net(img)
            accs = nn.CrossEntropyLoss()(accs, label)
            res = self.model(arch.get_arch_rep())
            torch.autograd.set_detect_anomaly(True)
            loss = nn.MSELoss()(res, accs.double())
            loss.backward()
            opt.step()

