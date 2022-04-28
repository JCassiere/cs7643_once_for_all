from pyexpat import model
import numpy as np
import torch
class ModelArch:
    names = 1
    def __init__(self, config_dict, n, d_c, k_c, e_c, name = None, acc = None) -> None:
        self.name = str(ModelArch.names) if name is None else name
        self.accuracy = acc
        self.config_dict = config_dict
        self.blocks = n
        self.depth_c = d_c
        self.kernel_c = k_c
        self.expansion_ratio_c = e_c
        ModelArch.names += 1

    @property
    def depth(self):
        return self.config_dict['depths']
    
    @property
    def kernel(self):
        return self.config_dict['kernel_sizes']

    @property
    def expansion_ratio(self):
        return self.config_dict['expansion_ratios']

    def get_arch_rep(self):
        depths = np.zeros((self.blocks, len(self.depth_c)))
        kernels = []
        expansion_r = []
        for i in range(self.blocks):
            depths[i][self.depth_c.index(self.depth[i])] = 1
            kernel = self.kernel[i]
            er = self.expansion_ratio[i]
            for j in range(self.depth[i]):
                vec = np.zeros(len(self.kernel_c))
                vec2 = np.zeros(len(self.expansion_ratio_c))
                vec[self.kernel_c.index(kernel[j])] = 1
                vec2[self.expansion_ratio_c.index(er[j])] = 1
                kernels.append(vec)
                expansion_r.append(vec2)
            for j in range(self.depth[i], max(self.depth_c)):
                vec = np.zeros(len(self.kernel_c))
                vec2 = np.zeros(len(self.expansion_ratio_c))
                kernels.append(vec)
                expansion_r.append(vec2)
        
        kernels = np.array(kernels)
        expansion_r = np.array(expansion_r)
        
        return torch.tensor(np.concatenate((depths.flatten(), kernels.flatten(), expansion_r.flatten()), axis=0))