import random
from ofa.accuracy_network import AccNetTrainer
from ofa.model_arch import ModelArch
import copy
from ofa.progressive_shrinking import get_network_config


class EvoSearch:
    def __init__(self, pop: int, cycles: int, samples: int) -> None:
        self.P = pop
        self.C = cycles
        self.S = samples

    def search(self, net, loader, device, batchsize=64, num_blocks = 5, kernel_choices = [3, 5, 7], depth_choices = [2, 3, 4], expansion_ratio_choices = [3, 4, 6]):
        population = []
        history = []
        acc_net_trainer = AccNetTrainer(net=net, num_samples=64, dataloader=loader, batch_size=batchsize, num_blocks=num_blocks, device=device, kernel_choices=kernel_choices, depth_choices=depth_choices, expansion_ratio_choices=expansion_ratio_choices)
        acc_net_trainer.train()
        
        for _ in range(self.P):
            model = random.choice(acc_net_trainer.arch_list)
            model.acc = acc_net_trainer.model(model.get_arch_rep())
            population.append(model)
            history.append(model)
        
        for _ in range(self.C):
            sample = []
            for j in range(self.S):
                candidate = random.choice(population)
                sample.append(candidate)
            
            parent = max(sample, key=lambda x: x.acc)
            child_config_dict = self.mutate(parent.config_dict, num_blocks, kernel_choices, depth_choices, expansion_ratio_choices)
            child = ModelArch(name=parent.name+"_mutated", config_dict=child_config_dict, n=num_blocks, d_c=depth_choices, k_c=kernel_choices, e_c=expansion_ratio_choices)
            child.acc = acc_net_trainer.model(child.get_arch_rep())
            population.append(child)
            history.append(child)
            dead = population.pop(0)
            del dead

        return max(history, key=lambda x: x.acc), history

    def mutate(self, config_dict, num_blocks, kernel_choices, depth_choices, expansion_ratio_choices):
        new_dict = copy.deepcopy(config_dict)
        block = random.randrange(num_blocks)
        depth = random.choice(depth_choices)
        block_kernels = []
        block_expansion_ratios = []
        for _ in range(depth):
            block_kernels.append(random.choice(kernel_choices))
            block_expansion_ratios.append(random.choice(expansion_ratio_choices))
        new_dict["depths"][block] = depth
        new_dict["kernel_sizes"][block] = block_kernels
        new_dict["expansion_ratios"][block] = block_expansion_ratios

        return new_dict


